/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/poly_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class MultiClassNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("BBoxes"),
                   "Input(BBoxes) of MultiClassNMS should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Scores"),
                   "Input(Scores) of MultiClassNMS should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of MultiClassNMS should not be null.");

    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");

    PADDLE_ENFORCE_EQ(box_dims.size(), 3,
                      "The rank of Input(BBoxes) must be 3.");
    PADDLE_ENFORCE_EQ(score_dims.size(), 3,
                      "The rank of Input(Scores) must be 3.");
    PADDLE_ENFORCE(box_dims[2] == 4 || box_dims[2] == 8 || box_dims[2] == 16 
                || box_dims[2] == 24 || box_dims[2] == 32,
                   "The 2nd dimension of Input(BBoxes) must be 4 or 8, "
                   "represents the layout of coordinate "
                   "[xmin, ymin, xmax, ymax] or "
                   "4 points: [x1, y1, x2, y2, x3, y3, x4, y4] or "
                   "8 points: [xi, yi] i= 1,2,...,8 or "
                   "12 points: [xi, yi] i= 1,2,...,12 or "
                   "16 points: [xi, yi] i= 1,2,...,16");
    PADDLE_ENFORCE_EQ(box_dims[1], score_dims[2],
                      "The 1st dimensiong of Input(BBoxes) must be equal to "
                      "3rd dimension of Input(Scores), which represents the "
                      "predicted bboxes.");

    // Here the box_dims[0] is not the real dimension of output.
    // It will be rewritten in the computing kernel.
    ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<framework::LoDTensor>("Scores")->type()),
        platform::CPUPlace());
  }
};

template <class T>
bool SortScorePairDescend(const std::pair<float, T>& pair1,
                          const std::pair<float, T>& pair2) {
  return pair1.first > pair2.first;
}

template <class T>
static inline void GetMaxScoreIndex(
    const std::vector<T>& scores, const T threshold, int top_k,
    std::vector<std::pair<T, int>>* sorted_indices) {
  for (size_t i = 0; i < scores.size(); ++i) {
    if (scores[i] > threshold) {
      sorted_indices->push_back(std::make_pair(scores[i], i));
    }
  }
  // Sort the score pair according to the scores in descending order
  std::stable_sort(sorted_indices->begin(), sorted_indices->end(),
                   SortScorePairDescend<int>);
  // Keep top_k scores if needed.
  if (top_k > -1 && top_k < static_cast<int>(sorted_indices->size())) {
    sorted_indices->resize(top_k);
  }
}

template <class T>
static inline T BBoxArea(const T* box, const bool normalized) {
  if (box[2] < box[0] || box[3] < box[1]) {
    // If coordinate values are is invalid
    // (e.g. xmax < xmin or ymax < ymin), return 0.
    return static_cast<T>(0.);
  } else {
    const T w = box[2] - box[0];
    const T h = box[3] - box[1];
    if (normalized) {
      return w * h;
    } else {
      // If coordinate values are not within range [0, 1].
      return (w + 1) * (h + 1);
    }
  }
}

template <class T>
static inline T JaccardOverlap(const T* box1, const T* box2,
                               const bool normalized) {
  if (box2[0] > box1[2] || box2[2] < box1[0] || box2[1] > box1[3] ||
      box2[3] < box1[1]) {
    return static_cast<T>(0.);
  } else {
    const T inter_xmin = std::max(box1[0], box2[0]);
    const T inter_ymin = std::max(box1[1], box2[1]);
    const T inter_xmax = std::min(box1[2], box2[2]);
    const T inter_ymax = std::min(box1[3], box2[3]);
    const T inter_w = inter_xmax - inter_xmin;
    const T inter_h = inter_ymax - inter_ymin;
    const T inter_area = inter_w * inter_h;
    const T bbox1_area = BBoxArea<T>(box1, normalized);
    const T bbox2_area = BBoxArea<T>(box2, normalized);
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <class T>
T PolyIoU(const T* box1, const T* box2, const size_t box_size,
          const bool normalized) {
  T bbox1_area = PolyArea<T>(box1, box_size, normalized);
  T bbox2_area = PolyArea<T>(box2, box_size, normalized);
  T inter_area = PolyOverlapArea<T>(box1, box2, box_size, normalized);
  if (bbox1_area == 0 || bbox2_area == 0 || inter_area == 0) {
    // If coordinate values are is invalid
    // if area size <= 0,  return 0.
    return T(0.);
  } else {
    return inter_area / (bbox1_area + bbox2_area - inter_area);
  }
}

template <typename T>
class MultiClassNMSKernel : public framework::OpKernel<T> {
 public:
  void NMSFast(const Tensor& bbox, const Tensor& scores,
               const T score_threshold, const T nms_threshold, const T eta,
               const int64_t top_k, std::vector<int>* selected_indices) const {
    // The total boxes for each instance.
    int64_t num_boxes = bbox.dims()[0];
    // 4: [xmin ymin xmax ymax]
    // 8: [x1 y1 x2 y2 x3 y3 x4 y4]
    // 16, 24, or 32: [x1 y1 x2 y2 ...  xn yn], n = 8, 12 or 16
    int64_t box_size = bbox.dims()[1];

    std::vector<T> scores_data(num_boxes);
    std::copy_n(scores.data<T>(), num_boxes, scores_data.begin());
    std::vector<std::pair<T, int>> sorted_indices;
    GetMaxScoreIndex(scores_data, score_threshold, top_k, &sorted_indices);

    selected_indices->clear();
    T adaptive_threshold = nms_threshold;
    const T* bbox_data = bbox.data<T>();

    while (sorted_indices.size() != 0) {
      const int idx = sorted_indices.front().second;
      bool keep = true;
      for (size_t k = 0; k < selected_indices->size(); ++k) {
        if (keep) {
          const int kept_idx = (*selected_indices)[k];
          T overlap = T(0.);
          // 4: [xmin ymin xmax ymax]
          if (box_size == 4) {
            overlap = JaccardOverlap<T>(bbox_data + idx * box_size,
                                        bbox_data + kept_idx * box_size, true);
          }
          // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
          if (box_size == 8 || box_size == 16 || box_size == 24 
                            || box_size == 32) {
            overlap =
                PolyIoU<T>(bbox_data + idx * box_size,
                           bbox_data + kept_idx * box_size, box_size, true);
          }
          keep = overlap <= adaptive_threshold;
        } else {
          break;
        }
      }
      if (keep) {
        selected_indices->push_back(idx);
      }
      sorted_indices.erase(sorted_indices.begin());
      if (keep && eta < 1 && adaptive_threshold > 0.5) {
        adaptive_threshold *= eta;
      }
    }
  }

  void MultiClassNMS(const framework::ExecutionContext& ctx,
                     const Tensor& scores, const Tensor& bboxes,
                     std::map<int, std::vector<int>>* indices,
                     int* num_nmsed_out) const {
    int64_t background_label = ctx.Attr<int>("background_label");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T nms_eta = static_cast<T>(ctx.Attr<float>("nms_eta"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));

    int64_t class_num = scores.dims()[0];
    int64_t predict_dim = scores.dims()[1];
    int num_det = 0;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;
      Tensor score = scores.Slice(c, c + 1);
      NMSFast(bboxes, score, score_threshold, nms_threshold, nms_eta, nms_top_k,
              &((*indices)[c]));
      num_det += (*indices)[c].size();
    }

    *num_nmsed_out = num_det;
    const T* scores_data = scores.data<T>();
    if (keep_top_k > -1 && num_det > keep_top_k) {
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto& it : *indices) {
        int label = it.first;
        const T* sdata = scores_data + label * predict_dim;
        const std::vector<int>& label_indices = it.second;
        for (size_t j = 0; j < label_indices.size(); ++j) {
          int idx = label_indices[j];
          PADDLE_ENFORCE_LT(idx, predict_dim);
          score_index_pairs.push_back(
              std::make_pair(sdata[idx], std::make_pair(label, idx)));
        }
      }
      // Keep top k results per image.
      std::stable_sort(score_index_pairs.begin(), score_index_pairs.end(),
                       SortScorePairDescend<std::pair<int, int>>);
      score_index_pairs.resize(keep_top_k);

      // Store the new indices.
      std::map<int, std::vector<int>> new_indices;
      for (size_t j = 0; j < score_index_pairs.size(); ++j) {
        int label = score_index_pairs[j].second.first;
        int idx = score_index_pairs[j].second.second;
        new_indices[label].push_back(idx);
      }
      new_indices.swap(*indices);
      *num_nmsed_out = keep_top_k;
    }
  }

  void MultiClassOutput(const Tensor& scores, const Tensor& bboxes,
                        const std::map<int, std::vector<int>>& selected_indices,
                        Tensor* outs) const {
    int64_t predict_dim = scores.dims()[1];
    int64_t box_size = bboxes.dims()[1];
    int64_t out_dim = bboxes.dims()[1] + 2;
    auto* scores_data = scores.data<T>();
    auto* bboxes_data = bboxes.data<T>();
    auto* odata = outs->data<T>();

    int count = 0;
    for (const auto& it : selected_indices) {
      int label = it.first;
      const T* sdata = scores_data + label * predict_dim;
      const std::vector<int>& indices = it.second;
      for (size_t j = 0; j < indices.size(); ++j) {
        int idx = indices[j];
        const T* bdata = bboxes_data + idx * box_size;
        odata[count * out_dim] = label;           // label
        odata[count * out_dim + 1] = sdata[idx];  // score
        // xmin, ymin, xmax, ymax or multi-points coordinates
        std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
        count++;
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<Tensor>("BBoxes");
    auto* scores = ctx.Input<Tensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");

    auto score_dims = scores->dims();

    int64_t batch_size = score_dims[0];
    int64_t class_num = score_dims[1];
    int64_t predict_dim = score_dims[2];
    int64_t box_dim = boxes->dims()[2];
    int64_t out_dim = boxes->dims()[2] + 2;

    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<size_t> batch_starts = {0};
    for (int64_t i = 0; i < batch_size; ++i) {
      Tensor ins_score = scores->Slice(i, i + 1);
      ins_score.Resize({class_num, predict_dim});

      Tensor ins_boxes = boxes->Slice(i, i + 1);
      ins_boxes.Resize({predict_dim, box_dim});

      std::map<int, std::vector<int>> indices;
      int num_nmsed_out = 0;
      MultiClassNMS(ctx, ins_score, ins_boxes, &indices, &num_nmsed_out);
      all_indices.push_back(indices);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      T* od = outs->mutable_data<T>({1}, ctx.GetPlace());
      od[0] = -1;
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      for (int64_t i = 0; i < batch_size; ++i) {
        Tensor ins_score = scores->Slice(i, i + 1);
        ins_score.Resize({class_num, predict_dim});

        Tensor ins_boxes = boxes->Slice(i, i + 1);
        ins_boxes.Resize({predict_dim, box_dim});

        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          MultiClassOutput(ins_score, ins_boxes, all_indices[i], &out);
        }
      }
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);

    outs->set_lod(lod);
  }
};

class MultiClassNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "(Tensor) A 3-D Tensor with shape "
             "[N, M, 4 or 8 16 24 32] represents the "
             "predicted locations of M bounding bboxes, N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4.");
    AddInput("Scores",
             "(Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 1st dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, defalut: 0) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(0);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score. If not provided, consider all boxes.");
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences aftern the filtering detections based on "
                 "score_threshold");
    AddAttr<float>("nms_threshold",
                   "(float, defalut: 0.3) "
                   "The threshold to be used in NMS.")
        .SetDefault(0.3);
    AddAttr<float>("nms_eta",
                   "(float) "
                   "The parameter for adaptive NMS.")
        .SetDefault(1.0);
    AddAttr<int>("keep_top_k",
                 "(int64_t) "
                 "Number of total bboxes to be kept per image after NMS "
                 "step. -1 means keeping all bboxes after NMS step.");
    AddOutput("Out",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 6] represents the "
              "detections. Each row has 6 values: "
              "[label, confidence, xmin, ymin, xmax, ymax] or "
              "(LoDTensor) A 2-D LoDTensor with shape [No, 10] represents the "
              "detections. Each row has 10 values: "
              "[label, confidence, x1, y1, x2, y2, x3, y3, x4, y4]. No is the "
              "total number of detections in this mini-batch."
              "For each instance, "
              "the offsets in first dimension are called LoD, the number of "
              "offset is N + 1, if LoD[i + 1] - LoD[i] == 0, means there is "
              "no detected bbox.");
    AddComment(R"DOC(
This operator is to do multi-class non maximum suppression (NMS) on a batched
of boxes and scores.

In the NMS step, this operator greedily selects a subset of detection bounding
boxes that have high scores larger than score_threshold, if providing this
threshold, then selects the largest nms_top_k confidences scores if nms_top_k
is larger than -1. Then this operator pruns away boxes that have high IOU
(intersection over union) overlap with already selected boxes by adaptive
threshold NMS based on parameters of nms_threshold and nms_eta.

Aftern NMS step, at most keep_top_k number of total bboxes are to be kept
per image if keep_top_k is larger than -1.

This operator support multi-class and batched inputs. It applying NMS
independently for each class. The outputs is a 2-D LoDTenosr, for each
image, the offsets in first dimension of LoDTensor are called LoD, the number
of offset is N + 1, where N is the batch size. If LoD[i + 1] - LoD[i] == 0,
means there is no detected bbox for this image. If there is no detected boxes
for all images, all the elements in LoD are 0, and the Out only contains one
value which is -1.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(multiclass_nms, ops::MultiClassNMSOp,
                  ops::MultiClassNMSOpMaker,
                  paddle::framework::EmptyGradOpMaker);
REGISTER_OP_CPU_KERNEL(multiclass_nms, ops::MultiClassNMSKernel<float>,
                       ops::MultiClassNMSKernel<double>);
