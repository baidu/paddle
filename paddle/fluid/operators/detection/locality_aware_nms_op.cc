/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
limitations under the License. */

#include <glog/logging.h>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/nms_util.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

class LocalityAwareNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("BBoxes"), "Input", "BBoxes",
                   "locality_aware_nms");
    OP_INOUT_CHECK(ctx->HasInput("Scores"), "Input", "Scores",
                   "locality_aware_nms");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "locality_aware_nms");

    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto score_size = score_dims.size();

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          score_size, 3,
          platform::errors::InvalidArgument(
              "The rank of Input(Scores) must be 3. But received %d.",
              score_size));
      PADDLE_ENFORCE_EQ(
          box_dims.size(), 3,
          platform::errors::InvalidArgument(
              "The rank of Input(BBoxes) must be 3. But received %d.",
              box_dims.size()));
      PADDLE_ENFORCE_EQ(
          box_dims[2] == 4 || box_dims[2] == 8 || box_dims[2] == 16 ||
              box_dims[2] == 24 || box_dims[2] == 32,
          true,
          platform::errors::InvalidArgument(
              "The last dimension of Input(BBoxes) must be 4 or 8, "
              "represents the layout of coordinate "
              "[xmin, ymin, xmax, ymax] or "
              "4 points: [x1, y1, x2, y2, x3, y3, x4, y4] or "
              "8 points: [xi, yi] i= 1,2,...,8 or "
              "12 points: [xi, yi] i= 1,2,...,12 or "
              "16 points: [xi, yi] i= 1,2,...,16. "
              "But received %d.",
              box_dims[2]));
      PADDLE_ENFORCE_EQ(
          box_dims[1], score_dims[2],
          platform::errors::InvalidArgument(
              "The 2nd dimension of Input(BBoxes) must be equal to "
              "last dimension of Input(Scores), which represents the "
              "predicted bboxes. But received the 2nd dimension of "
              "Input(BBoxes) was %d, last dimension of Input(Scores) was %d.",
              box_dims[1], score_dims[2]));
    }
    // Here the box_dims[0] is not the real dimension of output.
    // It will be rewritten in the computing kernel.
    ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Scores"),
        platform::CPUPlace());
  }
};

template <class T>
void PolyWeightedMerge(const T* box1, T* box2, const T score1, const T score2,
                       const size_t box_size) {
  for (size_t i = 0; i < box_size; ++i) {
    box2[i] = (box1[i] * score1 + box2[i] * score2) / (score1 + score2);
  }
}

template <class T>
void GetMaxScoreIndexWithLocalityAware(
    T* scores, T* bbox_data, int64_t box_size, const T threshold, int top_k,
    int64_t num_boxes, std::vector<std::pair<T, int>>* sorted_indices,
    const T nms_threshold, const bool normalized) {
  std::vector<bool> skip(num_boxes, true);
  int index = -1;
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (index > -1) {
      T overlap = T(0.);
      if (box_size == 4) {
        overlap = JaccardOverlap<T>(bbox_data + i * box_size,
                                    bbox_data + index * box_size, normalized);
      }
      // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
      if (box_size == 8 || box_size == 16 || box_size == 24 || box_size == 32) {
        overlap =
            PolyIoU<T>(bbox_data + i * box_size, bbox_data + index * box_size,
                       box_size, normalized);
      }

      if (overlap > nms_threshold) {
        PolyWeightedMerge(bbox_data + i * box_size,
                          bbox_data + index * box_size, scores[i],
                          scores[index], box_size);
        scores[index] += scores[i];
      } else {
        skip[index] = false;
        index = i;
      }
    } else {
      index = i;
    }
  }

  if (index > -1) {
    skip[index] = false;
  }
  for (int64_t i = 0; i < num_boxes; ++i) {
    if (scores[i] > threshold && skip[i] == false) {
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

template <typename T>
class LocalityAwareNMSKernel : public framework::OpKernel<T> {
 public:
  void LocalityAwareNMSFast(Tensor* bbox, Tensor* scores,
                            const T score_threshold, const T nms_threshold,
                            const T eta, const int64_t top_k,
                            std::vector<int>* selected_indices,
                            const bool normalized) const {
    // The total boxes for each instance.
    int64_t num_boxes = bbox->dims()[0];
    // 4: [xmin ymin xmax ymax]
    // 8: [x1 y1 x2 y2 x3 y3 x4 y4]
    // 16, 24, or 32: [x1 y1 x2 y2 ...  xn yn], n = 8, 12 or 16
    int64_t box_size = bbox->dims()[1];

    std::vector<std::pair<T, int>> sorted_indices;
    T adaptive_threshold = nms_threshold;
    T* bbox_data = bbox->data<T>();
    T* scores_data = scores->data<T>();

    GetMaxScoreIndexWithLocalityAware(
        scores_data, bbox_data, box_size, score_threshold, top_k, num_boxes,
        &sorted_indices, nms_threshold, normalized);

    selected_indices->clear();

    while (sorted_indices.size() != 0) {
      const int idx = sorted_indices.front().second;
      bool keep = true;
      for (int kept_idx : *selected_indices) {
        if (keep) {
          T overlap = T(0.);
          // 4: [xmin ymin xmax ymax]
          if (box_size == 4) {
            overlap =
                JaccardOverlap<T>(bbox_data + idx * box_size,
                                  bbox_data + kept_idx * box_size, normalized);
          }
          // 8: [x1 y1 x2 y2 x3 y3 x4 y4] or 16, 24, 32
          if (box_size == 8 || box_size == 16 || box_size == 24 ||
              box_size == 32) {
            overlap = PolyIoU<T>(bbox_data + idx * box_size,
                                 bbox_data + kept_idx * box_size, box_size,
                                 normalized);
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
    //    delete bbox_data;
  }

  void LocalityAwareNMS(const framework::ExecutionContext& ctx, Tensor* scores,
                        Tensor* bboxes, const int scores_size,
                        std::map<int, std::vector<int>>* indices,
                        int* num_nmsed_out) const {
    int64_t background_label = ctx.Attr<int>("background_label");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    bool normalized = ctx.Attr<bool>("normalized");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T nms_eta = static_cast<T>(ctx.Attr<float>("nms_eta"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));

    int num_det = 0;

    int64_t class_num = scores->dims()[0];
    Tensor bbox_slice, score_slice;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;

      score_slice = scores->Slice(c, c + 1);
      bbox_slice = *bboxes;

      LocalityAwareNMSFast(&bbox_slice, &score_slice, score_threshold,
                           nms_threshold, nms_eta, nms_top_k, &((*indices)[c]),
                           normalized);
      num_det += (*indices)[c].size();
    }

    *num_nmsed_out = num_det;
    const T* scores_data = scores->data<T>();
    if (keep_top_k > -1 && num_det > keep_top_k) {
      const T* sdata;
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto& it : *indices) {
        int label = it.first;

        sdata = scores_data + label * scores->dims()[1];

        const std::vector<int>& label_indices = it.second;
        for (int idx : label_indices) {
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
      for (auto& score_index_pair : score_index_pairs) {
        int label = score_index_pair.second.first;
        int idx = score_index_pair.second.second;
        new_indices[label].push_back(idx);
      }

      new_indices.swap(*indices);
      *num_nmsed_out = keep_top_k;
    }
  }

  void LocalityAwareNMSOutput(
      const platform::DeviceContext& ctx, const Tensor& scores,
      const Tensor& bboxes,
      const std::map<int, std::vector<int>>& selected_indices,
      const int scores_size, Tensor* outs, int* oindices = nullptr,
      const int offset = 0) const {
    int64_t predict_dim = scores.dims()[1];
    int64_t box_size = bboxes.dims()[1];
    if (scores_size == 2) {
      box_size = bboxes.dims()[2];
    }
    int64_t out_dim = box_size + 2;
    auto* scores_data = scores.data<T>();
    auto* bboxes_data = bboxes.data<T>();
    auto* odata = outs->data<T>();
    const T* sdata;
    Tensor bbox;
    bbox.Resize({scores.dims()[0], box_size});
    int count = 0;
    for (const auto& it : selected_indices) {
      int label = it.first;
      const std::vector<int>& indices = it.second;
      sdata = scores_data + label * predict_dim;
      for (int idx : indices) {
        odata[count * out_dim] = label;  // label
        const T* bdata;
        bdata = bboxes_data + idx * box_size;
        odata[count * out_dim + 1] = sdata[idx];  // score
        if (oindices != nullptr) {
          oindices[count] = offset + idx;
        }

        // xmin, ymin, xmax, ymax or multi-points coordinates
        std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
        count++;
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes_input = ctx.Input<LoDTensor>("BBoxes");
    auto* scores_input = ctx.Input<LoDTensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");
    auto score_dims = scores_input->dims();
    auto score_size = score_dims.size();
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    LoDTensor scores;
    LoDTensor boxes;
    TensorCopySync(*scores_input, platform::CPUPlace(), &scores);
    TensorCopySync(*boxes_input, platform::CPUPlace(), &boxes);
    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<size_t> batch_starts = {0};
    int64_t batch_size = score_dims[0];
    int64_t box_dim = boxes.dims()[2];
    int64_t out_dim = box_dim + 2;
    int num_nmsed_out = 0;
    Tensor boxes_slice, scores_slice;
    int n = batch_size;
    for (int i = 0; i < n; ++i) {
      scores_slice = scores.Slice(i, i + 1);
      scores_slice.Resize({score_dims[1], score_dims[2]});
      boxes_slice = boxes.Slice(i, i + 1);
      boxes_slice.Resize({score_dims[2], box_dim});

      std::map<int, std::vector<int>> indices;
      LocalityAwareNMS(ctx, &scores_slice, &boxes_slice, score_size, &indices,
                       &num_nmsed_out);
      all_indices.push_back(indices);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      T* od = outs->mutable_data<T>({1, 1}, ctx.GetPlace());
      od[0] = -1;
      batch_starts = {0, 1};
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      int offset = 0;
      int* oindices = nullptr;
      for (int i = 0; i < n; ++i) {
        scores_slice = scores.Slice(i, i + 1);
        boxes_slice = boxes.Slice(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice.Resize({score_dims[2], box_dim});

        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          LocalityAwareNMSOutput(dev_ctx, scores_slice, boxes_slice,
                                 all_indices[i], score_dims.size(), &out,
                                 oindices, offset);
        }
      }
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);
    outs->set_lod(lod);
  }
};

class LocalityAwareNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "Two types of bboxes are supported:"
             "1. (Tensor) A 3-D Tensor with shape "
             "[N, M, 4 or 8 16 24 32] represents the "
             "predicted locations of M bounding bboxes, N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4.");
    AddInput("Scores",
             "Two types of scores are supported:"
             "1. (Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 2nd dimension of BBoxes. ");
    AddAttr<int>(
        "background_label",
        "(int, default: -1) "
        "The index of background label, the background label will be ignored. "
        "If set to -1, then all categories will be considered.")
        .SetDefault(-1);
    AddAttr<float>("score_threshold",
                   "(float) "
                   "Threshold to filter out bounding boxes with low "
                   "confidence score. If not provided, consider all boxes.");
    AddAttr<int>("nms_top_k",
                 "(int64_t) "
                 "Maximum number of detections to be kept according to the "
                 "confidences after the filtering detections based on "
                 "score_threshold");
    AddAttr<float>("nms_threshold",
                   "(float, default: 0.3) "
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
    AddAttr<bool>("normalized",
                  "(bool, default true) "
                  "Whether detections are normalized.")
        .SetDefault(true);
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
This operator is to do locality-aware non maximum suppression (NMS) on a batched
of boxes and scores.
Firstly, this operator merge box and score according their IOU(intersection over union).
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
means there is no detected bbox for this image.

Please get more information from the following papers:
https://arxiv.org/abs/1704.03155.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    locality_aware_nms, ops::LocalityAwareNMSOp, ops::LocalityAwareNMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(locality_aware_nms, ops::LocalityAwareNMSKernel<float>,
                       ops::LocalityAwareNMSKernel<double>);
