/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

inline std::vector<size_t> GetNmsLodFromRoisNum(const Tensor* rois_num) {
  std::vector<size_t> rois_lod;
  auto* rois_num_data = rois_num->data<int>();
  rois_lod.push_back(static_cast<size_t>(0));
  for (int i = 0; i < rois_num->numel(); ++i) {
    rois_lod.push_back(rois_lod.back() + static_cast<size_t>(rois_num_data[i]));
  }
  return rois_lod;
}

class MultiClassNMSOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("BBoxes"), "Input", "BBoxes", "MultiClassNMS");
    OP_INOUT_CHECK(ctx->HasInput("Scores"), "Input", "Scores", "MultiClassNMS");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "MultiClassNMS");
    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto score_size = score_dims.size();

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(score_size == 2 || score_size == 3, true,
                        platform::errors::InvalidArgument(
                            "The rank of Input(Scores) must be 2 or 3"
                            ". But received rank = %d",
                            score_size));
      PADDLE_ENFORCE_EQ(box_dims.size(), 3,
                        platform::errors::InvalidArgument(
                            "The rank of Input(BBoxes) must be 3"
                            ". But received rank = %d",
                            box_dims.size()));
      if (score_size == 3) {
        PADDLE_ENFORCE_EQ(box_dims[2] == 4 || box_dims[2] == 8 ||
                              box_dims[2] == 16 || box_dims[2] == 24 ||
                              box_dims[2] == 32,
                          true,
                          platform::errors::InvalidArgument(
                              "The last dimension of Input"
                              "(BBoxes) must be 4 or 8, "
                              "represents the layout of coordinate "
                              "[xmin, ymin, xmax, ymax] or "
                              "4 points: [x1, y1, x2, y2, x3, y3, x4, y4] or "
                              "8 points: [xi, yi] i= 1,2,...,8 or "
                              "12 points: [xi, yi] i= 1,2,...,12 or "
                              "16 points: [xi, yi] i= 1,2,...,16"));
        PADDLE_ENFORCE_EQ(
            box_dims[1], score_dims[2],
            platform::errors::InvalidArgument(
                "The 2nd dimension of Input(BBoxes) must be equal to "
                "last dimension of Input(Scores), which represents the "
                "predicted bboxes."
                "But received box_dims[1](%s) != socre_dims[2](%s)",
                box_dims[1], score_dims[2]));
      } else {
        PADDLE_ENFORCE_EQ(box_dims[2], 4,
                          platform::errors::InvalidArgument(
                              "The last dimension of Input"
                              "(BBoxes) must be 4. But received dimension = %d",
                              box_dims[2]));
        PADDLE_ENFORCE_EQ(
            box_dims[1], score_dims[1],
            platform::errors::InvalidArgument(
                "The 2nd dimension of Input"
                "(BBoxes) must be equal to the 2nd dimension of Input(Scores). "
                "But received box dimension = %d, score dimension = %d",
                box_dims[1], score_dims[1]));
      }
    }
    // Here the box_dims[0] is not the real dimension of output.
    // It will be rewritten in the computing kernel.
    if (score_size == 3) {
      ctx->SetOutputDim("Out", {box_dims[1], box_dims[2] + 2});
    } else {
      ctx->SetOutputDim("Out", {-1, box_dims[2] + 2});
    }
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Out", std::max(ctx->GetLoDLevel("BBoxes"), 1));
    }
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
void SliceOneClass(const platform::DeviceContext& ctx,
                   const framework::Tensor& items, const int class_id,
                   framework::Tensor* one_class_item) {
  T* item_data = one_class_item->mutable_data<T>(ctx.GetPlace());
  const T* items_data = items.data<T>();
  const int64_t num_item = items.dims()[0];
  const int class_num = items.dims()[1];
  if (items.dims().size() == 3) {
    int item_size = items.dims()[2];
    for (int i = 0; i < num_item; ++i) {
      std::memcpy(item_data + i * item_size,
                  items_data + i * class_num * item_size + class_id * item_size,
                  sizeof(T) * item_size);
    }
  } else {
    for (int i = 0; i < num_item; ++i) {
      item_data[i] = items_data[i * class_num + class_id];
    }
  }
}

template <typename T>
class MultiClassNMSKernel : public framework::OpKernel<T> {
 public:
  void NMSFast(const Tensor& bbox, const Tensor& scores,
               const T score_threshold, const T nms_threshold, const T eta,
               const int64_t top_k, std::vector<int>* selected_indices,
               const bool normalized) const {
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
  }

  void MultiClassNMS(const framework::ExecutionContext& ctx,
                     const Tensor& scores, const Tensor& bboxes,
                     const int scores_size,
                     std::map<int, std::vector<int>>* indices,
                     int* num_nmsed_out) const {
    int64_t background_label = ctx.Attr<int>("background_label");
    int64_t nms_top_k = ctx.Attr<int>("nms_top_k");
    int64_t keep_top_k = ctx.Attr<int>("keep_top_k");
    bool normalized = ctx.Attr<bool>("normalized");
    T nms_threshold = static_cast<T>(ctx.Attr<float>("nms_threshold"));
    T nms_eta = static_cast<T>(ctx.Attr<float>("nms_eta"));
    T score_threshold = static_cast<T>(ctx.Attr<float>("score_threshold"));
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    int num_det = 0;

    int64_t class_num = scores_size == 3 ? scores.dims()[0] : scores.dims()[1];
    Tensor bbox_slice, score_slice;
    for (int64_t c = 0; c < class_num; ++c) {
      if (c == background_label) continue;
      if (scores_size == 3) {
        score_slice = scores.Slice(c, c + 1);
        bbox_slice = bboxes;
      } else {
        score_slice.Resize({scores.dims()[0], 1});
        bbox_slice.Resize({scores.dims()[0], 4});
        SliceOneClass<T>(dev_ctx, scores, c, &score_slice);
        SliceOneClass<T>(dev_ctx, bboxes, c, &bbox_slice);
      }
      NMSFast(bbox_slice, score_slice, score_threshold, nms_threshold, nms_eta,
              nms_top_k, &((*indices)[c]), normalized);
      if (scores_size == 2) {
        std::stable_sort((*indices)[c].begin(), (*indices)[c].end());
      }
      num_det += (*indices)[c].size();
    }

    *num_nmsed_out = num_det;
    const T* scores_data = scores.data<T>();
    if (keep_top_k > -1 && num_det > keep_top_k) {
      const T* sdata;
      std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
      for (const auto& it : *indices) {
        int label = it.first;
        if (scores_size == 3) {
          sdata = scores_data + label * scores.dims()[1];
        } else {
          score_slice.Resize({scores.dims()[0], 1});
          SliceOneClass<T>(dev_ctx, scores, label, &score_slice);
          sdata = score_slice.data<T>();
        }
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
      if (scores_size == 2) {
        for (const auto& it : new_indices) {
          int label = it.first;
          std::stable_sort(new_indices[label].begin(),
                           new_indices[label].end());
        }
      }
      new_indices.swap(*indices);
      *num_nmsed_out = keep_top_k;
    }
  }

  void MultiClassOutput(const platform::DeviceContext& ctx,
                        const Tensor& scores, const Tensor& bboxes,
                        const std::map<int, std::vector<int>>& selected_indices,
                        const int scores_size, Tensor* outs,
                        int* oindices = nullptr, const int offset = 0) const {
    int64_t class_num = scores.dims()[1];
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
      if (scores_size == 2) {
        SliceOneClass<T>(ctx, bboxes, label, &bbox);
      } else {
        sdata = scores_data + label * predict_dim;
      }

      for (int idx : indices) {
        odata[count * out_dim] = label;  // label
        const T* bdata;
        if (scores_size == 3) {
          bdata = bboxes_data + idx * box_size;
          odata[count * out_dim + 1] = sdata[idx];  // score
          if (oindices != nullptr) {
            oindices[count] = offset + idx;
          }
        } else {
          bdata = bbox.data<T>() + idx * box_size;
          odata[count * out_dim + 1] = *(scores_data + idx * class_num + label);
          if (oindices != nullptr) {
            oindices[count] = offset + idx * class_num + label;
          }
        }
        // xmin, ymin, xmax, ymax or multi-points coordinates
        std::memcpy(odata + count * out_dim + 2, bdata, box_size * sizeof(T));
        count++;
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* boxes = ctx.Input<LoDTensor>("BBoxes");
    auto* scores = ctx.Input<LoDTensor>("Scores");
    auto* outs = ctx.Output<LoDTensor>("Out");
    bool return_index = ctx.HasOutput("Index") ? true : false;
    auto index = ctx.Output<LoDTensor>("Index");
    bool has_roisnum = ctx.HasInput("RoisNum") ? true : false;
    auto rois_num = ctx.Input<Tensor>("RoisNum");
    auto score_dims = scores->dims();
    auto score_size = score_dims.size();
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();

    std::vector<std::map<int, std::vector<int>>> all_indices;
    std::vector<size_t> batch_starts = {0};
    int64_t batch_size = score_dims[0];
    int64_t box_dim = boxes->dims()[2];
    int64_t out_dim = box_dim + 2;
    int num_nmsed_out = 0;
    Tensor boxes_slice, scores_slice;
    int n = 0;
    if (has_roisnum) {
      n = score_size == 3 ? batch_size : rois_num->numel();
    } else {
      n = score_size == 3 ? batch_size : boxes->lod().back().size() - 1;
    }
    for (int i = 0; i < n; ++i) {
      std::map<int, std::vector<int>> indices;
      if (score_size == 3) {
        scores_slice = scores->Slice(i, i + 1);
        scores_slice.Resize({score_dims[1], score_dims[2]});
        boxes_slice = boxes->Slice(i, i + 1);
        boxes_slice.Resize({score_dims[2], box_dim});
      } else {
        std::vector<size_t> boxes_lod;
        if (has_roisnum) {
          boxes_lod = GetNmsLodFromRoisNum(rois_num);
        } else {
          boxes_lod = boxes->lod().back();
        }
        if (boxes_lod[i] == boxes_lod[i + 1]) {
          all_indices.push_back(indices);
          batch_starts.push_back(batch_starts.back());
          continue;
        }
        scores_slice = scores->Slice(boxes_lod[i], boxes_lod[i + 1]);
        boxes_slice = boxes->Slice(boxes_lod[i], boxes_lod[i + 1]);
      }
      MultiClassNMS(ctx, scores_slice, boxes_slice, score_size, &indices,
                    &num_nmsed_out);
      all_indices.push_back(indices);
      batch_starts.push_back(batch_starts.back() + num_nmsed_out);
    }

    int num_kept = batch_starts.back();
    if (num_kept == 0) {
      if (return_index) {
        outs->mutable_data<T>({0, out_dim}, ctx.GetPlace());
        index->mutable_data<int>({0, 1}, ctx.GetPlace());
      } else {
        T* od = outs->mutable_data<T>({1, 1}, ctx.GetPlace());
        od[0] = -1;
        batch_starts = {0, 1};
      }
    } else {
      outs->mutable_data<T>({num_kept, out_dim}, ctx.GetPlace());
      int offset = 0;
      int* oindices = nullptr;
      for (int i = 0; i < n; ++i) {
        if (score_size == 3) {
          scores_slice = scores->Slice(i, i + 1);
          boxes_slice = boxes->Slice(i, i + 1);
          scores_slice.Resize({score_dims[1], score_dims[2]});
          boxes_slice.Resize({score_dims[2], box_dim});
          if (return_index) {
            offset = i * score_dims[2];
          }
        } else {
          std::vector<size_t> boxes_lod;
          if (has_roisnum) {
            boxes_lod = GetNmsLodFromRoisNum(rois_num);
          } else {
            boxes_lod = boxes->lod().back();
          }
          if (boxes_lod[i] == boxes_lod[i + 1]) continue;
          scores_slice = scores->Slice(boxes_lod[i], boxes_lod[i + 1]);
          boxes_slice = boxes->Slice(boxes_lod[i], boxes_lod[i + 1]);
          if (return_index) {
            offset = boxes_lod[i] * score_dims[1];
          }
        }

        int64_t s = batch_starts[i];
        int64_t e = batch_starts[i + 1];
        if (e > s) {
          Tensor out = outs->Slice(s, e);
          if (return_index) {
            int* output_idx =
                index->mutable_data<int>({num_kept, 1}, ctx.GetPlace());
            oindices = output_idx + s;
          }
          MultiClassOutput(dev_ctx, scores_slice, boxes_slice, all_indices[i],
                           score_dims.size(), &out, oindices, offset);
        }
      }
    }
    if (ctx.HasOutput("NmsRoisNum")) {
      auto* nms_rois_num = ctx.Output<Tensor>("NmsRoisNum");
      nms_rois_num->mutable_data<int>({n}, ctx.GetPlace());
      int* num_data = nms_rois_num->data<int>();
      for (int i = 1; i <= n; i++) {
        num_data[i - 1] = batch_starts[i] - batch_starts[i - 1];
      }
      nms_rois_num->Resize({n});
    }

    framework::LoD lod;
    lod.emplace_back(batch_starts);
    if (return_index) {
      index->set_lod(lod);
    }
    outs->set_lod(lod);
  }
};

class MultiClassNMSOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("BBoxes",
             "Two types of bboxes are supported:"
             "1. (Tensor) A 3-D Tensor with shape "
             "[N, M, 4 or 8 16 24 32] represents the "
             "predicted locations of M bounding bboxes, N is the batch size. "
             "Each bounding box has four coordinate values and the layout is "
             "[xmin, ymin, xmax, ymax], when box size equals to 4."
             "2. (LoDTensor) A 3-D Tensor with shape [M, C, 4]"
             "M is the number of bounding boxes, C is the class number");
    AddInput("Scores",
             "Two types of scores are supported:"
             "1. (Tensor) A 3-D Tensor with shape [N, C, M] represents the "
             "predicted confidence predictions. N is the batch size, C is the "
             "class number, M is number of bounding boxes. For each category "
             "there are total M scores which corresponding M bounding boxes. "
             " Please note, M is equal to the 2nd dimension of BBoxes. "
             "2. (LoDTensor) A 2-D LoDTensor with shape [M, C]. "
             "M is the number of bbox, C is the class number. In this case, "
             "Input BBoxes should be the second case with shape [M, C, 4].");
    AddAttr<int>(
        "background_label",
        "(int, default: 0) "
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
means there is no detected bbox for this image.
)DOC");
  }
};

class MultiClassNMS2Op : public MultiClassNMSOp {
 public:
  MultiClassNMS2Op(const std::string& type,
                   const framework::VariableNameMap& inputs,
                   const framework::VariableNameMap& outputs,
                   const framework::AttributeMap& attrs)
      : MultiClassNMSOp(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    MultiClassNMSOp::InferShape(ctx);

    auto box_dims = ctx->GetInputDim("BBoxes");
    auto score_dims = ctx->GetInputDim("Scores");
    auto score_size = score_dims.size();
    if (score_size == 3) {
      ctx->SetOutputDim("Index", {box_dims[1], 1});
    } else {
      ctx->SetOutputDim("Index", {-1, 1});
    }
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("Index", std::max(ctx->GetLoDLevel("BBoxes"), 1));
    }
  }
};

class MultiClassNMS2OpMaker : public MultiClassNMSOpMaker {
 public:
  void Make() override {
    MultiClassNMSOpMaker::Make();
    AddOutput("Index",
              "(LoDTensor) A 2-D LoDTensor with shape [No, 1] represents the "
              "index of selected bbox. The index is the absolute index cross "
              "batches.")
        .AsIntermediate();
  }
};

class MultiClassNMS3Op : public MultiClassNMS2Op {
 public:
  MultiClassNMS3Op(const std::string& type,
                   const framework::VariableNameMap& inputs,
                   const framework::VariableNameMap& outputs,
                   const framework::AttributeMap& attrs)
      : MultiClassNMS2Op(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    MultiClassNMS2Op::InferShape(ctx);

    ctx->SetOutputDim("NmsRoisNum", {-1});
  }
};

class MultiClassNMS3OpMaker : public MultiClassNMS2OpMaker {
 public:
  void Make() override {
    MultiClassNMS2OpMaker::Make();
    AddInput("RoisNum",
             "(Tensor) The number of RoIs in shape (B),"
             "B is the number of images")
        .AsDispensable();
    AddOutput("NmsRoisNum", "(Tensor), The number of NMS RoIs in each image")
        .AsDispensable();
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    multiclass_nms, ops::MultiClassNMSOp, ops::MultiClassNMSOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(multiclass_nms, ops::MultiClassNMSKernel<float>,
                       ops::MultiClassNMSKernel<double>);
REGISTER_OPERATOR(
    multiclass_nms2, ops::MultiClassNMS2Op, ops::MultiClassNMS2OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(multiclass_nms2, ops::MultiClassNMSKernel<float>,
                       ops::MultiClassNMSKernel<double>);

REGISTER_OPERATOR(
    multiclass_nms3, ops::MultiClassNMS3Op, ops::MultiClassNMS3OpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(multiclass_nms3, ops::MultiClassNMSKernel<float>,
                       ops::MultiClassNMSKernel<double>);
