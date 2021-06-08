/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
const int kBoxDim = 4;

template <typename T>
void AppendRois(LoDTensor* out, int64_t offset, Tensor* to_add) {
  auto* out_data = out->data<T>();
  auto* to_add_data = to_add->data<T>();
  memcpy(out_data + offset, to_add_data, to_add->numel() * sizeof(T));
}

// Filter the ground-truth in RoIs and the RoIs with non-positive area.
// The ground-truth has max overlap with itself so the max_overlap is 1
// and the corresponding RoI will be removed.
template <typename T>
void FilterRoIs(const platform::DeviceContext& ctx, const Tensor& rpn_rois,
                const Tensor& max_overlap, Tensor* keep) {
  const T* rpn_rois_dt = rpn_rois.data<T>();
  const T* max_overlap_dt = max_overlap.data<T>();
  int rois_num = max_overlap.numel();
  keep->Resize({rois_num});
  int* keep_data = keep->mutable_data<int>(ctx.GetPlace());
  int keep_len = 0;
  for (int i = 0; i < rois_num; ++i) {
    if ((rpn_rois_dt[i * 4 + 2] - rpn_rois_dt[i * 4 + 0] + 1) > 0 &&
        (rpn_rois_dt[i * 4 + 3] - rpn_rois_dt[i * 4 + 1] + 1) > 0 &&
        max_overlap_dt[i] < 1.) {
      keep_data[keep_len++] = i;
    }
  }
  keep->Resize({keep_len});
}

class GenerateProposalLabelsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("RpnRois"), true,
        platform::errors::NotFound("Input(RpnRois) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("GtClasses"), true,
        platform::errors::NotFound("Input(GtClasses) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("IsCrowd"), true,
        platform::errors::NotFound("Input(IsCrowd) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("GtBoxes"), true,
        platform::errors::NotFound("Input(GtBoxes) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("ImInfo"), true,
        platform::errors::NotFound("Input(ImInfo) shouldn't be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Rois"), true,
        platform::errors::NotFound(
            "Output(Rois) of GenerateProposalLabelsOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("LabelsInt32"), true,
                      platform::errors::NotFound("Output(LabelsInt32) of "
                                                 "GenerateProposalLabelsOp "
                                                 "should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("BboxTargets"), true,
                      platform::errors::NotFound("Output(BboxTargets) of "
                                                 "GenerateProposalLabelsOp "
                                                 "should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BboxInsideWeights"), true,
        platform::errors::NotFound(
            "Output(BboxInsideWeights) of GenerateProposalLabelsOp "
            "should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BboxOutsideWeights"), true,
        platform::errors::NotFound(
            "Output(BboxOutsideWeights) of GenerateProposalLabelsOp "
            "should not be null"));

    auto rpn_rois_dims = ctx->GetInputDim("RpnRois");
    auto gt_boxes_dims = ctx->GetInputDim("GtBoxes");
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    PADDLE_ENFORCE_EQ(rpn_rois_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions size of Input(RpnRois) must be 2. "
                          "But received dimensions size=[%d], dimensions=[%s].",
                          rpn_rois_dims.size(), rpn_rois_dims));
    PADDLE_ENFORCE_EQ(gt_boxes_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions size of Input(GtBoxes) must be 2. "
                          "But received dimensions size=[%d], dimensions=[%s].",
                          gt_boxes_dims.size(), gt_boxes_dims));
    PADDLE_ENFORCE_EQ(im_info_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions size of Input(ImInfo) must be 2. But "
                          "received dimensions size=[%d], dimensions=[%s].",
                          im_info_dims.size(), im_info_dims));

    int class_nums = ctx->Attrs().Get<int>("class_nums");
    bool is_cascade_rcnn = ctx->Attrs().Get<bool>("is_cascade_rcnn");
    if (is_cascade_rcnn) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("MaxOverlap"), true,
          platform::errors::NotFound(
              "Input(MaxOverlap) of GenerateProposalLabelsOp "
              "should not be null when is_cascade_rcnn is True."));
    }

    ctx->SetOutputDim("Rois", {-1, 4});
    ctx->SetOutputDim("LabelsInt32", {-1, 1});
    ctx->SetOutputDim("BboxTargets", {-1, 4 * class_nums});
    ctx->SetOutputDim("BboxInsideWeights", {-1, 4 * class_nums});
    ctx->SetOutputDim("BboxOutsideWeights", {-1, 4 * class_nums});
    ctx->SetOutputDim("MaxOverlapWithGT", {-1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "RpnRois");
    return framework::OpKernelType(data_type, platform::CPUPlace());
  }
};

template <typename T>
void Concat(const platform::CPUDeviceContext& context,
            const Tensor& in_tensor_a, const Tensor& in_tensor_b,
            Tensor* out_tensor) {
  int axis = 0;
  std::vector<Tensor> inputs;
  inputs.emplace_back(in_tensor_a);
  inputs.emplace_back(in_tensor_b);
  math::ConcatFunctor<platform::CPUDeviceContext, T> concat_functor;
  concat_functor(context, inputs, axis, out_tensor);
}

template <typename T>
std::vector<std::vector<int>> SampleFgBgGt(
    const platform::CPUDeviceContext& context, Tensor* iou,
    const Tensor& is_crowd, const int batch_size_per_im,
    const float fg_fraction, const float fg_thresh, const float bg_thresh_hi,
    const float bg_thresh_lo, std::minstd_rand engine, const bool use_random,
    const bool is_cascade_rcnn, const Tensor& rpn_rois) {
  std::vector<int> fg_inds;
  std::vector<int> bg_inds;
  std::vector<int> mapped_gt_inds;
  int64_t gt_num = is_crowd.numel();
  const int* crowd_data = is_crowd.data<int>();
  T* proposal_to_gt_overlaps = iou->data<T>();
  int64_t row = iou->dims()[0];
  int64_t col = iou->dims()[1];
  float epsilon = 0.00001;
  // Follow the Faster RCNN's implementation
  for (int64_t i = 0; i < row; ++i) {
    const T* v = proposal_to_gt_overlaps + i * col;

    T max_overlap = *std::max_element(v, v + col);
    if ((i < gt_num) && (crowd_data[i])) {
      max_overlap = -1.0;
    }
    if (max_overlap >= fg_thresh) {
      // fg mapped gt label index
      for (int64_t j = 0; j < col; ++j) {
        T val = proposal_to_gt_overlaps[i * col + j];
        auto diff = std::abs(max_overlap - val);
        if (diff < epsilon) {
          fg_inds.emplace_back(i);
          mapped_gt_inds.emplace_back(j);
          break;
        }
      }
    } else if ((max_overlap >= bg_thresh_lo) && (max_overlap < bg_thresh_hi)) {
      bg_inds.emplace_back(i);
    } else {
      continue;
    }
  }

  std::vector<std::vector<int>> res;
  if (is_cascade_rcnn) {
    res.emplace_back(fg_inds);
    res.emplace_back(bg_inds);
    res.emplace_back(mapped_gt_inds);
  } else {
    // Reservoir Sampling
    // sampling fg
    std::uniform_real_distribution<float> uniform(0, 1);
    int fg_rois_per_im = std::floor(batch_size_per_im * fg_fraction);
    int fg_rois_this_image = fg_inds.size();
    int fg_rois_per_this_image = std::min(fg_rois_per_im, fg_rois_this_image);
    if (use_random) {
      const auto fg_size = static_cast<int64_t>(fg_inds.size());
      if (fg_size > fg_rois_per_this_image) {
        for (int64_t i = fg_rois_per_this_image; i < fg_size; ++i) {
          int rng_ind = std::floor(uniform(engine) * i);
          if (rng_ind < fg_rois_per_this_image) {
            std::iter_swap(fg_inds.begin() + rng_ind, fg_inds.begin() + i);
            std::iter_swap(mapped_gt_inds.begin() + rng_ind,
                           mapped_gt_inds.begin() + i);
          }
        }
      }
    }
    std::vector<int> new_fg_inds(fg_inds.begin(),
                                 fg_inds.begin() + fg_rois_per_this_image);
    std::vector<int> new_gt_inds(
        mapped_gt_inds.begin(),
        mapped_gt_inds.begin() + fg_rois_per_this_image);
    // sampling bg
    int bg_rois_per_image = batch_size_per_im - fg_rois_per_this_image;
    int bg_rois_this_image = bg_inds.size();
    int bg_rois_per_this_image =
        std::min(bg_rois_per_image, bg_rois_this_image);
    if (use_random) {
      const auto bg_size = static_cast<int64_t>(bg_inds.size());
      if (bg_size > bg_rois_per_this_image) {
        for (int64_t i = bg_rois_per_this_image; i < bg_size; ++i) {
          int rng_ind = std::floor(uniform(engine) * i);
          if (rng_ind < fg_rois_per_this_image)
            std::iter_swap(bg_inds.begin() + rng_ind, bg_inds.begin() + i);
        }
      }
    }
    std::vector<int> new_bg_inds(bg_inds.begin(),
                                 bg_inds.begin() + bg_rois_per_this_image);
    //
    res.emplace_back(new_fg_inds);
    res.emplace_back(new_bg_inds);
    res.emplace_back(new_gt_inds);
  }

  return res;
}

template <typename T>
void GatherBoxesLabels(const platform::CPUDeviceContext& context,
                       const Tensor& boxes, const Tensor& max_overlap,
                       const Tensor& gt_boxes, const Tensor& gt_classes,
                       const std::vector<int>& fg_inds,
                       const std::vector<int>& bg_inds,
                       const std::vector<int>& gt_inds, Tensor* sampled_boxes,
                       Tensor* sampled_labels, Tensor* sampled_gts,
                       Tensor* sampled_max_overlap) {
  int fg_num = fg_inds.size();
  int bg_num = bg_inds.size();
  Tensor fg_inds_t, bg_inds_t, gt_box_inds_t, gt_label_inds_t;
  int* fg_inds_data = fg_inds_t.mutable_data<int>({fg_num}, context.GetPlace());
  int* bg_inds_data = bg_inds_t.mutable_data<int>({bg_num}, context.GetPlace());
  int* gt_box_inds_data =
      gt_box_inds_t.mutable_data<int>({fg_num}, context.GetPlace());
  int* gt_label_inds_data =
      gt_label_inds_t.mutable_data<int>({fg_num}, context.GetPlace());
  std::copy(fg_inds.begin(), fg_inds.end(), fg_inds_data);
  std::copy(bg_inds.begin(), bg_inds.end(), bg_inds_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_box_inds_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_label_inds_data);

  Tensor fg_boxes, bg_boxes, fg_labels, bg_labels;
  fg_boxes.mutable_data<T>({fg_num, kBoxDim}, context.GetPlace());
  CPUGather<T>(context, boxes, fg_inds_t, &fg_boxes);
  bg_boxes.mutable_data<T>({bg_num, kBoxDim}, context.GetPlace());
  CPUGather<T>(context, boxes, bg_inds_t, &bg_boxes);
  Concat<T>(context, fg_boxes, bg_boxes, sampled_boxes);
  CPUGather<T>(context, gt_boxes, gt_box_inds_t, sampled_gts);
  fg_labels.mutable_data<int>({fg_num}, context.GetPlace());
  CPUGather<int>(context, gt_classes, gt_label_inds_t, &fg_labels);
  bg_labels.mutable_data<int>({bg_num}, context.GetPlace());
  math::set_constant(context, &bg_labels, 0);
  Concat<int>(context, fg_labels, bg_labels, sampled_labels);

  Tensor fg_max_overlap, bg_max_overlap;
  fg_max_overlap.mutable_data<T>({fg_num}, context.GetPlace());
  CPUGather<T>(context, max_overlap, fg_inds_t, &fg_max_overlap);
  bg_max_overlap.mutable_data<T>({bg_num}, context.GetPlace());
  CPUGather<T>(context, max_overlap, bg_inds_t, &bg_max_overlap);
  Concat<T>(context, fg_max_overlap, bg_max_overlap, sampled_max_overlap);
}

template <typename T>
std::vector<Tensor> SampleRoisForOneImage(
    const platform::CPUDeviceContext& context, const Tensor& rpn_rois_in,
    const Tensor& gt_classes, const Tensor& is_crowd, const Tensor& gt_boxes,
    const Tensor& im_info, const int batch_size_per_im, const float fg_fraction,
    const float fg_thresh, const float bg_thresh_hi, const float bg_thresh_lo,
    const std::vector<float>& bbox_reg_weights, const int class_nums,
    std::minstd_rand engine, bool use_random, bool is_cascade_rcnn,
    bool is_cls_agnostic, const Tensor& max_overlap) {
  // 1.1 map to original image
  auto im_scale = im_info.data<T>()[2];
  Tensor rpn_rois;
  rpn_rois.mutable_data<T>(rpn_rois_in.dims(), context.GetPlace());
  const T* rpn_rois_in_dt = rpn_rois_in.data<T>();
  T* rpn_rois_dt = rpn_rois.data<T>();

  for (int i = 0; i < rpn_rois.numel(); ++i) {
    rpn_rois_dt[i] = rpn_rois_in_dt[i] / im_scale;
  }

  int proposals_num = 1;

  if (is_cascade_rcnn) {
    Tensor keep;
    FilterRoIs<T>(context, rpn_rois, max_overlap, &keep);
    Tensor roi_filter;
    // Tensor box_filter;
    if (keep.numel() == 0) {
      math::SetConstant<platform::CPUDeviceContext, T> set_zero;
      roi_filter.mutable_data<T>({proposals_num, kBoxDim}, context.GetPlace());
      set_zero(context, &roi_filter, static_cast<T>(0));
    } else {
      proposals_num = keep.numel();
      roi_filter.mutable_data<T>({proposals_num, kBoxDim}, context.GetPlace());
      CPUGather<T>(context, rpn_rois, keep, &roi_filter);
    }
    T* roi_filter_dt = roi_filter.data<T>();
    memcpy(rpn_rois_dt, roi_filter_dt, roi_filter.numel() * sizeof(T));
    rpn_rois.Resize(roi_filter.dims());
  } else {
    proposals_num = rpn_rois.dims()[0];
  }
  // 1.2 compute overlaps
  proposals_num += gt_boxes.dims()[0];

  Tensor proposal_to_gt_overlaps;
  proposal_to_gt_overlaps.mutable_data<T>({proposals_num, gt_boxes.dims()[0]},
                                          context.GetPlace());

  Tensor boxes;
  boxes.mutable_data<T>({proposals_num, kBoxDim}, context.GetPlace());
  Concat<T>(context, gt_boxes, rpn_rois, &boxes);
  BboxOverlaps<T>(boxes, gt_boxes, &proposal_to_gt_overlaps);

  Tensor proposal_with_max_overlap;
  proposal_with_max_overlap.mutable_data<T>({proposals_num},
                                            context.GetPlace());

  MaxIoU<T>(proposal_to_gt_overlaps, &proposal_with_max_overlap);

  // Generate proposal index
  std::vector<std::vector<int>> fg_bg_gt =
      SampleFgBgGt<T>(context, &proposal_to_gt_overlaps, is_crowd,
                      batch_size_per_im, fg_fraction, fg_thresh, bg_thresh_hi,
                      bg_thresh_lo, engine, use_random, is_cascade_rcnn, boxes);
  std::vector<int> fg_inds = fg_bg_gt[0];
  std::vector<int> bg_inds = fg_bg_gt[1];
  std::vector<int> mapped_gt_inds = fg_bg_gt[2];  // mapped_gt_labels

  // Gather boxes and labels
  Tensor sampled_boxes, sampled_labels, sampled_gts, sampled_max_overlap;
  int fg_num = fg_inds.size();
  int bg_num = bg_inds.size();
  int boxes_num = fg_num + bg_num;
  framework::DDim bbox_dim({boxes_num, kBoxDim});
  sampled_boxes.mutable_data<T>(bbox_dim, context.GetPlace());
  sampled_labels.mutable_data<int>({boxes_num}, context.GetPlace());
  sampled_gts.mutable_data<T>({fg_num, kBoxDim}, context.GetPlace());
  sampled_max_overlap.mutable_data<T>({boxes_num}, context.GetPlace());
  GatherBoxesLabels<T>(context, boxes, proposal_with_max_overlap, gt_boxes,
                       gt_classes, fg_inds, bg_inds, mapped_gt_inds,
                       &sampled_boxes, &sampled_labels, &sampled_gts,
                       &sampled_max_overlap);

  // Compute targets
  Tensor bbox_targets_single;
  bbox_targets_single.mutable_data<T>(bbox_dim, context.GetPlace());
  BoxToDelta<T>(fg_num, sampled_boxes, sampled_gts, bbox_reg_weights.data(),
                false, &bbox_targets_single);

  // Scale rois
  Tensor sampled_rois;
  sampled_rois.mutable_data<T>(sampled_boxes.dims(), context.GetPlace());
  auto sampled_rois_et = framework::EigenTensor<T, 2>::From(sampled_rois);
  auto sampled_boxes_et = framework::EigenTensor<T, 2>::From(sampled_boxes);
  sampled_rois_et = sampled_boxes_et * im_scale;

  // Expand box targets
  Tensor bbox_targets, bbox_inside_weights, bbox_outside_weights;
  framework::DDim bbox_expand_dim({boxes_num, kBoxDim * class_nums});
  bbox_targets.mutable_data<T>(bbox_expand_dim, context.GetPlace());
  bbox_inside_weights.mutable_data<T>(bbox_expand_dim, context.GetPlace());
  bbox_outside_weights.mutable_data<T>(bbox_expand_dim, context.GetPlace());
  math::set_constant(context, &bbox_targets, 0.0);
  math::set_constant(context, &bbox_inside_weights, 0.0);
  math::set_constant(context, &bbox_outside_weights, 0.0);

  auto* bbox_targets_single_data = bbox_targets_single.data<T>();
  auto* sampled_labels_data = sampled_labels.data<int>();
  auto* bbox_targets_data = bbox_targets.data<T>();
  auto* bbox_inside_weights_data = bbox_inside_weights.data<T>();
  auto* bbox_outside_weights_data = bbox_outside_weights.data<T>();
  int width = kBoxDim * class_nums;
  for (int64_t i = 0; i < boxes_num; ++i) {
    int label = sampled_labels_data[i];
    if (label > 0) {
      if (is_cls_agnostic) {
        label = 1;
      }
      int dst_idx = i * width + kBoxDim * label;
      int src_idx = kBoxDim * i;
      bbox_targets_data[dst_idx] = bbox_targets_single_data[src_idx];
      bbox_targets_data[dst_idx + 1] = bbox_targets_single_data[src_idx + 1];
      bbox_targets_data[dst_idx + 2] = bbox_targets_single_data[src_idx + 2];
      bbox_targets_data[dst_idx + 3] = bbox_targets_single_data[src_idx + 3];
      bbox_inside_weights_data[dst_idx] = 1;
      bbox_inside_weights_data[dst_idx + 1] = 1;
      bbox_inside_weights_data[dst_idx + 2] = 1;
      bbox_inside_weights_data[dst_idx + 3] = 1;
      bbox_outside_weights_data[dst_idx] = 1;
      bbox_outside_weights_data[dst_idx + 1] = 1;
      bbox_outside_weights_data[dst_idx + 2] = 1;
      bbox_outside_weights_data[dst_idx + 3] = 1;
    }
  }
  std::vector<Tensor> res;
  res.emplace_back(sampled_rois);
  res.emplace_back(sampled_labels);
  res.emplace_back(bbox_targets);
  res.emplace_back(bbox_inside_weights);
  res.emplace_back(bbox_outside_weights);
  res.emplace_back(sampled_max_overlap);
  return res;
}

template <typename T>
class GenerateProposalLabelsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* rpn_rois = context.Input<LoDTensor>("RpnRois");
    auto* gt_classes = context.Input<LoDTensor>("GtClasses");
    auto* is_crowd = context.Input<LoDTensor>("IsCrowd");
    auto* gt_boxes = context.Input<LoDTensor>("GtBoxes");
    auto* im_info = context.Input<LoDTensor>("ImInfo");

    auto* rois = context.Output<LoDTensor>("Rois");
    auto* labels_int32 = context.Output<LoDTensor>("LabelsInt32");
    auto* bbox_targets = context.Output<LoDTensor>("BboxTargets");
    auto* bbox_inside_weights = context.Output<LoDTensor>("BboxInsideWeights");
    auto* bbox_outside_weights =
        context.Output<LoDTensor>("BboxOutsideWeights");
    auto* max_overlap_with_gt = context.Output<LoDTensor>("MaxOverlapWithGT");

    int batch_size_per_im = context.Attr<int>("batch_size_per_im");
    float fg_fraction = context.Attr<float>("fg_fraction");
    float fg_thresh = context.Attr<float>("fg_thresh");
    float bg_thresh_hi = context.Attr<float>("bg_thresh_hi");
    float bg_thresh_lo = context.Attr<float>("bg_thresh_lo");
    std::vector<float> bbox_reg_weights =
        context.Attr<std::vector<float>>("bbox_reg_weights");
    int class_nums = context.Attr<int>("class_nums");
    bool use_random = context.Attr<bool>("use_random");
    bool is_cascade_rcnn = context.Attr<bool>("is_cascade_rcnn");
    bool is_cls_agnostic = context.Attr<bool>("is_cls_agnostic");
    PADDLE_ENFORCE_EQ(
        rpn_rois->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateProposalLabelsOp rpn_rois needs 1 level of LoD. But "
            "received level of LoD is [%d], LoD is [%s].",
            rpn_rois->lod().size(), rpn_rois->lod()));
    PADDLE_ENFORCE_EQ(
        gt_classes->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateProposalLabelsOp gt_classes needs 1 level of LoD. But "
            "received level of LoD is [%d], LoD is [%s].",
            gt_classes->lod().size(), gt_classes->lod()));
    PADDLE_ENFORCE_EQ(
        is_crowd->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateProposalLabelsOp is_crowd needs 1 level of LoD. But "
            "received level of LoD is [%d], LoD is [%s].",
            is_crowd->lod().size(), is_crowd->lod()));
    PADDLE_ENFORCE_EQ(
        gt_boxes->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateProposalLabelsOp gt_boxes needs 1 level of LoD. But "
            "received level of LoD is [%d], LoD is [%s].",
            gt_boxes->lod().size(), gt_boxes->lod()));
    auto n = static_cast<int64_t>(rpn_rois->lod().back().size() - 1);
    int64_t rois_num = rpn_rois->dims()[0];
    int64_t gts_num = gt_boxes->dims()[0];
    int64_t init_num =
        is_cascade_rcnn ? rois_num + gts_num : n * batch_size_per_im;

    rois->mutable_data<T>({init_num, kBoxDim}, context.GetPlace());
    labels_int32->mutable_data<int>({init_num, 1}, context.GetPlace());
    bbox_targets->mutable_data<T>({init_num, kBoxDim * class_nums},
                                  context.GetPlace());
    bbox_inside_weights->mutable_data<T>({init_num, kBoxDim * class_nums},
                                         context.GetPlace());
    bbox_outside_weights->mutable_data<T>({init_num, kBoxDim * class_nums},
                                          context.GetPlace());
    max_overlap_with_gt->Resize({init_num});
    max_overlap_with_gt->mutable_data<T>(context.GetPlace());

    std::random_device rnd;
    std::minstd_rand engine;
    int seed = rnd();
    engine.seed(seed);

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_rois = 0;
    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    auto rpn_rois_lod = rpn_rois->lod().back();
    auto gt_classes_lod = gt_classes->lod().back();
    auto is_crowd_lod = is_crowd->lod().back();
    auto gt_boxes_lod = gt_boxes->lod().back();
    for (int i = 0; i < n; ++i) {
      if (rpn_rois_lod[i] == rpn_rois_lod[i + 1]) {
        lod0.emplace_back(num_rois);
        continue;
      }
      Tensor rpn_rois_slice =
          rpn_rois->Slice(rpn_rois_lod[i], rpn_rois_lod[i + 1]);
      Tensor gt_classes_slice =
          gt_classes->Slice(gt_classes_lod[i], gt_classes_lod[i + 1]);
      Tensor is_crowd_slice =
          is_crowd->Slice(is_crowd_lod[i], is_crowd_lod[i + 1]);
      Tensor gt_boxes_slice =
          gt_boxes->Slice(gt_boxes_lod[i], gt_boxes_lod[i + 1]);
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor max_overlap_slice;
      if (is_cascade_rcnn) {
        auto* max_overlap = context.Input<Tensor>("MaxOverlap");
        max_overlap_slice =
            max_overlap->Slice(rpn_rois_lod[i], rpn_rois_lod[i + 1]);
      } else {
        max_overlap_slice.mutable_data<T>({rpn_rois_slice.dims()[0]},
                                          context.GetPlace());
      }
      std::vector<Tensor> tensor_output = SampleRoisForOneImage<T>(
          dev_ctx, rpn_rois_slice, gt_classes_slice, is_crowd_slice,
          gt_boxes_slice, im_info_slice, batch_size_per_im, fg_fraction,
          fg_thresh, bg_thresh_hi, bg_thresh_lo, bbox_reg_weights, class_nums,
          engine, use_random, is_cascade_rcnn, is_cls_agnostic,
          max_overlap_slice);
      Tensor sampled_rois = tensor_output[0];
      Tensor sampled_labels_int32 = tensor_output[1];
      Tensor sampled_bbox_targets = tensor_output[2];
      Tensor sampled_bbox_inside_weights = tensor_output[3];
      Tensor sampled_bbox_outside_weights = tensor_output[4];
      Tensor sampled_max_overlap = tensor_output[5];

      AppendRois<T>(rois, kBoxDim * num_rois, &sampled_rois);
      AppendRois<int>(labels_int32, num_rois, &sampled_labels_int32);
      int64_t offset = kBoxDim * num_rois * class_nums;
      AppendRois<T>(bbox_targets, offset, &sampled_bbox_targets);
      AppendRois<T>(bbox_inside_weights, offset, &sampled_bbox_inside_weights);
      AppendRois<T>(bbox_outside_weights, offset,
                    &sampled_bbox_outside_weights);
      AppendRois<T>(max_overlap_with_gt, num_rois, &sampled_max_overlap);

      num_rois += sampled_rois.dims()[0];
      lod0.emplace_back(num_rois);
    }

    lod.emplace_back(lod0);
    rois->set_lod(lod);
    labels_int32->set_lod(lod);
    bbox_targets->set_lod(lod);
    bbox_inside_weights->set_lod(lod);
    bbox_outside_weights->set_lod(lod);
    rois->Resize({num_rois, kBoxDim});
    labels_int32->Resize({num_rois, 1});
    bbox_targets->Resize({num_rois, kBoxDim * class_nums});
    bbox_inside_weights->Resize({num_rois, kBoxDim * class_nums});
    bbox_outside_weights->Resize({num_rois, kBoxDim * class_nums});
    max_overlap_with_gt->Resize({num_rois});
    max_overlap_with_gt->set_lod(lod);
  }
};

class GenerateProposalLabelsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "RpnRois",
        "(LoDTensor), This input is a 2D LoDTensor with shape [N, 4]. "
        "N is the number of the GenerateProposalOp's output, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddInput("GtClasses",
             "(LoDTensor), This input is a 2D LoDTensor with shape [M, 1]. "
             "M is the number of groundtruth, "
             "each element is a class label of groundtruth.");
    AddInput(
        "IsCrowd",
        "(LoDTensor), This input is a 2D LoDTensor with shape [M, 1]. "
        "M is the number of groundtruth, "
        "each element is a flag indicates whether a groundtruth is crowd.");
    AddInput(
        "GtBoxes",
        "(LoDTensor), This input is a 2D LoDTensor with shape [M, 4]. "
        "M is the number of groundtruth, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddInput("ImInfo",
             "(Tensor), This input is a 2D Tensor with shape [B, 3]. "
             "B is the number of input images, "
             "each element consists of im_height, im_width, im_scale.");
    AddInput("MaxOverlap",
             "(LoDTensor), This input is a 1D LoDTensor with shape [N]."
             "N is the number of Input(RpnRois), "
             "each element is the maximum overlap between "
             "the proposal RoI and ground-truth.")
        .AsDispensable();

    AddOutput(
        "Rois",
        "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4]. "
        "P usuall equal to  batch_size_per_im * batch_size, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddOutput("LabelsInt32",
              "(LoDTensor), This output is a 2D LoDTensor with shape [P, 1], "
              "each element represents a class label of a roi");
    AddOutput("BboxTargets",
              "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4 * "
              "class_nums], "
              "each element represents a box label of a roi");
    AddOutput(
        "BboxInsideWeights",
        "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4 * "
        "class_nums], "
        "each element indicates whether a box should contribute to loss.");
    AddOutput(
        "BboxOutsideWeights",
        "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4 * "
        "class_nums], "
        "each element indicates whether a box should contribute to loss.");
    AddOutput("MaxOverlapWithGT",
              "(LoDTensor), This output is a 1D LoDTensor with shape [P], "
              "each element indicates the maxoverlap "
              "between output RoIs and ground-truth. "
              "The output RoIs may include ground-truth "
              "and the output maxoverlap may contain 1.");

    AddAttr<int>("batch_size_per_im", "Batch size of rois per images.");
    AddAttr<float>("fg_fraction",
                   "Foreground fraction in total batch_size_per_im.");
    AddAttr<float>(
        "fg_thresh",
        "Overlap threshold which is used to chose foreground sample.");
    AddAttr<float>("bg_thresh_hi",
                   "Overlap threshold upper bound which is used to chose "
                   "background sample.");
    AddAttr<float>("bg_thresh_lo",
                   "Overlap threshold lower bound which is used to chose "
                   "background sample.");
    AddAttr<std::vector<float>>("bbox_reg_weights", "Box regression weights.");
    AddAttr<int>("class_nums", "Class number.");
    AddAttr<bool>(
        "use_random",
        "Use random sampling to choose foreground and background boxes.")
        .SetDefault(true);
    AddAttr<bool>("is_cascade_rcnn",
                  "cascade rcnn sampling policy changed from stage 2.")
        .SetDefault(false);
    AddAttr<bool>(
        "is_cls_agnostic",
        "the box regress will only include fg and bg locations if set true ")
        .SetDefault(false);

    AddComment(R"DOC(
This operator can be, for given the GenerateProposalOp output bounding boxes and groundtruth,
to sample foreground boxes and background boxes, and compute loss target.

RpnRois is the output boxes of RPN and was processed by generate_proposal_op, these boxes
were combined with groundtruth boxes and sampled according to batch_size_per_im and fg_fraction,
If an instance with a groundtruth overlap greater than fg_thresh, then it was considered as a foreground sample.
If an instance with a groundtruth overlap greater than bg_thresh_lo and lower than bg_thresh_hi,
then it was considered as a background sample.
After all foreground and background boxes are chosen (so called Rois),
then we apply random sampling to make sure
the number of foreground boxes is no more than batch_size_per_im * fg_fraction.

For each box in Rois, we assign the classification (class label) and regression targets (box label) to it.
Finally BboxInsideWeights and BboxOutsideWeights are used to specify whether it would contribute to training loss.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    generate_proposal_labels, ops::GenerateProposalLabelsOp,
    ops::GenerateProposalLabelsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(generate_proposal_labels,
                       ops::GenerateProposalLabelsKernel<float>,
                       ops::GenerateProposalLabelsKernel<double>);

REGISTER_OP_VERSION(generate_proposal_labels)
    .AddCheckpoint(
        R"ROC(
              Upgrade of output [MaxOverlapWithGT])ROC",
        paddle::framework::compatible::OpVersionDesc().NewOutput(
            "MaxOverlapWithGT",
            "The maxoverlap between output RoIs and ground-truth."))
    .AddCheckpoint(
        R"ROC(
              Upgrade generate_proposal_labels add a new input [MaxOverlap])ROC",
        paddle::framework::compatible::OpVersionDesc().NewInput(
            "MaxOverlap", "MaxOverlap is dispensable."));
