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

#include <random>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

class RpnTargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Anchor"), "Input", "Anchor",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("GtBoxes"), "Input", "GtBoxes",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("IsCrowd"), "Input", "IsCrowd",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("ImInfo"), "Input", "ImInfo",
                   "rpn_target_assign");

    OP_INOUT_CHECK(ctx->HasOutput("LocationIndex"), "Output", "LocationIndex",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("ScoreIndex"), "Output", "ScoreIndex",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("TargetLabel"), "Output", "TargetLabel",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("TargetBBox"), "Output", "TargetBBox",
                   "rpn_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("BBoxInsideWeight"), "Output",
                   "BBoxInsideWeight", "rpn_target_assign");

    auto anchor_dims = ctx->GetInputDim("Anchor");
    auto gt_boxes_dims = ctx->GetInputDim("GtBoxes");
    auto im_info_dims = ctx->GetInputDim("ImInfo");
    PADDLE_ENFORCE_EQ(anchor_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The dimensions size of Input(Anchor) must be 2. But "
                          "received dimensions size=[%d], dimensions=[%s].",
                          anchor_dims.size(), anchor_dims));
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

    ctx->SetOutputDim("LocationIndex", {-1});
    ctx->SetOutputDim("ScoreIndex", {-1});
    ctx->SetOutputDim("TargetLabel", {-1, 1});
    ctx->SetOutputDim("TargetBBox", {-1, 4});
    ctx->SetOutputDim("BBoxInsideWeight", {-1, 4});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Anchor"),
        platform::CPUPlace());
  }
};

template <typename T>
void AppendRpns(LoDTensor* out, int64_t offset, Tensor* to_add) {
  auto* out_data = out->data<T>();
  auto* to_add_data = to_add->data<T>();
  memcpy(out_data + offset, to_add_data, to_add->numel() * sizeof(T));
}

template <typename T>
std::vector<Tensor> FilterStraddleAnchor(
    const platform::CPUDeviceContext& context, const Tensor* anchor,
    const float rpn_straddle_thresh, T im_height, T im_width) {
  std::vector<int> inds_inside;
  int anchor_num = anchor->dims()[0];
  auto* anchor_data = anchor->data<T>();
  if (rpn_straddle_thresh >= 0) {
    int index;
    for (int i = 0; i < anchor_num; ++i) {
      index = i * 4;
      if ((anchor_data[index + 0] >= -rpn_straddle_thresh) &&
          (anchor_data[index + 1] >= -rpn_straddle_thresh) &&
          (anchor_data[index + 2] < im_width + rpn_straddle_thresh) &&
          (anchor_data[index + 3] < im_height + rpn_straddle_thresh)) {
        inds_inside.emplace_back(i);
      }
    }
  } else {
    for (int i = 0; i < anchor_num; ++i) {
      inds_inside.emplace_back(i);
    }
  }
  int inside_num = inds_inside.size();
  Tensor inds_inside_t;
  int* inds_inside_data =
      inds_inside_t.mutable_data<int>({inside_num}, context.GetPlace());
  std::copy(inds_inside.begin(), inds_inside.end(), inds_inside_data);
  Tensor inside_anchor_t;
  T* inside_anchor_data =
      inside_anchor_t.mutable_data<T>({inside_num, 4}, context.GetPlace());
  Gather<T>(anchor->data<T>(), 4, inds_inside_data, inside_num,
            inside_anchor_data);
  std::vector<Tensor> res;
  res.emplace_back(inds_inside_t);
  res.emplace_back(inside_anchor_t);
  return res;
}

template <typename T>
Tensor FilterCrowdGt(const platform::CPUDeviceContext& context,
                     Tensor* gt_boxes, Tensor* is_crowd) {
  int gt_num = gt_boxes->dims()[0];
  std::vector<int> not_crowd_inds;
  auto* is_crowd_data = is_crowd->data<int>();
  for (int i = 0; i < gt_num; ++i) {
    if (is_crowd_data[i] == 0) {
      not_crowd_inds.emplace_back(i);
    }
  }
  int ncrowd_num = not_crowd_inds.size();
  Tensor ncrowd_gt_boxes;
  T* ncrowd_gt_boxes_data =
      ncrowd_gt_boxes.mutable_data<T>({ncrowd_num, 4}, context.GetPlace());
  Gather<T>(gt_boxes->data<T>(), 4, not_crowd_inds.data(), ncrowd_num,
            ncrowd_gt_boxes_data);
  return ncrowd_gt_boxes;
}

void ReservoirSampling(const int num, std::vector<int>* inds,
                       std::minstd_rand engine, bool use_random) {
  std::uniform_real_distribution<float> uniform(0, 1);
  size_t len = inds->size();
  if (len > static_cast<size_t>(num)) {
    if (use_random) {
      for (size_t i = num; i < len; ++i) {
        int rng_ind = std::floor(uniform(engine) * i);
        if (rng_ind < num)
          std::iter_swap(inds->begin() + rng_ind, inds->begin() + i);
      }
    }
    inds->resize(num);
  }
}

template <typename T>
void ScoreAssign(const T* anchor_by_gt_overlap_data,
                 const Tensor& anchor_to_gt_max, const Tensor& gt_to_anchor_max,
                 const int rpn_batch_size_per_im, const float rpn_fg_fraction,
                 const float rpn_positive_overlap,
                 const float rpn_negative_overlap, std::vector<int>* fg_inds,
                 std::vector<int>* bg_inds, std::vector<int>* tgt_lbl,
                 std::vector<int>* fg_fake, std::vector<T>* bbox_inside_weight,
                 std::minstd_rand engine, bool use_random) {
  float epsilon = 0.00001;
  int anchor_num = anchor_to_gt_max.dims()[0];
  int gt_num = gt_to_anchor_max.dims()[0];
  std::vector<int> target_label(anchor_num, -1);
  std::vector<int> fg_inds_fake;
  std::vector<int> bg_inds_fake;
  const T* anchor_to_gt_max_data = anchor_to_gt_max.data<T>();
  const T* gt_to_anchor_max_data = gt_to_anchor_max.data<T>();
  // TODO(buxingyuan): Match with Detectron now
  // but it seems here is a bug in two directions assignment
  // in which the later one may overwrites the former one.
  for (int64_t i = 0; i < anchor_num; ++i) {
    bool is_anchors_with_max_overlap = false;
    for (int64_t j = 0; j < gt_num; ++j) {
      T value = anchor_by_gt_overlap_data[i * gt_num + j];
      T diff = std::abs(value - gt_to_anchor_max_data[j]);
      if (diff < epsilon) {
        is_anchors_with_max_overlap = true;
        break;
      }
    }
    bool is_anchor_great_than_thresh =
        (anchor_to_gt_max_data[i] >= rpn_positive_overlap);
    if (is_anchors_with_max_overlap || is_anchor_great_than_thresh) {
      fg_inds_fake.push_back(i);
    }
  }

  // Reservoir Sampling
  int fg_num = 0;
  if (rpn_fg_fraction > 0 && rpn_batch_size_per_im > 0) {
    fg_num = static_cast<int>(rpn_fg_fraction * rpn_batch_size_per_im);
    ReservoirSampling(fg_num, &fg_inds_fake, engine, use_random);
  } else {
    fg_num = static_cast<int>(fg_inds_fake.size());
  }
  int fg_fake_num = static_cast<int>(fg_inds_fake.size());
  for (int64_t i = 0; i < fg_fake_num; ++i) {
    target_label[fg_inds_fake[i]] = 1;
  }

  for (int64_t i = 0; i < anchor_num; ++i) {
    if (anchor_to_gt_max_data[i] < rpn_negative_overlap) {
      bg_inds_fake.push_back(i);
    }
  }
  int bg_num = 0;
  if (rpn_fg_fraction > 0 && rpn_batch_size_per_im > 0) {
    bg_num = rpn_batch_size_per_im - fg_fake_num;
    ReservoirSampling(bg_num, &bg_inds_fake, engine, use_random);
    bg_num = static_cast<int>(bg_inds_fake.size());
  } else {
    bg_num = static_cast<int>(bg_inds_fake.size());
  }

  int fake_num = 0;
  for (int64_t i = 0; i < bg_num; ++i) {
    // fg fake found
    if (target_label[bg_inds_fake[i]] == 1) {
      fake_num++;
      fg_fake->emplace_back(fg_inds_fake[0]);
      for (int j = 0; j < 4; ++j) {
        bbox_inside_weight->emplace_back(T(0.));
      }
    }
    target_label[bg_inds_fake[i]] = 0;
  }

  for (int64_t i = 0; i < (fg_fake_num - fake_num) * 4; ++i) {
    bbox_inside_weight->emplace_back(T(1.));
  }

  for (int64_t i = 0; i < anchor_num; ++i) {
    if (target_label[i] == 1) {
      fg_inds->emplace_back(i);
      fg_fake->emplace_back(i);
    }
    if (target_label[i] == 0) bg_inds->emplace_back(i);
  }
  fg_num = fg_inds->size();
  bg_num = bg_inds->size();

  tgt_lbl->resize(fg_num + bg_num, 0);
  std::vector<int> fg_lbl(fg_num, 1);
  std::vector<int> bg_lbl(bg_num, 0);
  std::copy(fg_lbl.begin(), fg_lbl.end(), tgt_lbl->data());
  std::copy(bg_lbl.begin(), bg_lbl.end(), tgt_lbl->data() + fg_num);
}

template <typename T>
std::vector<Tensor> SampleRpnFgBgGt(const platform::CPUDeviceContext& ctx,
                                    const Tensor& anchor_by_gt_overlap,
                                    const int rpn_batch_size_per_im,
                                    const float rpn_positive_overlap,
                                    const float rpn_negative_overlap,
                                    const float rpn_fg_fraction,
                                    std::minstd_rand engine, bool use_random) {
  auto* anchor_by_gt_overlap_data = anchor_by_gt_overlap.data<T>();
  int anchor_num = anchor_by_gt_overlap.dims()[0];
  int gt_num = anchor_by_gt_overlap.dims()[1];

  std::vector<int> fg_inds;
  std::vector<int> bg_inds;
  std::vector<int> gt_inds;
  std::vector<int> tgt_lbl;
  std::vector<int> fg_fake;
  std::vector<T> bbox_inside_weight;
  // Calculate the max IoU between anchors and gt boxes
  // Map from anchor to gt box that has highest overlap
  auto place = ctx.GetPlace();
  Tensor anchor_to_gt_max, anchor_to_gt_argmax, gt_to_anchor_max;
  anchor_to_gt_max.mutable_data<T>({anchor_num}, place);
  int* argmax = anchor_to_gt_argmax.mutable_data<int>({anchor_num}, place);
  gt_to_anchor_max.mutable_data<T>({gt_num}, place);

  auto anchor_by_gt_overlap_et =
      framework::EigenMatrix<T>::From(anchor_by_gt_overlap);
  auto anchor_to_gt_max_et =
      framework::EigenVector<T>::Flatten(anchor_to_gt_max);
  auto gt_to_anchor_max_et =
      framework::EigenVector<T>::Flatten(gt_to_anchor_max);
  auto anchor_to_gt_argmax_et =
      framework::EigenVector<int>::Flatten(anchor_to_gt_argmax);
  anchor_to_gt_max_et =
      anchor_by_gt_overlap_et.maximum(Eigen::DSizes<int, 1>(1));
  anchor_to_gt_argmax_et =
      anchor_by_gt_overlap_et.argmax(1).template cast<int>();
  gt_to_anchor_max_et =
      anchor_by_gt_overlap_et.maximum(Eigen::DSizes<int, 1>(0));

  // Follow the Faster RCNN's implementation
  ScoreAssign(anchor_by_gt_overlap_data, anchor_to_gt_max, gt_to_anchor_max,
              rpn_batch_size_per_im, rpn_fg_fraction, rpn_positive_overlap,
              rpn_negative_overlap, &fg_inds, &bg_inds, &tgt_lbl, &fg_fake,
              &bbox_inside_weight, engine, use_random);

  int fg_num = fg_inds.size();
  int bg_num = bg_inds.size();
  int fg_fake_num = fg_fake.size();
  gt_inds.reserve(fg_fake_num);
  for (int i = 0; i < fg_fake_num; ++i) {
    gt_inds.emplace_back(argmax[fg_fake[i]]);
  }
  Tensor loc_index_t, score_index_t, tgt_lbl_t, gt_inds_t, bbox_inside_weight_t;
  int* loc_index_data = loc_index_t.mutable_data<int>({fg_fake_num}, place);
  int* score_index_data =
      score_index_t.mutable_data<int>({fg_num + bg_num}, place);
  int* tgt_lbl_data = tgt_lbl_t.mutable_data<int>({fg_num + bg_num}, place);
  int* gt_inds_data = gt_inds_t.mutable_data<int>({fg_fake_num}, place);
  T* bbox_inside_weight_data =
      bbox_inside_weight_t.mutable_data<T>({fg_fake_num, 4}, place);
  std::copy(fg_fake.begin(), fg_fake.end(), loc_index_data);
  std::copy(fg_inds.begin(), fg_inds.end(), score_index_data);
  std::copy(bg_inds.begin(), bg_inds.end(), score_index_data + fg_num);
  std::copy(tgt_lbl.begin(), tgt_lbl.end(), tgt_lbl_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_inds_data);
  std::copy(bbox_inside_weight.begin(), bbox_inside_weight.end(),
            bbox_inside_weight_data);
  std::vector<Tensor> loc_score_tgtlbl_gt;
  loc_score_tgtlbl_gt.emplace_back(loc_index_t);
  loc_score_tgtlbl_gt.emplace_back(score_index_t);
  loc_score_tgtlbl_gt.emplace_back(tgt_lbl_t);
  loc_score_tgtlbl_gt.emplace_back(gt_inds_t);
  loc_score_tgtlbl_gt.emplace_back(bbox_inside_weight_t);

  return loc_score_tgtlbl_gt;
}

template <typename T>
class RpnTargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* anchor = context.Input<Tensor>("Anchor");  // (H*W*A) * 4
    auto* gt_boxes = context.Input<LoDTensor>("GtBoxes");
    auto* is_crowd = context.Input<LoDTensor>("IsCrowd");
    auto* im_info = context.Input<LoDTensor>("ImInfo");

    auto* loc_index = context.Output<LoDTensor>("LocationIndex");
    auto* score_index = context.Output<LoDTensor>("ScoreIndex");
    auto* tgt_bbox = context.Output<LoDTensor>("TargetBBox");
    auto* tgt_lbl = context.Output<LoDTensor>("TargetLabel");
    auto* bbox_inside_weight = context.Output<LoDTensor>("BBoxInsideWeight");

    PADDLE_ENFORCE_EQ(gt_boxes->lod().size(), 1UL,
                      platform::errors::InvalidArgument(
                          "RpnTargetAssignOp gt_boxes needs 1 level of LoD. "
                          "But received level of LoD is [%d], LoD is [%s].",
                          gt_boxes->lod().size(), gt_boxes->lod()));
    PADDLE_ENFORCE_EQ(is_crowd->lod().size(), 1UL,
                      platform::errors::InvalidArgument(
                          "RpnTargetAssignOp is_crowd needs 1 level of LoD. "
                          "But received level of LoD is [%d], LoD is [%s].",
                          is_crowd->lod().size(), is_crowd->lod()));
    auto anchor_num = static_cast<int64_t>(anchor->dims()[0]);
    auto batch_num = static_cast<int64_t>(gt_boxes->lod().back().size() - 1);

    int rpn_batch_size_per_im = context.Attr<int>("rpn_batch_size_per_im");
    float rpn_straddle_thresh = context.Attr<float>("rpn_straddle_thresh");
    float rpn_positive_overlap = context.Attr<float>("rpn_positive_overlap");
    float rpn_negative_overlap = context.Attr<float>("rpn_negative_overlap");
    float rpn_fg_fraction = context.Attr<float>("rpn_fg_fraction");
    bool use_random = context.Attr<bool>("use_random");

    int64_t max_num = batch_num * rpn_batch_size_per_im;
    auto place = context.GetPlace();

    loc_index->mutable_data<int>({max_num}, place);
    score_index->mutable_data<int>({max_num}, place);
    tgt_bbox->mutable_data<T>({max_num, 4}, place);
    tgt_lbl->mutable_data<int>({max_num, 1}, place);
    bbox_inside_weight->mutable_data<T>({max_num, 4}, place);
    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    std::random_device rnd;
    std::minstd_rand engine;
    int seed = rnd();
    engine.seed(seed);

    framework::LoD lod_loc, loc_score;
    std::vector<size_t> lod0_loc(1, 0);
    std::vector<size_t> lod0_score(1, 0);

    int total_loc_num = 0;
    int total_score_num = 0;
    auto gt_boxes_lod = gt_boxes->lod().back();
    auto is_crowd_lod = is_crowd->lod().back();
    for (int i = 0; i < batch_num; ++i) {
      Tensor gt_boxes_slice =
          gt_boxes->Slice(gt_boxes_lod[i], gt_boxes_lod[i + 1]);
      Tensor is_crowd_slice =
          is_crowd->Slice(is_crowd_lod[i], is_crowd_lod[i + 1]);
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      auto* im_info_data = im_info_slice.data<T>();
      auto im_height = im_info_data[0];
      auto im_width = im_info_data[1];
      auto im_scale = im_info_data[2];

      // Filter straddle anchor
      std::vector<Tensor> filter_output = FilterStraddleAnchor<T>(
          dev_ctx, anchor, rpn_straddle_thresh, im_height, im_width);
      Tensor inds_inside = filter_output[0];
      Tensor inside_anchor = filter_output[1];

      // Filter crowd gt
      Tensor ncrowd_gt_boxes =
          FilterCrowdGt<T>(dev_ctx, &gt_boxes_slice, &is_crowd_slice);
      auto ncrowd_gt_boxes_et =
          framework::EigenTensor<T, 2>::From(ncrowd_gt_boxes);
      ncrowd_gt_boxes_et = ncrowd_gt_boxes_et * im_scale;

      Tensor anchor_by_gt_overlap;
      anchor_by_gt_overlap.mutable_data<T>(
          {inside_anchor.dims()[0], ncrowd_gt_boxes.dims()[0]}, place);
      BboxOverlaps<T>(inside_anchor, ncrowd_gt_boxes, &anchor_by_gt_overlap);

      auto loc_score_tgtlbl_gt = SampleRpnFgBgGt<T>(
          dev_ctx, anchor_by_gt_overlap, rpn_batch_size_per_im,
          rpn_positive_overlap, rpn_negative_overlap, rpn_fg_fraction, engine,
          use_random);

      Tensor sampled_loc_index = loc_score_tgtlbl_gt[0];
      Tensor sampled_score_index = loc_score_tgtlbl_gt[1];
      Tensor sampled_tgtlbl = loc_score_tgtlbl_gt[2];
      Tensor sampled_gt_index = loc_score_tgtlbl_gt[3];
      Tensor sampled_bbox_inside_weight = loc_score_tgtlbl_gt[4];

      int loc_num = sampled_loc_index.dims()[0];
      int score_num = sampled_score_index.dims()[0];
      // unmap to all anchor
      Tensor sampled_loc_index_unmap, sampled_score_index_unmap;
      sampled_loc_index_unmap.mutable_data<int>({loc_num}, place);
      sampled_score_index_unmap.mutable_data<int>({score_num}, place);
      Gather<int>(inds_inside.data<int>(), 1, sampled_loc_index.data<int>(),
                  loc_num, sampled_loc_index_unmap.data<int>());
      Gather<int>(inds_inside.data<int>(), 1, sampled_score_index.data<int>(),
                  score_num, sampled_score_index_unmap.data<int>());

      // get target bbox deltas
      Tensor sampled_anchor, sampled_gt, sampled_tgt_bbox;
      auto* sampled_anchor_data =
          sampled_anchor.mutable_data<T>({loc_num, 4}, place);
      auto* sampled_gt_data = sampled_gt.mutable_data<T>({loc_num, 4}, place);
      Gather<T>(anchor->data<T>(), 4, sampled_loc_index_unmap.data<int>(),
                loc_num, sampled_anchor_data);
      Gather<T>(ncrowd_gt_boxes.data<T>(), 4, sampled_gt_index.data<int>(),
                loc_num, sampled_gt_data);
      sampled_tgt_bbox.mutable_data<T>({loc_num, 4}, place);
      BoxToDelta<T>(loc_num, sampled_anchor, sampled_gt, nullptr, false,
                    &sampled_tgt_bbox);

      // Add anchor offset
      int anchor_offset = i * anchor_num;
      auto sampled_loc_index_unmap_et =
          framework::EigenTensor<int, 1>::From(sampled_loc_index_unmap);
      sampled_loc_index_unmap_et = sampled_loc_index_unmap_et + anchor_offset;
      auto sampled_score_index_unmap_et =
          framework::EigenTensor<int, 1>::From(sampled_score_index_unmap);
      sampled_score_index_unmap_et =
          sampled_score_index_unmap_et + anchor_offset;
      AppendRpns<int>(loc_index, total_loc_num, &sampled_loc_index_unmap);
      AppendRpns<int>(score_index, total_score_num, &sampled_score_index_unmap);
      AppendRpns<T>(tgt_bbox, total_loc_num * 4, &sampled_tgt_bbox);
      AppendRpns<int>(tgt_lbl, total_score_num, &sampled_tgtlbl);
      AppendRpns<T>(bbox_inside_weight, total_loc_num * 4,
                    &sampled_bbox_inside_weight);
      total_loc_num += loc_num;

      total_score_num += score_num;
      lod0_loc.emplace_back(total_loc_num);
      lod0_score.emplace_back(total_score_num);
    }

    PADDLE_ENFORCE_LE(
        total_loc_num, max_num,
        platform::errors::InvalidArgument(
            "The number of sampled bboxes should not be greater than the "
            "number of all anchor boxes(%d), but the number of sampled "
            "bboxes is :%d.",
            max_num, total_loc_num));
    PADDLE_ENFORCE_LE(
        total_score_num, max_num,
        platform::errors::InvalidArgument(
            "The number of sampled scores should not be greater than the "
            "number of all anchor boxes(%d), but the number of sampled "
            "scores is :%d.",
            max_num, total_score_num));

    lod_loc.emplace_back(lod0_loc);
    loc_score.emplace_back(lod0_score);
    loc_index->set_lod(lod_loc);
    score_index->set_lod(loc_score);
    tgt_bbox->set_lod(lod_loc);
    tgt_lbl->set_lod(loc_score);
    bbox_inside_weight->set_lod(lod_loc);
    loc_index->Resize({total_loc_num});
    score_index->Resize({total_score_num});
    tgt_bbox->Resize({total_loc_num, 4});
    tgt_lbl->Resize({total_score_num, 1});
    bbox_inside_weight->Resize({total_loc_num, 4});
  }
};

class RpnTargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Anchor",
             "(Tensor) input anchor is a 2-D Tensor with shape [H*W*A, 4].");
    AddInput("GtBoxes",
             "(LoDTensor) input ground-truth bbox with shape [K, 4].");
    AddInput("IsCrowd",
             "(LoDTensor) input which indicates ground-truth is crowd.");
    AddInput("ImInfo",
             "(LoDTensor) input image information with shape [N, 3]. "
             "N is the batch size, each image information includes height, "
             "width and scale.");
    AddAttr<int>("rpn_batch_size_per_im",
                 "Total number of RPN examples per image.")
        .SetDefault(256);
    AddAttr<float>(
        "rpn_straddle_thresh",
        "Remove RPN anchors that go outside the image by straddle_thresh "
        "pixels, "
        "Set to -1 or a large value, e.g. 100000, to disable pruning anchors.");
    AddAttr<float>(
        "rpn_positive_overlap",
        "Minimum overlap required between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a positive example.")
        .SetDefault(0.7);
    AddAttr<float>(
        "rpn_negative_overlap",
        "Maximum overlap allowed between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a negative examples.")
        .SetDefault(0.3);
    AddAttr<float>(
        "rpn_fg_fraction",
        "Target fraction of RoI minibatch that "
        "is labeled foreground (i.e. class > 0), 0-th class is background.")
        .SetDefault(0.25);
    AddAttr<bool>("use_random",
                  "A flag indicating whether to use a ReservoirSampling. "
                  "NOTE: DO NOT set this flag to false in training. "
                  "Setting this flag to false is only useful in unittest.")
        .SetDefault(true);
    AddOutput(
        "LocationIndex",
        "(Tensor), The indexes of foreground anchors in all RPN anchors, the "
        "shape of the LocationIndex is [F], F depends on the value of input "
        "tensor and attributes.");
    AddOutput(
        "ScoreIndex",
        "(Tensor), The indexes of foreground and background anchors in all "
        "RPN anchors(The rest anchors are ignored). The shape of the "
        "ScoreIndex is [F + B], F and B are sampled foreground and background "
        " number.");
    AddOutput("TargetBBox",
              "(Tensor), The target bbox deltas with shape "
              "[F, 4], F is the sampled foreground number.");
    AddOutput(
        "TargetLabel",
        "(Tensor<int>), The target labels of each anchor with shape "
        "[F + B, 1], F and B are sampled foreground and background number.");
    AddOutput("BBoxInsideWeight",
              "(Tensor), The bbox inside weight with shape "
              "[F, 4], F is the sampled foreground number.");
    AddComment(R"DOC(
This operator can be, for a given set of ground truth bboxes and the
anchors, to assign classification and regression targets to each prediction.
The ScoreIndex and LocationIndex will be generated according to the anchor-groundtruth IOU.
The rest anchors would not contibute to the RPN training loss

ScoreIndex is composed of foreground anchor indexes(positive labels) and
background anchor indexes(negative labels). LocationIndex is exactly same
as the foreground anchor indexes since we can not assign regression target to 
the background anchors.

The classification targets(TargetLabel) is a binary class label (of being
an object or not). Following the paper of Faster-RCNN, the positive labels
are two kinds of anchors: (i) the anchor/anchors with the highest IoU
overlap with a ground-truth box, or (ii) an anchor that has an IoU overlap
higher than rpn_positive_overlap(0.7) with any ground-truth box. Note that
a single ground-truth box may assign positive labels to multiple anchors.
A non-positive anchor is when its IoU ratio is lower than rpn_negative_overlap
(0.3) for all ground-truth boxes. Anchors that are neither positive nor
negative do not contribute to the training objective.

)DOC");
  }
};

class RetinanetTargetAssignOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Anchor",
             "(Tensor) input anchor is a 2-D Tensor with shape [H*W*A, 4].");
    AddInput("GtBoxes",
             "(LoDTensor) input ground-truth bbox with shape [K, 4].");
    AddInput("GtLabels",
             "(LoDTensor) input ground-truth label with shape [K, 1].");
    AddInput("IsCrowd",
             "(LoDTensor) input which indicates ground-truth is crowd.");
    AddInput("ImInfo",
             "(LoDTensor) input image information with shape [N, 3]. "
             "N is the batch size, each image information includes height, "
             "width and scale.");
    AddAttr<float>(
        "positive_overlap",
        "Minimum overlap required between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a positive example.")
        .SetDefault(0.5);
    AddAttr<float>(
        "negative_overlap",
        "Maximum overlap allowed between an anchor and ground-truth "
        "box for the (anchor, gt box) pair to be a negative examples.")
        .SetDefault(0.4);
    AddOutput(
        "LocationIndex",
        "(Tensor), The indexes of foreground anchors in all anchors, the "
        "shape of the LocationIndex is [F], F depends on the value of input "
        "tensor and attributes.");
    AddOutput(
        "ScoreIndex",
        "(Tensor), The indexes of foreground and background anchors in all "
        "RPN anchors(The rest anchors are ignored). The shape of the "
        "ScoreIndex is [F + B], F and B are foreground and background "
        " number.");
    AddOutput("TargetBBox",
              "(Tensor), The target bbox deltas with shape "
              "[F, 4], F is the foreground number.");
    AddOutput("TargetLabel",
              "(Tensor<int>), The target labels of each anchor with shape "
              "[F + B, 1], F and B are foreground and background number.");
    AddOutput("BBoxInsideWeight",
              "(Tensor), The bbox inside weight with shape "
              "[F, 4], F is the foreground number.");
    AddOutput("ForegroundNumber",
              "(Tensor), The foreground number. "
              "[1, 1].");
    AddComment(R"DOC(
    This layer can be, for given the Intersection-over-Union (IoU) overlap
    between anchors and ground truth boxes, to assign classification and
    regression targets to each anchor, these target labels are used for
    train retinanet. 
    
    Every anchor is assigned with a length C one-hot vector of
    classification targets, and a 4-vector of box regression targets,
    where C is the class number. The assignment rules are as followed:
    
    1. Anchors are assigned to ground-truth boxes when: (i) it has the highest
    IoU overlap with a ground-truth box, or (ii) it has an IoU overlap higher
    than positive_overlap(0.5) with any ground-truth box.
    
    2. Anchors are assigned to background when its IoU ratio is lower than
    negative_overlap (0.4) for all ground-truth boxes.

    When an anchor is assigned with a ground-truth box which is the i-th category,
    the i-th entry in its C vector of targets is set to 1 and all other entries
    are set to 0. When an anchor is assigned with background, all entries are set
    to 0. Anchors that are not assigned do not contribute to the training
    objective. The regression targets are the encoded ground-truth boxes
    associated with the assigned anchors.

)DOC");
  }
};

class RetinanetTargetAssignOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Anchor"), "Input", "Anchor",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("GtBoxes"), "Input", "GtBoxes",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("GtLabels"), "Input", "GtLabels",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("IsCrowd"), "Input", "IsCrowd",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasInput("ImInfo"), "Input", "ImInfo",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("LocationIndex"), "Output", "LocationIndex",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("ScoreIndex"), "Output", "ScoreIndex",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("TargetLabel"), "Output", "TargetLabel",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("TargetBBox"), "Output", "TargetBBox",
                   "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("BBoxInsideWeight"), "Output",
                   "BBoxInsideWeight", "retinanet_target_assign");
    OP_INOUT_CHECK(ctx->HasOutput("ForegroundNumber"), "Output",
                   "ForegroundNumber", "retinanet_target_assign");

    auto anchor_dims = ctx->GetInputDim("Anchor");
    auto gt_boxes_dims = ctx->GetInputDim("GtBoxes");
    auto gt_labels_dims = ctx->GetInputDim("GtLabels");
    auto im_info_dims = ctx->GetInputDim("ImInfo");

    PADDLE_ENFORCE_EQ(
        anchor_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(Anchor) should be 2, but received Anchor "
            "rank is :%d, Anchor shape is:[%s].",
            anchor_dims.size(), anchor_dims));
    PADDLE_ENFORCE_EQ(
        gt_boxes_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(GtBoxes) should be 2, but received GtBoxes "
            "rank is :%d, GtBoxes shape is:[%s].",
            gt_boxes_dims.size(), gt_boxes_dims));
    PADDLE_ENFORCE_EQ(
        gt_labels_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(GtLabels) should be 2, but received GtLabels "
            "rank is :%d, GtLabels shape is:[%s].",
            gt_labels_dims.size(), gt_labels_dims));
    PADDLE_ENFORCE_EQ(
        im_info_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(ImInfo) should be 2, but received ImInfo "
            "rank is :%d, ImInfo shape is:[%s].",
            im_info_dims.size(), im_info_dims));

    ctx->SetOutputDim("LocationIndex", {gt_labels_dims[0]});
    ctx->SetOutputDim("ScoreIndex", {gt_labels_dims[0]});
    ctx->SetOutputDim("TargetBBox", {gt_labels_dims[0], 4});
    ctx->SetOutputDim("TargetLabel", {gt_labels_dims[0], 1});
    ctx->SetOutputDim("BBoxInsideWeight", {gt_labels_dims[0], 4});
    ctx->SetOutputDim("ForegroundNumber", {gt_labels_dims[0], 1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Anchor"),
        platform::CPUPlace());
  }
};

template <typename T>
std::vector<Tensor> FilterCrowdGtBoxLabel(
    const platform::CPUDeviceContext& context, Tensor* gt_boxes,
    Tensor* gt_labels, Tensor* is_crowd) {
  int gt_num = gt_boxes->dims()[0];
  std::vector<int> not_crowd_inds;
  auto* is_crowd_data = is_crowd->data<int>();
  for (int i = 0; i < gt_num; ++i) {
    if (is_crowd_data[i] == 0) {
      not_crowd_inds.emplace_back(i);
    }
  }
  int ncrowd_num = not_crowd_inds.size();
  Tensor ncrowd_gt_boxes, ncrowd_gt_labels;
  T* ncrowd_gt_boxes_data =
      ncrowd_gt_boxes.mutable_data<T>({ncrowd_num, 4}, context.GetPlace());
  int* ncrowd_gt_labels_data =
      ncrowd_gt_labels.mutable_data<int>({ncrowd_num, 1}, context.GetPlace());
  Gather<T>(gt_boxes->data<T>(), 4, not_crowd_inds.data(), ncrowd_num,
            ncrowd_gt_boxes_data);
  Gather<int>(gt_labels->data<int>(), 1, not_crowd_inds.data(), ncrowd_num,
              ncrowd_gt_labels_data);
  std::vector<Tensor> res;
  res.emplace_back(ncrowd_gt_boxes);
  res.emplace_back(ncrowd_gt_labels);
  return res;
}

template <typename T>
std::vector<Tensor> GetAllFgBgGt(const platform::CPUDeviceContext& ctx,
                                 const Tensor& anchor_by_gt_overlap,
                                 const Tensor& ncrowd_gt_labels,
                                 const float positive_overlap,
                                 const float negative_overlap,
                                 std::minstd_rand engine) {
  auto* anchor_by_gt_overlap_data = anchor_by_gt_overlap.data<T>();
  int anchor_num = anchor_by_gt_overlap.dims()[0];
  int gt_num = anchor_by_gt_overlap.dims()[1];

  std::vector<int> fg_inds;
  std::vector<int> bg_inds;
  std::vector<int> gt_inds;
  std::vector<int> tgt_lbl;
  std::vector<int> fg_fake;
  std::vector<T> bbox_inside_weight;
  // Calculate the max IoU between anchors and gt boxes
  // Map from anchor to gt box that has highest overlap
  auto place = ctx.GetPlace();
  Tensor anchor_to_gt_max, anchor_to_gt_argmax, gt_to_anchor_max;
  anchor_to_gt_max.mutable_data<T>({anchor_num}, place);
  int* argmax = anchor_to_gt_argmax.mutable_data<int>({anchor_num}, place);
  gt_to_anchor_max.mutable_data<T>({gt_num}, place);

  auto anchor_by_gt_overlap_et =
      framework::EigenMatrix<T>::From(anchor_by_gt_overlap);
  auto anchor_to_gt_max_et =
      framework::EigenVector<T>::Flatten(anchor_to_gt_max);
  auto gt_to_anchor_max_et =
      framework::EigenVector<T>::Flatten(gt_to_anchor_max);
  auto anchor_to_gt_argmax_et =
      framework::EigenVector<int>::Flatten(anchor_to_gt_argmax);
  anchor_to_gt_max_et =
      anchor_by_gt_overlap_et.maximum(Eigen::DSizes<int, 1>(1));
  anchor_to_gt_argmax_et =
      anchor_by_gt_overlap_et.argmax(1).template cast<int>();
  gt_to_anchor_max_et =
      anchor_by_gt_overlap_et.maximum(Eigen::DSizes<int, 1>(0));

  ScoreAssign(anchor_by_gt_overlap_data, anchor_to_gt_max, gt_to_anchor_max, -1,
              -1, positive_overlap, negative_overlap, &fg_inds, &bg_inds,
              &tgt_lbl, &fg_fake, &bbox_inside_weight, engine, false);
  const int* gt_labels_data = ncrowd_gt_labels.data<int>();
  int64_t fg_num = fg_inds.size();
  for (int64_t i = 0; i < fg_num; ++i) {
    int gt_idx = argmax[fg_inds[i]];
    tgt_lbl[i] = gt_labels_data[gt_idx];
  }

  int bg_num = bg_inds.size();
  int fg_fake_num = fg_fake.size();
  gt_inds.reserve(fg_fake_num);
  for (int i = 0; i < fg_fake_num; ++i) {
    gt_inds.emplace_back(argmax[fg_fake[i]]);
  }

  Tensor loc_index_t, score_index_t, tgt_lbl_t, gt_inds_t, bbox_inside_weight_t;
  Tensor fg_num_t;
  int* loc_index_data = loc_index_t.mutable_data<int>({fg_fake_num}, place);
  int* score_index_data =
      score_index_t.mutable_data<int>({fg_num + bg_num}, place);
  int* tgt_lbl_data = tgt_lbl_t.mutable_data<int>({fg_num + bg_num}, place);
  int* gt_inds_data = gt_inds_t.mutable_data<int>({fg_fake_num}, place);
  int* fg_num_data = fg_num_t.mutable_data<int>({1}, place);
  T* bbox_inside_weight_data =
      bbox_inside_weight_t.mutable_data<T>({fg_fake_num, 4}, place);
  std::copy(fg_fake.begin(), fg_fake.end(), loc_index_data);
  std::copy(fg_inds.begin(), fg_inds.end(), score_index_data);
  std::copy(bg_inds.begin(), bg_inds.end(), score_index_data + fg_num);
  std::copy(tgt_lbl.begin(), tgt_lbl.end(), tgt_lbl_data);
  std::copy(gt_inds.begin(), gt_inds.end(), gt_inds_data);
  std::copy(bbox_inside_weight.begin(), bbox_inside_weight.end(),
            bbox_inside_weight_data);
  fg_num_data[0] = fg_fake.size() + 1;
  std::vector<Tensor> loc_score_tgtlbl_gt;
  loc_score_tgtlbl_gt.emplace_back(loc_index_t);
  loc_score_tgtlbl_gt.emplace_back(score_index_t);
  loc_score_tgtlbl_gt.emplace_back(tgt_lbl_t);
  loc_score_tgtlbl_gt.emplace_back(gt_inds_t);
  loc_score_tgtlbl_gt.emplace_back(bbox_inside_weight_t);
  loc_score_tgtlbl_gt.emplace_back(fg_num_t);

  return loc_score_tgtlbl_gt;
}

template <typename T>
class RetinanetTargetAssignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* anchor = context.Input<Tensor>("Anchor");  // (H*W*A) * 4
    auto* gt_boxes = context.Input<LoDTensor>("GtBoxes");
    auto* gt_labels = context.Input<LoDTensor>("GtLabels");
    auto* is_crowd = context.Input<LoDTensor>("IsCrowd");
    auto* im_info = context.Input<LoDTensor>("ImInfo");

    auto* loc_index = context.Output<LoDTensor>("LocationIndex");
    auto* score_index = context.Output<LoDTensor>("ScoreIndex");
    auto* tgt_bbox = context.Output<LoDTensor>("TargetBBox");
    auto* tgt_lbl = context.Output<LoDTensor>("TargetLabel");
    auto* bbox_inside_weight = context.Output<LoDTensor>("BBoxInsideWeight");
    auto* fg_num = context.Output<LoDTensor>("ForegroundNumber");

    PADDLE_ENFORCE_EQ(
        gt_boxes->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "The LoD level of Input(GtBoxes) should be 1, but received GtBoxes "
            "LoD level is :%d.",
            gt_boxes->lod().size()));
    PADDLE_ENFORCE_EQ(
        gt_labels->lod().size(), 1UL,
        platform::errors::InvalidArgument("The LoD level of Input(GtLabels) "
                                          "should be 1, but received GtLabels "
                                          "LoD level is :%d.",
                                          gt_labels->lod().size()));
    PADDLE_ENFORCE_EQ(
        is_crowd->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "The LoD level of Input(IsCrowd) should be 1, but received IsCrowd "
            "LoD level is :%d.",
            is_crowd->lod().size()));

    auto anchor_num = static_cast<int64_t>(anchor->dims()[0]);
    auto batch_num = static_cast<int64_t>(gt_boxes->lod().back().size() - 1);

    float positive_overlap = context.Attr<float>("positive_overlap");
    float negative_overlap = context.Attr<float>("negative_overlap");

    int64_t max_num = batch_num * anchor_num;
    auto place = context.GetPlace();

    loc_index->mutable_data<int>({max_num}, place);
    score_index->mutable_data<int>({max_num}, place);
    tgt_bbox->mutable_data<T>({max_num, 4}, place);
    tgt_lbl->mutable_data<int>({max_num, 1}, place);
    bbox_inside_weight->mutable_data<T>({max_num, 4}, place);
    fg_num->mutable_data<int>({batch_num, 1}, place);
    auto& dev_ctx = context.device_context<platform::CPUDeviceContext>();

    std::random_device rnd;
    std::minstd_rand engine;
    int seed = rnd();
    engine.seed(seed);

    framework::LoD lod_loc, loc_score, lod_fg;
    std::vector<size_t> lod0_loc(1, 0);
    std::vector<size_t> lod0_score(1, 0);
    std::vector<size_t> lod0_fg(1, 0);

    int total_loc_num = 0;
    int total_score_num = 0;
    int total_fg_num = 0;
    auto gt_boxes_lod = gt_boxes->lod().back();
    auto gt_labels_lod = gt_labels->lod().back();
    auto is_crowd_lod = is_crowd->lod().back();
    for (int i = 0; i < batch_num; ++i) {
      Tensor gt_boxes_slice =
          gt_boxes->Slice(gt_boxes_lod[i], gt_boxes_lod[i + 1]);
      Tensor gt_labels_slice =
          gt_labels->Slice(gt_labels_lod[i], gt_labels_lod[i + 1]);
      Tensor is_crowd_slice =
          is_crowd->Slice(is_crowd_lod[i], is_crowd_lod[i + 1]);
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      auto* im_info_data = im_info_slice.data<T>();
      auto im_height = im_info_data[0];
      auto im_width = im_info_data[1];
      auto im_scale = im_info_data[2];

      // Filter straddle anchor
      std::vector<Tensor> filter_output =
          FilterStraddleAnchor<T>(dev_ctx, anchor, -1, im_height, im_width);
      Tensor inds_inside = filter_output[0];
      Tensor inside_anchor = filter_output[1];

      // Filter crowd gt
      std::vector<Tensor> ncrowd_output = FilterCrowdGtBoxLabel<T>(
          dev_ctx, &gt_boxes_slice, &gt_labels_slice, &is_crowd_slice);
      Tensor ncrowd_gt_boxes = ncrowd_output[0];
      Tensor ncrowd_gt_labels = ncrowd_output[1];

      auto ncrowd_gt_boxes_et =
          framework::EigenTensor<T, 2>::From(ncrowd_gt_boxes);
      ncrowd_gt_boxes_et = ncrowd_gt_boxes_et * im_scale;

      Tensor anchor_by_gt_overlap;
      anchor_by_gt_overlap.mutable_data<T>(
          {inside_anchor.dims()[0], ncrowd_gt_boxes.dims()[0]}, place);
      BboxOverlaps<T>(inside_anchor, ncrowd_gt_boxes, &anchor_by_gt_overlap);

      auto loc_score_tgtlbl_gt =
          GetAllFgBgGt<T>(dev_ctx, anchor_by_gt_overlap, ncrowd_gt_labels,
                          positive_overlap, negative_overlap, engine);

      Tensor sampled_loc_index = loc_score_tgtlbl_gt[0];
      Tensor sampled_score_index = loc_score_tgtlbl_gt[1];
      Tensor sampled_tgtlbl = loc_score_tgtlbl_gt[2];
      Tensor sampled_gt_index = loc_score_tgtlbl_gt[3];
      Tensor sampled_bbox_inside_weight = loc_score_tgtlbl_gt[4];
      Tensor sampled_fg_num = loc_score_tgtlbl_gt[5];

      int loc_num = sampled_loc_index.dims()[0];
      int score_num = sampled_score_index.dims()[0];
      // unmap to all anchor
      Tensor sampled_loc_index_unmap, sampled_score_index_unmap;
      sampled_loc_index_unmap.mutable_data<int>({loc_num}, place);
      sampled_score_index_unmap.mutable_data<int>({score_num}, place);
      Gather<int>(inds_inside.data<int>(), 1, sampled_loc_index.data<int>(),
                  loc_num, sampled_loc_index_unmap.data<int>());
      Gather<int>(inds_inside.data<int>(), 1, sampled_score_index.data<int>(),
                  score_num, sampled_score_index_unmap.data<int>());

      // get target bbox deltas
      Tensor sampled_anchor, sampled_gt, sampled_tgt_bbox;
      auto* sampled_anchor_data =
          sampled_anchor.mutable_data<T>({loc_num, 4}, place);
      auto* sampled_gt_data = sampled_gt.mutable_data<T>({loc_num, 4}, place);
      Gather<T>(anchor->data<T>(), 4, sampled_loc_index_unmap.data<int>(),
                loc_num, sampled_anchor_data);
      Gather<T>(ncrowd_gt_boxes.data<T>(), 4, sampled_gt_index.data<int>(),
                loc_num, sampled_gt_data);
      sampled_tgt_bbox.mutable_data<T>({loc_num, 4}, place);
      BoxToDelta<T>(loc_num, sampled_anchor, sampled_gt, nullptr, false,
                    &sampled_tgt_bbox);

      // Add anchor offset
      int anchor_offset = i * anchor_num;
      auto sampled_loc_index_unmap_et =
          framework::EigenTensor<int, 1>::From(sampled_loc_index_unmap);
      sampled_loc_index_unmap_et = sampled_loc_index_unmap_et + anchor_offset;
      auto sampled_score_index_unmap_et =
          framework::EigenTensor<int, 1>::From(sampled_score_index_unmap);
      sampled_score_index_unmap_et =
          sampled_score_index_unmap_et + anchor_offset;
      AppendRpns<int>(loc_index, total_loc_num, &sampled_loc_index_unmap);
      AppendRpns<int>(score_index, total_score_num, &sampled_score_index_unmap);
      AppendRpns<T>(tgt_bbox, total_loc_num * 4, &sampled_tgt_bbox);
      AppendRpns<int>(tgt_lbl, total_score_num, &sampled_tgtlbl);
      AppendRpns<T>(bbox_inside_weight, total_loc_num * 4,
                    &sampled_bbox_inside_weight);
      AppendRpns<int>(fg_num, total_fg_num, &sampled_fg_num);

      total_loc_num += loc_num;
      total_score_num += score_num;
      total_fg_num += 1;
      lod0_loc.emplace_back(total_loc_num);
      lod0_score.emplace_back(total_score_num);
      lod0_fg.emplace_back(total_fg_num);
    }

    PADDLE_ENFORCE_LE(
        total_loc_num, max_num,
        platform::errors::InvalidArgument(
            "The number of sampled bboxes should not be greater than the "
            "number of all anchor boxes(%d), but the number of sampled "
            "bboxes is :%d.",
            max_num, total_loc_num));
    PADDLE_ENFORCE_LE(
        total_score_num, max_num,
        platform::errors::InvalidArgument(
            "The number of sampled scores should not be greater than the "
            "number of all anchor boxes(%d), but the number of sampled "
            "scores is :%d.",
            max_num, total_score_num));
    PADDLE_ENFORCE_LE(
        total_fg_num, batch_num,
        platform::errors::InvalidArgument(
            "The number of foreground numbers should not be greater than the "
            "batch size(%d), but the number of foreground numbers is :%d.",
            batch_num, total_fg_num));

    lod_loc.emplace_back(lod0_loc);
    loc_score.emplace_back(lod0_score);
    lod_fg.emplace_back(lod0_fg);
    loc_index->set_lod(lod_loc);
    score_index->set_lod(loc_score);
    tgt_bbox->set_lod(lod_loc);
    tgt_lbl->set_lod(loc_score);
    bbox_inside_weight->set_lod(lod_loc);
    fg_num->set_lod(lod_fg);
    loc_index->Resize({total_loc_num});
    score_index->Resize({total_score_num});
    tgt_bbox->Resize({total_loc_num, 4});
    tgt_lbl->Resize({total_score_num, 1});
    bbox_inside_weight->Resize({total_loc_num, 4});
    fg_num->Resize({total_fg_num, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    rpn_target_assign, ops::RpnTargetAssignOp, ops::RpnTargetAssignOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(rpn_target_assign, ops::RpnTargetAssignKernel<float>,
                       ops::RpnTargetAssignKernel<double>);
REGISTER_OPERATOR(
    retinanet_target_assign, ops::RetinanetTargetAssignOp,
    ops::RetinanetTargetAssignOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(retinanet_target_assign,
                       ops::RetinanetTargetAssignKernel<float>,
                       ops::RetinanetTargetAssignKernel<double>);
