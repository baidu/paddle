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

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detection/bbox_util.h"
#include "paddle/fluid/operators/detection/mask_util.h"
#include "paddle/fluid/operators/gather.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
const int kBoxDim = 4;

template <typename T>
void AppendMask(LoDTensor* out, int64_t offset, Tensor* to_add) {
  auto* out_data = out->data<T>();
  auto* to_add_data = to_add->data<T>();
  memcpy(out_data + offset, to_add_data, to_add->numel() * sizeof(T));
}

class GenerateMaskLabelsOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("ImInfo"), true,
        platform::errors::InvalidArgument("Input(ImInfo) shouldn't be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("GtClasses"), true,
                      platform::errors::InvalidArgument(
                          "Input(GtClasses) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("IsCrowd"), true,
        platform::errors::InvalidArgument("Input(IsCrowd) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("GtSegms"), true,
        platform::errors::InvalidArgument("Input(GtSegms) shouldn't be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Rois"), true,
        platform::errors::InvalidArgument("Input(Rois) shouldn't be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput("LabelsInt32"), true,
                      platform::errors::InvalidArgument(
                          "Input(LabelsInt32) shouldn't be null."));

    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("MaskRois"), true,
        platform::errors::InvalidArgument(
            "Output(MaskRois) of GenerateMaskLabelsOp should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("RoiHasMaskInt32"), true,
                      platform::errors::InvalidArgument(
                          "Output(RoiHasMaskInt32) of GenerateMaskLabelsOp "
                          "should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("MaskInt32"), true,
        platform::errors::InvalidArgument(
            "Output(MaskInt32) of GenerateMaskLabelsOp should not be null"));

    auto im_info_dims = ctx->GetInputDim("ImInfo");
    auto gt_segms_dims = ctx->GetInputDim("GtSegms");
    PADDLE_ENFORCE_EQ(im_info_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(ImInfo) must be 2."));
    PADDLE_ENFORCE_EQ(gt_segms_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "The rank of Input(GtSegms) must be 2."));
    PADDLE_ENFORCE_EQ(gt_segms_dims[1], 2,
                      platform::errors::InvalidArgument(
                          "The second dim of Input(GtSegms) must be 2."));
    int num_classes = ctx->Attrs().Get<int>("num_classes");
    int resolution = ctx->Attrs().Get<int>("resolution");

    ctx->SetOutputDim("MaskRois", {-1, 4});
    ctx->SetOutputDim("RoiHasMaskInt32", {-1, 1});
    ctx->SetOutputDim("MaskInt32", {-1, num_classes * resolution * resolution});
    if (!ctx->IsRuntime()) {
      ctx->SetLoDLevel("MaskRois", ctx->GetLoDLevel("Rois"));
      ctx->SetLoDLevel("RoiHasMaskInt32", ctx->GetLoDLevel("Rois"));
      ctx->SetLoDLevel("MaskInt32", ctx->GetLoDLevel("Rois"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "Rois");
    return framework::OpKernelType(data_type, platform::CPUPlace());
  }
};

/*
 * Expand masks from shape (#masks, M ** 2) to (#masks, #classes * M ** 2)
 * to encode class specific mask targets.
 */
template <typename T>
static inline void ExpandMaskTarget(const platform::CPUDeviceContext& ctx,
                                    const Tensor& masks,
                                    const Tensor& mask_class_labels,
                                    const int resolution, const int num_classes,
                                    Tensor* mask_targets) {
  const auto* masks_data = masks.data<uint8_t>();
  int64_t num_mask = masks.dims()[0];
  const int* mask_class_labels_data = mask_class_labels.data<int>();
  const int M = resolution * resolution;
  const int mask_dim = M * num_classes;

  int* mask_targets_data =
      mask_targets->mutable_data<int>({num_mask, mask_dim}, ctx.GetPlace());
  math::set_constant(ctx, mask_targets, -1);
  for (int64_t mask_id = 0; mask_id < num_mask; ++mask_id) {
    int cls = mask_class_labels_data[mask_id];
    int start = M * cls;
    if (cls > 0) {
      for (int i = 0; i < M; ++i) {
        mask_targets_data[mask_id * mask_dim + start + i] =
            static_cast<int>(masks_data[mask_id * M + i]);
      }
    }
  }
}

template <typename T>
std::vector<Tensor> SampleMaskForOneImage(
    const platform::CPUDeviceContext& ctx, const Tensor& im_info,
    const Tensor& gt_classes, const Tensor& is_crowd, const Tensor& gt_segms,
    const Tensor& rois, const Tensor& label_int32, const int num_classes,
    const int resolution, const framework::LoD& segm_length) {
  // Prepare the mask targets by associating one gt mask to each training roi
  // that has a fg (non-bg) class label.
  const auto gt_size = static_cast<int64_t>(gt_classes.dims()[0]);
  const auto roi_size = static_cast<int64_t>(rois.dims()[0]);
  const int* gt_classes_data = gt_classes.data<int>();
  const int* is_crowd_data = is_crowd.data<int>();
  const int* label_int32_data = label_int32.data<int>();
  PADDLE_ENFORCE_EQ(roi_size, label_int32.dims()[0],
                    platform::errors::InvalidArgument(
                        "The first dim of label [%d] is the different from "
                        "roi_size [%d], they should be same.",
                        label_int32.dims()[0], roi_size));

  std::vector<int> mask_gt_inds, fg_inds;
  std::vector<std::vector<std::vector<T>>> gt_polys;

  auto polys_num = segm_length[1];
  auto segm_lod_offset = framework::ConvertToOffsetBasedLoD(segm_length);
  auto lod1 = segm_lod_offset[1];
  auto lod2 = segm_lod_offset[2];
  const T* polys_data = gt_segms.data<T>();
  for (int64_t i = 0; i < gt_size; ++i) {
    if ((gt_classes_data[i] > 0) && (is_crowd_data[i] == 0)) {
      mask_gt_inds.emplace_back(i);

      // slice fg segmentation polys
      int poly_num = polys_num[i];
      std::vector<std::vector<T>> polys;
      int s_idx = lod1[i];
      for (int j = 0; j < poly_num; ++j) {
        int s = lod2[s_idx + j];
        int e = lod2[s_idx + j + 1];
        PADDLE_ENFORCE_NE(s, e,
                          platform::errors::InvalidArgument(
                              "The start point and the end point in the poly "
                              "segment [%d] should not be same, but received "
                              "the start point [%d] and the end point [%d].",
                              i, s, e));
        std::vector<T> plts(polys_data + s * 2, polys_data + e * 2);
        polys.push_back(plts);
      }
      gt_polys.push_back(polys);
    }
  }
  for (int64_t i = 0; i < roi_size; ++i) {
    if (label_int32_data[i] > 0) {
      fg_inds.emplace_back(i);
    }
  }
  int gt_num = mask_gt_inds.size();
  int fg_num = fg_inds.size();

  Tensor boxes_from_polys;
  boxes_from_polys.mutable_data<T>({gt_num, 4}, platform::CPUPlace());
  Poly2Boxes(gt_polys, boxes_from_polys.data<T>());

  std::vector<int> roi_has_mask =
      std::vector<int>(fg_inds.begin(), fg_inds.end());
  Tensor mask_class_labels;
  Tensor masks;
  Tensor rois_fg;

  auto im_scale = im_info.data<T>()[2];
  if (fg_num > 0) {
    // Class labels for the foreground rois
    mask_class_labels.mutable_data<int>({fg_num, 1}, ctx.GetPlace());
    Gather<int>(label_int32_data, 1, fg_inds.data(), fg_inds.size(),
                mask_class_labels.data<int>());

    auto* masks_data = masks.mutable_data<uint8_t>(
        {fg_num, resolution * resolution}, ctx.GetPlace());

    // Find overlap between all foreground rois and the bounding boxes
    // enclosing each segmentation
    T* rois_fg_data = rois_fg.mutable_data<T>({fg_num, 4}, ctx.GetPlace());
    Gather<T>(rois.data<T>(), 4, fg_inds.data(), fg_inds.size(),
              rois_fg.data<T>());

    for (int k = 0; k < rois_fg.numel(); ++k) {
      rois_fg_data[k] = rois_fg_data[k] / im_scale;
    }

    Tensor overlaps_bbfg_bbpolys;
    overlaps_bbfg_bbpolys.mutable_data<T>({fg_num, gt_num}, ctx.GetPlace());
    BboxOverlaps<T>(rois_fg, boxes_from_polys, &overlaps_bbfg_bbpolys);

    // Map from each fg rois to the index of the mask with highest overlap
    // (measured by bbox overlap)
    T* overlaps_bbfg_bbpolys_data = overlaps_bbfg_bbpolys.data<T>();
    std::vector<int> fg_masks_inds;
    for (int64_t i = 0; i < fg_num; ++i) {
      const T* v = overlaps_bbfg_bbpolys_data + i * gt_num;
      T max_overlap = std::numeric_limits<T>::min();
      int id = 0;
      for (int64_t j = 0; j < gt_num; ++j) {
        if (v[j] > max_overlap) {
          max_overlap = v[j];
          id = j;
        }
      }
      fg_masks_inds.push_back(id);
    }

    // add fg targets
    for (int64_t i = 0; i < fg_num; ++i) {
      int fg_polys_ind = fg_masks_inds[i];
      T* roi_fg = rois_fg_data + i * 4;
      uint8_t* mask = masks_data + i * resolution * resolution;
      Polys2MaskWrtBox(gt_polys[fg_polys_ind], roi_fg, resolution, mask);
    }
  } else {
    // The network cannot handle empty blobs, so we must provide a mask
    // We simply take the first bg roi, given it an all -1's mask (ignore
    // label), and label it with class zero (bg).
    int bg_num = 1;
    T* rois_fg_data = rois_fg.mutable_data<T>({bg_num, 4}, ctx.GetPlace());
    const T* rois_data = rois.data<T>();
    std::vector<int> bg_inds;
    for (int64_t i = 0; i < roi_size; ++i) {
      if (label_int32_data[i] == 0) {
        bg_inds.emplace_back(i);
        rois_fg_data[0] = rois_data[0] / im_scale;
        rois_fg_data[1] = rois_data[1] / im_scale;
        rois_fg_data[2] = rois_data[2] / im_scale;
        rois_fg_data[3] = rois_data[3] / im_scale;
        break;
      }
    }
    masks.mutable_data<uint8_t>({bg_num, resolution * resolution},
                                ctx.GetPlace());
    math::set_constant(ctx, &masks, -1);
    int* mask_class_labels_data =
        mask_class_labels.mutable_data<int>({bg_num, 1}, ctx.GetPlace());
    mask_class_labels_data[0] = 0;
    roi_has_mask = std::vector<int>(bg_inds.begin(), bg_inds.end());
  }

  Tensor masks_expand;
  ExpandMaskTarget<T>(ctx, masks, mask_class_labels, resolution, num_classes,
                      &masks_expand);

  T* rois_fg_data = rois_fg.data<T>();
  for (int k = 0; k < rois_fg.numel(); ++k) {
    rois_fg_data[k] = rois_fg_data[k] * im_scale;
  }

  Tensor roi_has_mask_t;
  int roi_has_mask_size = roi_has_mask.size();
  int* roi_has_mask_data =
      roi_has_mask_t.mutable_data<int>({roi_has_mask_size, 1}, ctx.GetPlace());
  std::copy(roi_has_mask.begin(), roi_has_mask.end(), roi_has_mask_data);

  std::vector<Tensor> res;
  res.emplace_back(rois_fg);
  res.emplace_back(roi_has_mask_t);
  res.emplace_back(masks_expand);
  return res;
}

template <typename T>
class GenerateMaskLabelsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* im_info = ctx.Input<LoDTensor>("ImInfo");
    auto* gt_classes = ctx.Input<LoDTensor>("GtClasses");
    auto* is_crowd = ctx.Input<LoDTensor>("IsCrowd");
    auto* gt_segms = ctx.Input<LoDTensor>("GtSegms");
    auto* rois = ctx.Input<LoDTensor>("Rois");
    auto* label_int32 = ctx.Input<LoDTensor>("LabelsInt32");

    auto* mask_rois = ctx.Output<LoDTensor>("MaskRois");
    auto* roi_has_mask_int32 = ctx.Output<LoDTensor>("RoiHasMaskInt32");
    auto* mask_int32 = ctx.Output<LoDTensor>("MaskInt32");

    int num_classes = ctx.Attr<int>("num_classes");
    int resolution = ctx.Attr<int>("resolution");

    PADDLE_ENFORCE_EQ(
        gt_classes->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateMaskLabelsOp gt_classes needs 1 level of LoD"));
    PADDLE_ENFORCE_EQ(
        is_crowd->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateMaskLabelsOp is_crowd needs 1 level of LoD"));
    PADDLE_ENFORCE_EQ(rois->lod().size(), 1UL,
                      platform::errors::InvalidArgument(
                          "GenerateMaskLabelsOp rois needs 1 level of LoD"));
    PADDLE_ENFORCE_EQ(
        label_int32->lod().size(), 1UL,
        platform::errors::InvalidArgument(
            "GenerateMaskLabelsOp label_int32 needs 1 level of LoD"));

    PADDLE_ENFORCE_EQ(
        gt_segms->lod().size(), 3UL,
        platform::errors::InvalidArgument(
            "GenerateMaskLabelsOp gt_segms needs 3 level of LoD"));

    auto n = static_cast<int64_t>(gt_classes->lod().back().size() - 1);
    PADDLE_ENFORCE_EQ(
        gt_segms->lod()[0].size() - 1, n,
        platform::errors::InvalidArgument(
            "Batchsize of Input(gt_segms) and Input(gt_classes) should be "
            "same, but received gt_segms[%d], gt_classes[%d].",
            gt_segms->lod()[0].size() - 1, n));

    int mask_dim = num_classes * resolution * resolution;
    int roi_num = rois->lod().back()[n];
    mask_rois->mutable_data<T>({roi_num, kBoxDim}, ctx.GetPlace());
    roi_has_mask_int32->mutable_data<int>({roi_num, 1}, ctx.GetPlace());
    mask_int32->mutable_data<int>({roi_num, mask_dim}, ctx.GetPlace());

    framework::LoD lod;
    std::vector<size_t> lod0(1, 0);

    int64_t num_mask = 0;
    auto& dev_ctx = ctx.device_context<platform::CPUDeviceContext>();

    auto gt_classes_lod = gt_classes->lod().back();
    auto is_crowd_lod = is_crowd->lod().back();
    auto rois_lod = rois->lod().back();
    auto label_int32_lod = label_int32->lod().back();
    auto gt_segms_lod = gt_segms->lod();

    for (int i = 0; i < n; ++i) {
      if (rois_lod[i] == rois_lod[i + 1]) {
        lod0.emplace_back(num_mask);
        continue;
      }
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor gt_classes_slice =
          gt_classes->Slice(gt_classes_lod[i], gt_classes_lod[i + 1]);
      Tensor is_crowd_slice =
          is_crowd->Slice(is_crowd_lod[i], is_crowd_lod[i + 1]);
      Tensor label_int32_slice =
          label_int32->Slice(label_int32_lod[i], label_int32_lod[i + 1]);
      Tensor rois_slice = rois->Slice(rois_lod[i], rois_lod[i + 1]);

      auto sub_lod_and_offset =
          framework::GetSubLoDAndAbsoluteOffset(gt_segms_lod, i, i + 1, 0);
      auto lod_length = sub_lod_and_offset.first;
      size_t s = sub_lod_and_offset.second.first;
      size_t e = sub_lod_and_offset.second.second;
      Tensor gt_segms_slice = gt_segms->Slice(s, e);

      std::vector<Tensor> tensor_output = SampleMaskForOneImage<T>(
          dev_ctx, im_info_slice, gt_classes_slice, is_crowd_slice,
          gt_segms_slice, rois_slice, label_int32_slice, num_classes,
          resolution, lod_length);

      Tensor sampled_mask_rois = tensor_output[0];
      Tensor sampled_roi_has_mask_int32 = tensor_output[1];
      Tensor sampled_mask_int32 = tensor_output[2];

      AppendMask<T>(mask_rois, kBoxDim * num_mask, &sampled_mask_rois);
      AppendMask<int>(roi_has_mask_int32, num_mask,
                      &sampled_roi_has_mask_int32);
      AppendMask<int>(mask_int32, mask_dim * num_mask, &sampled_mask_int32);

      num_mask += sampled_mask_rois.dims()[0];
      lod0.emplace_back(num_mask);
    }

    lod.emplace_back(lod0);
    mask_rois->set_lod(lod);
    roi_has_mask_int32->set_lod(lod);
    mask_int32->set_lod(lod);
    mask_rois->Resize({num_mask, kBoxDim});
    roi_has_mask_int32->Resize({num_mask, 1});
    mask_int32->Resize({num_mask, mask_dim});
  }
};

class GenerateMaskLabelsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ImInfo",
             "(Tensor), This input is a 2D Tensor with shape [B, 3]. "
             "B is the number of input images, "
             "each element consists of im_height, im_width, im_scale.");
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
        "GtSegms",
        "(LoDTensor), This input is a 2D LoDTensor with shape [S, 2], it's LoD "
        "level is 3. The LoD[0] represents the gt objects number of each "
        "instance. LoD[1] represents the segmentation counts of each objects. "
        "LoD[2] represents the polygons number of each segmentation. S the "
        "total number of polygons coordinate points. Each element is (x, y) "
        "coordinate points.");
    AddInput(
        "Rois",
        "(LoDTensor), This input is a 2D LoDTensor with shape [R, 4]. "
        "R is the number of rois which is the output of "
        "generate_proposal_labels, "
        "each element is a bounding box with (xmin, ymin, xmax, ymax) format.");
    AddInput("LabelsInt32",
             "(LoDTensor), This intput is a 2D LoDTensor with shape [R, 1], "
             "each element represents a class label of a roi");
    AddOutput(
        "MaskRois",
        "(LoDTensor), This output is a 2D LoDTensor with shape [P, 4]. "
        "P is the number of mask, "
        "each element is a bounding box with [xmin, ymin, xmax, ymax] format.");
    AddOutput("RoiHasMaskInt32",
              "(LoDTensor), This output is a 2D LoDTensor with shape [P, 1], "
              "each element represents the output mask rois index with regard "
              "to input rois");
    AddOutput("MaskInt32",
              "(LoDTensor), This output is a 4D LoDTensor with shape [P, Q], "
              "Q equal to num_classes * resolution * resolution");

    AddAttr<int>("num_classes", "Class number.");
    AddAttr<int>("resolution", "Resolution of mask.");

    AddComment(R"DOC(
This operator can be, for given the RoIs and corresponding labels,
to sample foreground RoIs. This mask branch also has
a :math: `K \\times M^{2}` dimensional output targets for each foreground
RoI, which encodes K binary masks of resolution M x M, one for each of the
K classes. This mask targets are used to compute loss of mask branch.
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    generate_mask_labels, ops::GenerateMaskLabelsOp,
    ops::GenerateMaskLabelsOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(generate_mask_labels,
                       ops::GenerateMaskLabelsKernel<float>);
