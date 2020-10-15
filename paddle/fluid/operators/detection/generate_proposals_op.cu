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

#include <paddle/fluid/memory/allocation/allocator.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

namespace {

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

int const kThreadsPerBlock = sizeof(uint64_t) * 8;

static const double kBBoxClipDefault = std::log(1000.0 / 16.0);

struct RangeInitFunctor {
  int start_;
  int delta_;
  int *out_;
  __device__ void operator()(size_t i) { out_[i] = start_ + i * delta_; }
};

template <typename T>
static void SortDescending(const platform::CUDADeviceContext &ctx,
                           const Tensor &value, Tensor *value_out,
                           Tensor *index_out) {
  int num = static_cast<int>(value.numel());
  Tensor index_in_t;
  int *idx_in = index_in_t.mutable_data<int>({num}, ctx.GetPlace());
  platform::ForRange<platform::CUDADeviceContext> for_range(ctx, num);
  for_range(RangeInitFunctor{0, 1, idx_in});

  int *idx_out = index_out->mutable_data<int>({num}, ctx.GetPlace());

  const T *keys_in = value.data<T>();
  T *keys_out = value_out->mutable_data<T>({num}, ctx.GetPlace());

  // Determine temporary device storage requirements
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      nullptr, temp_storage_bytes, keys_in, keys_out, idx_in, idx_out, num);
  // Allocate temporary storage
  auto place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  auto d_temp_storage = memory::Alloc(place, temp_storage_bytes);

  // Run sorting operation
  cub::DeviceRadixSort::SortPairsDescending<T, int>(
      d_temp_storage->ptr(), temp_storage_bytes, keys_in, keys_out, idx_in,
      idx_out, num);
}

template <typename T>
struct BoxDecodeAndClipFunctor {
  const T *anchor;
  const T *deltas;
  const T *var;
  const int *index;
  const T *im_info;

  T *proposals;

  BoxDecodeAndClipFunctor(const T *anchor, const T *deltas, const T *var,
                          const int *index, const T *im_info, T *proposals)
      : anchor(anchor),
        deltas(deltas),
        var(var),
        index(index),
        im_info(im_info),
        proposals(proposals) {}

  T bbox_clip_default{static_cast<T>(kBBoxClipDefault)};

  __device__ void operator()(size_t i) {
    int k = index[i] * 4;
    T axmin = anchor[k];
    T aymin = anchor[k + 1];
    T axmax = anchor[k + 2];
    T aymax = anchor[k + 3];

    T w = axmax - axmin + 1.0;
    T h = aymax - aymin + 1.0;
    T cx = axmin + 0.5 * w;
    T cy = aymin + 0.5 * h;

    T dxmin = deltas[k];
    T dymin = deltas[k + 1];
    T dxmax = deltas[k + 2];
    T dymax = deltas[k + 3];

    T d_cx, d_cy, d_w, d_h;
    if (var) {
      d_cx = cx + dxmin * w * var[k];
      d_cy = cy + dymin * h * var[k + 1];
      d_w = exp(Min(dxmax * var[k + 2], bbox_clip_default)) * w;
      d_h = exp(Min(dymax * var[k + 3], bbox_clip_default)) * h;
    } else {
      d_cx = cx + dxmin * w;
      d_cy = cy + dymin * h;
      d_w = exp(Min(dxmax, bbox_clip_default)) * w;
      d_h = exp(Min(dymax, bbox_clip_default)) * h;
    }

    T oxmin = d_cx - d_w * 0.5;
    T oymin = d_cy - d_h * 0.5;
    T oxmax = d_cx + d_w * 0.5 - 1.;
    T oymax = d_cy + d_h * 0.5 - 1.;

    proposals[i * 4] = Max(Min(oxmin, im_info[1] - 1.), 0.);
    proposals[i * 4 + 1] = Max(Min(oymin, im_info[0] - 1.), 0.);
    proposals[i * 4 + 2] = Max(Min(oxmax, im_info[1] - 1.), 0.);
    proposals[i * 4 + 3] = Max(Min(oymax, im_info[0] - 1.), 0.);
  }

  __device__ __forceinline__ T Min(T a, T b) const { return a > b ? b : a; }

  __device__ __forceinline__ T Max(T a, T b) const { return a > b ? a : b; }
};

template <typename T, int BlockSize>
static __global__ void FilterBBoxes(const T *bboxes, const T *im_info,
                                    const T min_size, const int num,
                                    int *keep_num, int *keep) {
  T im_h = im_info[0];
  T im_w = im_info[1];
  T im_scale = im_info[2];

  int cnt = 0;
  __shared__ int keep_index[BlockSize];

  CUDA_KERNEL_LOOP(i, num) {
    keep_index[threadIdx.x] = -1;
    __syncthreads();

    int k = i * 4;
    T xmin = bboxes[k];
    T ymin = bboxes[k + 1];
    T xmax = bboxes[k + 2];
    T ymax = bboxes[k + 3];

    T w = xmax - xmin + 1.0;
    T h = ymax - ymin + 1.0;
    T cx = xmin + w / 2.;
    T cy = ymin + h / 2.;

    T w_s = (xmax - xmin) / im_scale + 1.;
    T h_s = (ymax - ymin) / im_scale + 1.;

    if (w_s >= min_size && h_s >= min_size && cx <= im_w && cy <= im_h) {
      keep_index[threadIdx.x] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int size = (num - i) < BlockSize ? num - i : BlockSize;
      for (int j = 0; j < size; ++j) {
        if (keep_index[j] > -1) {
          keep[cnt++] = keep_index[j];
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    keep_num[0] = cnt;
  }
}

static __device__ inline float IoU(const float *a, const float *b) {
  float left = max(a[0], b[0]), right = min(a[2], b[2]);
  float top = max(a[1], b[1]), bottom = min(a[3], b[3]);
  float width = max(right - left + 1, 0.f), height = max(bottom - top + 1, 0.f);
  float inter_s = width * height;
  float s_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
  float s_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
  return inter_s / (s_a + s_b - inter_s);
}

static __global__ void NMSKernel(const int n_boxes,
                                 const float nms_overlap_thresh,
                                 const float *dev_boxes, uint64_t *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
      min(n_boxes - row_start * kThreadsPerBlock, kThreadsPerBlock);
  const int col_size =
      min(n_boxes - col_start * kThreadsPerBlock, kThreadsPerBlock);

  __shared__ float block_boxes[kThreadsPerBlock * 4];
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 4 + 0] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_boxes[threadIdx.x * 4 + 1] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_boxes[threadIdx.x * 4 + 2] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_boxes[threadIdx.x * 4 + 3] =
        dev_boxes[(kThreadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = kThreadsPerBlock * row_start + threadIdx.x;
    const float *cur_box = dev_boxes + cur_box_idx * 4;
    int i = 0;
    uint64_t t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (IoU(cur_box, block_boxes + i * 4) > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_boxes, kThreadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

template <typename T>
static void NMS(const platform::CUDADeviceContext &ctx, const Tensor &proposals,
                const Tensor &sorted_indices, const T nms_threshold,
                Tensor *keep_out) {
  int boxes_num = proposals.dims()[0];
  const int col_blocks = DIVUP(boxes_num, kThreadsPerBlock);
  dim3 blocks(DIVUP(boxes_num, kThreadsPerBlock),
              DIVUP(boxes_num, kThreadsPerBlock));
  dim3 threads(kThreadsPerBlock);

  const T *boxes = proposals.data<T>();
  auto place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  framework::Vector<uint64_t> mask(boxes_num * col_blocks);
  NMSKernel<<<blocks, threads>>>(boxes_num, nms_threshold, boxes,
                                 mask.CUDAMutableData(BOOST_GET_CONST(
                                     platform::CUDAPlace, ctx.GetPlace())));

  std::vector<uint64_t> remv(col_blocks);
  memset(&remv[0], 0, sizeof(uint64_t) * col_blocks);

  std::vector<int> keep_vec;
  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / kThreadsPerBlock;
    int inblock = i % kThreadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {
      ++num_to_keep;
      keep_vec.push_back(i);
      uint64_t *p = &mask[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  int *keep = keep_out->mutable_data<int>({num_to_keep}, ctx.GetPlace());
  memory::Copy(place, keep, platform::CPUPlace(), keep_vec.data(),
               sizeof(int) * num_to_keep, ctx.stream());
  ctx.Wait();
}

template <typename T>
static std::pair<Tensor, Tensor> ProposalForOneImage(
    const platform::CUDADeviceContext &ctx, const Tensor &im_info,
    const Tensor &anchors, const Tensor &variances,
    const Tensor &bbox_deltas,  // [M, 4]
    const Tensor &scores,       // [N, 1]
    int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size,
    float eta) {
  // 1. pre nms
  Tensor scores_sort, index_sort;
  SortDescending<T>(ctx, scores, &scores_sort, &index_sort);
  int num = scores.numel();
  int pre_nms_num = (pre_nms_top_n <= 0 || pre_nms_top_n > num) ? scores.numel()
                                                                : pre_nms_top_n;
  scores_sort.Resize({pre_nms_num, 1});
  index_sort.Resize({pre_nms_num, 1});

  // 2. box decode and clipping
  Tensor proposals;
  proposals.mutable_data<T>({pre_nms_num, 4}, ctx.GetPlace());

  {
    platform::ForRange<platform::CUDADeviceContext> for_range(ctx, pre_nms_num);
    for_range(BoxDecodeAndClipFunctor<T>{
        anchors.data<T>(), bbox_deltas.data<T>(), variances.data<T>(),
        index_sort.data<int>(), im_info.data<T>(), proposals.data<T>()});
  }

  // 3. filter
  Tensor keep_index, keep_num_t;
  keep_index.mutable_data<int>({pre_nms_num}, ctx.GetPlace());
  keep_num_t.mutable_data<int>({1}, ctx.GetPlace());
  min_size = std::max(min_size, 1.0f);
  auto stream = ctx.stream();
  FilterBBoxes<T, 512><<<1, 512, 0, stream>>>(
      proposals.data<T>(), im_info.data<T>(), min_size, pre_nms_num,
      keep_num_t.data<int>(), keep_index.data<int>());
  int keep_num;
  const auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());
  memory::Copy(platform::CPUPlace(), &keep_num, gpu_place,
               keep_num_t.data<int>(), sizeof(int), ctx.stream());
  ctx.Wait();
  keep_index.Resize({keep_num});

  Tensor scores_filter, proposals_filter;
  // Handle the case when there is no keep index left
  if (keep_num == 0) {
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    proposals_filter.mutable_data<T>({1, 4}, ctx.GetPlace());
    scores_filter.mutable_data<T>({1, 1}, ctx.GetPlace());
    set_zero(ctx, &proposals_filter, static_cast<T>(0));
    set_zero(ctx, &scores_filter, static_cast<T>(0));
    return std::make_pair(proposals_filter, scores_filter);
  }
  proposals_filter.mutable_data<T>({keep_num, 4}, ctx.GetPlace());
  scores_filter.mutable_data<T>({keep_num, 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals, keep_index, &proposals_filter);
  GPUGather<T>(ctx, scores_sort, keep_index, &scores_filter);

  if (nms_thresh <= 0) {
    return std::make_pair(proposals_filter, scores_filter);
  }

  // 4. nms
  Tensor keep_nms;
  NMS<T>(ctx, proposals_filter, keep_index, nms_thresh, &keep_nms);
  if (post_nms_top_n > 0 && post_nms_top_n < keep_nms.numel()) {
    keep_nms.Resize({post_nms_top_n});
  }

  Tensor scores_nms, proposals_nms;
  proposals_nms.mutable_data<T>({keep_nms.numel(), 4}, ctx.GetPlace());
  scores_nms.mutable_data<T>({keep_nms.numel(), 1}, ctx.GetPlace());
  GPUGather<T>(ctx, proposals_filter, keep_nms, &proposals_nms);
  GPUGather<T>(ctx, scores_filter, keep_nms, &scores_nms);

  return std::make_pair(proposals_nms, scores_nms);
}
}  // namespace

template <typename DeviceContext, typename T>
class CUDAGenerateProposalsKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *scores = context.Input<Tensor>("Scores");
    auto *bbox_deltas = context.Input<Tensor>("BboxDeltas");
    auto *im_info = context.Input<Tensor>("ImInfo");
    auto anchors = GET_DATA_SAFELY(context.Input<Tensor>("Anchors"), "Input",
                                   "Anchors", "GenerateProposals");
    auto variances = GET_DATA_SAFELY(context.Input<Tensor>("Variances"),
                                     "Input", "Variances", "GenerateProposals");

    auto *rpn_rois = context.Output<LoDTensor>("RpnRois");
    auto *rpn_roi_probs = context.Output<LoDTensor>("RpnRoiProbs");

    int pre_nms_top_n = context.Attr<int>("pre_nms_topN");
    int post_nms_top_n = context.Attr<int>("post_nms_topN");
    float nms_thresh = context.Attr<float>("nms_thresh");
    float min_size = context.Attr<float>("min_size");
    float eta = context.Attr<float>("eta");
    PADDLE_ENFORCE_GE(eta, 1.,
                      platform::errors::InvalidArgument(
                          "Not support adaptive NMS. The attribute 'eta' "
                          "should not less than 1. But received eta=[%d]",
                          eta));

    auto &dev_ctx = context.template device_context<DeviceContext>();

    auto scores_dim = scores->dims();
    int64_t num = scores_dim[0];
    int64_t c_score = scores_dim[1];
    int64_t h_score = scores_dim[2];
    int64_t w_score = scores_dim[3];

    auto bbox_dim = bbox_deltas->dims();
    int64_t c_bbox = bbox_dim[1];
    int64_t h_bbox = bbox_dim[2];
    int64_t w_bbox = bbox_dim[3];

    Tensor bbox_deltas_swap, scores_swap;
    bbox_deltas_swap.mutable_data<T>({num, h_bbox, w_bbox, c_bbox},
                                     dev_ctx.GetPlace());
    scores_swap.mutable_data<T>({num, h_score, w_score, c_score},
                                dev_ctx.GetPlace());

    math::Transpose<DeviceContext, T, 4> trans;
    std::vector<int> axis = {0, 2, 3, 1};
    trans(dev_ctx, *bbox_deltas, &bbox_deltas_swap, axis);
    trans(dev_ctx, *scores, &scores_swap, axis);

    anchors.Resize({anchors.numel() / 4, 4});
    variances.Resize({variances.numel() / 4, 4});

    rpn_rois->mutable_data<T>({bbox_deltas->numel() / 4, 4},
                              context.GetPlace());
    rpn_roi_probs->mutable_data<T>({scores->numel(), 1}, context.GetPlace());

    T *rpn_rois_data = rpn_rois->data<T>();
    T *rpn_roi_probs_data = rpn_roi_probs->data<T>();

    auto place = BOOST_GET_CONST(platform::CUDAPlace, dev_ctx.GetPlace());
    auto cpu_place = platform::CPUPlace();

    int64_t num_proposals = 0;
    std::vector<size_t> offset(1, 0);
    std::vector<int> tmp_num;

    for (int64_t i = 0; i < num; ++i) {
      Tensor im_info_slice = im_info->Slice(i, i + 1);
      Tensor bbox_deltas_slice = bbox_deltas_swap.Slice(i, i + 1);
      Tensor scores_slice = scores_swap.Slice(i, i + 1);

      bbox_deltas_slice.Resize({h_bbox * w_bbox * c_bbox / 4, 4});
      scores_slice.Resize({h_score * w_score * c_score, 1});

      std::pair<Tensor, Tensor> box_score_pair =
          ProposalForOneImage<T>(dev_ctx, im_info_slice, anchors, variances,
                                 bbox_deltas_slice, scores_slice, pre_nms_top_n,
                                 post_nms_top_n, nms_thresh, min_size, eta);

      Tensor &proposals = box_score_pair.first;
      Tensor &scores = box_score_pair.second;

      memory::Copy(place, rpn_rois_data + num_proposals * 4, place,
                   proposals.data<T>(), sizeof(T) * proposals.numel(),
                   dev_ctx.stream());
      memory::Copy(place, rpn_roi_probs_data + num_proposals, place,
                   scores.data<T>(), sizeof(T) * scores.numel(),
                   dev_ctx.stream());
      dev_ctx.Wait();
      num_proposals += proposals.dims()[0];
      offset.emplace_back(num_proposals);
      tmp_num.push_back(proposals.dims()[0]);
    }
    if (context.HasOutput("RpnRoisNum")) {
      auto *rpn_rois_num = context.Output<Tensor>("RpnRoisNum");
      rpn_rois_num->mutable_data<int>({num}, context.GetPlace());
      int *num_data = rpn_rois_num->data<int>();
      memory::Copy(place, num_data, cpu_place, &tmp_num[0], sizeof(int) * num,
                   dev_ctx.stream());
      rpn_rois_num->Resize({num});
    }
    framework::LoD lod;
    lod.emplace_back(offset);
    rpn_rois->set_lod(lod);
    rpn_roi_probs->set_lod(lod);
    rpn_rois->Resize({num_proposals, 4});
    rpn_roi_probs->Resize({num_proposals, 1});
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(generate_proposals,
                        ops::CUDAGenerateProposalsKernel<
                            paddle::platform::CUDADeviceContext, float>);
