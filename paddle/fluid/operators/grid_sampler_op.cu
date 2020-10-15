/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/grid_sampler_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

static __forceinline__ __device__ bool in_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}

template <typename T>
static __forceinline__ __device__ void atomic_add(T* data, int h, int w, int sH,
                                                  int sW, int H, int W,
                                                  T delta) {
  if (in_bounds(h, w, H, W)) {
    platform::CudaAtomicAdd(data + h * sH + w * sW, delta);
  }
}

template <typename T>
static __forceinline__ __device__ T _unnormalize(T coord, int size,
                                                 bool align_corners) {
  if (align_corners) {
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T>
static __forceinline__ __device__ T clip_indexes(T in, int max_value) {
  return min(static_cast<T>(max_value), max(in, static_cast<T>(0)));
}

template <typename T>
static __forceinline__ __device__ T reflect_indexes(T in, int twice_low,
                                                    int twice_high) {
  if (twice_low == twice_high) {
    return static_cast<T>(0);
  }
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = fabs(in - min);
  T extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    return extra + min;
  } else {
    return span - extra + min;
  }
}

template <typename T>
static __forceinline__ __device__ T compute_positions(T coord, int size,
                                                      PaddingMode padding_mode,
                                                      bool align_corners) {
  coord = _unnormalize<T>(coord, size, align_corners);
  if (padding_mode == PaddingMode::border) {
    coord = clip_indexes(coord, size - 1);
  } else if (padding_mode == PaddingMode::reflect) {
    if (align_corners) {
      coord = reflect_indexes(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_indexes(coord, -1, 2 * size - 1);
    }
    coord = clip_indexes(coord, size - 1);
  }
  return coord;
}

template <typename T>
static __forceinline__ __device__ T _unnormalize_with_mask(T coord, int size,
                                                           bool align_corners,
                                                           T* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<T>(size - 1) / 2;
    return ((coord + 1.f) / 2) * (size - 1);
  } else {
    *grad_in = static_cast<T>(size) / 2;
    return ((coord + 1.f) * size - 1) / 2;
  }
}

template <typename T>
static __forceinline__ __device__ T clip_indexes_with_mask(T in, int clip_limit,
                                                           T* grad_in) {
  if (in <= static_cast<T>(0)) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  } else {
    T max = static_cast<T>(clip_limit - 1);
    if (in >= max) {
      *grad_in = static_cast<T>(0);
      return max;
    } else {
      *grad_in = static_cast<T>(1);
      return in;
    }
  }
}

template <typename T>
static __forceinline__ __device__ T
reflect_indexes_with_mask(T in, int twice_low, int twice_high, T* grad_in) {
  if (twice_low == twice_high) {
    *grad_in = static_cast<T>(0);
    return static_cast<T>(0);
  }
  int grad_in_mult_;
  T min = static_cast<T>(twice_low) / 2;
  T span = static_cast<T>(twice_high - twice_low) / 2;
  in = in - min;
  if (in < static_cast<T>(0)) {
    grad_in_mult_ = -1;
    in = -in;
  } else {
    grad_in_mult_ = 1;
  }
  T extra = fmod(in, span);
  int flips = static_cast<int>(floor(in / span));
  if (flips % 2 == 0) {
    *grad_in = static_cast<T>(grad_in_mult_);
    return extra + min;
  } else {
    *grad_in = static_cast<T>(-grad_in_mult_);
    return span - extra + min;
  }
}

template <typename T>
static __forceinline__ __device__ T
compute_positions_with_mask(T coord, int size, PaddingMode padding_mode,
                            bool align_corners, T* grad_in) {
  T grad_clip, grad_refl;
  coord = _unnormalize_with_mask<T>(coord, size, align_corners, grad_in);
  if (padding_mode == PaddingMode::border) {
    coord = clip_indexes_with_mask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_clip;
  } else if (padding_mode == PaddingMode::reflect) {
    if (align_corners) {
      coord = reflect_indexes_with_mask(coord, 0, 2 * (size - 1), &grad_refl);
    } else {
      coord = reflect_indexes_with_mask(coord, -1, 2 * size - 1, &grad_refl);
    }
    coord = clip_indexes_with_mask(coord, size, &grad_clip);
    *grad_in = (*grad_in) * grad_refl * grad_clip;
  }

  return coord;
}

template <typename T>
__global__ void grid_sample_cuda_kernel(const int nthreads, int n, int out_c,
                                        int out_h, int out_w, int in_h,
                                        int in_w, const T* input, const T* grid,
                                        T* output, const Mode mode,
                                        const PaddingMode padding_mode,
                                        bool align_corners) {
  int inp_sN = out_c * in_h * in_w;

  int inp_sC = in_h * in_w;
  int inp_sH = in_w;
  int inp_sW = 1;
  int grid_sN = out_h * out_w * 2;
  int grid_sH = out_w * 2;
  int grid_sW = 2;
  int grid_sCoor = 1;
  int out_sN = out_c * out_h * out_w;
  int out_sC = out_h * out_w;
  int out_sH = out_w;
  int out_sW = 1;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_w;
    const int h = (index / out_w) % out_h;
    const int n = index / (out_h * out_w);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];

    ix = compute_positions(ix, in_w, padding_mode, align_corners);
    iy = compute_positions(iy, in_h, padding_mode, align_corners);

    if (mode == Mode::bilinear) {
      int ix_nw = static_cast<int>(floor(ix));
      int iy_nw = static_cast<int>(floor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      T nw = (ix_se - ix) * (iy_se - iy);
      T ne = (ix - ix_sw) * (iy_sw - iy);
      T sw = (ix_ne - ix) * (iy - iy_ne);
      T se = (ix - ix_nw) * (iy - iy_nw);

      auto inp_offset_NC = n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < out_c;
           ++c, inp_offset_NC += inp_sC, out_ptr_NCHW += out_sC) {
        *out_ptr_NCHW = static_cast<T>(0);
        if (in_bounds(iy_nw, ix_nw, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_nw * inp_sH + ix_nw * inp_sW] * nw;
        }
        if (in_bounds(iy_ne, ix_ne, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_ne * inp_sH + ix_ne * inp_sW] * ne;
        }
        if (in_bounds(iy_sw, ix_sw, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_sw * inp_sH + ix_sw * inp_sW] * sw;
        }
        if (in_bounds(iy_se, ix_se, in_h, in_w)) {
          *out_ptr_NCHW +=
              input[inp_offset_NC + iy_se * inp_sH + ix_se * inp_sW] * se;
        }
      }
    } else if (mode == Mode::nearest) {
      int ix_nearest = static_cast<int>(std::nearbyint(ix));
      int iy_nearest = static_cast<int>(std::nearbyint(iy));
      auto inp_offset_NC = n * inp_sN;
      auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;
      for (int c = 0; c < out_c;
           ++c, inp_offset_NC += inp_sC, out_ptr_NCHW += out_sC) {
        if (in_bounds(iy_nearest, ix_nearest, in_h, in_w)) {
          *out_ptr_NCHW =
              input[inp_offset_NC + iy_nearest * inp_sH + ix_nearest * inp_sW];
        } else {
          *out_ptr_NCHW = static_cast<T>(0);
        }
      }
    }
  }
}

template <typename T>
class GridSampleOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.cuda_device_context();
    auto align_corners = ctx.Attr<bool>("align_corners");
    auto padding_mode_s = ctx.Attr<std::string>("padding_mode");
    auto mode_s = ctx.Attr<std::string>("mode");
    PaddingMode padding_mode;
    Mode mode;
    if (padding_mode_s == "border") {
      padding_mode = PaddingMode::border;
    } else if (padding_mode_s == "reflection") {
      padding_mode = PaddingMode::reflect;
    } else {
      padding_mode = PaddingMode::zeros;
    }

    if (mode_s == "nearest") {
      mode = Mode::nearest;
    } else {
      mode = Mode::bilinear;
    }

    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    const int n = grid->dims()[0];
    const int out_h = grid->dims()[1];
    const int out_w = grid->dims()[2];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];
    VLOG(3) << "n: " << n << "; c: " << c << "; out_h: " << out_h
            << "; out_w: " << out_w;
    auto* output = ctx.Output<Tensor>("Output");
    auto* output_data = output->mutable_data<T>(ctx.GetPlace());

    VLOG(3) << "set constant";
    math::SetConstant<paddle::platform::CUDADeviceContext, T>()(
        dev_ctx, output, static_cast<T>(0));
    int count = static_cast<int>(n * out_h * out_w);

    auto cu_stream = dev_ctx.stream();

    int block = 512;
    int grid_size = (count + block - 1) / block;
    grid_sample_cuda_kernel<T><<<block, grid_size, 0, cu_stream>>>(
        count, n, c, out_h, out_w, in_h, in_w, input->data<T>(),
        grid->data<T>(), output_data, mode, padding_mode, align_corners);
  }
};

template <typename T>
__global__ void grid_sampler_cuda_backward_kernel(
    const int nthreads, const T* grad_output, const T* input, const T* grid,
    int n, int out_c, int out_h, int out_w, int in_h, int in_w, T* grad_input,
    T* grad_grid, const Mode mode, const PaddingMode padding_mode,
    bool align_corners) {
  int inp_sN = out_c * in_h * in_w;
  int inp_sC = in_h * in_w;
  int inp_sH = in_w;
  int inp_sW = 1;
  int grid_sN = out_h * out_w * 2;
  int grid_sH = out_w * 2;
  int grid_sW = 2;
  int grid_sCoor = 1;

  int gOut_sN = out_c * out_h * out_w;
  int gOut_sC = out_h * out_w;
  int gOut_sH = out_w;
  int gOut_sW = 1;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % out_w;
    const int h = (index / out_w) % out_h;
    const int n = index / (out_h * out_w);
    const int grid_offset = n * grid_sN + h * grid_sH + w * grid_sW;

    T ix = grid[grid_offset];
    T iy = grid[grid_offset + grid_sCoor];

    T gix_mult, giy_mult;
    ix = compute_positions_with_mask(ix, in_w, padding_mode, align_corners,
                                     &gix_mult);
    iy = compute_positions_with_mask(iy, in_h, padding_mode, align_corners,
                                     &giy_mult);

    if (mode == Mode::bilinear) {
      int ix_nw = static_cast<int>(floor(ix));
      int iy_nw = static_cast<int>(floor(iy));
      int ix_ne = ix_nw + 1;
      int iy_ne = iy_nw;
      int ix_sw = ix_nw;
      int iy_sw = iy_nw + 1;
      int ix_se = ix_nw + 1;
      int iy_se = iy_nw + 1;

      T nw = (ix_se - ix) * (iy_se - iy);
      T ne = (ix - ix_sw) * (iy_sw - iy);
      T sw = (ix_ne - ix) * (iy - iy_ne);
      T se = (ix - ix_nw) * (iy - iy_nw);

      T gix = static_cast<T>(0), giy = static_cast<T>(0);
      int gOut_offset = n * gOut_sN + h * gOut_sH + w * gOut_sW;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      int inp_offset_NC = n * inp_sN;
      for (int c = 0; c < out_c; ++c, inp_offset_NC += inp_sC,
               gInp_ptr_NC += inp_sC, gOut_offset += gOut_sC) {
        T gOut = grad_output[gOut_offset];

        atomic_add(gInp_ptr_NC, iy_nw, ix_nw, inp_sH, inp_sW, in_h, in_w,
                   nw * gOut);
        atomic_add(gInp_ptr_NC, iy_ne, ix_ne, inp_sH, inp_sW, in_h, in_w,
                   ne * gOut);
        atomic_add(gInp_ptr_NC, iy_sw, ix_sw, inp_sH, inp_sW, in_h, in_w,
                   sw * gOut);
        atomic_add(gInp_ptr_NC, iy_se, ix_se, inp_sH, inp_sW, in_h, in_w,
                   se * gOut);

        if (in_bounds(iy_nw, ix_nw, in_h, in_w)) {
          T nw_val = input[inp_offset_NC + iy_nw * inp_sH + ix_nw * inp_sW];
          gix -= nw_val * (iy_se - iy) * gOut;
          giy -= nw_val * (ix_se - ix) * gOut;
        }
        if (in_bounds(iy_ne, ix_ne, in_h, in_w)) {
          T ne_val = input[inp_offset_NC + iy_ne * inp_sH + ix_ne * inp_sW];
          gix += ne_val * (iy_sw - iy) * gOut;
          giy -= ne_val * (ix - ix_sw) * gOut;
        }
        if (in_bounds(iy_sw, ix_sw, in_h, in_w)) {
          T sw_val = input[inp_offset_NC + iy_sw * inp_sH + ix_sw * inp_sW];
          gix -= sw_val * (iy - iy_ne) * gOut;
          giy += sw_val * (ix_ne - ix) * gOut;
        }
        if (in_bounds(iy_se, ix_se, in_h, in_w)) {
          T se_val = input[inp_offset_NC + iy_se * inp_sH + ix_se * inp_sW];
          gix += se_val * (iy - iy_nw) * gOut;
          giy += se_val * (ix - ix_nw) * gOut;
        }
      }

      if (grad_grid != nullptr) {
        T* gGrid_ptr_NHW = grad_grid + index * grid_sW;
        gGrid_ptr_NHW[0] = gix_mult * gix;
        gGrid_ptr_NHW[1] = giy_mult * giy;
      }
    } else if (mode == Mode::nearest) {
      int ix_nearest = static_cast<int>(std::nearbyint(ix));
      int iy_nearest = static_cast<int>(std::nearbyint(iy));

      int gOut_offset = n * gOut_sN + h * gOut_sH + w * gOut_sW;
      T* gInp_ptr_NC = grad_input + n * inp_sN;
      for (int c = 0; c < out_c;
           ++c, gInp_ptr_NC += inp_sC, gOut_offset += gOut_sC) {
        atomic_add(gInp_ptr_NC, iy_nearest, ix_nearest, inp_sH, inp_sW, in_h,
                   in_w, grad_output[gOut_offset]);
      }

      if (grad_grid != nullptr) {
        T* gGrid_ptr_NHW = grad_grid + index * grid_sW;
        gGrid_ptr_NHW[0] = static_cast<T>(0);
        gGrid_ptr_NHW[1] = static_cast<T>(0);
      }
    }
  }
}

template <typename T>
class GridSampleGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.cuda_device_context();
    auto align_corners = ctx.Attr<bool>("align_corners");
    auto padding_mode_s = ctx.Attr<std::string>("padding_mode");
    auto mode_s = ctx.Attr<std::string>("mode");

    PaddingMode padding_mode;
    Mode mode;
    if (padding_mode_s == "border") {
      padding_mode = PaddingMode::border;
    } else if (padding_mode_s == "reflection") {
      padding_mode = PaddingMode::reflect;
    } else {
      padding_mode = PaddingMode::zeros;
    }

    if (mode_s == "nearest") {
      mode = Mode::nearest;
    } else {
      mode = Mode::bilinear;
    }

    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));

    const int n = grid->dims()[0];
    const int out_h = grid->dims()[1];
    const int out_w = grid->dims()[2];
    const int c = input->dims()[1];
    const int in_h = input->dims()[2];
    const int in_w = input->dims()[3];

    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<paddle::platform::CUDADeviceContext, T>()(
        ctx.template device_context<paddle::platform::CUDADeviceContext>(),
        input_grad, static_cast<T>(0));

    T* grid_grad_data = nullptr;
    if (ctx.HasOutput(framework::GradVarName("Grid"))) {
      auto* grid_grad = ctx.Output<Tensor>(framework::GradVarName("Grid"));
      grid_grad_data = grid_grad->mutable_data<T>(ctx.GetPlace());
      math::SetConstant<paddle::platform::CUDADeviceContext, T>()(
          ctx.template device_context<paddle::platform::CUDADeviceContext>(),
          grid_grad, static_cast<T>(0));
    }

    int count = static_cast<int>(n * out_h * out_w);
    auto cu_stream = dev_ctx.stream();
    int block = 512;
    int grid_size = (count + block - 1) / block;
    grid_sampler_cuda_backward_kernel<T><<<block, grid_size, 0, cu_stream>>>(
        count, output_grad->data<T>(), input->data<T>(), grid->data<T>(), n, c,
        out_h, out_w, in_h, in_w, input_grad->data<T>(), grid_grad_data, mode,
        padding_mode, align_corners);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(grid_sampler, ops::GridSampleOpCUDAKernel<float>,
                        ops::GridSampleOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(grid_sampler_grad,
                        ops::GridSampleGradOpCUDAKernel<float>,
                        ops::GridSampleGradOpCUDAKernel<double>);
