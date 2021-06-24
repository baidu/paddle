/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/fake_quantize_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void FindAbsMaxKernel(const T* in, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  extern __shared__ char* shared_max_data_tmp[];
  auto shared_max_data = reinterpret_cast<T*>(shared_max_data_tmp);
  if (gridDim.x > 1) {
    shared_max_data[tid] = T(0);
    for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
      T tmp = abs(in[i]);
      if (tmp > shared_max_data[tid]) {
        shared_max_data[tid] = tmp;
      }
    }
  } else {
    if (bid < n) {
      shared_max_data[tid] = abs(in[bid]);
    } else {
      shared_max_data[tid] = T(0);
    }
  }
  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
struct FindAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const T* in,
                  const int num, T* out) {
    int block = 1024;
    int grid = (block - 1 + num) / block;
    grid = (grid > block) ? block : grid;

    framework::Tensor max;
    T* max_data =
        max.mutable_data<T>(framework::make_ddim({grid}), ctx.GetPlace());
    FindAbsMaxKernel<T><<<grid, block, 1024 * sizeof(T), ctx.stream()>>>(
        in, num, max_data);
    FindAbsMaxKernel<T><<<1, block, 1024 * sizeof(T), ctx.stream()>>>(
        max_data, grid, out);
  }
};

template struct FindAbsMaxFunctor<platform::CUDADeviceContext, float>;
template struct FindAbsMaxFunctor<platform::CUDADeviceContext,
                                  paddle::platform::float16>;

template <typename T>
__global__ void FindChannelAbsMaxKernelQuantAxis0(const T* in, const int n,
                                                  const int c, T* out) {
  int tid = threadIdx.x;
  int channel_size = n / c;
  const T* in_c = in + blockIdx.x * channel_size;
  extern __shared__ T shared_max_data[];
  shared_max_data[tid] = T(0);
  for (int i = tid; i < channel_size; i += blockDim.x) {
    T tmp = fabs(in_c[i]);
    if (tmp > shared_max_data[tid]) {
      shared_max_data[tid] = tmp;
    }
  }
  __syncthreads();
  for (int i = blockDim.x / 2; i > 0; i >>= 1) {
    if (tid < i && (shared_max_data[tid] < shared_max_data[tid + i])) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[blockIdx.x] = shared_max_data[0];
  }
}

template <typename T>
__global__ void FindChannelAbsMaxKernelQuantAxis1(const T* in, const int n,
                                                  const int cin, const int cout,
                                                  T* out) {
  extern __shared__ T shared_max_data[];
  int cout_wh_size = n / cin;
  int wh_size = n / (cin * cout);

  int tid = threadIdx.x;
  int bid = blockIdx.x;
  const T* in_current = in + tid * cout_wh_size + bid * wh_size;
  shared_max_data[tid] = T(0);
  for (int i = 0; i < wh_size; i++) {
    T tmp = fabs(in_current[i]);
    if (tmp > shared_max_data[tid]) {
      shared_max_data[tid] = tmp;
    }
  }
  __syncthreads();

  int len = blockDim.x;
  for (int i = (len + 1) / 2; i > 0; len = i, i = (i + 1) / 2) {
    if (tid < i && tid + i < len &&
        shared_max_data[tid] < shared_max_data[tid + i]) {
      shared_max_data[tid] = shared_max_data[tid + i];
    }
    if (i == 1) {
      i = 0;  // break the loop
    }
    __syncthreads();
  }
  if (tid == 0 && shared_max_data[0] > out[bid]) {
    out[bid] = shared_max_data[0];
  }
}

template <typename T>
struct FindChannelAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in_tensor, const int quant_axis,
                  T* out_abs_max) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    const int num = in_tensor.numel();
    auto in_dims = in_tensor.dims();
    const T* in_data = in_tensor.data<T>();
    if (quant_axis == 0) {
      int cout = in_dims[0];
      int grid = cout;
      int block = 1024;
      FindChannelAbsMaxKernelQuantAxis0<
          T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
          in_data, num, cout, out_abs_max);
    } else if (quant_axis == 1) {
      int cin = in_dims[0];
      int cout = in_dims[1];
      int grid = cout;
      int max_threads = 1024;

#ifdef PADDLE_WITH_HIP
      hipMemset(out_abs_max, 0, sizeof(T) * cout);
#else
      cudaMemset(out_abs_max, 0, sizeof(T) * cout);
#endif

      for (int i = 0; i < cin / max_threads; i++) {
        int block = max_threads;
        FindChannelAbsMaxKernelQuantAxis1<
            T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
            in_data, num, cin, cout, out_abs_max);
        in_data += num / cin;
      }

      int block = cin % max_threads;
      if (block > 0) {
        FindChannelAbsMaxKernelQuantAxis1<
            T><<<grid, block, block * sizeof(T), ctx.stream()>>>(
            in_data, num, in_dims[0], in_dims[1], out_abs_max);
      }
    }
  }
};

template struct FindChannelAbsMaxFunctor<platform::CUDADeviceContext, float>;

template <typename T>
__global__ void ClipAndQuantKernel(const T* in, const T* scale,
                                   const int bin_cnt, const int n, T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  T s = scale[0];
  T inv_s = inverse(s);
  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    T x = in[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out[i] = round(v);
  }
}

template <typename T>
__global__ void ClipAndQuantDequantKernel(const T* in, const T* scale,
                                          const int bin_cnt, const int n,
                                          T* out) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  T s = scale[0];
  T bin_cnt_t = static_cast<T>(bin_cnt);

  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    T x = in[i];
    x = x > s ? s : x;
    x = x < -s ? -s : x;
    x = (bin_cnt_t / s) * x;

    x = static_cast<T>(round(static_cast<float>(x)));
    out[i] = (x * s) / bin_cnt_t;
  }
}

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    int num = in.numel();
    int block = 1024;
    int grid = (block - 1 + num) / block;

    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    ClipAndQuantKernel<T><<<grid, block, 0, ctx.stream()>>>(
        in_data, scale_data, bin_cnt, num, out_data);
  }
};

template struct ClipAndFakeQuantFunctor<platform::CUDADeviceContext, float>;

template <typename T>
struct ClipAndFakeQuantDequantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    int num = in.numel();
    int block = 1024;
    int grid = (block - 1 + num) / block;

    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    ClipAndQuantDequantKernel<T><<<grid, block, 0, ctx.stream()>>>(
        in_data, scale_data, bin_cnt, num, out_data);
  }
};

// ChannelClipAndQuantKernel for quant_axis is 0
template <typename T>
__global__ void ChannelClipAndQuantKernelQuantAxis0(const T* in, const T* scale,
                                                    const int bin_cnt,
                                                    const int n, const int c,
                                                    T* out) {
  int tid = threadIdx.x;

  int channel_size = n / c;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;

  T s = scale[blockIdx.x];
  T inv_s = inverse(s);

  for (int i = tid; i < channel_size; i += blockDim.x) {
    T x = in_c[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out_c[i] = round(v);
  }
}

// ChannelClipAndQuantKernel for quant_axis is 1
template <typename T>
__global__ void ChannelClipAndQuantKernelQuantAxis1(const T* in, const T* scale,
                                                    const int bin_cnt,
                                                    const int n, const int cin,
                                                    const int cout, T* out) {
  T s = scale[blockIdx.x % cout];
  T inv_s = inverse(s);

  int wh_size = n / (cin * cout);
  const T* in_c = in + blockIdx.x * wh_size;
  T* out_c = out + blockIdx.x * wh_size;

  for (int i = threadIdx.x; i < wh_size; i += blockDim.x) {
    T x = in_c[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out_c[i] = round(v);
  }
}

template <typename T>
struct ChannelClipAndFakeQuantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));

    int num = in.numel();
    auto in_dims = in.dims();
    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    if (quant_axis == 0) {
      int grid = in_dims[0];
      int block = 1024;
      ChannelClipAndQuantKernelQuantAxis0<T><<<grid, block, 0, ctx.stream()>>>(
          in_data, scale_data, bin_cnt, num, in_dims[0], out_data);
    } else if (quant_axis == 1) {
      int grid = in_dims[0] * in_dims[1];
      int block = 1024;
      ChannelClipAndQuantKernelQuantAxis1<T><<<grid, block, 0, ctx.stream()>>>(
          in_data, scale_data, bin_cnt, num, in_dims[0], in_dims[1], out_data);
    }
  }
};

template struct ChannelClipAndFakeQuantFunctor<platform::CUDADeviceContext,
                                               float>;

template <typename T>
__global__ void FindRangeAbsMaxAndFillArray(const T* cur_scale,
                                            const T* last_scale,
                                            const int64_t* iter,
                                            const int window_size, T* scale_arr,
                                            T* out_scale, int* need_find_max,
                                            int* out_size) {
  int it = iter[0];
  int idx = it % window_size;
  T removed = scale_arr[idx];
  T cur = cur_scale[0];
  scale_arr[idx] = cur;
  T max = last_scale[0];
  out_scale[0] = max < cur ? cur : max;
  if (fabs(removed - max) < 1e-6) {
    need_find_max[0] = 1;
    out_size[0] = it > window_size ? window_size : it;
  } else {
    need_find_max[0] = 0;
  }
}

template <typename T>
struct FindRangeAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale) {
    const auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());

    T* scale_arr = scales_arr->mutable_data<T>(gpu_place);
    T* out_scale_data = out_scale->mutable_data<T>(gpu_place);

    framework::Tensor need_find_max, out_size;
    int* find_max = need_find_max.mutable_data<int>({1}, gpu_place);
    int* out_size_data = out_size.mutable_data<int>({1}, gpu_place);

    FindRangeAbsMaxAndFillArray<T><<<1, 1, 0, ctx.stream()>>>(
        cur_scale.data<T>(), last_scale.data<T>(), iter.data<int64_t>(),
        window_size, scale_arr, out_scale_data, find_max, out_size_data);

    int g_find_max;
    memory::Copy(platform::CPUPlace(), &g_find_max, gpu_place, find_max,
                 sizeof(int), ctx.stream());
    ctx.Wait();
    if (g_find_max) {
      int len;
      memory::Copy(platform::CPUPlace(), &len, gpu_place, out_size_data,
                   sizeof(int), ctx.stream());
      ctx.Wait();
      FindAbsMaxFunctor<platform::CUDADeviceContext, T>()(ctx, scale_arr, len,
                                                          out_scale_data);
    }
  }
};

template struct FindRangeAbsMaxFunctor<platform::CUDADeviceContext, float>;

template <typename T>
struct FindMovingAverageAbsMaxFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in_accum,
                  const framework::Tensor& in_state, const T* cur_scale,
                  const float rate, framework::Tensor* out_state,
                  framework::Tensor* out_accum, framework::Tensor* out_scale) {
    const auto gpu_place = BOOST_GET_CONST(platform::CUDAPlace, ctx.GetPlace());

    T accum;
    T state;
    T scale;
    memory::Copy(platform::CPUPlace(), &accum, gpu_place, in_accum.data<T>(),
                 sizeof(T), ctx.stream());
    memory::Copy(platform::CPUPlace(), &state, gpu_place, in_state.data<T>(),
                 sizeof(T), ctx.stream());
    memory::Copy(platform::CPUPlace(), &scale, gpu_place, cur_scale, sizeof(T),
                 ctx.stream());
    ctx.Wait();

    T rate_t = static_cast<T>(rate);
    state = rate_t * state + static_cast<T>(1.0);
    accum = rate_t * accum + scale;
    scale = accum / state;

    memory::Copy(gpu_place, out_accum->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &accum, sizeof(T), ctx.stream());
    memory::Copy(gpu_place, out_state->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &state, sizeof(T), ctx.stream());
    memory::Copy(gpu_place, out_scale->mutable_data<T>(gpu_place),
                 platform::CPUPlace(), &scale, sizeof(T), ctx.stream());
    ctx.Wait();
  }
};

// ChannelClipAndQuantDequantKernel for quant_axis is 0
template <typename T>
__global__ void ChannelClipAndQuantDequantKernelQuantAxis0(
    const T* in, const T* scale, const int bin_cnt, const int n, const int c,
    T* out) {
  int tid = threadIdx.x;

  int channel_size = n / c;
  const T* in_c = in + blockIdx.x * channel_size;
  T* out_c = out + blockIdx.x * channel_size;

  T s = scale[blockIdx.x];
  T inv_s = inverse(s);

  for (int i = tid; i < channel_size; i += blockDim.x) {
    T x = in_c[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out_c[i] = round(v) * s / bin_cnt;
  }
}

// ChannelClipAndQuantDequantKernel for quant_axis is 1
template <typename T>
__global__ void ChannelClipAndQuantDequantKernelQuantAxis1(
    const T* in, const T* scale, const int bin_cnt, const int n, const int cin,
    const int cout, T* out) {
  T s = scale[blockIdx.x % cout];
  T inv_s = inverse(s);

  int wh_size = n / (cin * cout);
  const T* in_c = in + blockIdx.x * wh_size;
  T* out_c = out + blockIdx.x * wh_size;

  for (int i = threadIdx.x; i < wh_size; i += blockDim.x) {
    T x = in_c[i];
    T v = x > s ? s : x;
    v = v < -s ? -s : v;
    v = bin_cnt * inv_s * v;
    out_c[i] = round(v) * s / bin_cnt;
  }
}

template <typename T>
struct ChannelClipFakeQuantDequantFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));

    int num = in.numel();
    auto in_dims = in.dims();

    const T* in_data = in.data<T>();
    const T* scale_data = scale.data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());

    if (quant_axis == 0) {
      int grid = in_dims[0];
      int block = 1024;
      ChannelClipAndQuantDequantKernelQuantAxis0<
          T><<<grid, block, 0, ctx.stream()>>>(in_data, scale_data, bin_cnt,
                                               num, in_dims[0], out_data);
    } else if (quant_axis == 1) {
      int grid = in_dims[0] * in_dims[1];
      int block = 1024;

      ChannelClipAndQuantDequantKernelQuantAxis1<
          T><<<grid, block, 0, ctx.stream()>>>(
          in_data, scale_data, bin_cnt, num, in_dims[0], in_dims[1], out_data);
    }
  }
};

template struct ChannelClipFakeQuantDequantFunctor<platform::CUDADeviceContext,
                                                   float>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
using float16 = paddle::platform::float16;
REGISTER_OP_CUDA_KERNEL(fake_quantize_abs_max,
                        ops::FakeQuantizeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(fake_quantize_dequantize_abs_max,
                        ops::FakeQuantizeDequantizeAbsMaxKernel<CUDA, float>,
                        ops::FakeQuantizeDequantizeAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(fake_channel_wise_quantize_abs_max,
                        ops::FakeChannelWiseQuantizeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(fake_quantize_range_abs_max,
                        ops::FakeQuantizeRangeAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(
    fake_quantize_moving_average_abs_max,
    ops::FakeQuantizeMovingAverageAbsMaxKernel<CUDA, float>);
REGISTER_OP_CUDA_KERNEL(moving_average_abs_max_scale,
                        ops::MovingAverageAbsMaxScaleKernel<CUDA, float>,
                        ops::MovingAverageAbsMaxScaleKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_quantize_dequantize_moving_average_abs_max,
    ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel<CUDA, float>,
    ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(stright_throuth_estimator_grad,
                        ops::StrightThroughEstimatorGradKernel<CUDA, float>,
                        ops::StrightThroughEstimatorGradKernel<CUDA, float16>);
REGISTER_OP_CUDA_KERNEL(
    fake_channel_wise_quantize_dequantize_abs_max,
    ops::FakeChannelWiseQuantizeDequantizeAbsMaxKernel<CUDA, float>);
