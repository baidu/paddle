// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <limits>
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/log_softmax_op.h"
#include "paddle/fluid/platform/cuda_device_function.h"

namespace paddle {
namespace operators {

#define LAUNCH_WARP_FORWAR_COMPUTE(near_greater_power_of_two)                \
  case near_greater_power_of_two:                                            \
    ComputeLogSoftmaxForwardInWarp<                                          \
        T, AccT, near_greater_power_of_two><<<blocks, threads, 0, stream>>>( \
        dst, src, outer_size, dim_size);                                     \
    break;

template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceSum(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T sum_val = platform::CudaShuffleXorSync(0xFFFFFFFF, value, offset);
    value = value + sum_val;
  }
  return value;
}

template <typename T, int KernelWarpSize>
__device__ __forceinline__ T WarpReduceMax(T value) {
#pragma unroll
  for (int offset = KernelWarpSize / 2; offset > 0; offset /= 2) {
    T max_val = platform::CudaShuffleXorSync(0xFFFFFFFF, value, offset);
    value = max(value, max_val);
  }
  return value;
}

int GetNearGreaterPowerOfTwo(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) {
    ++log2_value;
  }
  return 1 << log2_value;
}

template <typename T, typename AccT, int NearGreaterPowerOfTwo>
__global__ void ComputeLogSoftmaxForwardInWarp(T *dst, const T *src,
                                               int batch_size,
                                               int element_count) {
  constexpr int near_greater_power_of_two = NearGreaterPowerOfTwo;
  constexpr int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  constexpr int warp_iter = near_greater_power_of_two / kernel_warp_size;
  int batch_id = blockDim.y * blockIdx.x + threadIdx.y;

  int thread_in_warp_idx = threadIdx.x;

  // 1.read data from global memory to registers
  AccT elements[warp_iter];
  // set effective_element_count as the num of elements when warps do effective
  // work
  // set effective_element_count as 0, when warps do ineffective work
  int effective_element_count = (batch_id < batch_size) ? element_count : 0;
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < effective_element_count) {
      elements[it] =
          static_cast<AccT>(src[batch_id * element_count + element_index]);
    } else {
      elements[it] = -std::numeric_limits<AccT>::infinity();
    }
  }

  // 2.compute max_value. For each thread, loop all registers to find max
  AccT max_value = elements[0];
#pragma unroll
  for (int it = 1; it < warp_iter; ++it) {
    max_value = (max_value > elements[it]) ? max_value : elements[it];
  }
  max_value = WarpReduceMax<AccT, kernel_warp_size>(max_value);

  // 3.For each warp, accumulate all thread registers
  AccT sum = 0.0f;
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    sum += std::exp(elements[it] - max_value);
  }
  sum = WarpReduceSum<AccT, kernel_warp_size>(sum);

  // 4.store result.
  sum = std::log(sum);
#pragma unroll
  for (int it = 0; it < warp_iter; ++it) {
    int element_index = thread_in_warp_idx + it * kernel_warp_size;
    if (element_index < effective_element_count) {
      dst[batch_id * element_count + element_index] =
          static_cast<T>(elements[it] - max_value - sum);
    } else {
      break;
    }
  }
}

template <typename T, typename AccT>
void LaunchSoftmaxForwardForLastAxis(T *dst, const T *src, int dim_size,
                                     int outer_size, gpuStream_t stream) {
  int threads_per_block = 128;
  int near_greater_power_of_two = GetNearGreaterPowerOfTwo(dim_size);
  int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  int warps_per_block = (threads_per_block / kernel_warp_size);
  int blocks = (outer_size + warps_per_block - 1) / warps_per_block;
  dim3 threads(kernel_warp_size, warps_per_block, 1);

  switch (near_greater_power_of_two) {
    LAUNCH_WARP_FORWAR_COMPUTE(1);
    LAUNCH_WARP_FORWAR_COMPUTE(2);
    LAUNCH_WARP_FORWAR_COMPUTE(4);     // dim_size: 3~4
    LAUNCH_WARP_FORWAR_COMPUTE(8);     // dim_size: 5~8
    LAUNCH_WARP_FORWAR_COMPUTE(16);    // dim_size: 9~16
    LAUNCH_WARP_FORWAR_COMPUTE(32);    // dim_size: 17~32
    LAUNCH_WARP_FORWAR_COMPUTE(64);    // dim_size: 33~64
    LAUNCH_WARP_FORWAR_COMPUTE(128);   // dim_size 65~128
    LAUNCH_WARP_FORWAR_COMPUTE(256);   // dim_size 129~256
    LAUNCH_WARP_FORWAR_COMPUTE(512);   // dim_size 257~512
    LAUNCH_WARP_FORWAR_COMPUTE(1024);  // dim_size 513~1024

    default:
      break;
  }
}

template <typename T>
class LogSoftmaxKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *x = context.Input<framework::Tensor>("X");
    auto *out = context.Output<framework::Tensor>("Out");
    const auto *input_data = x->data<T>();
    auto *output_data = out->mutable_data<T>(context.GetPlace());

    const int rank = x->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    int dim_size = x->dims()[axis];
    int inner_size = 1;
    for (int i = axis + 1; i < x->dims().size(); ++i) {
      inner_size *= x->dims()[i];
    }
    int outer_size = SizeToAxis(axis, x->dims());
    gpuStream_t stream = context.cuda_device_context().stream();

    if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
      LaunchSoftmaxForwardForLastAxis<T, MPDType>(output_data, input_data,
                                                  dim_size, outer_size, stream);
    } else {
      LogSoftmaxFunctor<platform::CUDADeviceContext, T>()(
          context.template device_context<platform::CUDADeviceContext>(), x,
          out, axis);
    }
  }
};

// Backward below
#define LAUNCH_WARP_BACKWARD_COMPUTE(near_greater_power_of_two)              \
  case near_greater_power_of_two:                                            \
    ComputeLogSoftmaxBackwardInWarp<                                         \
        T, AccT, near_greater_power_of_two><<<blocks, threads, 0, stream>>>( \
        output, grad_output, grad_input, outer_size, dim_size);              \
    break;

template <typename T, typename AccT, int NearGreaterPowerOfTwo>
__global__ void ComputeLogSoftmaxBackwardInWarp(const T *output,
                                                const T *grad_output,
                                                T *grad_input, int batch_size,
                                                int element_count) {
  constexpr int near_greater_power_of_two = NearGreaterPowerOfTwo;
  constexpr int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  constexpr int warp_iter = near_greater_power_of_two / kernel_warp_size;
  int batch_id = blockDim.y * blockIdx.x + threadIdx.y;

  int thread_in_warp_idx = threadIdx.x % kernel_warp_size;

  // 1.read data from global memory to registers
  AccT output_register[warp_iter];
  AccT grad_output_register[warp_iter];
  int effective_element_count = (batch_id < batch_size) ? element_count : 0;
  for (int iter = 0; iter < warp_iter; ++iter) {
    int element_index = thread_in_warp_idx + iter * kernel_warp_size;
    if (element_index < effective_element_count) {
      output_register[iter] =
          static_cast<AccT>(output[batch_id * element_count + element_index]);
      grad_output_register[iter] = static_cast<AccT>(
          grad_output[batch_id * element_count + element_index]);
    } else {
      output_register[iter] = AccT(0);
      grad_output_register[iter] = AccT(0);
    }
  }

  // 2. For each warp, accumulate all thread registers
  AccT sum = grad_output_register[0];
#pragma unroll
  for (int iter = 1; iter < warp_iter; ++iter) {
    sum += grad_output_register[iter];
  }
  sum = WarpReduceSum<AccT, kernel_warp_size>(sum);

// 3. write result in grad_input
#pragma unroll
  for (int iter = 0; iter < warp_iter; ++iter) {
    int element_index = thread_in_warp_idx + iter * kernel_warp_size;
    if (element_index < effective_element_count) {
      grad_input[batch_id * element_count + element_index] = static_cast<T>(
          (grad_output_register[iter] - std::exp(output_register[iter]) * sum));
    }
  }
}

template <typename T, typename AccT>
void LaunchSoftmaxBackwardForLastAxis(T *grad_input, const T *grad_output,
                                      const T *output, int dim_size,
                                      int outer_size, gpuStream_t stream) {
  int threads_per_block = 128;
  int near_greater_power_of_two = GetNearGreaterPowerOfTwo(dim_size);
  int kernel_warp_size =
      (near_greater_power_of_two < 32) ? near_greater_power_of_two : 32;
  int warps_per_block = (threads_per_block / kernel_warp_size);
  int blocks = (outer_size + warps_per_block - 1) / warps_per_block;
  dim3 threads(kernel_warp_size, warps_per_block, 1);

  switch (near_greater_power_of_two) {
    LAUNCH_WARP_BACKWARD_COMPUTE(1);     // dim_size: 1
    LAUNCH_WARP_BACKWARD_COMPUTE(2);     // dim_size: 2
    LAUNCH_WARP_BACKWARD_COMPUTE(4);     // dim_size: 3~4
    LAUNCH_WARP_BACKWARD_COMPUTE(8);     // dim_size: 5~8
    LAUNCH_WARP_BACKWARD_COMPUTE(16);    // dim_size: 9~16
    LAUNCH_WARP_BACKWARD_COMPUTE(32);    // dim_size: 17~32
    LAUNCH_WARP_BACKWARD_COMPUTE(64);    // dim_size: 33~64
    LAUNCH_WARP_BACKWARD_COMPUTE(128);   // dim_size: 65~128
    LAUNCH_WARP_BACKWARD_COMPUTE(256);   // dim_size: 129~256
    LAUNCH_WARP_BACKWARD_COMPUTE(512);   // dim_size: 257~512
    LAUNCH_WARP_BACKWARD_COMPUTE(1024);  // dim_size: 513~1024

    default:
      break;
  }
}

template <typename T>
class LogSoftmaxGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
  using MPDType = typename details::MPTypeTrait<T>::Type;

 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const auto *out = context.Input<framework::Tensor>("Out");
    const auto *g_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *g_x = context.Output<framework::Tensor>(framework::GradVarName("X"));

    const auto *out_data = out->data<T>();
    const auto *g_out_data = g_out->data<T>();
    auto *g_x_data = g_x->mutable_data<T>(context.GetPlace());

    const int rank = out->dims().size();
    const int axis = CanonicalAxis(context.Attr<int>("axis"), rank);

    int dim_size = out->dims()[axis];
    int inner_size = 1;
    for (int i = axis + 1; i < out->dims().size(); ++i) {
      inner_size *= out->dims()[i];
    }
    int outer_size = SizeToAxis(axis, out->dims());
    gpuStream_t stream = context.cuda_device_context().stream();

    if (inner_size == 1 && dim_size <= 1024 && dim_size * sizeof(T) <= 4096) {
      LaunchSoftmaxBackwardForLastAxis<T, MPDType>(
          g_x_data, g_out_data, out_data, dim_size, outer_size, stream);
    } else {
      LogSoftmaxGradFunctor<platform::CUDADeviceContext, T>()(
          context.template device_context<platform::CUDADeviceContext>(), out,
          g_out, g_x, axis);
    }
  }
};

}  // operators
}  // paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    log_softmax, ops::LogSoftmaxKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxKernel<plat::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    log_softmax_grad, ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, float>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, double>,
    ops::LogSoftmaxGradKernel<plat::CUDADeviceContext, plat::float16>);
