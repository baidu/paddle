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

#include "paddle/fluid/operators/cast_op.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

// aligned vector generates vectorized load/store on CUDA
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize(const T* pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  }
  return 1;
}

template <typename InT, typename OutT, int VecSize>
__global__ void VecCastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  using LoadT = AlignedVector<InT, VecSize>;
  using StoreT = AlignedVector<OutT, VecSize>;
  for (int64_t i = idx * VecSize; i < N;
       i += blockDim.x * gridDim.x * VecSize) {
    InT in_vec[VecSize];
    LoadT* in_value = reinterpret_cast<LoadT*>(&in_vec);
    *in_value = *reinterpret_cast<const LoadT*>(&in[i]);

    OutT out_vec[VecSize];
#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      out_vec[ii] = static_cast<OutT>(in_vec[ii]);
    }

    *(reinterpret_cast<StoreT*>(&out[i])) =
        *reinterpret_cast<StoreT*>(&out_vec[0]);
  }
}

template <typename InT, typename OutT>
__global__ void CastCUDAKernel(const InT* in, const int64_t N, OutT* out) {
  CUDA_KERNEL_LOOP(index, N) { out[index] = static_cast<OutT>(in[index]); }
}

template <typename InT>
struct CastOpFunctor<platform::CUDADeviceContext, InT> {
  const framework::Tensor* in_;
  framework::Tensor* out_;
  const platform::CUDADeviceContext& ctx_;
  CastOpFunctor(const framework::Tensor* in, framework::Tensor* out,
                const platform::CUDADeviceContext& ctx)
      : in_(in), out_(out), ctx_(ctx) {}

  template <typename OutT>
  void apply() const {
    auto* in = in_->data<InT>();
    auto size = in_->numel();
    auto* out = out_->mutable_data<OutT>(ctx_.GetPlace());
    platform::GpuLaunchConfig config =
        platform::GetGpuLaunchConfig1D(ctx_, size);
    int vec_size = VectorizedSize<OutT>(out);
    if (!std::is_same<InT, OutT>::value && vec_size == 4 && size % 4 == 0) {
      VecCastCUDAKernel<InT, OutT, 4><<<
          config.block_per_grid, config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    } else {
      CastCUDAKernel<InT, OutT><<<config.block_per_grid,
                                  config.thread_per_block, 0, ctx_.stream()>>>(
          in, size, out);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
#define REGISTER_CAST_BASE(op_name, ...)                                      \
  REGISTER_OP_CUDA_KERNEL(                                                    \
      op_name, ops::CastOpKernel<paddle::platform::CUDADeviceContext, float>, \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext, double>,         \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext, int>,            \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext, int64_t>,        \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext, bool>,           \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext, uint8_t>,        \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext,                  \
                        paddle::platform::float16>,                           \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext,                  \
                        paddle::platform::complex<float>>,                    \
      ops::CastOpKernel<paddle::platform::CUDADeviceContext,                  \
                        paddle::platform::complex<double>>,                   \
      ##__VA_ARGS__);

#define REGISTER_CAST_EX(op_name)                                           \
  REGISTER_CAST_BASE(op_name,                                               \
                     ops::CastOpKernel<paddle::platform::CUDADeviceContext, \
                                       paddle::platform::bfloat16>)

#if !defined(PADDLE_WITH_HIP)
REGISTER_CAST_EX(cast)
#else
REGISTER_CAST_BASE(cast)
#endif
