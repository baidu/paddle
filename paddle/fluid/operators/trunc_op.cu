/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/trunc_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
class TruncFunctor {
 public:
  __device__ TruncFunctor(const T x) : x_(x) {}
  __device__ T operator()() { return trunc(x_); }

 public:
  const T x_;
};

template <>
class TruncFunctor<int> {
 public:
  __device__ TruncFunctor(const int x) : x_(x) {}
  __device__ int operator()() { return x_; }

 public:
  const int x_;
};

template <>
class TruncFunctor<int64_t> {
 public:
  __device__ TruncFunctor(const int64_t x) : x_(x) {}
  __device__ int64_t operator()() { return x_; }

 public:
  const int64_t x_;
};

template <typename T>
__global__ void Trunc(const T* x, T* out, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) {
    TruncFunctor<T> functor(x[index]);
    out[index] = functor();
  }
}

template <typename T>
__global__ void TruncGrad(T* dx, int64_t N) {
  CUDA_KERNEL_LOOP(index, N) { dx[index] = static_cast<T>(0.0); }
}

template <typename T>
class TruncCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out = context.Output<Tensor>("Out");

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());

    int64_t numel = x->numel();

    int theads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + theads - 1) / theads;

    Trunc<<<blocks, theads>>>(x_data, out_data, numel);
  }
};

template <typename T>
class TruncCUDAGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* dout = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));

    const auto* dout_data = dout->data<T>();
    auto* dx_data = dx->mutable_data<T>(context.GetPlace());

    int64_t numel = dout->numel();

    int theads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + theads - 1) / theads;

    TruncGrad<<<blocks, theads>>>(dx_data, numel);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(trunc, ops::TruncCUDAKernel<float>,
                        ops::TruncCUDAKernel<double>, ops::TruncCUDAKernel<int>,
                        ops::TruncCUDAKernel<int64_t>);

REGISTER_OP_CUDA_KERNEL(trunc_grad, ops::TruncCUDAGradKernel<float>,
                        ops::TruncCUDAGradKernel<double>,
                        ops::TruncCUDAGradKernel<int>,
                        ops::TruncCUDAGradKernel<int64_t>);
