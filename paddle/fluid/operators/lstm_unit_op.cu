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

/* Acknowledgement: the following code is strongly inspired by
https://github.com/caffe2/caffe2/blob/master/caffe2/operators/lstm_unit_op_gpu.cu
*/

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cross_entropy_op.h"
#include "paddle/fluid/operators/lstm_unit_op.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

template <typename Dtype>
__device__ Dtype cuda_sigmoid(const Dtype x) {
  return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype cuda_tanh(const Dtype x) {
  return Dtype(1 - exp(-2. * x)) / (Dtype(1) + exp(-2. * x));
}

template <typename T>
__global__ void LSTMUnitKernel(const int nthreads, const int dim,
                               const T* C_prev, const T* X, T* C, T* H,
                               const T forget_bias) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;

    const T* X_offset = X + 4 * dim * n;
    const T i = cuda_sigmoid(X_offset[d]);
    const T f = cuda_sigmoid(X_offset[1 * dim + d] + forget_bias);
    const T o = cuda_sigmoid(X_offset[2 * dim + d]);
    const T g = cuda_tanh(X_offset[3 * dim + d]);
    const T c_prev = C_prev[index];
    const T c = f * c_prev + i * g;
    C[index] = c;
    const T tanh_c = cuda_tanh(c);
    H[index] = o * tanh_c;
  }
}

template <typename T>
__global__ void LSTMUnitGradientKernel(const int nthreads, const int dim,
                                       const T* C_prev, const T* X, const T* C,
                                       const T* C_diff, const T* H_diff,
                                       T* C_prev_diff, T* X_diff,
                                       const T forget_bias) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const T* X_offset = X + 4 * dim * n;
    T* c_prev_diff = C_prev_diff + index;
    T* X_diff_offset = X_diff + 4 * dim * n;
    T* i_diff = X_diff_offset + d;
    T* f_diff = X_diff_offset + 1 * dim + d;
    T* o_diff = X_diff_offset + 2 * dim + d;
    T* g_diff = X_diff_offset + 3 * dim + d;

    const T i = cuda_sigmoid(X_offset[d]);
    const T f = cuda_sigmoid(X_offset[1 * dim + d] + forget_bias);
    const T o = cuda_sigmoid(X_offset[2 * dim + d]);
    const T g = cuda_tanh(X_offset[3 * dim + d]);
    const T c_prev = C_prev[index];
    const T c = C[index];
    const T tanh_c = cuda_tanh(c);
    const T c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    *c_prev_diff = c_term_diff * f;
    *i_diff = c_term_diff * g * i * (1 - i);
    *f_diff = c_term_diff * c_prev * f * (1 - f);
    *o_diff = H_diff[index] * tanh_c * o * (1 - o);
    *g_diff = c_term_diff * i * (1 - g * g);
  }
}

template <typename T>
class LstmUnitOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));

    auto* x_tensor = ctx.Input<framework::Tensor>("X");
    auto* c_prev_tensor = ctx.Input<framework::Tensor>("C_prev");
    auto* c_tensor = ctx.Output<framework::Tensor>("C");
    auto* h_tensor = ctx.Output<framework::Tensor>("H");

    auto forget_bias = static_cast<T>(ctx.Attr<float>("forget_bias"));

    int b_size = c_tensor->dims()[0];
    int D = c_tensor->dims()[1];

    const T* X = x_tensor->data<T>();
    const T* C_prev = c_prev_tensor->data<T>();

    T* C = c_tensor->mutable_data<T>(ctx.GetPlace());
    T* H = h_tensor->mutable_data<T>(ctx.GetPlace());

    int block = 512;
    int n = b_size * D;
    int grid = (n + block - 1) / block;

    LSTMUnitKernel<T><<<grid, block>>>(n, D, C_prev, X, C, H, forget_bias);
  }
};

template <typename T>
class LstmUnitGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CUDAPlace."));

    auto x_tensor = ctx.Input<Tensor>("X");
    auto c_prev_tensor = ctx.Input<Tensor>("C_prev");
    auto c_tensor = ctx.Input<Tensor>("C");
    auto h_tensor = ctx.Input<Tensor>("H");

    auto hdiff_tensor = ctx.Input<Tensor>(framework::GradVarName("H"));
    auto cdiff_tensor = ctx.Input<Tensor>(framework::GradVarName("C"));

    auto xdiff_tensor = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto c_prev_diff_tensor =
        ctx.Output<Tensor>(framework::GradVarName("C_prev"));

    auto* X = x_tensor->data<T>();
    auto* C_prev = c_prev_tensor->data<T>();
    auto* C = c_tensor->data<T>();

    auto* H_diff = hdiff_tensor->data<T>();
    auto* C_diff = cdiff_tensor->data<T>();

    auto* C_prev_diff = c_prev_diff_tensor->mutable_data<T>(ctx.GetPlace());
    auto* X_diff = xdiff_tensor->mutable_data<T>(ctx.GetPlace());

    int N = c_tensor->dims()[0];
    int D = c_tensor->dims()[1];

    auto forget_bias = static_cast<T>(ctx.Attr<float>("forget_bias"));

    int block = 512;
    int n = N * D;
    int grid = (n + block - 1) / block;

    LSTMUnitGradientKernel<T><<<grid, block>>>(
        n, D, C_prev, X, C, C_diff, H_diff, C_prev_diff, X_diff, forget_bias);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(lstm_unit, ops::LstmUnitOpCUDAKernel<float>,
                        ops::LstmUnitOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(lstm_unit_grad, ops::LstmUnitGradOpCUDAKernel<float>,
                        ops::LstmUnitGradOpCUDAKernel<double>);
