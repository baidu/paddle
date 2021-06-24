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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/diagonal_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T, int X_DIM_SIZE, int OUT_DIM_SIZE>
__global__ void Diagonal(const T* data1, T* data2, const int64_t offset_,
                         int64_t axis1_, int64_t axis2_, int64_t* x_stride,
                         int64_t* out_stride, int64_t numel, bool is_grad) {
  CUDA_KERNEL_LOOP(idx, numel) {
    int64_t idx_dim[X_DIM_SIZE] = {0};
    int64_t temp = 0;
    for (size_t i = 0; i < X_DIM_SIZE - 1; i++) {
      idx_dim[i] = (idx - temp) / x_stride[i];
      temp = temp + idx_dim[i] * x_stride[i];
    }
    idx_dim[X_DIM_SIZE - 1] = idx - temp;

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    int64_t out_dim[OUT_DIM_SIZE] = {0};
    int temp_pos = 0;
    for (int i = 0; i < X_DIM_SIZE; i++) {
      if (i != axis1_ && i != axis2_) {
        out_dim[temp_pos] = idx_dim[i];
        temp_pos++;
      }
    }
    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis2_dim;
      flag = true;
    }
    if (!is_grad) {
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < OUT_DIM_SIZE - 1; i++) {
          idx_output = idx_output + out_dim[i] * out_stride[i];
        }
        idx_output = idx_output + out_dim[OUT_DIM_SIZE - 1];
        data2[idx_output] = data1[idx];
      }
    } else {
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < OUT_DIM_SIZE - 1; i++) {
          idx_output = idx_output + out_dim[i] * out_stride[i];
        }
        idx_output = idx_output + out_dim[OUT_DIM_SIZE - 1];
        data2[idx] = data1[idx_output];
      } else {
        data2[idx] = static_cast<T>(0);
      }
    }
  }
}

template <typename T>
class DiagonalCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const auto* input_data = input->data<T>();
    auto input_dim = input->dims().Get();
    auto input_dim_size = input->dims().size();

    std::vector<int64_t> res_in = vectorize(framework::stride(input->dims()));
    paddle::framework::Tensor input_stride_tensor;
    framework::TensorFromVector<int64_t>(res_in, context.device_context(),
                                         &input_stride_tensor);
    int64_t* input_stride = input_stride_tensor.data<int64_t>();

    auto* output = context.Output<framework::Tensor>("Out");
    auto* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = output->dims().Get();
    auto output_dim_size = output->dims().size();

    std::vector<int64_t> res_out = vectorize(framework::stride(output->dims()));
    paddle::framework::Tensor output_stride_tensor;
    framework::TensorFromVector<int64_t>(res_out, context.device_context(),
                                         &output_stride_tensor);
    int64_t* output_stride = output_stride_tensor.data<int64_t>();

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;
    int64_t numel = input->numel();

    int threads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + threads - 1) / threads;

    switch (input_dim_size) {
      case 2:
        Diagonal<T, 2, 1><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 3:
        Diagonal<T, 3, 2><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 4:
        Diagonal<T, 4, 3><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 5:
        Diagonal<T, 5, 4><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 6:
        Diagonal<T, 6, 5><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 7:
        Diagonal<T, 7, 6><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 8:
        Diagonal<T, 8, 7><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      case 9:
        Diagonal<T, 9, 8><<<blocks, threads>>>(input_data, output_data, offset_,
                                               axis1_, axis2_, input_stride,
                                               output_stride, numel, false);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 10, but received %d.",
            input_dim_size));
    }
  }
};

template <typename T>
class DiagonalGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const auto* dout_data = dout->data<T>();
    auto dout_dim = dout->dims().Get();
    auto dout_dim_size = dout->dims().size();

    std::vector<int64_t> res_dout = vectorize(framework::stride(dout->dims()));
    paddle::framework::Tensor dout_stride_tensor;
    framework::TensorFromVector<int64_t>(res_dout, context.device_context(),
                                         &dout_stride_tensor);
    int64_t* dout_stride = dout_stride_tensor.data<int64_t>();

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    auto* dx_data = dx->mutable_data<T>(context.GetPlace());
    auto dx_dim = dx->dims().Get();
    auto dx_dim_size = dx->dims().size();

    std::vector<int64_t> res_dx = vectorize(framework::stride(dx->dims()));
    paddle::framework::Tensor dx_stride_tensor;
    framework::TensorFromVector<int64_t>(res_dx, context.device_context(),
                                         &dx_stride_tensor);
    int64_t* dx_stride = dx_stride_tensor.data<int64_t>();

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? dx_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? dx_dim_size + axis2 : axis2;

    int64_t numel = dx->numel();

    int threads = PADDLE_CUDA_NUM_THREADS;
    int blocks = (numel + threads - 1) / threads;

    switch (dx_dim_size) {
      case 2:
        Diagonal<T, 2, 1><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 3:
        Diagonal<T, 3, 2><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 4:
        Diagonal<T, 4, 3><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 5:
        Diagonal<T, 5, 4><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 6:
        Diagonal<T, 6, 5><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 7:
        Diagonal<T, 7, 6><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 8:
        Diagonal<T, 8, 7><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      case 9:
        Diagonal<T, 9, 8><<<blocks, threads>>>(dout_data, dx_data, offset_,
                                               axis1_, axis2_, dx_stride,
                                               dout_stride, numel, true);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of output(input@Grad) should be less than 10, but "
            "received %d.",
            dx_dim_size));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(diagonal, ops::DiagonalCUDAKernel<int>,
                        ops::DiagonalCUDAKernel<int64_t>,
                        ops::DiagonalCUDAKernel<float>,
                        ops::DiagonalCUDAKernel<double>,
                        ops::DiagonalCUDAKernel<plat::float16>,
                        ops::DiagonalCUDAKernel<bool>);

REGISTER_OP_CUDA_KERNEL(diagonal_grad, ops::DiagonalGradCUDAKernel<int>,
                        ops::DiagonalGradCUDAKernel<int64_t>,
                        ops::DiagonalGradCUDAKernel<float>,
                        ops::DiagonalGradCUDAKernel<double>,
                        ops::DiagonalGradCUDAKernel<plat::float16>);
