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

#ifndef PADDLE_WITH_HIP
// HIP not support cudnnSpatialTfGridGeneratorForward

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

using framework::Tensor;
using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::CudnnDataLayout;
using ScopedSpatialTransformerDescriptor =
    platform::ScopedSpatialTransformerDescriptor;
template <typename T>
using CudnnDataType = platform::CudnnDataType<T>;

template <typename T>
class CUDNNGridSampleOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::InvalidArgument(
                          "It must use CUDAPlace when using CUDA Kernel"));
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    auto* output = ctx.Output<Tensor>("Output");

    int n = input->dims()[0];
    int c = input->dims()[1];
    int out_h = grid->dims()[1];
    int out_w = grid->dims()[2];
    const int size[4] = {n, c, out_h, out_w};

    const T* input_data = input->data<T>();
    const T* grid_data = grid->data<T>();
    T* output_data =
        output->mutable_data<T>({n, c, out_h, out_w}, ctx.GetPlace());

    ScopedSpatialTransformerDescriptor st_desc;
    cudnnSpatialTransformerDescriptor_t cudnn_st_desc =
        st_desc.descriptor<T>(4, size);

    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor output_desc;
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        DataLayout::kNCHW, framework::vectorize<int>(input->dims()));
    cudnnTensorDescriptor_t cudnn_output_desc = output_desc.descriptor<T>(
        DataLayout::kNCHW, framework::vectorize<int>(output->dims()));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSpatialTfSamplerForward(
        handle, cudnn_st_desc, CudnnDataType<T>::kOne(), cudnn_input_desc,
        input_data, grid_data, CudnnDataType<T>::kZero(), cudnn_output_desc,
        output_data));
  }
};

template <typename T>
class CUDNNGridSampleGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::InvalidArgument(
                          "It must use CUDAPlace when using CUDA Kernel"));
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();
    auto* input = ctx.Input<Tensor>("X");
    auto* grid = ctx.Input<Tensor>("Grid");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Output"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* grid_grad = ctx.Output<Tensor>(framework::GradVarName("Grid"));

    auto output_grad_dims = output_grad->dims();
    const int n = output_grad_dims[0];
    const int c = output_grad_dims[1];
    const int h = output_grad_dims[2];
    const int w = output_grad_dims[3];
    const int size[4] = {n, c, h, w};

    ScopedSpatialTransformerDescriptor st_dest;
    cudnnSpatialTransformerDescriptor_t cudnn_st_dest =
        st_dest.descriptor<T>(4, size);

    const T* input_data = input->data<T>();
    const T* grid_data = grid->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    T* input_grad_data =
        input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
    T* grid_grad_data =
        grid_grad->mutable_data<T>({n, h, w, 2}, ctx.GetPlace());

    ScopedTensorDescriptor input_desc;
    ScopedTensorDescriptor input_grad_desc;
    ScopedTensorDescriptor output_grad_desc;
    cudnnTensorDescriptor_t cudnn_input_desc = input_desc.descriptor<T>(
        DataLayout::kNCHW, framework::vectorize<int>(input->dims()));
    cudnnTensorDescriptor_t cudnn_input_grad_desc =
        input_grad_desc.descriptor<T>(
            DataLayout::kNCHW, framework::vectorize<int>(input_grad->dims()));
    cudnnTensorDescriptor_t cudnn_output_grad_desc =
        output_grad_desc.descriptor<T>(
            DataLayout::kNCHW, framework::vectorize<int>(output_grad->dims()));

    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cudnnSpatialTfSamplerBackward(
            handle, cudnn_st_dest, CudnnDataType<T>::kOne(), cudnn_input_desc,
            input_data, CudnnDataType<T>::kZero(), cudnn_input_grad_desc,
            input_grad_data, CudnnDataType<T>::kOne(), cudnn_output_grad_desc,
            output_grad_data, grid_data, CudnnDataType<T>::kZero(),
            grid_grad_data));
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
REGISTER_OP_KERNEL(grid_sampler, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNGridSampleOpKernel<float>,
                   paddle::operators::CUDNNGridSampleOpKernel<double>);
REGISTER_OP_KERNEL(grid_sampler_grad, CUDNN, plat::CUDAPlace,
                   paddle::operators::CUDNNGridSampleGradOpKernel<float>,
                   paddle::operators::CUDNNGridSampleGradOpKernel<double>);

#endif  // PADDLE_WITH_HIP
