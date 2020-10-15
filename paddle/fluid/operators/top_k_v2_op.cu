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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/fluid/operators/top_k_v2_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define FIXED_BLOCK_DIM_BASE(dim, ...) \
  case (dim): {                        \
    constexpr auto kBlockDim = (dim);  \
    __VA_ARGS__;                       \
  } break

#define FIXED_BLOCK_DIM(...)                \
  FIXED_BLOCK_DIM_BASE(256, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(128, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_BASE(64, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_BASE(32, ##__VA_ARGS__)

template <typename DeviceContext, typename T>
class TopkV2OpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");
    auto* indices = ctx.Output<Tensor>("Indices");

    // get the attributes
    int k = static_cast<int>(ctx.Attr<int>("k"));
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    const bool& sorted = static_cast<bool>(ctx.Attr<bool>("sorted"));
    const bool& largest = static_cast<bool>(ctx.Attr<bool>("largest"));

    // get the input dims
    const auto& in_dims = input->dims();
    // calcluate the real axis
    if (axis < 0) axis += in_dims.size();

    auto* k_t = ctx.Input<Tensor>("K");
    if (k_t) {
      Tensor k_host;
      framework::TensorCopySync(*k_t, platform::CPUPlace(), &k_host);
      k = k_host.data<int>()[0];
      framework::DDim output_dims = output->dims();
      output_dims[axis] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    const auto& out_dims = output->dims();

    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    if (axis == in_dims.size() - 1) {
      // if get the topK from the last axis
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      const auto& dev_ctx = ctx.cuda_device_context();

      if (k > input_width) k = input_width;

      if ((input_width <= 1024 || k >= 128 || k == input_width)) {
        if (SortTopk<T>(dev_ctx, input, input_width, input_height, k, output,
                        indices, largest)) {
          // Successed, return.
          return;
        } else {
          LOG(INFO) << "TopKOP: Some errors happened when use cub sorting, use "
                       "default topk kernel.";
        }
      }

      // NOTE: pass lds and dim same to input width.
      // NOTE: old matrix implementation of stride is different to eigen.
      const int kMaxHeight = 2048;
      int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
      switch (GetDesiredBlockDim(input_width)) {
        FIXED_BLOCK_DIM(
            KeMatrixTopK<T, 5,
                         kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                output_data, k, indices_data, input_data, input_width,
                input_width, static_cast<int>(k), gridx, input_height,
                largest));
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "the input data shape has error in the topk cuda kernel."));
      }
    } else {
      // if get topK not from the last axis, will tranpose the tensor and get
      // TopK

      // first step, prepare the trans args for the tranpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);

      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(output->dims());
      for (int i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
        trans_out_dims[i] = out_dims[trans[i]];
      }
      // second step, tranpose the input
      Tensor trans_input;
      trans_input.mutable_data<T>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      const auto& dev_ctx = ctx.cuda_device_context();
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_input, trans);
      // third step, calcluate the topk
      // allocate the tmp cuda memory for the tmp result
      Tensor trans_ind;
      trans_ind.mutable_data<int64_t>(trans_out_dims, ctx.GetPlace());
      Tensor trans_out;
      trans_out.mutable_data<T>(trans_out_dims, ctx.GetPlace());

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      if (k > input_width) k = input_width;

      if ((input_width <= 1024 || k >= 128 || k == input_width)) {
        if (SortTopk<T>(dev_ctx, &trans_input, input_width, input_height, k,
                        &trans_out, &trans_ind, largest)) {
          // last step, tranpose back the indices and output
          TransCompute<platform::CUDADeviceContext, int64_t>(
              ndims, dev_ctx, trans_ind, indices, trans);
          TransCompute<platform::CUDADeviceContext, T>(
              ndims, dev_ctx, trans_out, output, trans);
          return;
        } else {
          LOG(INFO) << "TopKOP: Some errors happened when use cub sorting, use "
                       "default topk kernel.";
        }
      }

      const int kMaxHeight = 2048;
      int gridx = input_height < kMaxHeight ? input_height : kMaxHeight;
      switch (GetDesiredBlockDim(input_width)) {
        FIXED_BLOCK_DIM(
            KeMatrixTopK<T, 5,
                         kBlockDim><<<gridx, kBlockDim, 0, dev_ctx.stream()>>>(
                trans_out.data<T>(), k, trans_ind.data<int64_t>(),
                trans_input.data<T>(), input_width, input_width,
                static_cast<int>(k), gridx, input_height, largest));
        default:
          PADDLE_THROW(platform::errors::Fatal(
              "the input data shape has error in the topk cuda kernel."));
      }

      // last step, tranpose back the indices and output
      TransCompute<platform::CUDADeviceContext, int64_t>(
          ndims, dev_ctx, trans_ind, indices, trans);
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, trans_out,
                                                   output, trans);
    }
  }
};

#undef FIXED_BLOCK_DIM_BASE
#undef FIXED_BLOCK_DIM
template <typename DeviceContext, typename T>
class TopkV2OpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* x = context.Input<Tensor>("X");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    int axis = context.Attr<int>("axis");

    const auto& in_dims = x->dims();
    const auto& out_dims = indices->dims();

    // get the real the axis and the k
    if (axis < 0) axis += in_dims.size();
    const int& k = out_dims[axis];
    const int& raw_height = in_dims[axis];

    // allocate the cuda memory for the x_grad
    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();

    int pre, n, post;
    GetDims(in_dims, axis, &pre, &n, &post);

    // calcluate the block and grid num
    auto& dev_ctx = context.cuda_device_context();
    auto ComputeBlockSize = [](int col) {
      if (col > 512)
        return 1024;
      else if (col > 256 && col <= 512)
        return 512;
      else if (col > 128 && col <= 256)
        return 256;
      else if (col > 64 && col <= 128)
        return 128;
      else
        return 64;
    };
    int block_size = ComputeBlockSize(post * k);
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
    int grid_size = std::min(max_blocks, pre);

    // lanuch the cuda kernel to assign the grad
    AssignGradWithAxis<T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
        out_grad_data, indices_data, x_grad_data, pre, post, n, k);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    top_k_v2,
    paddle::operators::TopkV2OpCUDAKernel<paddle::platform::CUDADeviceContext,
                                          float>,
    paddle::operators::TopkV2OpCUDAKernel<paddle::platform::CUDADeviceContext,
                                          double>,
    paddle::operators::TopkV2OpCUDAKernel<paddle::platform::CUDADeviceContext,
                                          int>,
    paddle::operators::TopkV2OpCUDAKernel<paddle::platform::CUDADeviceContext,
                                          int64_t>,
    paddle::operators::TopkV2OpCUDAKernel<paddle::platform::CUDADeviceContext,
                                          paddle::platform::float16>);

REGISTER_OP_CUDA_KERNEL(
    top_k_v2_grad, paddle::operators::TopkV2OpGradCUDAKernel<
                       paddle::platform::CUDADeviceContext, float>,
    paddle::operators::TopkV2OpGradCUDAKernel<
        paddle::platform::CUDADeviceContext, double>,
    paddle::operators::TopkV2OpGradCUDAKernel<
        paddle::platform::CUDADeviceContext, int>,
    paddle::operators::TopkV2OpGradCUDAKernel<
        paddle::platform::CUDADeviceContext, int64_t>,
    paddle::operators::TopkV2OpGradCUDAKernel<
        paddle::platform::CUDADeviceContext, paddle::platform::float16>);
