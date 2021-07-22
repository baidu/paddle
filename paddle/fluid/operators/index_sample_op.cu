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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/index_sample_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/cuda_device_function.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename T, typename IndexT = int>
__global__ void IndexSampleForward(const IndexT* index, const T* in_data,
                                   T* out_data, size_t index_length,
                                   size_t input_length, size_t batch_size) {
  int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  int index_j = blockDim.y * blockIdx.y + threadIdx.y;
  int index_idx = index_j * index_length + index_i;
  int in_idx = index_j * input_length + index_i;

  if (index_i < index_length & index_j < batch_size) {
    IndexT sample_idx = index[index_idx];
    out_data[index_idx] = in_data[in_idx - index_i + sample_idx];
  }
}

template <typename T, typename IndexT = int>
__global__ void IndexSampleGrad(const IndexT* index, T* in_grad,
                                const T* out_grad, size_t index_length,
                                size_t input_length, size_t batch_size,
                                bool same_data_in_row = true) {
  int index_i = blockDim.x * blockIdx.x + threadIdx.x;
  int index_j = blockDim.y * blockIdx.y + threadIdx.y;
  int index_idx = index_j * index_length + index_i;
  int in_idx = index_j * input_length + index_i;

  if (index_i < index_length & index_j < batch_size) {
    IndexT sample_idx = index[index_idx];
    if (same_data_in_row) {
      platform::CudaAtomicAdd(&(in_grad[in_idx - index_i + sample_idx]),
                              out_grad[sample_idx]);
    } else {
      in_grad[in_idx - index_i + sample_idx] = out_grad[index_idx];
    }
  }
}

template <typename T>
class IndexSampleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("X");
    auto* index = ctx.Input<LoDTensor>("Index");
    auto* output = ctx.Output<LoDTensor>("Out");

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                            index_type == framework::proto::VarType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));
    const auto* in_data = input->data<T>();
    auto* out_data = output->mutable_data<T>(ctx.GetPlace());
    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();

    auto input_dim = input->dims();
    auto index_dim = index->dims();
    size_t batch_size = input_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];

    auto block_width = platform::RoundToPowerOfTwo(index_length);
    int block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;

    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      IndexSampleForward<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
          index_data, in_data, out_data, index_length, input_length,
          batch_size);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      IndexSampleForward<T, int><<<grid_dim, block_dim, 0, stream>>>(
          index_data, in_data, out_data, index_length, input_length,
          batch_size);
    }
  }
};

template <typename T>
class IndexSampleGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* output_grad = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* index = ctx.Input<LoDTensor>("Index");

    const auto* output_grad_data = output_grad->data<T>();
    auto* input_grad_data = input_grad->mutable_data<T>(ctx.GetPlace());

    const auto& index_type = index->type();
    bool index_type_match = index_type == framework::proto::VarType::INT64 ||
                            index_type == framework::proto::VarType::INT32;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto stream =
        ctx.template device_context<platform::CUDADeviceContext>().stream();
    auto input_num = input_grad->numel();
    auto input_dim = input_grad->dims();
    auto index_dim = index->dims();
    size_t batch_size = index_dim[0];
    size_t input_length = input_dim[1];
    size_t index_length = index_dim[1];
    bool same_data_in_index_row = index_length == 1 ? false : true;

    auto block_width = platform::RoundToPowerOfTwo(index_length);
    auto block_height =
        platform::RoundToPowerOfTwo(index_length * batch_size) / block_width;
    dim3 block_dim(block_width, block_height);
    dim3 grid_dim((index_length + block_dim.x - 1) / block_dim.x,
                  (batch_size + block_dim.y - 1) / block_dim.y);

    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    set_zero(dev_ctx, input_grad, static_cast<T>(0));

    if (index_type == framework::proto::VarType::INT64) {
      const int64_t* index_data = index->data<int64_t>();
      IndexSampleGrad<T, int64_t><<<grid_dim, block_dim, 0, stream>>>(
          index_data, input_grad_data, output_grad_data, index_length,
          input_length, batch_size, same_data_in_index_row);
    } else if (index_type == framework::proto::VarType::INT32) {
      const int* index_data = index->data<int>();
      IndexSampleGrad<T, int><<<grid_dim, block_dim, 0, stream>>>(
          index_data, input_grad_data, output_grad_data, index_length,
          input_length, batch_size, same_data_in_index_row);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace paddle {
namespace operators {

using DDim = framework::DDim;

template <typename T, typename IndexT = int>
__global__ void IndexSampleCUDAInner(const T* input, int64_t input_width,
        const IndexT* index, int64_t index_width, int64_t numel,
        T* output) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numel) {
        // ignore the idle thread
        return ;
    }

    int64_t input_idx = (int64_t)(idx / index_width) * input_width + index[idx];

    output[idx] = input[input_idx];
}

template <typename T>
class IndexSampleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *input_var = ctx.InputVar("X");
    auto *index_var = ctx.InputVar("Index");

    auto &input_tensor = input_var->Get<LoDTensor>();
    auto &index_tensor = index_var->Get<LoDTensor>();

    auto *out_var = ctx.OutputVar("Out");
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

    const auto &index_type = index_tensor.type();
    bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                            index_type == framework::proto::VarType::INT64;
    PADDLE_ENFORCE_EQ(index_type_match, true,
                      platform::errors::InvalidArgument(
                          "Input(Index) holds the wrong type, it holds %s, but "
                          "desires to be %s or %s",
                          paddle::framework::DataTypeToString(index_type),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT32),
                          paddle::framework::DataTypeToString(
                              framework::proto::VarType::INT64)));

    auto input_dims = input_tensor.dims();
    int64_t input_width = input_dims[1];

    auto index_dims = index_tensor.dims();
    int64_t index_width = index_dims[1];

    T* output = static_cast<T*>(out_tensor->mutable_data<T>(ctx.GetPlace()));

    if (index_type == framework::proto::VarType::INT32) {
      IndexSampleCUDAInner<T, int><<<1, index_tensor.numel()>>>(input_tensor.data<T>(), input_width,
              index_tensor.data<int>(), index_width, index_tensor.numel(),
              output);
    } else if (index_type == framework::proto::VarType::INT64) {
      IndexSampleCUDAInner<T, int64_t><<<1, index_tensor.numel()>>>(input_tensor.data<T>(), input_width,
              index_tensor.data<int64_t>(), index_width, index_tensor.numel(),
              output);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    index_sample,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    index_sample_grad,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::IndexSampleGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
