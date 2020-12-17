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

#include <set>
#include <vector>

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
struct SelectedRowsAdd<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(
        in1_height, input2.height(),
        platform::errors::InvalidArgument("The two inputs height must be equal."
                                          "But recieved first input height  = "
                                          "[%d], second input height = [%d]",
                                          in1_height, input2.height()));
    output->set_height(in1_height);

    framework::Vector<int64_t> in1_rows(input1.rows());
    auto& in2_rows = input2.rows();
    std::vector<int64_t> out_rows;
    out_rows.reserve(in1_rows.size() + in2_rows.size());

    // concat rows
    out_rows.insert(out_rows.end(), in1_rows.begin(), in1_rows.end());
    out_rows.insert(out_rows.end(), in2_rows.begin(), in2_rows.end());
    output->set_rows(out_rows);

    auto* out_value = output->mutable_value();
    auto& in1_value = input1.value();
    auto& in2_value = input2.value();

    auto in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel, in2_value.numel() / in2_rows.size(),
        platform::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But recieved first input width = [%d], second input width = [%d]",
            in1_row_numel, in2_value.numel() / in2_rows.size()));
    PADDLE_ENFORCE_EQ(
        in1_row_numel, out_value->numel() / out_rows.size(),
        platform::errors::InvalidArgument(
            "The input and oupput width must be equal."
            "But recieved input width = [%d], output width = [%d]",
            in1_row_numel, out_value->numel() / out_rows.size()));

    auto* out_data = out_value->data<T>();
    auto* in1_data = in1_value.data<T>();

    auto in1_place = input1.place();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(in1_place), true,
                      platform::errors::InvalidArgument(
                          "The running enviroment is not on the GPU place."));
    auto in2_place = input2.place();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(in2_place), true,
                      platform::errors::InvalidArgument(
                          "The running enviroment is not on the GPU place."));
    auto out_place = context.GetPlace();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(out_place), true,
                      platform::errors::InvalidArgument(
                          "The running enviroment is not on the GPU place."));

    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, out_place), out_data,
                 BOOST_GET_CONST(platform::CUDAPlace, in1_place), in1_data,
                 in1_value.numel() * sizeof(T), context.stream());

    auto* in2_data = in2_value.data<T>();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, out_place),
                 out_data + in1_value.numel(),
                 BOOST_GET_CONST(platform::CUDAPlace, in2_place), in2_data,
                 in2_value.numel() * sizeof(T), context.stream());
  }
};

template struct SelectedRowsAdd<platform::CUDADeviceContext, float>;
template struct SelectedRowsAdd<platform::CUDADeviceContext, double>;

namespace {
template <typename T, int block_size>
__global__ void SelectedRowsAddTensorKernel(const T* selected_rows,
                                            const int64_t* rows, T* tensor_out,
                                            int64_t row_numel) {
  const int ty = blockIdx.x;
  int tid = threadIdx.x;

  selected_rows += ty * row_numel;
  tensor_out += rows[ty] * row_numel;

  for (int index = tid; index < row_numel; index += block_size) {
    // Since index in rows of SelectedRows can be duplicate, we can not use
    // tensor_out[index] += selected_rows[index]; Instead, we have to use
    // AtomicAdd to avoid concurrent write error.
    paddle::platform::CudaAtomicAdd(tensor_out + index, selected_rows[index]);
  }
}
}  // namespace

template <typename T>
struct SelectedRowsAddTensor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output) {
    auto in1_height = input1.height();
    auto in2_dims = input2.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_EQ(
        in1_height, in2_dims[0],
        platform::errors::InvalidArgument(
            "The two inputs height must be equal."
            "But recieved first input height = [%d], first input height = [%d]",
            in1_height, in2_dims[0]));
    PADDLE_ENFORCE_EQ(
        in1_height, out_dims[0],
        platform::errors::InvalidArgument(
            "The input and output height must be equal."
            "But recieved input height = [%d], output height = [%d]",
            in1_height, out_dims[0]));

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel, input2.numel() / in1_height,
        platform::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But recieved first input width = [%d], second input width = [%d]",
            in1_row_numel, input2.numel() / in1_height));
    PADDLE_ENFORCE_EQ(
        in1_row_numel, output->numel() / in1_height,
        platform::errors::InvalidArgument(
            "The input and output width must be equal."
            "But recieved input width = [%d], output width = [%d]",
            in1_row_numel, output->numel() / in1_height));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = input2.data<T>();
    auto* out_data = output->data<T>();

    SetConstant<platform::CUDADeviceContext, T> functor;
    functor(context, output, static_cast<T>(0));

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid(in1_rows.size(), 1);
    SelectedRowsAddTensorKernel<
        T, block_size><<<grid, threads, 0, context.stream()>>>(
        in1_data, in1_rows.CUDAData(context.GetPlace()), out_data,
        in1_row_numel);

    auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    auto in2_eigen = framework::EigenVector<T>::Flatten(input2);
    out_eigen.device(*context.eigen_device()) = out_eigen + in2_eigen;
  }
};

template struct SelectedRowsAddTensor<platform::CUDADeviceContext, float>;
template struct SelectedRowsAddTensor<platform::CUDADeviceContext, double>;
template struct SelectedRowsAdd<platform::CUDADeviceContext, platform::float16>;
template struct SelectedRowsAddTensor<platform::CUDADeviceContext,
                                      platform::float16>;

template <typename T>
struct SelectedRowsAddTo<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& input1,
                  const int64_t input2_offset,
                  framework::SelectedRows* input2) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(
        in1_height, input2->height(),
        platform::errors::InvalidArgument("The two inputs height must be equal."
                                          "But recieved first input height = "
                                          "[%d], second input height = [%d]",
                                          in1_height, input2->height()));

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    if (in1_rows.size()) {
      in2_rows.Extend(in1_rows.begin(), in1_rows.end());
    }

    auto in1_place = input1.place();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(in1_place), true,
                      platform::errors::InvalidArgument(
                          "The running enviroment is not on the GPU place."));
    auto in2_place = input2->place();
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(in1_place), true,
                      platform::errors::InvalidArgument(
                          "The running enviroment is not on the GPU place."));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->data<T>();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, in2_place),
                 in2_data + input2_offset,
                 BOOST_GET_CONST(platform::CUDAPlace, in1_place), in1_data,
                 in1_value.numel() * sizeof(T), context.stream());
  }
};

template struct SelectedRowsAddTo<platform::CUDADeviceContext, float>;
template struct SelectedRowsAddTo<platform::CUDADeviceContext, double>;
template struct SelectedRowsAddTo<platform::CUDADeviceContext, int>;
template struct SelectedRowsAddTo<platform::CUDADeviceContext, int64_t>;
template struct SelectedRowsAddTo<platform::CUDADeviceContext,
                                  platform::float16>;

namespace {
template <typename T, int block_size>
__global__ void SelectedRowsAddToTensorKernel(const T* selected_rows,
                                              const int64_t* rows,
                                              T* tensor_out,
                                              int64_t row_numel) {
  const int ty = blockIdx.x;
  int tid = threadIdx.x;

  selected_rows += ty * row_numel;
  tensor_out += rows[ty] * row_numel;

  for (int index = tid; index < row_numel; index += block_size) {
    // Since index in rows of SelectedRows can be duplicate, we have to use
    // Atomic Operation to avoid concurrent write error.
    paddle::platform::CudaAtomicAdd(tensor_out + index, selected_rows[index]);
  }
}
}  // namespace

template <typename T>
struct SelectedRowsAddToTensor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(
        in1_height, in2_dims[0],
        platform::errors::InvalidArgument("The two inputs height must be equal."
                                          "But recieved first input height = "
                                          "[%d], second input height = [%d]",
                                          in1_height, in2_dims[0]));

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel, input2->numel() / in1_height,
        platform::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But recieved first input width = [%d], second input width = [%d]",
            in1_row_numel, input2->numel() / in1_height));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = input2->data<T>();
    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid(in1_rows.size(), 1);
    SelectedRowsAddToTensorKernel<
        T, block_size><<<grid, threads, 0, context.stream()>>>(
        in1_data, in1_rows.CUDAData(context.GetPlace()), in2_data,
        in1_row_numel);
  }
};

template struct SelectedRowsAddToTensor<platform::CUDADeviceContext, float>;
template struct SelectedRowsAddToTensor<platform::CUDADeviceContext, double>;
template struct SelectedRowsAddToTensor<platform::CUDADeviceContext, int>;
template struct SelectedRowsAddToTensor<platform::CUDADeviceContext, int64_t>;
template struct SelectedRowsAddToTensor<platform::CUDADeviceContext,
                                        platform::float16>;

namespace scatter {

template <typename T, int block_size>
__global__ void MergeAddKernel(const T* input, const int64_t* input_rows,
                               T* out, const int64_t* out_rows,
                               size_t out_rows_size, int64_t row_numel) {
  const int ty = blockIdx.x;
  int tid = threadIdx.x;
  __shared__ size_t out_idx;

  if (tid == 0) {
    for (size_t i = 0; i < out_rows_size; i++) {
      if (input_rows[ty] == out_rows[i]) {
        out_idx = i;
      }
    }
  }

  __syncthreads();

  input += ty * row_numel;
  out += out_idx * row_numel;
  for (int index = tid; index < row_numel; index += block_size) {
    paddle::platform::CudaAtomicAdd(out + index, input[index]);
  }
}

template <typename T>
struct MergeAdd<platform::CUDADeviceContext, T> {
  framework::SelectedRows operator()(const platform::CUDADeviceContext& context,
                                     const framework::SelectedRows& input,
                                     const bool sorted_result = false) {
    framework::SelectedRows out;
    (*this)(context, input, &out);
    return out;
  }

  void operator()(const platform::CUDADeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output,
                  const bool sorted_result = false) {
    framework::Vector<int64_t> input_rows(input.rows());
    if (input_rows.size() == 0) {
      return;
    }

    framework::SelectedRows& out = *output;
    std::set<int64_t> row_set(input_rows.begin(), input_rows.end());
    std::vector<int64_t> merge_rows_cpu(row_set.begin(), row_set.end());
    framework::Vector<int64_t> merge_rows(merge_rows_cpu);

    auto input_width = input.value().dims()[1];

    out.set_rows(merge_rows);
    out.set_height(input.height());
    out.mutable_value()->mutable_data<T>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CUDADeviceContext, T> constant_functor;
    constant_functor(context, out.mutable_value(), static_cast<T>(0));

    auto* out_data = out.mutable_value()->data<T>();
    auto* input_data = input.value().data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);
    dim3 grid1(input_rows.size(), 1);

    MergeAddKernel<T, 256><<<grid1, threads, 0, context.stream()>>>(
        input_data, input_rows.CUDAData(context.GetPlace()), out_data,
        out.mutable_rows()->CUDAMutableData(context.GetPlace()),
        out.rows().size(), input_width);
  }

  void operator()(const platform::CUDADeviceContext& context,
                  const std::vector<const framework::SelectedRows*>& inputs,
                  framework::SelectedRows* output,
                  const bool sorted_result = false) {
    if (inputs.size() == 0) {
      VLOG(3) << "no input! return";
      return;
    }
    const framework::SelectedRows* has_value_input = nullptr;
    for (auto* in : inputs) {
      if (in->rows().size() > 0) {
        has_value_input = in;
        break;
      }
    }
    if (has_value_input == nullptr) {
      VLOG(3) << "no input has value! just return" << std::endl;
      return;
    }
    auto input_width = has_value_input->value().dims()[1];
    auto input_height = has_value_input->height();
    framework::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      PADDLE_ENFORCE_EQ(input_width, input->value().dims()[1],
                        platform::errors::InvalidArgument(
                            "All input should have same "
                            "dimension except for the first one."));
      PADDLE_ENFORCE_EQ(input_height, input->height(),
                        platform::errors::InvalidArgument(
                            "All input should have same height."));
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }
    std::vector<int64_t> merge_rows_cpu(merged_row_set.begin(),
                                        merged_row_set.end());
    framework::Vector<int64_t> merge_rows(merge_rows_cpu);

    out.set_rows(merge_rows);
    out.set_height(input_height);
    out.mutable_value()->mutable_data<T>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CUDADeviceContext, T> constant_functor;
    constant_functor(context, out.mutable_value(), static_cast<T>(0));

    auto* out_data = out.mutable_value()->data<T>();

    const int block_size = 256;
    dim3 threads(block_size, 1);

    for (auto* input : inputs) {
      if (input->rows().size() == 0) {
        continue;
      }
      auto* input_data = input->value().data<T>();
      auto& input_rows = input->rows();
      dim3 grid1(input_rows.size(), 1);

      MergeAddKernel<T, 256><<<grid1, threads, 0, context.stream()>>>(
          input_data, input_rows.CUDAData(context.GetPlace()), out_data,
          out.mutable_rows()->CUDAMutableData(context.GetPlace()),
          out.rows().size(), input_width);
    }
  }
};

template struct MergeAdd<platform::CUDADeviceContext, float>;
template struct MergeAdd<platform::CUDADeviceContext, double>;
template struct MergeAdd<platform::CUDADeviceContext, int>;
template struct MergeAdd<platform::CUDADeviceContext, int64_t>;
template struct MergeAdd<platform::CUDADeviceContext, platform::float16>;
template struct MergeAdd<platform::CUDADeviceContext, platform::complex64>;
template struct MergeAdd<platform::CUDADeviceContext, platform::complex128>;

template <typename T, int block_size>
__global__ void UpdateToTensorKernel(const T* selected_rows,
                                     const int64_t* rows, const ScatterOps& op,
                                     T* tensor_out, int64_t row_numel) {
  const int ty = blockIdx.x;
  int tid = threadIdx.x;

  selected_rows += ty * row_numel;
  tensor_out += rows[ty] * row_numel;
  // FIXME(typhoonzero): use macro fix the below messy code.
  switch (op) {
    case ScatterOps::ASSIGN:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] = selected_rows[index];
      }
      break;
    case ScatterOps::ADD:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] += selected_rows[index];
      }
      break;
    case ScatterOps::SUB:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] -= selected_rows[index];
      }
      break;
    case ScatterOps::SUBBY:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] = selected_rows[index] - tensor_out[index];
      }
      break;
    case ScatterOps::MUL:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] *= selected_rows[index];
      }
      break;
    case ScatterOps::DIV:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] /= selected_rows[index];
      }
      break;
    case ScatterOps::DIVBY:
      for (int index = tid; index < row_numel; index += block_size) {
        tensor_out[index] = selected_rows[index] / tensor_out[index];
      }
      break;
  }
}

template <typename T>
struct UpdateToTensor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const ScatterOps& op, const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    // NOTE: Use SelectedRowsAddToTensor for better performance
    //       no additional MergeAdd called.
    MergeAdd<platform::CUDADeviceContext, T> merge_func;
    auto merged_in1 = merge_func(context, input1);

    auto in1_height = merged_in1.height();
    auto in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(
        in1_height, in2_dims[0],
        platform::errors::InvalidArgument("The two inputs height must be equal."
                                          "But recieved first input height = "
                                          "[%d], second input height = [%d]",
                                          in1_height, in2_dims[0]));

    auto& in1_value = merged_in1.value();
    auto& in1_rows = merged_in1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(
        in1_row_numel, input2->numel() / in1_height,
        platform::errors::InvalidArgument(
            "The two inputs width must be equal."
            "But recieved first input width = [%d], second input width = [%d]",
            in1_row_numel, input2->numel() / in1_height));

    auto* in1_data = in1_value.template data<T>();
    auto* in2_data = input2->data<T>();

    dim3 threads(platform::PADDLE_CUDA_NUM_THREADS, 1);
    dim3 grid(in1_rows.size(), 1);
    UpdateToTensorKernel<T, platform::PADDLE_CUDA_NUM_THREADS><<<
        grid, threads, 0, context.stream()>>>(in1_data, in1_rows.cuda_data(),
                                              op, in2_data, in1_row_numel);
  }
};
}  // namespace scatter
}  // namespace math
}  // namespace operators
}  // namespace paddle
