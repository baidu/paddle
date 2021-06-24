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

#pragma once
#include <memory.h>
#include <cstring>
#include <vector>

#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {

using framework::Tensor;

/**
 * A thin wrapper for gathering on cpu tensor
 * Return a new tensor from source tensor, gathered according to index
 * input[src]: type-T source Tensor
 * input[index]: type-IndexT index Tensor (1-D)
 * return: output tensor
 */
template <typename T, typename IndexT = int>
void CPUGather(const platform::DeviceContext& ctx, const Tensor& src,
               const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.GetPlace()), true,
      platform::errors::PreconditionNotMet("It should be running on the CPU."));
  // check index of shape 1-D
  if (index.dims().size() == 2) {
    PADDLE_ENFORCE_EQ(
        index.dims()[1], 1,
        platform::errors::InvalidArgument(
            "index.dims()[1] should be 1 when index.dims().size() = 2"
            "in gather_op, but received value is [%d].",
            index.dims()[1]));
  } else {
    PADDLE_ENFORCE_EQ(index.dims().size(), 1,
                      platform::errors::InvalidArgument(
                          "index.dims().size() should be 1 or 2 in gather_op,"
                          "but received shape's size is [%d].",
                          index.dims().size()));
  }
  int64_t index_size = index.dims()[0];

  auto src_dims = src.dims();

  const T* p_src = src.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // slice size
  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) slice_size *= src_dims[i];

  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < index_size; ++i) {
    IndexT index_ = p_index[i];
    memcpy(p_output + i * slice_size, p_src + index_ * slice_size, slice_bytes);
  }
}

template <typename T, typename IndexT = int>
void CPUGatherNd(const platform::DeviceContext& ctx, const Tensor& input,
                 const Tensor& index, Tensor* output) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.GetPlace()), true,
      platform::errors::PreconditionNotMet("It should be running on the CPU."));

  auto index_dims = index.dims();
  auto index_dims_size = index_dims.size();
  auto input_dims = input.dims();
  auto input_dims_size = input_dims.size();

  const T* p_input = input.data<T>();
  const IndexT* p_index = index.data<IndexT>();
  T* p_output = output->data<T>();

  // final dim
  int64_t end_size = index_dims[index_dims_size - 1];
  // remain dim
  auto remain_ddim = framework::slice_ddim(index_dims, 0, index_dims_size - 1);
  int64_t remain_numel = framework::product(remain_ddim);
  // slice size
  int64_t slice_size = 1;
  for (int64_t i = end_size; i < input_dims_size; ++i) {
    slice_size *= input_dims[i];
  }
  const size_t slice_bytes = slice_size * sizeof(T);

  for (int64_t i = 0; i < remain_numel; ++i) {
    int64_t index_ = 0;
    int64_t temp = 1;
    for (int64_t j = end_size - 1; j >= 0; --j) {
      IndexT index_value = p_index[i * end_size + j];
      PADDLE_ENFORCE_LT(
          index_value, input_dims[j],
          platform::errors::InvalidArgument(
              "Input(index[-1)] has wrong value, it is [%d]", index_value));
      PADDLE_ENFORCE_GE(
          index_value, 0UL,
          platform::errors::InvalidArgument(
              "The value of Input(index) must be no less than 0"));

      index_ += (index_value * temp);
      temp *= input_dims[j];
    }
    memcpy(p_output + i * slice_size, p_input + index_ * slice_size,
           slice_bytes);
  }
}

template <typename T, typename U>
void GatherV2Function(const Tensor* input, const Tensor* index, int axis,
                      Tensor* out, const paddle::platform::Place& place) {
  auto* index_data = index->data<U>();
  int index_size = index->numel();
  int input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  int axis_index = axis;

  int input_index_dim_size = input_dim[axis_index];
  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_LT(index_data[i], input_index_dim_size,
                      platform::errors::InvalidArgument(
                          "The element of Index must be less than the size of "
                          "input dim size of axis which is %d, but received "
                          "index element which is %d in the %d index.",
                          input_index_dim_size, index_data[i], i));
  }

  int inner_dim_size = 1;
  int outer_dim_size = 1;
  std::vector<int> out_dim_vec;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  out_dim_vec.push_back(index_size);
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  auto out_dim = framework::make_ddim(out_dim_vec);

  out->Resize(out_dim);
  auto* out_data = out->mutable_data<T>(place);

  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    (i * input_size / inner_dim_size);
        out_data[out_index] = input_data[index];
        out_index++;
      }
    }
  }
}

template <typename T, typename U>
void GatherV2GradFunction(const Tensor* input, const Tensor* index,
                          const int axis, Tensor* out,
                          const paddle::platform::Place& place) {
  auto* index_data = index->data<U>();

  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  int axis_index = axis;
  int input_index_dim_size = input_dim[axis_index];

  int inner_dim_size = 1;
  int outer_dim_size = 1;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  auto* out_data = out->mutable_data<T>(place);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  auto out_dim = out->dims();
  int out_index_dim_size = out_dim[axis_index];
  operators::math::set_constant(*dev_ctx, out, 0.0);

  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < input_index_dim_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index = k + index_data[j] * outer_dim_size +
                    i * outer_dim_size * out_index_dim_size;
        out_data[index] += input_data[j * outer_dim_size + k];
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle
