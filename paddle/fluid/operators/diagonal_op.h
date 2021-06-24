// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
template <typename T>

std::vector<T> ComputeDimStride(const std::vector<T> dim) {
  size_t dim_size = dim.size();
  std::vector<T> dim_strides;
  dim_strides.resize(dim_size);
  for (size_t i = 0; i < dim_size - 1; i++) {
    size_t temp_stride = 1;
    for (size_t j = i + 1; j < dim_size; j++) {
      temp_stride = temp_stride * dim[j];
    }
    dim_strides[i] = temp_stride;
  }
  dim_strides[dim_size - 1] = 1;
  return dim_strides;
}
template <typename T>
class DiagonalKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    const T* input_data = input->data<T>();
    auto input_dim = vectorize(input->dims());
    auto input_dim_size = input_dim.size();

    auto* output = context.Output<framework::Tensor>("Out");
    T* output_data = output->mutable_data<T>(context.GetPlace());
    auto output_dim = vectorize(output->dims());

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? input_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? input_dim_size + axis2 : axis2;

    std::vector<int64_t> input_stride = ComputeDimStride(input_dim);
    std::vector<int64_t> output_stride = ComputeDimStride(output_dim);

    int64_t numel = input->numel();

    for (int64_t idx = 0; idx < numel; idx++) {
      std::vector<int64_t> idx_dim(input_dim_size);
      int64_t temp = 0;
      for (size_t i = 0; i < input_dim_size; i++) {
        idx_dim[i] = (idx - temp) / input_stride[i];
        temp = temp + idx_dim[i] * input_stride[i];
      }

      int64_t axis1_dim = idx_dim[axis1_];
      int64_t axis2_dim = idx_dim[axis2_];

      idx_dim.erase(idx_dim.begin() + std::max(axis1_, axis2_));
      idx_dim.erase(idx_dim.begin() + std::min(axis1_, axis2_));

      bool flag = false;
      if (offset_ == 0 && axis1_dim == axis2_dim) {
        idx_dim.push_back(axis1_dim);
        flag = true;
      } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
        idx_dim.push_back(axis1_dim);
        flag = true;
      } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
        idx_dim.push_back(axis2_dim);
        flag = true;
      }
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < idx_dim.size(); i++) {
          idx_output = idx_output + idx_dim[i] * output_stride[i];
        }
        output_data[idx_output] = input_data[idx];
      }
    }
  }
};

template <typename T>
class DiagonalGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* dout =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    const T* dout_data = dout->data<T>();
    auto dout_dim = vectorize(dout->dims());

    auto* dx =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());
    auto dx_dim = vectorize(dx->dims());
    auto dx_dim_size = dx_dim.size();

    const int64_t offset_ = context.Attr<int>("offset");
    const int64_t axis1 = context.Attr<int>("axis1");
    int64_t axis1_ = axis1 < 0 ? dx_dim_size + axis1 : axis1;
    const int64_t axis2 = context.Attr<int>("axis2");
    int64_t axis2_ = axis2 < 0 ? dx_dim_size + axis2 : axis2;

    std::vector<int64_t> dout_stride = ComputeDimStride(dout_dim);
    std::vector<int64_t> dx_stride = ComputeDimStride(dx_dim);

    int64_t numel = dx->numel();

    for (int64_t idx = 0; idx < numel; idx++) {
      std::vector<int64_t> idx_dim(dx_dim_size);
      int64_t temp = 0;
      for (size_t i = 0; i < dx_dim_size; i++) {
        idx_dim[i] = (idx - temp) / dx_stride[i];
        temp = temp + idx_dim[i] * dx_stride[i];
      }

      int64_t axis1_dim = idx_dim[axis1_];
      int64_t axis2_dim = idx_dim[axis2_];

      idx_dim.erase(idx_dim.begin() + std::max(axis1_, axis2_));
      idx_dim.erase(idx_dim.begin() + std::min(axis1_, axis2_));

      bool flag = false;
      if (offset_ == 0 && axis1_dim == axis2_dim) {
        idx_dim.push_back(axis1_dim);
        flag = true;
      } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
        idx_dim.push_back(axis1_dim);
        flag = true;
      } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
        idx_dim.push_back(axis2_dim);
        flag = true;
      }
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < idx_dim.size(); i++) {
          idx_output = idx_output + idx_dim[i] * dout_stride[i];
        }
        dx_data[idx] = dout_data[idx_output];
      } else {
        dx_data[idx] = static_cast<T>(0);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
