// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace platform {
using framework::Tensor;

template <typename T>
cudnnDataType_t ToCudnnDataType(const T& t) {
  auto type = framework::ToDataType(t);
  return ToCudnnDataType(type);
}

template <>
cudnnDataType_t ToCudnnDataType(const framework::proto::VarType::Type& t) {
  cudnnDataType_t type = CUDNN_DATA_FLOAT;
  switch (t) {
    case framework::proto::VarType::FP16:
      type = CUDNN_DATA_HALF;
      break;
    case framework::proto::VarType::FP32:
      type = CUDNN_DATA_FLOAT;
      break;
    case framework::proto::VarType::FP64:
      type = CUDNN_DATA_DOUBLE;
      break;
    default:
      break;
  }
  return type;
}

class ActivationDescriptor {
 public:
  using T = cudnnActivationStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE(dynload::cudnnDestroyActivationDescriptor(t));
        t = nullptr;
      }
    }
  };
  ActivationDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE(dynload::cudnnCreateActivationDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  template <typename T>
  void set(cudnnActivationMode_t mode, const T& coef) {
    CUDNN_ENFORCE(dynload::cudnnSetActivationDescriptor(
        desc_.get(), mode, CUDNN_NOT_PROPAGATE_NAN, static_cast<double>(coef)));
  }

  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

class TensorDescriptor {
 public:
  using T = cudnnTensorStruct;
  struct Deleter {
    void operator()(T* t) {
      if (t != nullptr) {
        PADDLE_ENFORCE(dynload::cudnnDestroyTensorDescriptor(t));
        t = nullptr;
      }
    }
  };
  TensorDescriptor() {
    T* raw_ptr;
    PADDLE_ENFORCE(dynload::cudnnCreateTensorDescriptor(&raw_ptr));
    desc_.reset(raw_ptr);
  }
  T* desc() { return desc_.get(); }
  T* desc() const { return desc_.get(); }
  void set(const Tensor& tensor, const int groups = 1) {
    auto dims = framework::vectorize2int(tensor.dims());
    std::vector<int> strides(dims.size());
    strides[dims.size() - 1] = 1;
    for (int i = dims.size() - 2; i >= 0; i--) {
      strides[i] = dims[i + 1] * strides[i + 1];
    }
    std::vector<int> dims_with_group(dims.begin(), dims.end());
    if (groups > 1) {
      dims_with_group[1] = dims_with_group[1] / groups;
    }
    PADDLE_ENFORCE(dynload::cudnnSetTensorNdDescriptor(
        desc_.get(), ToCudnnDataType(tensor.type()), dims_with_group.size(),
        dims_with_group.data(), strides.data()));
  }

 private:
  std::unique_ptr<T, Deleter> desc_;
};

}  // namespace platform
}  // namespace paddle
