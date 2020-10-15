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
#include <cmath>
#include <memory>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace math {

template <typename DeviceContext, typename T>
struct TransposeNormal {
  // for dims >= 7 situation
  void operator()(const DeviceContext& context, const framework::Tensor& in,
                  framework::Tensor* out, const std::vector<int>& axis);
};

template <typename DeviceContext, typename T, int Rank>
struct Transpose {
  void operator()(const DeviceContext& context, const framework::Tensor& in,
                  framework::Tensor* out, const std::vector<int>& axis);
};

template <typename DeviceContext, typename T>
struct SetConstant {
  void operator()(const DeviceContext& context, framework::Tensor* tensor,
                  T num);
};

template <typename Place>
void set_constant_with_place(const platform::DeviceContext& context,
                             framework::Tensor* tensor, float value);

void set_constant(const platform::DeviceContext& context,
                  framework::Tensor* tensor, float value);

template <typename DeviceContext, typename T>
struct RowwiseAdd {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& vec, framework::Tensor* output);
};

template <typename DeviceContext, typename T>
struct ElementwiseAddTo {
  // dst = dst + src
  void operator()(DeviceContext* ctx, const framework::Tensor& src,
                  framework::Tensor* dst);
};

template <typename DeviceContext, typename T>
struct ColwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseSum {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

template <typename DeviceContext, typename T>
struct RowwiseMean {
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  framework::Tensor* vec);
};

#ifdef PADDLE_WITH_XPU
template <typename U>
struct TensorSetConstantXPU {
  TensorSetConstantXPU(framework::Tensor* tensor, U value)
      : tensor_(tensor), value_(value) {}
  template <typename T>
  void apply() const {
    int dev_id = -1;
    xpu_current_device(&dev_id);
    if (dev_id >= 64) {
      // if dev_id >= 64, the device is a simulator device, -64 to get real
      // dev_id
      dev_id -= 64;
    }
    auto xpu = platform::XPUPlace(dev_id);
    auto* begin = tensor_->mutable_data<T>(xpu);
    int numel = tensor_->numel();
    std::unique_ptr<T[]> data_cpu(new T[numel]);
    std::fill(data_cpu.get(), data_cpu.get() + numel, static_cast<T>(value_));
    memory::Copy(xpu, begin, platform::CPUPlace(),
                 static_cast<void*>(data_cpu.get()), numel * sizeof(T));
  }
  framework::Tensor* tensor_;
  U value_;
};
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
