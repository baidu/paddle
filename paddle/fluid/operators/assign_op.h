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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
class AssignFunctor {
 public:
  AssignFunctor(framework::Variable *out,
                const platform::DeviceContext &dev_ctx)
      : out_(out), dev_ctx_(dev_ctx) {}

  void operator()(const framework::LoDTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<framework::LoDTensor>();
    copy_tensor(lod_tensor, &out_tensor);
  }

  void operator()(const framework::LoDTensorArray &array) const {
    auto &out_array = *out_->GetMutable<framework::LoDTensorArray>();
    out_array.resize(array.size());
    for (size_t i = 0; i < array.size(); ++i) {
      copy_tensor(array[i], &out_array[i]);
    }
  }

  void operator()(const framework::SelectedRows &rows) const {
    framework::SelectedRows &out_rows =
        *out_->GetMutable<framework::SelectedRows>();
    out_rows.set_rows(rows.rows());
    out_rows.set_height(rows.height());
    auto &t = rows.value();
    auto *m = out_rows.mutable_value();
    framework::TensorCopy(t, t.place(), m);
  }

  template <typename T>
  void operator()(const T &v) const {
    PADDLE_ENFORCE_EQ(
        true, false,
        platform::errors::PermissionDenied(
            "Not support type for assign op with type %s", typeid(T).name()));
  }

 private:
  void copy_tensor(const framework::LoDTensor &lod_tensor,
                   framework::LoDTensor *out) const {
    if (lod_tensor.numel() == 0) return;
    auto &out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
};

}  // namespace operators
}  // namespace paddle
