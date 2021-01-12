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
class PinnedMemcpyFunctor {
 public:
  PinnedMemcpyFunctor(framework::Variable *out,
                      const platform::DeviceContext &dev_ctx,
                      const bool to_pinned)
      : out_(out), dev_ctx_(dev_ctx), to_pinned_(to_pinned) {}

  void operator()(const framework::LoDTensor &lod_tensor) const {
    auto &out_tensor = *out_->GetMutable<framework::LoDTensor>();

    if (to_pinned_) {
      framework::TensorCopy(lod_tensor, platform::CUDAPinnedPlace(), dev_ctx_,
                            &out_tensor);
    } else {
      framework::TensorCopy(lod_tensor, dev_ctx_.GetPlace(), dev_ctx_,
                            &out_tensor);
    }
    out_tensor.set_lod(lod_tensor.lod());
  }

  void operator()(const framework::LoDTensorArray &array) const {
    // (JZ-LIANT) to support LoDTensorArray
    PADDLE_THROW(platform::errors::Unimplemented(
        "Pinned Memcpy for LoDTensorArray is NOT support yet."));
  }

  void operator()(const framework::SelectedRows &rows) const {
    // (JZ-LIANT) to support SelectedRows
    PADDLE_THROW(platform::errors::Unimplemented(
        "Pinned Memcpy for SelectedRows is NOT support yet."));
  }

  template <typename T>
  void operator()(const T &v) const {
    PADDLE_ENFORCE_EQ(true, false,
                      platform::errors::PermissionDenied(
                          "Not support type for Pinned Memcpy  op with type %s",
                          typeid(T).name()));
  }

 private:
  framework::Variable *out_;
  const platform::DeviceContext &dev_ctx_;
  const bool to_pinned_;
};

}  // namespace operators
}  // namespace paddle
