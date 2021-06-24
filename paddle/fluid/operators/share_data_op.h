/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class ShareDataKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar("X");
    auto *out_var = ctx.OutputVar("Out");
    if (in_var->IsType<framework::LoDTensor>()) {
      const auto &origin_tensor = in_var->Get<framework::LoDTensor>();
      auto *detach_tensor = out_var->GetMutable<framework::LoDTensor>();
      detach_tensor->ShareDataWith(origin_tensor);
    } else {
      const auto &origin_selected_rows = in_var->Get<framework::SelectedRows>();
      auto *detach_selected_rows =
          out_var->GetMutable<framework::SelectedRows>();
      detach_selected_rows->mutable_value()->ShareDataWith(
          origin_selected_rows.value());
    }
  }
};
}  // namespace operators
}  // namespace paddle
