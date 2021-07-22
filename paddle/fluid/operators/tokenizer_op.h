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

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename T>
class TokenizerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<std::vector<std::string>>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");

    output->Resize(framework::make_ddim({static_cast<int64_t>(input->size())}));
    auto* out_data = output->mutable_data<T>(ctx.GetPlace());
    // only support cpu now
    VLOG(0) << "input size: " << input->size();
    for (size_t i = 0; i < input->size(); ++i) {
      VLOG(0) << "input[" << i << "] = " << input->at(i)
              << ", size: " << input->at(i).size();
      out_data[i] = input->at(i).size();
    }
  }
};

}  // namespace operators
}  // namespace paddle
