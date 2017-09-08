/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <vector>
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename Place, typename T>
class ConcatKernel : public framework::OpKernel {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto ins = ctx.MultiInput<framework::Tensor>("X");
    auto* out = ctx.Output<framework::Tensor>("Out");
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    size_t N = ins.size();
    int output_axis_dim = 0;
    int before = 1;
    int after = 1;
    for (size_t i = 0; i < N; i++) {
      output_axis_dim += ins[i]->dims()[axis];
    }
    auto& input_zero = ins[0];
    for (int i = 0; i < input_zero->dims().size(); i++) {
      if (i == axis) {
        continue;
      }
      if (i < axis) {
        before *= input_zero->dims()[i];
      } else {
        after *= input_zero->dims()[i];
      }
    }
    int output_offset = 0;
    for (size_t i = 0; i < N; i++) {
      auto& in = ins[i];
      auto axis_dim = in->dims()[axis];
      for (int j = 0; j < before; j++) {
        int len = axis_dim * after * sizeof(T);
        const T* src = in->data<T>() + axis_dim * after * j;
        T* out_data = out->mutable_data<T>(platform::CPUPlace());
        T* dest = out_data + output_offset + output_axis_dim * after * j;
        memcpy(dest, src, len);
      }
      output_offset += axis_dim * after;
    }
  }
};

}  // namespace operators
}  // namespace paddle
