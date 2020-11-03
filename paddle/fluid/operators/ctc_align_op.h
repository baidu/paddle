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

#include <string.h>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class CTCAlignKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* output = ctx.Output<LoDTensor>("Output");
    size_t blank = static_cast<size_t>(ctx.Attr<int>("blank"));
    bool merge_repeated = ctx.Attr<bool>("merge_repeated");
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    auto input_dims = input->dims();
    const T* input_data = input->data<T>();

    // support tensor input, no lod information
    if (input->lod().empty()) {
      size_t padding_value =
          static_cast<size_t>(ctx.Attr<int>("padding_value"));
      auto* input_length = ctx.Input<LoDTensor>("InputLength");
      const T* input_length_data = input_length->data<T>();

      auto* output_length = ctx.Output<LoDTensor>("OutputLength");
      T* output_length_data = output_length->mutable_data<T>(ctx.GetPlace());

      for (size_t batch_id = 0; batch_id < (unsigned)input_dims[0];
           batch_id++) {
        T prev_token = -1;
        size_t output_idx = 0;
        for (size_t i = 0; i < (unsigned)input_length_data[batch_id]; i++) {
          size_t input_ind = batch_id * input_dims[1] + i;
          if ((unsigned)input_data[input_ind] != blank &&
              !(merge_repeated && input_data[input_ind] == prev_token)) {
            output_data[batch_id * input_dims[1] + output_idx] =
                input_data[input_ind];
            ++output_idx;
          }
          prev_token = input_data[input_ind];
        }
        output_length_data[batch_id] = output_idx;
        for (size_t j = output_idx; j < (unsigned)input_dims[1]; j++)
          output_data[batch_id * input_dims[1] + j] = padding_value;
      }
    } else {
      const size_t level = 0;
      auto input_lod = framework::ToAbsOffset(input->lod());

      // check input dims and lod
      PADDLE_ENFORCE_EQ(
          input_dims[0], static_cast<int64_t>(input_lod[level].back()),
          platform::errors::InvalidArgument(
              "The first dimension %d of CTCAlign operator Input(Input) should "
              "be equal to "
              "the sum of all sequences' lengths %d.",
              input_dims[0], static_cast<int64_t>(input_lod[level].back())));

      const size_t num_sequences = input_lod[level].size() - 1;

      // merge repeated tokens and delete blank
      size_t output_idx = 0;
      std::vector<size_t> output_lod0(1, 0);
      for (size_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
        T prev_token = -1;
        for (size_t i = input_lod[level][seq_idx];
             i < input_lod[level][seq_idx + 1]; ++i) {
          if ((unsigned)input_data[i] != blank &&
              !(merge_repeated && input_data[i] == prev_token)) {
            output_data[output_idx] = input_data[i];
            ++output_idx;
          }
          prev_token = input_data[i];
        }
        output_lod0.push_back(output_idx);
      }

      // set output lod
      framework::LoD output_lod;
      output_lod.push_back(output_lod0);
      output->set_lod(output_lod);
      // resize output dims
      output->Resize({static_cast<int64_t>(output_lod0.back()), 1});
      // for empty sequence
      if (output_lod0.back() == 0) {
        output->Resize({1, 1});
        output_data = output->mutable_data<T>(ctx.GetPlace());
        output_data[0] = -1;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
