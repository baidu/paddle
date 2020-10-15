/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MergeIdsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto place = ctx.GetPlace();
    if (!platform::is_cpu_place(place)) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "MergeIds do not support GPU kernel"));
    }

    const auto ids = ctx.MultiInput<framework::LoDTensor>("Ids");
    const auto row_ids = ctx.MultiInput<framework::LoDTensor>("Rows");
    const auto x_tensors = ctx.MultiInput<framework::LoDTensor>("X");
    auto outs = ctx.MultiOutput<framework::LoDTensor>("Out");

    PADDLE_ENFORCE_EQ(row_ids.size(), x_tensors.size(),
                      platform::errors::InvalidArgument(
                          "the number of Rows and X should be the same"));
    PADDLE_ENFORCE_EQ(ids.size(), outs.size(),
                      platform::errors::InvalidArgument(
                          "the number of Ids and Out should be the same"));

    int64_t row_ids_size = 0;
    int64_t row_size = 0;
    int64_t embedding_size = 0;

    for (size_t i = 0; i < x_tensors.size(); ++i) {
      const auto *x_tensor = x_tensors[i];
      const auto *row_id = row_ids[i];

      if (embedding_size == 0) {
        embedding_size = x_tensor->dims()[1];
      }
      PADDLE_ENFORCE_EQ(embedding_size, x_tensor->dims()[1],
                        platform::errors::InvalidArgument(
                            "embedding size of all input should be the same"));
      row_size += x_tensor->dims()[0];
      row_ids_size += row_id->dims()[0];
    }

    PADDLE_ENFORCE_EQ(
        row_size, row_ids_size,
        platform::errors::InvalidArgument(
            "the merged X dim[0] and merged Rows dim[0] should be the same"));

    std::unordered_map<int64_t, std::tuple<int64_t, int64_t>>
        selected_rows_idx_map;
    for (size_t i = 0; i < x_tensors.size(); ++i) {
      const auto *row_id = row_ids[i];

      for (auto j = 0; j < row_id->numel(); ++j) {
        int64_t key = row_id->data<int64_t>()[j];
        std::tuple<int64_t, int64_t> val = std::make_tuple(i, j);
        selected_rows_idx_map.insert(std::make_pair(key, val));
      }
    }
    PADDLE_ENFORCE_EQ(row_ids_size, selected_rows_idx_map.size(),
                      platform::errors::InvalidArgument(
                          "the rows and tensor map size should be the same"));

    for (size_t i = 0; i < outs.size(); ++i) {
      auto *out_ids = ids[i];
      auto *out = outs[i];

      out->set_lod(out_ids->lod());

      auto nums = out_ids->dims()[0];
      auto *out_data = out->mutable_data<T>(
          framework::make_ddim({nums, embedding_size}), place);
      for (auto j = 0; j < nums; ++j) {
        auto id = out_ids->data<int64_t>()[j];
        auto row_tuple = selected_rows_idx_map.at(id);
        auto row_idx = std::get<1>(row_tuple);
        const auto *x_tensor = x_tensors[std::get<0>(row_tuple)];

        memcpy(out_data + embedding_size * j,
               x_tensor->data<T>() + row_idx * embedding_size,
               sizeof(T) * embedding_size);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
