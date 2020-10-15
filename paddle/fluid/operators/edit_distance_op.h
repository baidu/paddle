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
#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/mixed_vector.h"
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

template <typename Place, typename T>
class EditDistanceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    auto* out_t = ctx.Output<framework::Tensor>("Out");

    auto* x1_t = ctx.Input<framework::LoDTensor>("Hyps");
    auto* x2_t = ctx.Input<framework::LoDTensor>("Refs");
    auto* sequence_num = ctx.Output<framework::Tensor>("SequenceNum");
    int64_t* seq_num_data = sequence_num->mutable_data<int64_t>(ctx.GetPlace());
    auto batch_size = x1_t->dims()[0];

    auto normalized = ctx.Attr<bool>("normalized");

    framework::Vector<size_t> hyp_lod(batch_size + 1);
    framework::Vector<size_t> ref_lod(batch_size + 1);

    bool use_length = ctx.HasInput("HypsLength");

    if (use_length) {
      // build lod when using padding
      auto hyp_length_ptr =
          ctx.Input<framework::Tensor>("HypsLength")->data<int64_t>();
      auto ref_length_ptr =
          ctx.Input<framework::Tensor>("RefsLength")->data<int64_t>();

      for (auto i = 0; i < batch_size; i++) {
        hyp_lod[i + 1] = hyp_lod[i] + hyp_length_ptr[i];
        ref_lod[i + 1] = ref_lod[i] + ref_length_ptr[i];
      }

    } else {
      hyp_lod = x1_t->lod()[0];
      ref_lod = x2_t->lod()[0];
    }

    if (normalized) {
      for (size_t i = 1; i < ref_lod.size(); ++i) {
        PADDLE_ENFORCE_GT(ref_lod[i], ref_lod[i - 1],
                          platform::errors::InvalidArgument(
                              "Reference string %d is empty.", i));
      }
    }
    auto num_strs = hyp_lod.size() - 1;
    *seq_num_data = static_cast<int64_t>(num_strs);

    out_t->Resize({static_cast<int64_t>(num_strs), 1});
    out_t->mutable_data<float>(ctx.GetPlace());
    auto out = out_t->data<T>();

    T distance = 0.0;
    for (size_t num = 0; num < num_strs; ++num) {
      auto m = static_cast<int64_t>(hyp_lod[num + 1] - hyp_lod[num]);
      auto n = static_cast<int64_t>(ref_lod[num + 1] - ref_lod[num]);

      if (m == 0) {
        distance = n;
      } else if (n == 0) {
        distance = m;
      } else {
        framework::Tensor dist_t;
        dist_t.Resize({m + 1, n + 1});
        dist_t.mutable_data<T>(ctx.GetPlace());
        auto dist = dist_t.data<T>();
        auto hyp_offset = use_length ? num * x1_t->dims()[1] : hyp_lod[num];
        auto ref_offset = use_length ? num * x2_t->dims()[1] : ref_lod[num];
        auto x1 = x1_t->data<int64_t>() + hyp_offset;
        auto x2 = x2_t->data<int64_t>() + ref_offset;
        for (int64_t i = 0; i < m + 1; ++i) {
          dist[i * (n + 1)] = i;
        }
        for (int64_t j = 0; j < n + 1; ++j) {
          dist[j] = j;
        }
        for (int64_t i = 1; i < m + 1; ++i) {
          for (int64_t j = 1; j < n + 1; ++j) {
            int cost = x1[i - 1] == x2[j - 1] ? 0 : 1;
            int dels = dist[(i - 1) * (n + 1) + j] + 1;
            int ins = dist[i * (n + 1) + (j - 1)] + 1;
            int subs = dist[(i - 1) * (n + 1) + (j - 1)] + cost;
            dist[i * (n + 1) + j] = std::min(dels, std::min(ins, subs));
          }
        }
        distance = dist[m * (n + 1) + n];
      }

      if (normalized) {
        PADDLE_ENFORCE_GT(n, 0UL,
                          platform::errors::InvalidArgument(
                              "The reference string (#%d) cannot be empty "
                              "when Attr(normalized) is enabled.",
                              n));
        distance = distance / n;
      }
      out[num] = distance;
    }
  }
};

}  // namespace operators
}  // namespace paddle
