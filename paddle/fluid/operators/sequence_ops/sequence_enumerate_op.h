//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {
using LoDTensor = framework::LoDTensor;

template <typename DeviceContext, typename T>
class SequenceEnumerateKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<LoDTensor>("Out");
    int win_size = context.Attr<int>("win_size");
    auto pad_value = static_cast<T>(context.Attr<int>("pad_value"));

    PADDLE_ENFORCE_EQ(
        in->lod().empty(), false,
        platform::errors::InvalidArgument(
            "Input(X) Tensor of SequenceEnumerateOp does not contain "
            "LoD information."));

    auto in_dims = in->dims();
    auto lod0 = in->lod()[0];
    PADDLE_ENFORCE_EQ(
        static_cast<uint64_t>(in_dims[0]), lod0.back(),
        platform::errors::InvalidArgument(
            "The actual input data's size mismatched with LoD information."
            "Received input data size is %d (actual) vs %d (loD information).",
            static_cast<uint64_t>(in_dims[0]), lod0.back()));
    PADDLE_ENFORCE_EQ(
        in_dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "Input(X) of SequenceEnumerate operator's rank should be 2."
            "Received %d instead.",
            in_dims.size()));
    PADDLE_ENFORCE_EQ(in_dims[1], 1,
                      platform::errors::InvalidArgument(
                          "Input(X) of SequenceEnumerate operator's 2nd "
                          "dimension should be 1. Received %d instead.",
                          in_dims[1]));

    // Generate enumerate sequence set
    auto in_data = in->data<T>();
    out->Resize({in_dims[0], win_size});
    out->set_lod(in->lod());
    auto out_data = out->mutable_data<T>(context.GetPlace());
    for (size_t i = 0; i < lod0.size() - 1; ++i) {
      if (lod0[i] == lod0[i + 1]) continue;
      int start = lod0[i];
      int end = lod0[i + 1];

      int copy_size = win_size < end - start + 1 ? win_size : end - start + 1;
      int mid = end + 1 - copy_size;
      int pad_num = win_size - copy_size;
      copy_size *= sizeof(T);
      for (int idx = start; idx < mid; ++idx) {
        std::memcpy(out_data, in_data + idx, copy_size);
        out_data += win_size;
      }
      for (int idx = mid; idx < end; ++idx) {
        copy_size -= sizeof(T);
        pad_num++;
        std::memcpy(out_data, in_data + idx, copy_size);
        T* pdata = out_data + copy_size / sizeof(T);
        for (int i = 0; i < pad_num; ++i) {
          pdata[i] = pad_value;
        }
        out_data += win_size;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
