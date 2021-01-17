//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_ASCEND
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
#include "paddle/fluid/framework/tensor.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class AscendTriggerCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
#ifdef PADDLE_WITH_ASCEND
    auto ascend_ptr = paddle::framework::AscendInstance::GetInstance();
    auto graph_idx = ctx.Attr<int>("graph_idx");
    VLOG(4) << "AscendTrigger Kernel, begin to run graph: " << graph_idx;
    auto inputs = ctx.MultiInput<framework::Tensor>("FeedList");
    auto outputs = ctx.MultiOutput<framework::Tensor>("FetchList");
    ascend_ptr->RunAscendSubgraph(graph_idx, inputs, &outputs);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Please compile WITH_ASCEND option to enable ascend_trigger op"));
#endif
  }
};

}  // namespace operators
}  // namespace paddle
