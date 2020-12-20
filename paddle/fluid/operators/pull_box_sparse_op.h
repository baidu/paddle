//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
static void PullBoxSparseFunctor(const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::Tensor>("Ids");
  auto outputs = ctx.MultiOutput<framework::Tensor>("Out");
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  // BoxPS only supports float now
  std::vector<float *> all_values(slot_size);
  std::vector<int64_t> slot_lengths(slot_size);
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = slot->numel();
    auto *output = outputs[i]->mutable_data<T>(ctx.GetPlace());
    all_values[i] = output;
  }
#ifdef PADDLE_WITH_BOX_PS
  auto hidden_size = ctx.Attr<int>("size");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PullSparse(ctx.GetPlace(), all_keys, all_values, slot_lengths,
                      hidden_size, 0);
#endif
#ifdef PADDLE_WITH_PSLIB
  auto hidden_size = ctx.Attr<int>("size");
  auto gpu_ps_ptr = paddle::framework::PSGPUWrapper::GetInstance();
  gpu_ps_ptr->PullSparse(ctx.GetPlace(), 0, all_keys, all_values, slot_lengths,
                        hidden_size);
#endif
}

template <typename T>
static void PushBoxSparseFunctor(const framework::ExecutionContext &ctx) {
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto d_output =
      ctx.MultiInput<framework::Tensor>(framework::GradVarName("Out"));
  const auto slot_size = inputs.size();
  std::vector<const uint64_t *> all_keys(slot_size);
  std::vector<const float *> all_grad_values(slot_size);
  std::vector<int64_t> slot_lengths(slot_size);
  int batch_size = -1;
  for (size_t i = 0; i < slot_size; i++) {
    const auto *slot = inputs[i];
    const uint64_t *single_slot_keys =
        reinterpret_cast<const uint64_t *>(slot->data<int64_t>());
    all_keys[i] = single_slot_keys;
    slot_lengths[i] = slot->numel();
    int cur_batch_size =
        slot->lod().size() ? slot->lod()[0].size() - 1 : slot->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                        platform::errors::PreconditionNotMet(
                            "The batch size of all input slots should be same, "
                            "please cheack"));
    }
    const float *grad_value = d_output[i]->data<float>();
    all_grad_values[i] = grad_value;
  }
#ifdef PADDLE_WITH_BOX_PS
  auto hidden_size = ctx.Attr<int>("size");
  auto box_ptr = paddle::framework::BoxWrapper::GetInstance();
  box_ptr->PushSparseGrad(ctx.GetPlace(), all_keys, all_grad_values,
                          slot_lengths, hidden_size, 0, batch_size);
#endif
#ifdef PADDLE_WITH_PSLIB
  auto hidden_size = ctx.Attr<int>("size");
  auto gpu_ps_ptr = paddle::framework::PSGPUWrapper::GetInstance();
  gpu_ps_ptr->PushSparseGrad(ctx.GetPlace(), 0, all_keys, all_grad_values, slot_lengths,
                        hidden_size, batch_size);
#endif
}

using LoDTensor = framework::LoDTensor;
template <typename T>
class PullBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PullBoxSparseFunctor<T>(ctx);
  }
};

template <typename T>
class PushBoxSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushBoxSparseFunctor<T>(ctx);
  }
};
}  // namespace operators
}  // namespace paddle
