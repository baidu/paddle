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

#include "paddle/fluid/framework/details/broadcast_op_handle.h"

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace framework {
namespace details {

void BroadcastOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  if (places_.size() == 1) return;

  // The input and output may have dummy vars.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);
  auto out_var_handles = DynamicCast<VarHandle>(outputs_);

  PADDLE_ENFORCE_EQ(in_var_handles.size(), 1UL,
                    platform::errors::PreconditionNotMet(
                        "The number of inputs should be 1, but got %d.",
                        in_var_handles.size()));
  PADDLE_ENFORCE_EQ(out_var_handles.size(), places_.size(),
                    platform::errors::PreconditionNotMet(
                        "The number of outputs and the number of places should "
                        "be equal, but got the number of outputs is %d and the "
                        "number of places is %d.",
                        out_var_handles.size(), places_.size()));

  VarHandle *in_var_handle = in_var_handles[0];

  BroadcastOneVar(*in_var_handle, out_var_handles, local_exec_scopes_);
}

void BroadcastOpHandle::BroadcastOneVar(
    const VarHandle &in_var_handle,
    const std::vector<VarHandle *> &out_var_handles,
    const std::vector<Scope *> &var_scopes) {
  auto *in_var =
      var_scopes.at(in_var_handle.scope_idx())->FindVar(in_var_handle.name());
  PADDLE_ENFORCE_NOT_NULL(
      in_var, platform::errors::NotFound("Variable %s is not found in scopes.",
                                         in_var_handle.name()));
  Tensor &in_tensor = VariableVisitor::GetMutableTensor(in_var);
  if (UNLIKELY(!in_tensor.IsInitialized())) {
    VLOG(3) << "in var " << in_var_handle.name() << "not inited, return!";
    return;
  }

  InitOutputValue(in_var_handle, out_var_handles);

  if (platform::is_cpu_place(in_tensor.place())) {
    WaitInputVarGenerated();
    for (auto *out_var_handle : out_var_handles) {
      if (out_var_handle->IsTheSameVar(in_var_handle)) {
        continue;
      }
      auto &out_p = out_var_handle->place();
      auto *out_var = var_scopes.at(out_var_handle->scope_idx())
                          ->FindVar(out_var_handle->name());

      RunAndRecordEvent(out_p, [in_tensor, out_var] {
        paddle::framework::TensorCopy(
            in_tensor, platform::CPUPlace(),
            &VariableVisitor::GetMutableTensor(out_var));
      });
    }
  } else if (platform::is_gpu_place(in_tensor.place())) {
#if defined(PADDLE_WITH_NCCL)
    VarHandle *out_handle = nullptr;
    int root_id =
        BOOST_GET_CONST(platform::CUDAPlace, in_tensor.place()).device;
    std::vector<std::function<void()>> broadcast_calls;

    int type = platform::ToNCCLDataType(in_tensor.type());
    size_t numel = static_cast<size_t>(in_tensor.numel());

    for (auto out_var_handle : out_var_handles) {
      Variable *out_var = var_scopes.at(out_var_handle->scope_idx())
                              ->FindVar(out_var_handle->name());

      int dst_id =
          BOOST_GET_CONST(platform::CUDAPlace, out_var_handle->place()).device;

      auto &nccl_ctx = nccl_ctxs_->at(dst_id);

      void *send_recv_buffer = nullptr;
      if (root_id == dst_id) {
        send_recv_buffer = const_cast<void *>(in_tensor.data<void>());
        out_handle = out_var_handle;
      } else {
        send_recv_buffer = VariableVisitor::GetMutableTensor(out_var)
                               .Resize(in_tensor.dims())
                               .mutable_data(out_var_handle->place());
      }

      broadcast_calls.emplace_back(
          [send_recv_buffer, numel, type, root_id, &nccl_ctx] {
            PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBcast(
                send_recv_buffer, numel, static_cast<ncclDataType_t>(type),
                root_id, nccl_ctx.comm_, nccl_ctx.stream()));
          });
    }

    WaitInputVarGenerated();
    this->RunAndRecordEvent([&] {
      {
        platform::NCCLGroupGuard guard;
        for (auto &call : broadcast_calls) {
          call();
        }
      }

      if (!out_handle->IsTheSameVar(in_var_handle)) {
        auto out_var = var_scopes.at(in_var_handle.scope_idx())
                           ->FindVar(out_var_handles[0]->name());
        paddle::framework::TensorCopy(
            in_tensor, in_var_handle.place(),
            *(dev_ctxes_.at(in_var_handle.place())),
            &VariableVisitor::GetMutableTensor(out_var));
      }
    });
    for (auto &p : places_) {
      nccl_ctxs_->DevCtx(p)->Wait();
    }
#else
    PADDLE_THROW(
        platform::errors::PreconditionNotMet("Not compiled with NCLL."));
#endif
  } else {
#if defined(PADDLE_WITH_XPU_BKCL)
    VarHandle *out_handle = nullptr;
    int root_id = BOOST_GET_CONST(platform::XPUPlace, in_tensor.place()).device;
    std::vector<std::function<void()>> broadcast_calls;

    int type = platform::ToBKCLDataType(in_tensor.type());
    size_t numel = static_cast<size_t>(in_tensor.numel());

    for (auto out_var_handle : out_var_handles) {
      Variable *out_var = var_scopes.at(out_var_handle->scope_idx())
                              ->FindVar(out_var_handle->name());

      int dst_id =
          BOOST_GET_CONST(platform::XPUPlace, out_var_handle->place()).device;

      auto &bkcl_ctx = bkcl_ctxs_->at(dst_id);

      void *send_recv_buffer = nullptr;
      if (root_id == dst_id) {
        send_recv_buffer = const_cast<void *>(in_tensor.data<void>());
        out_handle = out_var_handle;
      } else {
        send_recv_buffer = VariableVisitor::GetMutableTensor(out_var)
                               .Resize(in_tensor.dims())
                               .mutable_data(out_var_handle->place());
      }

      broadcast_calls.emplace_back([send_recv_buffer, numel, type, root_id,
                                    &bkcl_ctx] {
        PADDLE_ENFORCE_EQ(
            bkcl_broadcast(bkcl_ctx.comm(), send_recv_buffer, send_recv_buffer,
                           numel, static_cast<BKCLDataType>(type), root_id,
                           nullptr),
            BKCL_SUCCESS,
            platform::errors::Unavailable("bkcl_broadcast failed"));
      });
    }

    WaitInputVarGenerated();
    this->RunAndRecordEvent([&] {
      {
        PADDLE_ENFORCE_EQ(
            bkcl_group_start(), BKCL_SUCCESS,
            platform::errors::Unavailable("bkcl_group_start failed"));
        for (auto &call : broadcast_calls) {
          call();
        }
        PADDLE_ENFORCE_EQ(
            bkcl_group_end(), BKCL_SUCCESS,
            platform::errors::Unavailable("bkcl_group_end failed"));
      }

      if (!out_handle->IsTheSameVar(in_var_handle)) {
        auto out_var = var_scopes.at(in_var_handle.scope_idx())
                           ->FindVar(out_var_handles[0]->name());
        paddle::framework::TensorCopy(
            in_tensor, in_var_handle.place(),
            *(dev_ctxes_.at(in_var_handle.place())),
            &VariableVisitor::GetMutableTensor(out_var));
      }
    });
#else
    PADDLE_THROW(
        platform::errors::PreconditionNotMet("Not compiled with BKCL."));
#endif
  }
}

void BroadcastOpHandle::InitOutputValue(
    const VarHandle &in_var_handle,
    const std::vector<VarHandle *> &out_var_handles) const {
  auto &var_scopes = local_exec_scopes_;
  auto *in_var =
      var_scopes.at(in_var_handle.scope_idx())->FindVar(in_var_handle.name());

  Tensor &in_tensor = VariableVisitor::GetMutableTensor(in_var);

  // NOTE: The tensors' Place of input and output must be all on GPU or all on
  // CPU.
  for (auto *out_var_handle : out_var_handles) {
    if (out_var_handle->IsTheSameVar(in_var_handle)) {
      continue;
    }
    auto t_out_p = out_var_handle->place();
    auto *out_var = var_scopes.at(out_var_handle->scope_idx())
                        ->FindVar(out_var_handle->name());
    PADDLE_ENFORCE_NOT_NULL(out_var, platform::errors::NotFound(
                                         "Variable %s is not found in scopes.",
                                         out_var_handle->name()));
    if (is_gpu_place(in_tensor.place())) {
      PADDLE_ENFORCE_EQ(platform::is_gpu_place(t_out_p), true,
                        platform::errors::PreconditionNotMet(
                            "Places of input and output must be all on GPU."));
    } else {
      t_out_p = platform::CPUPlace();
    }
    VariableVisitor::ShareDimsAndLoD(*in_var, out_var);
    VariableVisitor::GetMutableTensor(out_var).mutable_data(t_out_p,
                                                            in_tensor.type());
  }
}

std::string BroadcastOpHandle::Name() const { return "broadcast"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
