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
#include "paddle/fluid/framework/details/op_handle_base.h"

#include <map>
#include <unordered_set>

namespace paddle {
namespace framework {
namespace details {
std::string OpHandleBase::DebugString() const {
  std::stringstream ss;
  ss << Name() << "(";
  for (auto *var : inputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ") --> (";
  for (auto *var : outputs_) {
    ss << var->DebugString() << ", ";
  }
  ss << ")\n";
  return ss.str();
}

OpHandleBase::~OpHandleBase() PADDLE_MAY_THROW {
#ifdef PADDLE_WITH_CUDA
  for (auto &ev : events_) {
    if (ev.second) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaEventDestroy(ev.second));
    }
  }
#endif
}

void OpHandleBase::InitCUDA() {
#ifdef PADDLE_WITH_CUDA
  for (auto &p : dev_ctxes_) {
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, p.first).device;
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(dev_id));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaEventCreateWithFlags(&events_[dev_id], cudaEventDisableTiming));
  }
  if (IsMultiDeviceTransfer() && dev_ctxes_.size() > 0) {
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        int dev_id =
            BOOST_GET_CONST(platform::CUDAPlace, out_var_handle->place())
                .device;
        out_var_handle->SetGenerateEvent(events_.at(dev_id));
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(
        dev_ctxes_.size(), 1UL,
        platform::errors::InvalidArgument(
            "Operator %s should have only one dev_ctx, but got %d.", Name(),
            dev_ctxes_.size()));
    auto &place = dev_ctxes_.begin()->first;
    int dev_id = BOOST_GET_CONST(platform::CUDAPlace, place).device;
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        PADDLE_ENFORCE_EQ(
            platform::is_same_place(place, out_var_handle->place()), true,
            platform::errors::InvalidArgument(
                "The place of output(%s) is not consistent with the "
                "place of current op(%s).",
                out_var_handle->Name(), Name()));
        out_var_handle->SetGenerateEvent(events_.at(dev_id));
      }
    }
  }
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Paddle can't use CUDA device since it's not compiled with CUDA,"
      "Please recompile or reinstall Paddle with GPU support."));
#endif
}

void OpHandleBase::InitXPU() {
#ifdef PADDLE_WITH_XPU
  if (IsMultiDeviceTransfer() && dev_ctxes_.size() > 0) {
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        // TODO(liuyuhui): XPU now don't support sync events, add later.
      }
    }
  } else {
    PADDLE_ENFORCE_EQ(dev_ctxes_.size(), 1UL,
                      platform::errors::InvalidArgument(
                          "%s should have only one dev_ctx.", Name()));
    auto &place = dev_ctxes_.begin()->first;
    int dev_id = BOOST_GET_CONST(platform::XPUPlace, place).device;
    PADDLE_ENFORCE_EQ(
        xpu_set_device(dev_id), XPU_SUCCESS,
        platform::errors::PreconditionNotMet("xpu_set_device failed"));
    for (auto &out_var : outputs_) {
      auto *out_var_handle = dynamic_cast<VarHandle *>(out_var);
      if (out_var_handle) {
        PADDLE_ENFORCE_EQ(
            platform::is_same_place(place, out_var_handle->place()), true,
            platform::errors::InvalidArgument(
                "The place of output(%s) is not consistent with the "
                "place of current op(%s).",
                out_var_handle->Name(), Name()));
      }
    }
  }
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Paddle can't use XPU device since it's not compiled with XPU,"
      "Please recompile or reinstall Paddle with XPU support."));
#endif
}

void OpHandleBase::Run(DeviceType use_device) {
#ifdef PADDLE_WITH_CUDA
  if (events_.empty() && use_device == p::kCUDA && dev_ctxes_.size() > 0) {
    InitCUDA();
  }
#else
  PADDLE_ENFORCE_NE(
      use_device, p::kCUDA,
      platform::errors::InvalidArgument(
          "Argument use_device should not be kCUDA when Paddle is not "
          "compiled with CUDA."));
#endif

  if (use_device == p::kXPU && dev_ctxes_.size() > 0) {
#ifdef PADDLE_WITH_XPU
    InitXPU();
#else
    PADDLE_ENFORCE_NE(
        use_device, p::kXPU,
        platform::errors::InvalidArgument(
            "Argument use_device should not be kXPU when Paddle is not "
            "compiled with XPU."));
#endif
  }

  // skip running current op, used with inplace_addto_op_pass
  if (skip_running_) {
    VLOG(4) << "skip running: " << Name();
    return;
  }

  RunImpl();
}

void OpHandleBase::RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) {
#ifdef PADDLE_WITH_CUDA
  PADDLE_ENFORCE_NOT_NULL(waited_ctx, platform::errors::InvalidArgument(
                                          "Argument waited_ctx is NULL."));
  if (platform::is_cpu_place(waited_ctx->GetPlace()) || events_.empty()) {
    for (auto &dev_ctx : dev_ctxes_) {
      PADDLE_ENFORCE_NOT_NULL(
          dev_ctx.second,
          platform::errors::InvalidArgument("The device context is NULL."));
      dev_ctx.second->Wait();
    }
  } else {
    auto stream =
        static_cast<platform::CUDADeviceContext *>(waited_ctx)->stream();
    for (auto &ev : events_) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamWaitEvent(stream, ev.second, 0));
    }
  }
#else
  for (auto &dev_ctx : dev_ctxes_) {
    dev_ctx.second->Wait();
  }
#endif
}

void OpHandleBase::AddInput(VarHandleBase *in) {
  this->inputs_.emplace_back(in);
  node_->inputs.push_back(in->Node());
  in->AddOutput(this, this->Node());
}

void OpHandleBase::AddOutput(VarHandleBase *out) {
  outputs_.emplace_back(out);
  node_->outputs.push_back(out->Node());
  out->AddInput(this, this->Node());
}

void OpHandleBase::WaitInputVarGenerated(bool wait_for_feed) {
  for (auto in_var : inputs_) {
    if (NeedWait(in_var)) {
      // Dummy Variable is used to represent dependencies between operators, so
      // there doesn't add event for it.
      auto *in_var_handle = dynamic_cast<VarHandle *>(in_var);
      if (in_var_handle) {
        auto &place = in_var_handle->place();
        if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
          auto stream =
              static_cast<platform::CUDADeviceContext *>(dev_ctxes_.at(place))
                  ->stream();
          PADDLE_ENFORCE_CUDA_SUCCESS(
              cudaStreamWaitEvent(stream, in_var_handle->GetEvent(), 0));
#else
          PADDLE_THROW(
              platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
        }
        // There are nothing to do when the place is CPUPlace.
      }
    } else {
      // NOTE(zhiqiu): Special case when using fetch_async_op_handle may lead to
      // nodetermination due to parallel execution of cuda memory operation. Eg:
      // execute stream: CPU->GPU copy (feed)
      // fetch stream: GPU->CUDAPinned (fetch)
      if (in_var && wait_for_feed) {
        auto *in_var_handle = dynamic_cast<VarHandle *>(in_var);
        if (in_var_handle) {
          auto &place = in_var_handle->place();
          if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
            platform::DeviceContextPool &pool =
                platform::DeviceContextPool::Instance();
            auto stream =
                static_cast<platform::CUDADeviceContext *>(pool.Get(place))
                    ->stream();
            PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
#else
            PADDLE_THROW(platform::errors::PreconditionNotMet(
                "Not compiled with CUDA."));
#endif
          }
        }
      }
    }
  }
}

void OpHandleBase::WaitInputVarGenerated(const platform::Place &place) {
  for (auto in_var : inputs_) {
    if (NeedWait(in_var)) {
      // Dummy Variable is used to represent dependencies between operators,
      // so there doesn't add event for it.
      auto *in_var_handle = dynamic_cast<VarHandle *>(in_var);
      if (in_var_handle) {
        if (platform::is_gpu_place(in_var_handle->place())) {
#ifdef PADDLE_WITH_CUDA
          auto stream = static_cast<platform::CUDADeviceContext *>(
                            dev_ctxes_.at(in_var_handle->place()))
                            ->stream();
          PADDLE_ENFORCE_CUDA_SUCCESS(
              cudaStreamWaitEvent(stream, in_var_handle->GetEvent(), 0));
#else
          PADDLE_THROW(
              platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
        }
        // There are nothing to do when the place is CPUPlace.
      }
    }
  }
}

size_t OpHandleBase::NoDummyInputSize() const {
  size_t cnt = 0;
  for (auto *in : inputs_) {
    if (dynamic_cast<DummyVarHandle *>(in) == nullptr) {
      ++cnt;
    }
  }
  return cnt;
}

bool OpHandleBase::NeedWait(VarHandleBase *in_var) {
  return in_var && in_var->GeneratedOp();
}

void OpHandleBase::RunAndRecordEvent(const std::function<void()> &callback) {
  callback();
#ifdef PADDLE_WITH_CUDA
  if (!events_.empty()) {  // Use event
    for (auto &p : dev_ctxes_) {
      auto dev_id = BOOST_GET_CONST(platform::CUDAPlace, p.first).device;
      auto *cuda_dev_ctx = static_cast<platform::CUDADeviceContext *>(p.second);
      VLOG(10) << "cudadevicecontext:" << cuda_dev_ctx << ", dev_id:" << dev_id;
      PADDLE_ENFORCE_CUDA_SUCCESS(
          cudaEventRecord(events_.at(dev_id), cuda_dev_ctx->stream()));
    }
  }
#endif
}

void OpHandleBase::RunAndRecordEvent(platform::Place p,
                                     const std::function<void()> &callback) {
#ifdef PADDLE_WITH_CUDA
  if (platform::is_cpu_place(p) || events_.empty()) {
    callback();
  } else {
    auto *ctx = dev_ctxes_.at(p);
    auto *cuda_ctx = static_cast<platform::CUDADeviceContext *>(ctx);
    cuda_ctx->RecordEvent(
        events_.at(BOOST_GET_CONST(platform::CUDAPlace, p).device), callback);
  }
#else
  callback();
#endif
}

size_t OpHandleBase::NotReadyInputSize() const {
  std::unordered_set<VarHandleBase *> res;
  for (auto *var : inputs_) {
    if (var->GeneratedOp() != nullptr) {
      res.emplace(var);
    }
  }
  return res.size();
}

void OpHandleBase::SetLocalExecScopes(
    const std::unordered_map<Scope *, Scope *> &scope_map) {
  local_exec_scopes_.clear();
  auto scopes = GetLocalScopes();
  for (auto *scope : scopes) {
    auto iter = scope_map.find(scope);
    PADDLE_ENFORCE_NE(
        iter, scope_map.end(),
        platform::errors::NotFound("Local scope not found in scope map."));
    local_exec_scopes_.emplace_back(iter->second);
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
