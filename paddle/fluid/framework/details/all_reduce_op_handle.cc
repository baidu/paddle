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
#include "paddle/fluid/framework/details/all_reduce_op_handle.h"

#include <algorithm>

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_NCCL
DECLARE_bool(sync_nccl_allreduce);
#endif

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_NCCL)
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLCommunicator *ctxs)
    : NCCLOpHandleBase(node, places, ctxs), local_scopes_(local_scopes) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "The number of places and the number of local scopes "
                        "should be equal, but got number of places is %d and "
                        "number of local scopes is %d.",
                        places_.size(), local_scopes_.size()));
}
#elif defined(PADDLE_WITH_XPU_BKCL)
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::BKCLCommunicator *ctxs)
    : BKCLOpHandleBase(node, places, ctxs), local_scopes_(local_scopes) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "The number of places and the number of local scopes "
                        "should be equal, but got number of places is %d and "
                        "number of local scopes is %d.",
                        places_.size(), local_scopes_.size()));
}
#else
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size(),
                    platform::errors::InvalidArgument(
                        "The number of places and the number of local scopes "
                        "should be equal, but got number of places is %d and "
                        "number of local scopes is %d.",
                        places_.size(), local_scopes_.size()));
}
#endif

void AllReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  WaitInputVarGenerated();
  std::vector<VarHandleBase *> inputs = this->Inputs();
  std::vector<VarHandleBase *> outputs = this->Outputs();
  auto in_var_handles = DynamicCast<VarHandle>(inputs);
  auto out_var_handles = DynamicCast<VarHandle>(outputs);
  AllReduceImpl(in_var_handles, out_var_handles);
}

void AllReduceOpHandle::AllReduceImpl(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<VarHandle *> &out_var_handles) {
  size_t num_places = places_.size();
  PADDLE_ENFORCE_EQ(in_var_handles.size(), num_places,
                    platform::errors::InvalidArgument(
                        "The NoDummyInputSize should be equal "
                        "to the number of places, but got NoDummyInputSize is "
                        "%d and the number of places is %d.",
                        in_var_handles.size(), num_places));
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      platform::errors::InvalidArgument(
          "The NoDummyInputSize and NoDummyOutputSize should be "
          "equal, but got NoDummyInputSize is %d and NoDummyOutputSize is %d.",
          in_var_handles.size(), out_var_handles.size()));
  PADDLE_ENFORCE_EQ(
      local_exec_scopes_.size(), num_places,
      platform::errors::InvalidArgument(
          "The number of local scopes should be equal "
          "to the number of places, but got the number of local scopes is "
          "%d and the number of places is %d.",
          in_var_handles.size(), num_places));

  std::vector<const void *> lod_tensor_data;
  std::vector<platform::Place> places;
  lod_tensor_data.reserve(num_places);
  places.reserve(num_places);
  int64_t numel = -1;
  bool is_gpu_place = false;
#if defined(PADDLE_WITH_XPU_BKCL)
  bool is_xpu_place = false;
#endif
  auto dtype = static_cast<framework::proto::VarType::Type>(0);
  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    auto &local_scope = local_exec_scopes_[i];
    auto var = local_scope->FindVar(in_var_handles[i]->name());
    PADDLE_ENFORCE_NOT_NULL(var, platform::errors::NotFound(
                                     "Variable %s is not found in local scope.",
                                     in_var_handles[i]->name()));
    auto &lod_tensor = var->Get<LoDTensor>();

    if (i == 0) {
      numel = static_cast<int64_t>(lod_tensor.numel());
      // only enforce place0, we will enforce other palce numel == place0 numel
      PADDLE_ENFORCE_GT(
          numel, 0,
          platform::errors::PreconditionNotMet(
              "The numel of tensor %s should be > 0, but got numel is %d.",
              in_var_handles[i]->name(), numel));
      dtype = lod_tensor.type();
      is_gpu_place = platform::is_gpu_place(lod_tensor.place());
#if defined(PADDLE_WITH_XPU_BKCL)
      is_xpu_place = platform::is_xpu_place(lod_tensor.place());
#endif
    }
    PADDLE_ENFORCE_EQ(
        numel, static_cast<int64_t>(lod_tensor.numel()),
        platform::errors::PreconditionNotMet(
            "The size of tensors of the same variable in different local "
            "scopes should be equal."));
    PADDLE_ENFORCE_EQ(
        dtype, lod_tensor.type(),
        platform::errors::PreconditionNotMet(
            "The dtype of tensors of the same variable in different local "
            "scopes should be equal."));
#if defined(PADDLE_WITH_XPU_BKCL)
    PADDLE_ENFORCE_EQ(is_xpu_place, platform::is_xpu_place(lod_tensor.place()),
                      platform::errors::PreconditionNotMet(
                          "The place type of tensors of the same variable "
                          "in different local scopes should be equal."));
#endif
    PADDLE_ENFORCE_EQ(is_gpu_place, platform::is_gpu_place(lod_tensor.place()),
                      platform::errors::PreconditionNotMet(
                          "The place type of tensors of the same variable "
                          "in different local scopes should be equal."));

    lod_tensor_data.emplace_back(lod_tensor.data<void>());
    places.emplace_back(lod_tensor.place());

    VLOG(10) << "place:" << i << ", input_name:" << in_var_handles[i]->name()
             << ", out_name:" << out_var_handles[i]->name();

    PADDLE_ENFORCE_EQ(
        in_var_handles[i]->name(), out_var_handles[i]->name(),
        platform::errors::InvalidArgument(
            "The name of input and output of all_reduce op should be equal, "
            "but got input is %s and output is %s.",
            in_var_handles[i]->name(), out_var_handles[i]->name()));
  }

  std::vector<std::string> grad_var_names;
  grad_var_names.reserve(num_places);
  for (auto &out_var : out_var_handles) {
    grad_var_names.emplace_back(out_var->Name());
  }

  AllReduceFunc(lod_tensor_data, dtype, numel, places, grad_var_names);
}

void AllReduceOpHandle::AllReduceFunc(
    std::vector<const void *> lod_tensor_data,
    const framework::proto::VarType::Type &dtype, int64_t numel,
    const std::vector<platform::Place> &places,
    const std::vector<std::string> &out_var_names) {
  if (is_gpu_place(places[0])) {
#if defined(PADDLE_WITH_NCCL)
    PADDLE_ENFORCE_NOT_NULL(nccl_ctxs_,
                            platform::errors::InvalidArgument(
                                "The nccl context should not be NULL."));
    ncclDataType_t nccl_dtype = platform::ToNCCLDataType(dtype);
    std::vector<std::function<void()>> all_reduce_calls;
    for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
      auto &p = places[i];
      void *buffer = const_cast<void *>(lod_tensor_data.at(i));
      all_reduce_calls.emplace_back([=] {
        NCCLAllReduce(p, buffer, buffer, numel, nccl_dtype, ncclSum);
      });
    }
    NCCLAllReduceFunc(all_reduce_calls);
#else
    PADDLE_THROW(
        platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
  } else if (is_xpu_place(places[0])) {
#if defined(PADDLE_WITH_XPU_BKCL)
    PADDLE_ENFORCE_NOT_NULL(bkcl_ctxs_,
                            platform::errors::InvalidArgument(
                                "The bkcl context should not be NULL."));
    BKCLDataType bkcl_dtype = platform::ToBKCLDataType(dtype);
    std::vector<std::function<void()>> all_reduce_calls;
    for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
      auto &p = places[i];
      void *buffer = const_cast<void *>(lod_tensor_data.at(i));
      all_reduce_calls.emplace_back([=] {
        BKCLAllReduce(p, buffer, buffer, numel, bkcl_dtype, BKCL_ADD);
      });
    }
    BKCLAllReduceFunc(all_reduce_calls);
#else
    PADDLE_THROW(
        platform::errors::PreconditionNotMet("Not compiled with BKCL."));
#endif
  } else {  // Special handle CPU only Operator's gradient. Like CRF
    auto &trg = *local_exec_scopes_[0]
                     ->FindVar(out_var_names[0])
                     ->GetMutable<LoDTensor>();

    // Reduce All Tensor to trg in CPU
    ReduceBufferData func(lod_tensor_data, trg.data<void>(), numel);
    VisitDataType(trg.type(), func);

    for (size_t i = 1; i < local_exec_scopes_.size(); ++i) {
      auto &scope = local_exec_scopes_[i];
      auto &p = places[i];
      auto *var = scope->FindVar(out_var_names[i]);

      size_t size = numel * SizeOfType(trg.type());
      RunAndRecordEvent(p, [&trg, var, p, size] {
        auto dst_ptr = var->GetMutable<framework::LoDTensor>()->data<void>();
        platform::CPUPlace cpu_place;
        memory::Copy(cpu_place, dst_ptr, cpu_place, trg.data<void>(), size);
      });
    }
  }
  VLOG(10) << Name() << " size:" << numel * SizeOfType(dtype);
}

#if defined(PADDLE_WITH_XPU_BKCL)
void AllReduceOpHandle::BKCLAllReduceFunc(
    const std::vector<std::function<void()>> &all_reduce_calls) {
  this->RunAndRecordEvent([&] {
    if (all_reduce_calls.size() == 1UL) {
      all_reduce_calls[0]();
    } else {
      PADDLE_ENFORCE_EQ(
          bkcl_group_start(), BKCL_SUCCESS,
          platform::errors::PreconditionNotMet("bkcl_group_start failed"));
      for (auto &call : all_reduce_calls) {
        call();
      }
      PADDLE_ENFORCE_EQ(
          bkcl_group_end(), BKCL_SUCCESS,
          platform::errors::PreconditionNotMet("bkcl_group_end failed"));
    }
  });
}
#endif

#if defined(PADDLE_WITH_NCCL)
void AllReduceOpHandle::NCCLAllReduceFunc(
    const std::vector<std::function<void()>> &all_reduce_calls) {
  this->RunAndRecordEvent([&] {
    if (all_reduce_calls.size() == 1UL) {
      // Do not use NCCLGroup when manage NCCL by per thread per device
      all_reduce_calls[0]();
    } else {
      platform::NCCLGroupGuard guard;
      for (auto &call : all_reduce_calls) {
        call();
      }
    }
  });

  SyncNCCLAllReduce();
}

void AllReduceOpHandle::SyncNCCLAllReduce() {
  if (FLAGS_sync_nccl_allreduce) {
    for (auto &p : places_) {
      int dev_id = BOOST_GET_CONST(platform::CUDAPlace, p).device;
      auto *nccl_ctxs =
          nccl_ctxs_->GetRunEnvNCCLCtx(run_order_, use_hierarchical_allreduce_);
      auto &nccl_ctx = nccl_ctxs->at(dev_id);
      auto stream = nccl_ctx.stream();
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaGetLastError());
    }
  }
}
#endif

std::string AllReduceOpHandle::Name() const { return "all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
