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
#include <math.h>
#include <algorithm>
#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/profiler.h"

#ifdef PADDLE_WITH_CUDA
DECLARE_bool(sync_nccl_allreduce);
#endif
DECLARE_bool(check_nan_inf);

namespace paddle {
namespace framework {
namespace details {

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places,
                                     const platform::NCCLCommunicator *ctxs)
    : NCCLOpHandleBase(node, places, ctxs), local_scopes_(local_scopes) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
}
#else
AllReduceOpHandle::AllReduceOpHandle(ir::Node *node,
                                     const std::vector<Scope *> &local_scopes,
                                     const std::vector<platform::Place> &places)
    : OpHandleBase(node), local_scopes_(local_scopes), places_(places) {
  PADDLE_ENFORCE_EQ(places_.size(), local_scopes_.size());
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
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), num_places,
      "The NoDummyInputSize should be equal to the number of places.");
  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), out_var_handles.size(),
      "The NoDummyInputSize and NoDummyOutputSize should be equal.");
  PADDLE_ENFORCE_EQ(local_exec_scopes_.size(), num_places);

  std::vector<const void *> lod_tensor_data;
  std::vector<platform::Place> places;
  lod_tensor_data.reserve(num_places);
  places.reserve(num_places);
  int64_t numel = -1;
  bool is_gpu_place = false;
  auto dtype = static_cast<framework::proto::VarType::Type>(0);
  for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
    auto &local_scope = local_exec_scopes_[i];
    auto var = local_scope->FindVar(in_var_handles[i]->name());
    PADDLE_ENFORCE_NOT_NULL(var, "%s is not found int scope.",
                            in_var_handles[i]->name());
    auto &lod_tensor = var->Get<LoDTensor>();

    if (i == 0) {
      numel = static_cast<int64_t>(lod_tensor.numel());
      dtype = lod_tensor.type();
      is_gpu_place = platform::is_gpu_place(lod_tensor.place());
    }
    PADDLE_ENFORCE_EQ(numel, static_cast<int64_t>(lod_tensor.numel()));
    PADDLE_ENFORCE_EQ(dtype, lod_tensor.type());
    PADDLE_ENFORCE_EQ(is_gpu_place, platform::is_gpu_place(lod_tensor.place()));

    lod_tensor_data.emplace_back(lod_tensor.data<void>());
    places.emplace_back(lod_tensor.place());

    VLOG(10) << "place:" << i << ", input_name:" << in_var_handles[i]->name()
             << ", out_name:" << out_var_handles[i]->name();

    PADDLE_ENFORCE_EQ(in_var_handles[i]->name(), out_var_handles[i]->name(),
                      "The name of input and output should be equal.");
  }

  std::vector<std::string> grad_var_names;
  grad_var_names.reserve(num_places);
  for (auto &out_var : out_var_handles) {
    grad_var_names.emplace_back(out_var->Name());
  }

  AllReduceFunc(lod_tensor_data, dtype, numel, places, grad_var_names);
}

template <typename T>
struct AllReduceRangeFunctor {
  explicit AllReduceRangeFunctor(const T *a) : a_(a) {}
  inline HOSTDEVICE void operator()(size_t id) const {
    float r = static_cast<float>(a_[id]);
    PADDLE_ENFORCE(std::isnan(r) == 0, "%f is nan", r);
    PADDLE_ENFORCE(std::isinf(r) == 0, "%f is inf", r);
  }
  const T *a_;
};

struct TensorCheckerVisitor {
  TensorCheckerVisitor(const std::string &var_name,
                       const framework::Tensor &tensor,
                       const platform::Place &place)
      : var_name_(var_name), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() const {
    if (!std::is_floating_point<T>::value) {
      VLOG(10) << var_name_
               << " need not to check, it's type is not float point";
      return;
    }

    AllReduceRangeFunctor<T> func(tensor_.data<T>());
    if (platform::is_gpu_place(place_)) {
#ifdef PADDLE_WITH_CUDA
      auto *dev_ctx = reinterpret_cast<platform::CUDADeviceContext *>(
          platform::DeviceContextPool::Instance().Get(place_));
      platform::ForRange<platform::CUDADeviceContext> for_range(
          *dev_ctx, tensor_.numel());
      for_range(func);
#else
      PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
      return;
    }

    auto *dev_ctx = reinterpret_cast<platform::CPUDeviceContext *>(
        platform::DeviceContextPool::Instance().Get(place_));
    platform::ForRange<platform::CPUDeviceContext> for_range(*dev_ctx,
                                                             tensor_.numel());
    for_range(func);
  }

  std::string var_name_;
  const framework::Tensor &tensor_;
  const platform::Place &place_;
};

void AllReduceOpHandle::CheckNanOrInf(const std::string &op_type,
                                      const framework::Scope &scope,
                                      const std::string &var_name,
                                      const platform::Place &place) {
  auto *var = scope.FindVar(var_name);
  PADDLE_ENFORCE_NOT_NULL(var, "can't find var:%s", var_name);

  const Tensor *tensor{nullptr};
  if (var->IsType<framework::LoDTensor>()) {
    tensor = &var->Get<framework::LoDTensor>();
  } else if (var->IsType<framework::SelectedRows>()) {
    tensor = &var->Get<framework::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  /*
  PADDLE_ENFORCE(!framework::TensorContainsInf(*tensor),
          "Operator %s output Tensor %s contains Inf", op_type, var_name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(*tensor),
          "Operator %s output Tensor %s contains NAN", op_type, var_name);
          */

  TensorCheckerVisitor vistor(var_name, *tensor, place);
  VisitDataType(tensor->type(), vistor);
}

void AllReduceOpHandle::AllReduceFunc(
    std::vector<const void *> lod_tensor_data,
    const framework::proto::VarType::Type &dtype, int64_t numel,
    const std::vector<platform::Place> &places,
    const std::vector<std::string> &out_var_names) {
  if (is_gpu_place(places[0])) {
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
    PADDLE_ENFORCE_NOT_NULL(nccl_ctxs_, "nccl_ctxs should not be nullptr.");
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
    PADDLE_THROW("Not compiled with CUDA.");
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

  if (FLAGS_check_nan_inf) {
    for (size_t i = 0; i < local_exec_scopes_.size(); ++i) {
      VLOG(10) << Name() << " check_nan_inf " << out_var_names[i];
      CheckNanOrInf(Name(), *local_exec_scopes_[i], out_var_names[i],
                    places_[i]);
    }
  }

  VLOG(10) << Name() << " size:" << numel * SizeOfType(dtype);
}

#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
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
      int dev_id = boost::get<platform::CUDAPlace>(p).device;
      auto *nccl_ctxs =
          nccl_ctxs_->GetRunEnvNCCLCtx(run_order_, use_hierarchical_allreduce_);
      auto &nccl_ctx = nccl_ctxs->at(dev_id);
      auto stream = nccl_ctx.stream();
      cudaError_t e_sync = cudaStreamSynchronize(stream);
      if (e_sync != 0) {
        LOG(FATAL) << "cudaStreamSynchronize " << cudaGetErrorString(e_sync);
      }

      cudaError_t e_get = cudaGetLastError();
      if (e_get != 0) {
        LOG(FATAL) << "cudaGetLastError  " << cudaGetErrorString(e_get)
                   << " errno:" << e_get;
      }
    }
  }
}
#endif

std::string AllReduceOpHandle::Name() const { return "all_reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
