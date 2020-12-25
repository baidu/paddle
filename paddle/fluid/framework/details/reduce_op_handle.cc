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

#include "paddle/fluid/framework/details/reduce_op_handle.h"

#include <memory>

#include "paddle/fluid/framework/details/container_cast.h"
#include "paddle/fluid/framework/details/reduce_and_gather.h"
#include "paddle/fluid/framework/details/variable_visitor.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/profiler.h"

DEFINE_bool(
    cpu_deterministic, false,
    "Whether to make the result of computation deterministic in CPU side.");

namespace paddle {
namespace framework {
namespace details {

std::once_flag CollectiveContext::init_flag_;
std::unique_ptr<CollectiveContext> CollectiveContext::context_;

static inline std::string GetRemoteVarName(const std::string &var_name,
                                           int trainer_id) {
  return string::Sprintf("%s_merged_tmp@trainer_%d", var_name, trainer_id);
}

void ReduceOpHandle::Wait(
    const std::map<platform::Place, platform::DeviceContext *> &dev_ctxes) {
  // TODO(gongwb): use event wait?
  for (auto &dev_ctx : dev_ctxes) {
    dev_ctx.second->Wait();
  }
}

void ReduceOpHandle::RunImpl() {
  platform::RecordEvent record_event(Name());

  if (places_.size() == 1) return;
  // the input and output may have dummy var.
  auto in_var_handles = DynamicCast<VarHandle>(inputs_);

  PADDLE_ENFORCE_EQ(
      in_var_handles.size(), places_.size(),
      platform::errors::InvalidArgument(
          "The number of inputs should equal to the number of places, but got "
          "the number of inputs is %d and the number of places is %d.",
          in_var_handles.size(), places_.size()));

  VarHandle *out_var_handle;
  {
    auto out_var_handles = DynamicCast<VarHandle>(outputs_);

    PADDLE_ENFORCE_EQ(out_var_handles.size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The number of output should be one, but got %d.",
                          out_var_handles.size()));
    out_var_handle = out_var_handles.front();
  }

  auto in_0_handle = in_var_handles[0];

  auto &var_scopes = local_exec_scopes_;

  auto pre_in_var =
      var_scopes.at(in_0_handle->scope_idx())->FindVar(in_0_handle->name());

  PADDLE_ENFORCE_NOT_NULL(pre_in_var, platform::errors::NotFound(
                                          "Variable %s is not found in scope.",
                                          in_0_handle->name()));

  // NOTE: The Places of all input tensor must be all on CPU or all on GPU.
  std::vector<platform::Place> in_places;  // used to get dev_ctx
  for (auto *in_handle : in_var_handles) {
    in_places.emplace_back(in_handle->place());
    auto in_var =
        var_scopes.at(in_handle->scope_idx())->FindVar(in_handle->name());

    PADDLE_ENFORCE_NOT_NULL(
        in_var, platform::errors::NotFound("Variable %s is not found in scope.",
                                           in_handle->name()));

    VariableVisitor::EnforceShapeAndDTypeEQ(*pre_in_var, *in_var);
  }

  auto out_var = var_scopes.at(out_var_handle->scope_idx())
                     ->FindVar(out_var_handle->name());

  PADDLE_ENFORCE_NOT_NULL(
      out_var, platform::errors::NotFound("Variable %s is not found in scope.",
                                          out_var_handle->name()));

  // NOTE: The tensors' Place of input and output must be all on GPU or all on
  // CPU.
  auto in_p = VariableVisitor::GetMutableTensor(pre_in_var).place();
  platform::Place t_out_p;
  if (platform::is_gpu_place(in_p)) {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(out_var_handle->place()), true,
                      platform::errors::PreconditionNotMet(
                          "Places of input and output must be all on GPU."));
    t_out_p = out_var_handle->place();
  } else {
    t_out_p = platform::CPUPlace();
  }

  if (pre_in_var->IsType<framework::SelectedRows>()) {
    this->RunAndRecordEvent([&] {
      std::vector<const SelectedRows *> in_selected_rows =
          GetInputValues<SelectedRows>(in_var_handles, var_scopes);

      const CollectiveContext &collective_context =
          *CollectiveContext::GetInstance();
      VLOG(10) << "GatherSelectedRows CollectiveContext:"
               << collective_context.String();

      // TODO(gongwb): add cpu support
      if (collective_context.endpoints_.size() <= 1 ||
          is_cpu_place(in_places[0]) || is_cpu_place(t_out_p)) {
        GatherLocalSelectedRowsFunctor functor(
            in_selected_rows, in_places, dev_ctxes_, t_out_p,
            out_var->GetMutable<framework::SelectedRows>());
        WaitInputVarGenerated();
        functor();
        return;
      }
    });
  } else {
    std::vector<const LoDTensor *> lod_tensors =
        GetInputValues<LoDTensor>(in_var_handles, var_scopes);

    if (paddle::platform::is_cpu_place(lod_tensors[0]->place())) {
      WaitInputVarGenerated();
      this->RunAndRecordEvent([&] {
        // FIXME(zcd): The order of summing is important,
        // especially when the type of data is float or double.
        // For example, the result of `a+b+c+d` may be different
        // with the result of `c+a+b+d`, so the summing order should be fixed.
        if (!FLAGS_cpu_deterministic) {
          ReduceLoDTensor func(lod_tensors,
                               out_var->GetMutable<framework::LoDTensor>());
          VisitDataType(lod_tensors[0]->type(), func);
        } else {
          // We sum lod_tensors to reduce_sum_trg which is in local_scopes_0
          // here, but it doesn't mean reduce_sum_trg must be in local_scopes_0.
          auto &reduce_sum_trg = *this->local_exec_scopes_[0]
                                      ->FindVar(out_var_handle->name())
                                      ->GetMutable<framework::LoDTensor>();
          ReduceLoDTensor func(lod_tensors, &reduce_sum_trg);
          VisitDataType(lod_tensors[0]->type(), func);

          auto trg = out_var->GetMutable<framework::LoDTensor>();
          if (reduce_sum_trg.data<void>() != trg->data<void>()) {
            TensorCopy(reduce_sum_trg, platform::CPUPlace(), trg);
          }
        }
      });
    } else if (paddle::platform::is_gpu_place(lod_tensors[0]->place())) {
#if defined(PADDLE_WITH_NCCL)
      auto pre_in = pre_in_var->Get<framework::LoDTensor>();
      VariableVisitor::ShareDimsAndLoD(*pre_in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(
          out_var_handle->place(), pre_in.type());

      auto out_p = out_var_handle->place();
      int root_id = BOOST_GET_CONST(platform::CUDAPlace, out_p).device;
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < var_scopes.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = *lod_tensors[i];

        int dev_id = BOOST_GET_CONST(platform::CUDAPlace, p).device;
        auto &nccl_ctx = nccl_ctxs_->at(dev_id);

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root_id == dev_id) {
          recvbuffer =
              out_var->GetMutable<framework::LoDTensor>()->mutable_data(
                  out_var_handle->place());
        }

        int type = platform::ToNCCLDataType(lod_tensor.type());
        size_t numel = static_cast<size_t>(lod_tensor.numel());
        all_reduce_calls.emplace_back(
            [buffer, recvbuffer, type, numel, root_id, &nccl_ctx] {
              PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclReduce(
                  buffer, recvbuffer, numel, static_cast<ncclDataType_t>(type),
                  ncclSum, root_id, nccl_ctx.comm_, nccl_ctx.stream()));
            });
      }

      WaitInputVarGenerated();
      this->RunAndRecordEvent([&] {
        platform::NCCLGroupGuard guard;
        for (auto &call : all_reduce_calls) {
          call();
        }
      });
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
    } else if (paddle::platform::is_xpu_place(lod_tensors[0]->place())) {
#if defined(PADDLE_WITH_XPU_BKCL)
      auto pre_in = pre_in_var->Get<framework::LoDTensor>();
      VariableVisitor::ShareDimsAndLoD(*pre_in_var, out_var);
      VariableVisitor::GetMutableTensor(out_var).mutable_data(
          out_var_handle->place(), pre_in.type());

      auto out_p = out_var_handle->place();
      int root_id = BOOST_GET_CONST(platform::XPUPlace, out_p).device;
      std::vector<std::function<void()>> all_reduce_calls;
      for (size_t i = 0; i < var_scopes.size(); ++i) {
        auto &p = in_places[i];
        auto &lod_tensor = *lod_tensors[i];

        int dev_id = BOOST_GET_CONST(platform::XPUPlace, p).device;
        auto &bkcl_ctx = bkcl_ctxs_->at(dev_id);

        void *buffer = const_cast<void *>(lod_tensor.data<void>());
        void *recvbuffer = nullptr;
        if (root_id == dev_id) {
          recvbuffer =
              out_var->GetMutable<framework::LoDTensor>()->mutable_data(
                  out_var_handle->place());
        }

        int type = platform::ToBKCLDataType(lod_tensor.type());
        size_t numel = static_cast<size_t>(lod_tensor.numel());
        all_reduce_calls.emplace_back([buffer, recvbuffer, type, numel, root_id,
                                       &bkcl_ctx] {
          PADDLE_ENFORCE_EQ(bkcl_reduce(bkcl_ctx.comm(), buffer, recvbuffer,
                                        numel, static_cast<BKCLDataType>(type),
                                        BKCL_ADD, root_id, nullptr),
                            BKCL_SUCCESS, platform::errors::Unavailable(
                                              "bkcl_all_reduce failed"));
        });
      }

      WaitInputVarGenerated();
      this->RunAndRecordEvent([&] {
        PADDLE_ENFORCE_EQ(
            bkcl_group_start(), BKCL_SUCCESS,
            platform::errors::Unavailable("bkcl_group_start failed"));
        for (auto &call : all_reduce_calls) {
          call();
        }
        PADDLE_ENFORCE_EQ(
            bkcl_group_end(), BKCL_SUCCESS,
            platform::errors::Unavailable("bkcl_group_end failed"));
      });
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with XPU."));
#endif
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The place of tensor should be CPUPlace, CUDAPlace or XPUPlace, but "
          "got %s.",
          lod_tensors[0]->place()));
    }
  }
}

template <typename T>
std::vector<const T *> ReduceOpHandle::GetInputValues(
    const std::vector<VarHandle *> &in_var_handles,
    const std::vector<Scope *> &var_scopes) const {
  std::vector<const T *> in_selected_rows;
  for (auto *in_handle : in_var_handles) {
    auto &in_sr = var_scopes.at(in_handle->scope_idx())
                      ->FindVar(in_handle->name())
                      ->Get<T>();
    in_selected_rows.emplace_back(&in_sr);
  }
  return in_selected_rows;
}

std::string ReduceOpHandle::Name() const { return "reduce"; }
}  // namespace details
}  // namespace framework
}  // namespace paddle
