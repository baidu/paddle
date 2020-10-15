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

#include <sys/types.h>
#include <algorithm>
#include <memory>

#include "glog/logging.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/communicator_common.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace operators {
namespace distributed {

class RPCClient;

using LoDTensor = framework::LoDTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T>
void RecvSparseLodTensor(const CommContext &rpc_ctx,
                         const framework::Scope &scope) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto cpu_place = platform::CPUPlace();
  auto &cpu_ctx = *pool.Get(cpu_place);

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();
  std::vector<const float *> tensors;
  std::vector<distributed::VarHandlePtr> rets;
  std::vector<std::string> recv_varnames;
  for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
    auto &recv_var_name = rpc_ctx.splited_varnames[i];
    VLOG(4) << "recv " << recv_var_name << " from " << rpc_ctx.epmap[i];
    local_scope->Var(recv_var_name);
    // sparse param in recv_scope is LoDTensor
    rets.push_back(rpc_client->AsyncGetVarNoBarrier(
        rpc_ctx.epmap[i], cpu_ctx, *local_scope.get(), recv_var_name,
        recv_var_name));
    recv_varnames.push_back(recv_var_name);
  }

  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, platform::errors::ExecutionTimeout(
                                               "internal error in RPCClient"));
    auto &recv_var_name = recv_varnames[i];
    auto *local_var = local_scope->FindVar(recv_var_name);
    const auto *value = local_var->Get<framework::LoDTensor>().data<float>();
    tensors.push_back(value);
  }

  auto *merged_var = scope.FindVar(rpc_ctx.var_name);

  if (merged_var == nullptr || !merged_var->IsInitialized()) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("%s must initialized at first."));
  }
  auto dims1 = merged_var->Get<framework::LoDTensor>().dims()[1];
  int64_t height = 0;
  for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
    auto *splited_var = local_scope->FindVar(rpc_ctx.splited_varnames[i]);
    height += splited_var->Get<framework::LoDTensor>().dims()[0];
  }

  PADDLE_ENFORCE_EQ(
      merged_var->Get<framework::LoDTensor>().dims()[0], height,
      platform::errors::InvalidArgument(
          "Received variable must has same dimension with local variable."));

  auto *merged_t = merged_var->GetMutable<framework::LoDTensor>();
  auto *merged_d = merged_t->mutable_data<float>(cpu_place);

  auto pserver_num = rpc_ctx.splited_varnames.size();
  for (int x = 0; x < height; ++x) {
    auto id = x % pserver_num;
    auto idx = x / pserver_num;
    std::memcpy(merged_d + x * dims1, tensors[id] + idx * dims1,
                sizeof(float) * dims1);
  }
}

template <typename T>
void RecvGeoSparseRecords(const CommContext &rpc_ctx,
                          const framework::Scope &scope) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto cpu_place = platform::CPUPlace();
  auto &cpu_ctx = *pool.Get(cpu_place);

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
    auto &recv_var_name = rpc_ctx.splited_varnames[i];
    local_scope->Var(recv_var_name);
    VLOG(4) << "recv " << recv_var_name << " from " << rpc_ctx.epmap[i];
    // sparse param in recv_scope is LoDTensor
    rets.push_back(rpc_client->AsyncGetVar(rpc_ctx.epmap[i], cpu_ctx,
                                           *local_scope.get(), recv_var_name,
                                           recv_var_name, recv_var_name));
  }

  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, platform::errors::ExecutionTimeout(
                                               "internal error in RPCClient"));
  }

  int64_t height = 0;
  int64_t ids_num = 0;
  int64_t width = 0;

  std::vector<int64_t> all_ids;
  auto pserver_num = rpc_ctx.splited_varnames.size();

  for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
    auto &recv_var_name = rpc_ctx.splited_varnames[i];
    auto *recv_var = local_scope->FindVar(recv_var_name);
    auto &recv_t = recv_var->Get<framework::SelectedRows>();

    height += recv_t.height();
    ids_num += recv_t.rows().size();
    width = recv_t.value().dims()[1];

    if (rpc_ctx.is_distributed) {
      std::copy(recv_t.rows().begin(), recv_t.rows().end(),
                std::back_inserter(all_ids));
    } else {
      std::transform(recv_t.rows().begin(), recv_t.rows().end(),
                     std::back_inserter(all_ids),
                     [&](int64_t id) { return id * pserver_num + i; });
    }
  }

  auto *var = scope.FindVar(rpc_ctx.var_name);
  auto *t_ = var->GetMutable<framework::SelectedRows>();
  T *out_data =
      t_->mutable_value()->mutable_data<T>({ids_num, width}, cpu_place);
  t_->set_height(height);
  t_->set_rows(all_ids);

  int64_t cnt = 0;
  for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
    auto &recv_var_name = rpc_ctx.splited_varnames[i];
    auto *recv_var = local_scope->FindVar(recv_var_name);
    auto &recv_t = recv_var->Get<framework::SelectedRows>();

    auto rows = recv_t.rows().size();
    const T *in_data = recv_t.value().data<T>();
    std::copy_n(in_data, rows * width, out_data + cnt);
    cnt += rows * width;
  }
  t_->SyncIndex();
}

template <typename T>
void RecvLodTensor(const CommContext &rpc_ctx, const framework::Scope &scope) {
  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  std::vector<distributed::VarHandlePtr> rets;

  // variable do not spilt
  if (rpc_ctx.origin_varnames.size() == 1 &&
      rpc_ctx.splited_varnames.size() == 1) {
    auto varname = rpc_ctx.origin_varnames[0];
    const auto place =
        scope.FindVar(varname)->Get<framework::LoDTensor>().place();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &ctx = *pool.Get(place);
    VLOG(4) << "recv " << varname << " from " << rpc_ctx.epmap[0] << " in gpu? "
            << platform::is_gpu_place(place);
    rets.push_back(rpc_client->AsyncGetVarNoBarrier(rpc_ctx.epmap[0], ctx,
                                                    scope, varname, varname));

    for (size_t i = 0; i < rets.size(); i++) {
      PADDLE_ENFORCE_NE(
          rets[i]->Wait(), 0U,
          platform::errors::ExecutionTimeout("internal error in RPCClient"));
    }

    VLOG(3) << "ParameterRecv out " << rpc_ctx.var_name;
    return;
  } else {
    PADDLE_ENFORCE(false, platform::errors::Unimplemented(
                              "ParameterRecv can not recv dense with multi "
                              "parts now, add it soon."));
  }
}

template <typename T>
void ParameterRecv<T>::operator()(const CommContext &rpc_ctx,
                                  const framework::Scope &scope,
                                  bool geo_records) {
  VLOG(3) << "ParameterRecv in " << rpc_ctx.var_name;

  PADDLE_ENFORCE_GE(rpc_ctx.origin_varnames.size(), 1,
                    platform::errors::InvalidArgument(
                        "origin_varnames.size() >= 1 is permitted"));

  if (rpc_ctx.is_sparse) {
    if (geo_records) {
      RecvGeoSparseRecords<T>(rpc_ctx, scope);
    } else {
      RecvSparseLodTensor<T>(rpc_ctx, scope);
    }
  } else {
    RecvLodTensor<T>(rpc_ctx, scope);
  }

  VLOG(3) << "ParameterRecv out " << rpc_ctx.var_name;
}
template <typename T>
void ParameterRecv<T>::operator()(const CommContext &rpc_ctx,
                                  const framework::Scope &scope) {
  this->operator()(rpc_ctx, scope, false);
}

template struct ParameterRecv<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
