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

#include "paddle/fluid/operators/distributed/parameter_send.h"
#include <memory>
#include <utility>
#include "glog/logging.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/communicator_common.h"
#include "paddle/fluid/operators/distributed/distributed.h"
#include "paddle/fluid/operators/distributed/request_handler.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Scope;
class Tensor;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

class RPCClient;

using LoDTensor = framework::LoDTensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

typedef std::vector<std::pair<std::string, std::string>> EP_SPLIT_TABLE_PAIRS;

inline EP_SPLIT_TABLE_PAIRS GetMultiFieldCommContext(
    const CommContext &rpc_ctx, const framework::Scope &scope,
    int multi_parts) {
  EP_SPLIT_TABLE_PAIRS table_pairs;

  auto *send_var = scope.FindVar(rpc_ctx.var_name);
  if (send_var->IsType<framework::SelectedRows>()) {
    PADDLE_ENFORCE_GE(multi_parts, 1,
                      platform::errors::InvalidArgument(
                          "multi_parts must == 1 in parameter send, now is: %d",
                          multi_parts));

    for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
      table_pairs.push_back(
          std::make_pair(rpc_ctx.epmap[i], rpc_ctx.splited_varnames[i]));
    }

  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "GetMultiFieldCommContext unsupported LoDTensor current!"));
  }

  return table_pairs;
}  // namespace distributed

void SendByNotifyRPC(const CommContext &rpc_ctx,
                     const framework::Scope &scope) {
  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto &send_var_name = rpc_ctx.var_name;
  std::vector<distributed::VarHandlePtr> rets;

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  if (NeedSend(scope, send_var_name)) {
    for (size_t j = 0; j < rpc_ctx.epmap.size(); j++) {
      auto &endpoint = rpc_ctx.epmap[j];
      VLOG(4) << "sending " << send_var_name << " to " << endpoint;
      rets.push_back(rpc_client->AsyncDistributeNotify(endpoint, cpu_ctx, scope,
                                                       send_var_name));
      VLOG(4) << "send var " << send_var_name << " by notify RPC done";
    }
  } else {
    VLOG(3) << "don't send non-initialized variable: " << rpc_ctx.var_name;
  }

  for (auto &handle : rets) {
    PADDLE_ENFORCE_NE(handle->Wait(), 0U, platform::errors::ExecutionTimeout(
                                              "internal error in RPCClient"));
  }
}

template <typename T>
void ParameterSend<T>::operator()(const CommContext &rpc_ctx,
                                  const framework::Scope &scope, bool sync,
                                  int multi_parts) {
  if (rpc_ctx.var_name == STEP_COUNTER) {
    SendByNotifyRPC(rpc_ctx, scope);
    return;
  }

  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &cpu_ctx = *pool.Get(platform::CPUPlace());

  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(rpc_ctx.trainer_id);

  std::vector<distributed::VarHandlePtr> rets;
  auto *send_var = scope.FindVar(rpc_ctx.var_name);

  if (send_var->IsType<framework::LoDTensor>()) {
    size_t out_num = rpc_ctx.splited_varnames.size();
    if (out_num > 1) {
      auto &send_tensor = send_var->Get<framework::LoDTensor>();
      auto &send_tensor_dims = send_tensor.dims();
      std::vector<framework::DDim> outs_dims;
      outs_dims.reserve(out_num);

      // infer output shape
      PADDLE_ENFORCE_EQ(
          rpc_ctx.height_sections.size(), out_num,
          platform::errors::InvalidArgument("tensor split sections size"
                                            "should be equal to output size."));
      for (size_t i = 0; i < out_num; ++i) {
        auto dim = send_tensor_dims;
        dim[0] = rpc_ctx.height_sections[i];
        outs_dims.push_back(dim);
      }

      // create output var in local scope
      size_t row_offset = 0;
      for (size_t i = 0; i < out_num; ++i) {
        framework::Tensor *out = local_scope->Var(rpc_ctx.splited_varnames[i])
                                     ->GetMutable<framework::LoDTensor>();
        *out = send_tensor.Slice(row_offset, row_offset + outs_dims[i][0]);
        row_offset += outs_dims[i][0];
      }
    } else {
      auto &send_tensor = send_var->Get<framework::LoDTensor>();
      framework::Tensor *out = local_scope->Var(rpc_ctx.splited_varnames[0])
                                   ->GetMutable<framework::LoDTensor>();
      out->ShareDataWith(send_tensor);
    }

    for (size_t i = 0; i < rpc_ctx.splited_varnames.size(); i++) {
      auto &send_var_name = rpc_ctx.splited_varnames[i];
      auto &endpoint = rpc_ctx.epmap[i];
      VLOG(4) << " send var name: " << send_var_name
              << "endpoint: " << endpoint;
      if (NeedSend(*local_scope.get(), send_var_name)) {
        VLOG(3) << "sending " << send_var_name << " to " << endpoint;
        rets.push_back(rpc_client->AsyncSendVar(
            endpoint, cpu_ctx, *local_scope.get(), send_var_name));
        VLOG(4) << "send var " << send_var_name << " async handle done";
      } else {
        VLOG(3) << "don't send non-initialized variable: "
                << rpc_ctx.splited_varnames[i];
      }
    }
  } else if (send_var->IsType<framework::SelectedRows>()) {
    auto &send_slr = send_var->Get<framework::SelectedRows>();

    auto &send_rows = send_slr.rows();
    if (send_rows.size() == 0) {
      LOG(WARNING)
          << "WARNING: The variable sent to pserver is empty, which "
             "may cause an unknown error. Please check the state of "
             "use_double_buffer in pyreader/dataloader async mode, you need to "
             "turn it false.";
    }

    std::vector<std::vector<size_t>> outs_rows_idx;
    std::vector<std::vector<size_t>> outs_dense_idx;

    auto table_pairs = GetMultiFieldCommContext(rpc_ctx, scope, 1);
    outs_rows_idx.resize(table_pairs.size());
    outs_dense_idx.resize(table_pairs.size());

    auto row_numel = send_slr.value().numel() / send_slr.value().dims()[0];
    auto *src = send_slr.value().data<T>();

    // create output var in local scope
    std::vector<framework::SelectedRows *> outs;
    for (auto &table : table_pairs) {
      auto *out =
          local_scope->Var(table.second)->GetMutable<framework::SelectedRows>();
      outs.push_back(out);
    }

    if (!rpc_ctx.is_distributed) {
      auto pserver_num = rpc_ctx.epmap.size();

      // split rows index into output sparse vars
      for (size_t i = 0; i < send_rows.size(); ++i) {
        auto ep_idx = send_rows[i] % pserver_num;
        auto id = send_rows[i] / pserver_num;
        outs_rows_idx[ep_idx].push_back(id);
        outs_dense_idx[ep_idx].push_back(i);
      }

      auto place = platform::CPUPlace();

      for (size_t out_idx = 0; out_idx < rpc_ctx.splited_varnames.size();
           out_idx++) {
        auto rows_idx = outs_rows_idx[out_idx];

        auto dims = send_slr.GetCompleteDims();
        dims[0] = rows_idx.size();
        outs[out_idx]->set_height(rpc_ctx.height_sections[out_idx]);
        outs[out_idx]->mutable_rows()->clear();
        outs[out_idx]->mutable_value()->mutable_data<T>(dims, send_slr.place());

        if (rows_idx.size() > 0) {
          for (auto idx : rows_idx) {
            outs[out_idx]->mutable_rows()->push_back(idx);
          }
          auto dst = outs[out_idx]->mutable_value()->mutable_data<T>(place);
          for (size_t j = 0; j < rows_idx.size(); j++) {
            if (platform::is_cpu_place(place)) {
              memory::Copy(platform::CPUPlace(), dst + j * row_numel,
                           platform::CPUPlace(),
                           src + outs_dense_idx[out_idx][j] * row_numel,
                           sizeof(T) * row_numel);
            } else {
              PADDLE_THROW(
                  platform::errors::Unimplemented("do not support GPU now"));
            }
          }
        }
        PADDLE_ENFORCE_EQ(
            rows_idx.size(), outs[out_idx]->rows().size(),
            platform::errors::InvalidArgument(
                "rows should has the same size with tensor dim 0"));
      }
    } else {
      auto pserver_num = rpc_ctx.epmap.size();

      // split rows index into output sparse vars
      for (size_t i = 0; i < send_rows.size(); ++i) {
        auto out_idx = send_rows[i] % pserver_num;
        outs_rows_idx[out_idx].push_back(send_rows[i]);
        outs_dense_idx[out_idx].push_back(i);
      }

      auto place = platform::CPUPlace();

      for (size_t out_idx = 0; out_idx < rpc_ctx.splited_varnames.size();
           out_idx++) {
        auto rows_idx = outs_rows_idx[out_idx];

        auto dims = send_slr.GetCompleteDims();
        dims[0] = rows_idx.size();

        outs[out_idx]->set_height(rpc_ctx.height_sections[out_idx]);
        outs[out_idx]->mutable_rows()->clear();
        outs[out_idx]->mutable_value()->mutable_data<T>(dims, send_slr.place());

        if (rows_idx.size() > 0) {
          for (auto idx : rows_idx) {
            outs[out_idx]->mutable_rows()->push_back(idx);
          }
          auto dst = outs[out_idx]->mutable_value()->mutable_data<T>(place);
          for (size_t j = 0; j < rows_idx.size(); j++) {
            if (platform::is_cpu_place(place)) {
              memory::Copy(platform::CPUPlace(), dst + j * row_numel,
                           platform::CPUPlace(),
                           src + outs_dense_idx[out_idx][j] * row_numel,
                           sizeof(T) * row_numel);
            } else {
              PADDLE_THROW(
                  platform::errors::Unimplemented("do not support GPU now"));
            }
          }
        }
        PADDLE_ENFORCE_EQ(
            rows_idx.size(), outs[out_idx]->rows().size(),
            platform::errors::InvalidArgument(
                "rows should has the same size with tensor dim 0"));
      }
    }

    for (size_t i = 0; i < table_pairs.size(); i++) {
      auto &send_var_name = table_pairs[i].second;
      auto &endpoint = table_pairs[i].first;
      auto need_send = NeedSend(*local_scope.get(), send_var_name);

      VLOG(4) << "send var name: " << send_var_name
              << " send var endpoint: " << endpoint
              << " need send: " << need_send;

      if (need_send) {
        VLOG(4) << "sending " << send_var_name << " to " << endpoint;

        rets.push_back(rpc_client->AsyncSendVar(
            endpoint, cpu_ctx, *local_scope.get(), send_var_name));
        VLOG(4) << "send var " << send_var_name << " async handle done";
      } else {
        VLOG(4) << "don't send non-initialized variable: "
                << rpc_ctx.splited_varnames[i];
      }
    }
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "unsupported var type: %s to send!", send_var->Type()));
  }

  VLOG(4) << "Prepare to send var " << rpc_ctx.var_name;
  if (sync) {
    for (auto &handle : rets) {
      VLOG(4) << "Wait send var to pserver handle: " << handle;
      PADDLE_ENFORCE_NE(handle->Wait(), 0U, platform::errors::ExecutionTimeout(
                                                "internal error in RPCClient"));
    }
  }
}

template struct ParameterSend<float>;

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
