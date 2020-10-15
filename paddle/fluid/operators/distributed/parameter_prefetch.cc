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

#include "paddle/fluid/operators/distributed/parameter_prefetch.h"
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/distributed/distributed.h"

namespace paddle {
namespace framework {
class ExecutionContext;
class Scope;
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

static void SplitIdsIntoMultipleVarsBySection(
    const std::vector<int64_t> &in_ids,
    const std::vector<std::string> &in_varnames, const int tables,
    const int pservers, const bool is_distibuted, framework::Scope *scope,
    std::vector<std::vector<int64_t>> *splited_ids,
    std::vector<std::vector<int64_t>> *origin_ids) {
  PADDLE_ENFORCE_EQ(
      in_varnames.size(), tables,
      platform::errors::OutOfRange(
          "send varnames size: %d not equal table number: %d, internal error",
          in_varnames.size(), tables));

  PADDLE_ENFORCE_LE(
      tables, pservers,
      platform::errors::OutOfRange("table number %d not equal or less than "
                                   "pserver number: %d, internal error",
                                   tables, pservers));

  auto place = platform::CPUPlace();

  std::set<int64_t> st(in_ids.begin(), in_ids.end());
  std::vector<int64_t> all_ids;
  all_ids.assign(st.begin(), st.end());

  splited_ids->resize(tables);
  origin_ids->resize(tables);

  if (is_distibuted) {
    for (auto &id : all_ids) {
      auto pserver_id = id % pservers;
      (*splited_ids)[pserver_id].push_back(id);
      (*origin_ids)[pserver_id].push_back(id);
    }
  } else {
    for (auto &id : all_ids) {
      auto pserver_id = id % pservers;
      (*origin_ids)[pserver_id].push_back(id);
      id = id / pservers;
      (*splited_ids)[pserver_id].push_back(id);
    }
  }

  for (size_t i = 0; i < in_varnames.size(); ++i) {
    auto *id_tensor =
        scope->Var(in_varnames[i])->GetMutable<framework::LoDTensor>();

    auto &ids = (*splited_ids)[i];
    if (!ids.empty()) {
      auto *id_tensor_data = id_tensor->mutable_data<int64_t>(
          framework::make_ddim({static_cast<int64_t>(ids.size()), 1}), place);
      memcpy(id_tensor_data, ids.data(), sizeof(int64_t) * ids.size());
    }
  }
}

typedef std::vector<std::pair<std::string, std::string>> TableAndEndpoints;

void prefetch_core(
    const std::vector<int64_t> &ids, const TableAndEndpoints &tables,
    const framework::ExecutionContext &context, const framework::Scope &scope,
    const bool is_distributed,
    std::unordered_map<int64_t, std::vector<float>> *recved_vec_map) {
  distributed::RPCClient *rpc_client =
      distributed::RPCClient::GetInstance<RPCCLIENT_T>(
          context.Attr<int>("trainer_id"));

  int pservers = context.Attr<int>("pserver_num");

  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &actual_ctx = *pool.Get(platform::CPUPlace());

  std::unique_ptr<framework::Scope> local_scope = scope.NewTmpScope();

  std::vector<std::string> in_var_names;
  std::vector<std::string> out_var_names;
  for (size_t i = 0; i < tables.size(); ++i) {
    in_var_names.push_back("prefetch_send@" + tables[i].second);
    out_var_names.push_back("prefetch_recv@" + tables[i].second);
  }

  std::vector<std::vector<int64_t>> split_ids;
  std::vector<std::vector<int64_t>> origin_ids;
  SplitIdsIntoMultipleVarsBySection(ids, in_var_names, tables.size(), pservers,
                                    is_distributed, local_scope.get(),
                                    &split_ids, &origin_ids);

  // create output var in local scope
  for (auto &name : out_var_names) {
    local_scope->Var(name)->GetMutable<framework::LoDTensor>();
  }

  std::vector<distributed::VarHandlePtr> rets;
  for (size_t i = 0; i < in_var_names.size(); i++) {
    if (NeedSend(*local_scope.get(), in_var_names[i])) {
      VLOG(3) << "sending " << in_var_names[i] << " to " << tables[i].second
              << " to get " << out_var_names[i] << " back";
      rets.push_back(rpc_client->AsyncPrefetchVar(
          tables[i].second, actual_ctx, *local_scope.get(), in_var_names[i],
          out_var_names[i], tables[i].first));
    } else {
      VLOG(3) << "don't send no-initialied variable: " << out_var_names[i];
    }
  }
  for (size_t i = 0; i < rets.size(); i++) {
    PADDLE_ENFORCE_NE(rets[i]->Wait(), 0U, platform::errors::ExecutionTimeout(
                                               "internal error in RPCClient"));
  }

  for (size_t o_idx = 0; o_idx < out_var_names.size(); ++o_idx) {
    auto &ids_in_this_section = origin_ids[o_idx];

    if (!ids_in_this_section.empty()) {
      auto &prefetch_out_var =
          local_scope->Var(out_var_names[o_idx])->Get<framework::LoDTensor>();
      const auto *out_var_data = prefetch_out_var.data<float>();
      auto &dims = prefetch_out_var.dims();

      PADDLE_ENFORCE_EQ(dims.size(), 2,
                        platform::errors::InvalidArgument(
                            "The size of Tensor dims must be 2."));
      PADDLE_ENFORCE_EQ(ids_in_this_section.size(), dims[0],
                        platform::errors::InvalidArgument(
                            "The size of ids in this section must equal to "
                            "dims[0]: %s, but got %s",
                            dims[0], ids_in_this_section.size()));

      auto row_numel = dims[1];

      for (int64_t i = 0; i < dims[0]; ++i) {
        auto origin_id = ids_in_this_section[i];
        std::vector<float> vecs(row_numel);

        std::copy_n(out_var_data + i * row_numel, row_numel, vecs.begin());
        (*recved_vec_map)[origin_id] = vecs;
      }
    } else {
      VLOG(3) << "ids in this section is empty";
    }
  }
}

void prefetch(const std::string &id_name, const std::string &out_name,
              const std::string &persistable_var_name,
              const bool is_distributed,
              const std::vector<std::string> &table_names,
              const std::vector<std::string> &endpoints,
              const framework::ExecutionContext &context,
              const framework::Scope &scope) {
  prefetchs({id_name}, {out_name}, persistable_var_name, is_distributed,
            table_names, endpoints, context, scope);
}

void prefetchs(const std::vector<std::string> &id_var_names,
               const std::vector<std::string> &out_var_names,
               const std::string &persistable_var_name,
               const bool is_distributed,
               const std::vector<std::string> &table_names,
               const std::vector<std::string> &endpoints,
               const framework::ExecutionContext &context,
               const framework::Scope &scope) {
  auto vec_dim_1 = 0;
  auto vec_dim_0 = 0;
  framework::Variable *var = scope.FindVar(persistable_var_name);

  if (var->IsType<SelectedRows>()) {
    vec_dim_1 = var->Get<framework::SelectedRows>().value().dims()[1];
  } else {
    vec_dim_0 = var->Get<framework::LoDTensor>().dims()[0];
    vec_dim_1 = var->Get<framework::LoDTensor>().dims()[1];
  }

  PADDLE_ENFORCE_GT(vec_dim_1, 0,
                    platform::errors::InvalidArgument(
                        "lookup table var's dim must gather than 0"));

  const auto place =
      scope.FindVar(id_var_names[0])->Get<framework::LoDTensor>().place();

  std::vector<std::vector<int64_t>> ids_group;
  std::vector<int64_t> ids_union;
  std::vector<framework::LoD> ids_lods;
  TableAndEndpoints tables;

  for (auto &id_name : id_var_names) {
    auto &id_tensor = scope.FindVar(id_name)->Get<framework::LoDTensor>();
    std::vector<int64_t> ids;
    TensorToVector(id_tensor, context.device_context(), &ids);
    ids_union.insert(ids_union.end(), ids.begin(), ids.end());
    ids_group.push_back(ids);
    ids_lods.push_back(id_tensor.lod());
  }

  std::unordered_set<int64_t> s(ids_union.begin(), ids_union.end());
  ids_union.assign(s.begin(), s.end());

  for (auto &i : ids_union) {
    PADDLE_ENFORCE_GE(
        i, 0, platform::errors::OutOfRange(
                  "each element in embedding should be larger or equal 0"));
    if (!is_distributed) {
      PADDLE_ENFORCE_LT(
          i, vec_dim_0,
          platform::errors::OutOfRange(
              "embedding id must in [0, %d) when is_distributed False",
              vec_dim_0));
    }
  }

  for (size_t i = 0; i < table_names.size(); i++) {
    tables.push_back(std::make_pair(table_names[i], endpoints[i]));
  }

  std::unordered_map<int64_t, std::vector<float>> recved_vec_map;
  prefetch_core(ids_union, tables, context, scope, is_distributed,
                &recved_vec_map);

  auto padding_idx = distributed::kNoPadding;

  if (context.HasAttr("padding_idx")) {
    padding_idx = context.Attr<int64_t>("padding_idx");
  }

  for (size_t i = 0; i < out_var_names.size(); i++) {
    std::vector<int64_t> ids = ids_group[i];
    auto ids_size = ids.size();
    auto *out_t =
        scope.FindVar(out_var_names[i])->GetMutable<framework::LoDTensor>();
    out_t->set_lod(ids_lods[i]);
    out_t->Resize(
        framework::make_ddim({static_cast<int64_t>(ids_size), vec_dim_1}));
    auto *out_d = out_t->mutable_data<float>(place);

    if (platform::is_cpu_place(out_t->place())) {
      for (auto idx = 0; idx < static_cast<int>(ids_size); idx++) {
        const auto &id = ids[idx];
        if (padding_idx != distributed::kNoPadding && id == padding_idx) {
          memset(out_d + idx * vec_dim_1, 0, sizeof(float) * vec_dim_1);
        } else {
          std::copy_n(recved_vec_map[id].begin(), vec_dim_1,
                      out_d + idx * vec_dim_1);
        }
      }
    } else {
#ifdef PADDLE_WITH_CUDA
      for (auto idx = 0; idx < static_cast<int>(ids_size); idx++) {
        const auto &id = ids[idx];
        auto stream = context.cuda_device_context().stream();
        if (padding_idx != distributed::kNoPadding && id == padding_idx) {
          platform::GpuMemsetAsync(out_d + idx * vec_dim_1, 0,
                                   sizeof(float) * vec_dim_1, stream);
        } else {
          auto &cpu_place =
              BOOST_GET_CONST(platform::CPUPlace,
                              paddle::platform::CPUDeviceContext().GetPlace());
          auto &gpu_place =
              BOOST_GET_CONST(platform::CUDAPlace, out_t->place());
          memory::Copy(gpu_place, out_d + idx * vec_dim_1, cpu_place,
                       &recved_vec_map[id][0], sizeof(float) * vec_dim_1,
                       stream);
        }
      }
#else
      PADDLE_ENFORCE(true, platform::errors::PermissionDenied(
                               "Paddle is not compiled with GPU!"));
#endif
    }
  }
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
