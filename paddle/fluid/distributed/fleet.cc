/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/distributed/fleet.h"
#include "paddle/fluid/distributed/service/communicator.h"
#include "paddle/fluid/distributed/table/table.h"

namespace paddle {
namespace distributed {

using framework::LoDTensor;
using framework::ProgramDesc;
using framework::VarDesc;
using framework::Variable;

const uint32_t MAX_FEASIGN_NUM = 1024 * 100 * 100;
std::shared_ptr<FleetWrapper> FleetWrapper::s_instance_ = NULL;
bool FleetWrapper::is_initialized_ = false;

std::shared_ptr<paddle::distributed::PSCore> FleetWrapper::pserver_ptr_ = NULL;

void FleetWrapper::SetClient2ClientConfig(int request_timeout_ms,
                                          int connect_timeout_ms,
                                          int max_retry) {
  client2client_request_timeout_ms_ = request_timeout_ms;
  client2client_connect_timeout_ms_ = connect_timeout_ms;
  client2client_max_retry_ = max_retry;
}

void FleetWrapper::LoadSparseOnServer(const std::string& path,
                                      const std::string& meta,
                                      uint32_t table_id) {
  VLOG(3) << "load sparse table " << table_id << " with " << path << " meta "
          << meta;
  pserver_ptr_->_server_ptr->table(table_id)->load(path, meta);
}

void FleetWrapper::InitServer(
    const std::string& dist_desc,
    const std::vector<std::string>& host_sign_list, int index, int trainers,
    const std::vector<framework::ProgramDesc>& server_sub_program) {
  if (!is_initialized_) {
    VLOG(3) << "Going to init server";
    pserver_ptr_ = std::shared_ptr<paddle::distributed::PSCore>(
        new paddle::distributed::PSCore());
    pserver_ptr_->init_server(dist_desc, &host_sign_list, host_sign_list.size(),
                              index, trainers, server_sub_program);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Server can be initialized only once";
  }
}

// void FleetWrapper::InitWorker(
//     const std::string& dist_desc, const std::vector<uint64_t>&
//     host_sign_list, Scope* scope, const RpcCtxMap& send_ctx, const
//     std::unordered_map<uint64_t, std::vector<std::string>>&
//         dense_varnames,
//     const std::map<std::string, std::string>& envs, int node_num, int index)
//     {
//   if (!is_initialized_) {
//     VLOG(3) << "Going to init worker";

//     Communicator::InitInstance<AsyncCommunicator>(
//         send_ctx, dense_varnames, dist_desc, host_sign_list, scope, envs);

//     pserver_ptr_ = std::shared_ptr<paddle::distributed::PSCore>(
//         new paddle::distributed::PSCore());
//     pserver_ptr_->init_worker(dist_desc, _regions,
//                               const_cast<uint64_t*>(host_sign_list.data()),
//                               node_num, index);
//     is_initialized_ = true;
//   } else {
//     VLOG(3) << "Worker can be initialized only once";
//   }
// }

void FleetWrapper::InitWorker(
    const std::string& dist_desc,
    const std::vector<std::string>& host_sign_list, Scope* scope,
    const RpcCtxMap& send_ctx,
    const std::unordered_map<uint64_t, std::vector<std::string>>&
        dense_varnames,
    const std::map<std::string, std::string>& envs, int node_num, int index) {
  if (!is_initialized_) {
    VLOG(3) << "Going to init worker";

    Communicator::InitInstance<AsyncCommunicator>(
        send_ctx, dense_varnames, dist_desc, host_sign_list, scope, envs);

    pserver_ptr_ = std::shared_ptr<paddle::distributed::PSCore>(
        new paddle::distributed::PSCore());
    pserver_ptr_->init_worker(dist_desc, _regions, &host_sign_list, node_num,
                              index);
    is_initialized_ = true;
  } else {
    VLOG(3) << "Worker can be initialized only once";
  }
}

void FleetWrapper::StopServer() {
  VLOG(3) << "Going to stop server";
  auto* communicator = Communicator::GetInstance();
  auto status = communicator->_worker_ptr->stop_server();
  status.wait();
}

void FleetWrapper::FinalizeWorker() {
  VLOG(3) << "Going to finalize worker";
  pserver_ptr_->finalize_worker();
}

void FleetWrapper::BarrierWithTable(uint32_t barrier_type) {
  VLOG(3) << "Going to Barrier worker";
  auto* communicator = Communicator::GetInstance();
  communicator->BarrierWithTable(barrier_type, 10800000);
}

uint64_t FleetWrapper::RunServer(const std::string& ip, uint32_t port) {
  VLOG(3) << "Going to run server with ip " << ip << " port " << port;
  auto ret = pserver_ptr_->run_server(ip, port);
  return ret;
}

std::vector<uint64_t> FleetWrapper::GetClientsInfo() {
  VLOG(3) << "Going to get client info";
  return pserver_ptr_->get_client_info();
  return std::vector<uint64_t>();
}

void FleetWrapper::CreateClient2ClientConnection() {
  VLOG(3) << "Going to create client2client connection";
  pserver_ptr_->create_client2client_connection(
      client2client_request_timeout_ms_, client2client_connect_timeout_ms_,
      client2client_max_retry_);
}

std::future<int32_t> FleetWrapper::PullSparseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim) {
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (auto name : var_names) {
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }

  bool training = true;
  return pserver_ptr_->_worker_ptr->pull_sparse(pull_result_ptr.data(),
                                                table_id, fea_keys->data(),
                                                fea_keys->size(), training);
}

void FleetWrapper::PullSparseVarsSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names, std::vector<uint64_t>* fea_keys,
    std::vector<std::vector<float>>* fea_values, int fea_value_dim,
    const std::vector<std::string>& var_emb_names) {
  std::vector<std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  fea_keys->clear();
  fea_keys->resize(0);
  fea_keys->reserve(MAX_FEASIGN_NUM);
  for (size_t var_index = 0; var_index < var_names.size(); ++var_index) {
    const std::string& name = var_names[var_index];
    Variable* var = scope.FindVar(name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();

    // skip slots which do not have embedding
    const std::string& emb_name = var_emb_names[var_index];
    Variable* emb_var = scope.FindVar(emb_name);
    if (emb_var == nullptr) {
      continue;
    }

    for (auto i = 0u; i < len; ++i) {
      if (ids[i] == 0u) {
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  fea_values->resize(fea_keys->size() + 1);
  for (auto& t : *fea_values) {
    t.resize(fea_value_dim);
  }
  std::vector<float*> pull_result_ptr;
  for (auto& t : *fea_values) {
    pull_result_ptr.push_back(t.data());
  }
  bool training = true;
  auto status = pserver_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys->data(), fea_keys->size(),
      training);
  pull_sparse_status.push_back(std::move(status));
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit_);
      exit(-1);
    }
  }
}

// is_training is true means training, false means inference, the behavior is
// different on pserver

void FleetWrapper::PullSparseToTensorSync(const uint64_t table_id, int fea_dim,
                                          uint64_t padding_id,
                                          platform::Place place,
                                          bool is_training,
                                          std::vector<const LoDTensor*>* inputs,
                                          std::vector<LoDTensor*>* outputs) {
  std::vector<uint64_t> fea_keys;
  std::vector<float*> pull_result_ptr;
  fea_keys.reserve(MAX_FEASIGN_NUM / 100);
  pull_result_ptr.reserve(MAX_FEASIGN_NUM / 100);
  std::vector<float> init_value(fea_dim, 0);
  framework::LoDTensor* output = nullptr;
  float* output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  for (size_t index = 0; index < inputs->size(); ++index) {
    const framework::LoDTensor* tensor = inputs->at(index);
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len += fea_dim) {
      if (!output || output_len == size_t(output->numel())) {
        ++output_index;
        CHECK(output_index < outputs->size());  // NOLINT
        output = outputs->at(output_index);
        output->set_lod(tensor->lod());
        output_data = output->mutable_data<float>(place);
        output_len = 0;
        CHECK(output->numel() % fea_dim == 0);  // NOLINT
        CHECK(output_data != nullptr);          // NOLINT
      }
      uint64_t real_id = static_cast<uint64_t>(ids[i]);
      if (real_id == padding_id) {
        memcpy(output_data + output_len, init_value.data(),
               sizeof(float) * fea_dim);
        continue;
      }
      fea_keys.push_back(real_id);
      pull_result_ptr.push_back(output_data + output_len);
    }
  }
  auto* communicator = Communicator::GetInstance();
  auto status = communicator->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys.data(), fea_keys.size(),
      is_training);
  status.wait();
  auto ret = status.get();
  if (ret != 0) {
    LOG(ERROR) << "fleet pull sparse failed, status[" << ret << "]";
    sleep(sleep_seconds_before_fail_exit_);
  }
}

void FleetWrapper::PullDenseVarsAsync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names,
    std::vector<std::future<int32_t>>* pull_dense_status, bool in_cpu) {
  auto& regions = _regions[tid];
  regions.clear();
  regions.resize(var_names.size());
  for (auto i = 0u; i < var_names.size(); ++i) {
    std::string varname = var_names[i];
    if (!in_cpu) {
      varname = var_names[i] + "pin";
    }
    Variable* var = scope.FindVar(varname);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::distributed::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }
  auto status = pserver_ptr_->_worker_ptr->pull_dense(regions.data(),
                                                      regions.size(), tid);
  pull_dense_status->push_back(std::move(status));
}

void FleetWrapper::PullDenseVarsSync(
    const Scope& scope, const uint64_t tid,
    const std::vector<std::string>& var_names) {
  auto& regions = _regions[tid];
  regions.clear();
  regions.reserve(var_names.size());
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* w = tensor->data<float>();
    paddle::distributed::Region reg(w, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  auto* communicator = Communicator::GetInstance();
  auto status = communicator->_worker_ptr->pull_dense(regions.data(),
                                                      regions.size(), tid);
  status.wait();
}

void FleetWrapper::PushDenseParamSync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {
  auto place = platform::CPUPlace();
  std::vector<paddle::distributed::Region> regions;
  for (auto& t : var_names) {
    Variable* var = scope.FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    float* g = tensor->mutable_data<float>(place);
    paddle::distributed::Region reg(g, tensor->numel());
    regions.emplace_back(std::move(reg));
  }
  auto* communicator = Communicator::GetInstance();
  auto push_status = communicator->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  CHECK(status == 0) << "push dense param failed, status[" << status << "]";
}

void FleetWrapper::PushDenseVarsSync(
    Scope* scope, const uint64_t table_id,
    const std::vector<std::string>& var_names) {}

void FleetWrapper::PushDenseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<std::string>& var_names,
    std::vector<std::future<int32_t>>* push_sparse_status, float scale_datanorm,
    int batch_size) {
  auto* communicator = Communicator::GetInstance();
  PADDLE_ENFORCE_EQ(
      communicator->Check(table_id), true,
      platform::errors::InvalidArgument(
          "can not find table: %s, please check your config", table_id));
  communicator->Send(var_names, scope);
}

void FleetWrapper::PushSparseVarsAsync(
    const Scope& scope, const uint64_t table_id,
    const std::string& grad_varname,
    std::vector<std::future<int32_t>>* push_sparse_status) {
  std::vector<std::string> varnames;
  varnames.push_back(grad_varname);

  auto* communicator = Communicator::GetInstance();
  PADDLE_ENFORCE_EQ(
      communicator->Check(table_id), true,
      platform::errors::InvalidArgument(
          "can not find table: %s, please check your config", table_id));
  communicator->Send(varnames, scope);
}

void FleetWrapper::PushSparseVarsWithLabelAsync(
    const Scope& scope, const uint64_t table_id,
    const std::vector<uint64_t>& fea_keys, const std::vector<float>& fea_labels,
    const std::vector<std::string>& sparse_key_names,
    const std::vector<std::string>& sparse_grad_names, const int emb_dim,
    std::vector<std::vector<float>>* push_values,
    std::vector<std::future<int32_t>>* push_sparse_status, const int batch_size,
    const bool use_cvm, const bool dump_slot,
    std::vector<uint64_t>* sparse_push_keys, const bool no_cvm) {
  // not support
  return;
}

void FleetWrapper::PushSparseFromTensorWithLabelAsync(
    const Scope& scope, const uint64_t table_id, int fea_dim,
    uint64_t padding_id, bool scale_sparse, const std::string& accesor,
    const std::string& click_name, platform::Place place,
    const std::vector<std::string>& input_names,
    std::vector<const LoDTensor*>* inputs,
    std::vector<const LoDTensor*>* outputs) {
  // not support
  return;
}

void FleetWrapper::LoadModel(const std::string& path, const std::string& mode) {
  auto* communicator = Communicator::GetInstance();
  auto ret = communicator->_worker_ptr->load(path, mode);
  // auto ret = pserver_ptr_->_worker_ptr->load(path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model from path:" << path << " failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
}

void FleetWrapper::LoadModelOneTable(const uint64_t table_id,
                                     const std::string& path, const int mode) {
  auto* communicator = Communicator::GetInstance();
  auto ret =
      communicator->_worker_ptr->load(table_id, path, std::to_string(mode));
  // auto ret =
  //    pserver_ptr_->_worker_ptr->load(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "load model of table id: " << table_id
               << ", from path: " << path << " failed";
  }
}

void FleetWrapper::SaveModel(const std::string& path, const int mode) {
  auto* communicator = Communicator::GetInstance();
  auto ret = communicator->_worker_ptr->save(path, std::to_string(mode));
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {
    LOG(ERROR) << "save model failed";
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
}

void FleetWrapper::SaveModelOneTable(const uint64_t table_id,
                                     const std::string& path, const int mode) {
  auto* communicator = Communicator::GetInstance();
  auto ret =
      communicator->_worker_ptr->save(table_id, path, std::to_string(mode));
  ret.wait();
  if (ret.get() != 0) {
    LOG(ERROR) << "save model of table id: " << table_id
               << ", to path: " << path << " failed";
  }
}

void FleetWrapper::RecvAndSaveTable(const uint64_t table_id,
                                    const std::string& path) {
  auto* communicator = Communicator::GetInstance();
  auto ret = communicator->_worker_ptr->recv_and_save_table(table_id, path);
  if (ret != 0) {
    LOG(ERROR) << "save model of table id: " << table_id
               << ", to path: " << path << " failed";
  }
}

void FleetWrapper::PrintTableStat(const uint64_t table_id) {
  auto* communicator = Communicator::GetInstance();
  auto ret = communicator->_worker_ptr->print_table_stat(table_id);
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "print table stat failed";
  }
}

void FleetWrapper::ShrinkSparseTable(int table_id, int threshold) {
  auto* communicator = Communicator::GetInstance();
  auto ret =
      communicator->_worker_ptr->shrink(table_id, std::to_string(threshold));
  ret.wait();
  int32_t err_code = ret.get();
  if (err_code == -1) {
    LOG(ERROR) << "shrink sparse table stat failed";
  }
}

void FleetWrapper::ClearModel() {
  auto ret = pserver_ptr_->_worker_ptr->clear();
  ret.wait();
}

void FleetWrapper::ClearOneTable(const uint64_t table_id) {
  auto ret = pserver_ptr_->_worker_ptr->clear(table_id);
  ret.wait();
}

void FleetWrapper::ShrinkDenseTable(int table_id, Scope* scope,
                                    std::vector<std::string> var_list,
                                    float decay, int emb_dim) {
  std::vector<paddle::distributed::Region> regions;
  for (std::string& name : var_list) {
    if (name.find("batch_sum") != std::string::npos) {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      VLOG(3) << "prepare shrink dense batch_sum";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      float* g = tensor->data<float>();

      // show_batch_sum += N * log(decay)
      std::string size_name = name;
      size_name.replace(size_name.find("batch_sum"), size_name.length(),
                        "batch_size");
      Variable* var_size = scope->FindVar(size_name);
      CHECK(var_size != nullptr) << "var[" << size_name << "] not found";
      VLOG(3) << "shrink dense batch_sum: " << name << ", " << size_name;
      float* g_size = var_size->GetMutable<LoDTensor>()->data<float>();

      for (int k = 0; k < tensor->numel(); k += emb_dim) {
        g[k] = g[k] + g_size[k] * log(decay);
      }
      paddle::distributed::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    } else {
      Variable* var = scope->FindVar(name);
      CHECK(var != nullptr) << "var[" << name << "] not found";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      float* g = tensor->data<float>();
      paddle::distributed::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }
  }
  auto push_status = pserver_ptr_->_worker_ptr->push_dense_param(
      regions.data(), regions.size(), table_id);
  push_status.wait();
  auto status = push_status.get();
  if (status != 0) {
    // PADDLE_THORW(platform::errors::Fatal(
    //    "push shrink dense param failed, status is [%d].", status));
    sleep(sleep_seconds_before_fail_exit_);
    exit(-1);
  }
}

void FleetWrapper::ClientFlush() {
  auto ret = pserver_ptr_->_worker_ptr->flush();
  ret.wait();
}

int FleetWrapper::RegisterClientToClientMsgHandler(int msg_type,
                                                   MsgHandlerFunc handler) {
  VLOG(3) << "calling FleetWrapper::RegisterClientToClientMsgHandler";
  VLOG(3) << "pserver_ptr_=" << pserver_ptr_;
  VLOG(3) << "_worker_ptr=" << pserver_ptr_->_worker_ptr;
  return pserver_ptr_->_worker_ptr->registe_client2client_msg_handler(msg_type,
                                                                      handler);
}

std::future<int32_t> FleetWrapper::SendClientToClientMsg(
    int msg_type, int to_client_id, const std::string& msg) {
  return pserver_ptr_->_worker_ptr->send_client2client_msg(msg_type,
                                                           to_client_id, msg);
}

std::default_random_engine& FleetWrapper::LocalRandomEngine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;

    engine_wrapper_t() {
      struct timespec tp;
      clock_gettime(CLOCK_REALTIME, &tp);
      double cur_time = tp.tv_sec + tp.tv_nsec * 1e-9;
      static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++, (uint64_t)(cur_time * 1000)};
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

size_t FleetWrapper::GetAbsoluteSum(size_t start, size_t end, size_t level,
                                    const framework::LoD& lod) {
  if (level >= lod.size() - 1) {
    return end - start;
  }
  size_t ret = 0;
  for (size_t i = start; i < end - 1; ++i) {
    size_t pos1 = lod[level][i];
    size_t pos2 = lod[level][i + 1];
    ret += GetAbsoluteSum(pos1, pos2, level + 1, lod);
  }
  return ret;
}

}  // end namespace distributed
}  // end namespace paddle
