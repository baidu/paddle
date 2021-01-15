// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/distributed/service/brpc_ps_server.h"

#include <thread>  // NOLINT
#include "Eigen/Dense"
#include "butil/endpoint.h"
#include "iomanip"
#include "paddle/fluid/distributed/table/table.h"
#include "paddle/fluid/framework/archive.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace distributed {

int32_t BrpcPsServer::initialize() {
  auto &service_config = _config.downpour_server_param().service_param();
  if (!service_config.has_service_class()) {
    LOG(ERROR) << "miss service_class in ServerServiceParameter";
    return -1;
  }
  auto *service =
      CREATE_PSCORE_CLASS(PsBaseService, service_config.service_class());
  if (service == NULL) {
    LOG(ERROR) << "service is unregistered, service_name:"
               << service_config.service_class();
    return -1;
  }

  _service.reset(service);
  if (service->configure(this) != 0 || service->initialize() != 0) {
    LOG(ERROR) << "service initialize failed, service_name:"
               << service_config.service_class();
    return -1;
  }
  if (_server.AddService(service, brpc::SERVER_DOESNT_OWN_SERVICE) != 0) {
    LOG(ERROR) << "service add to brpc failed, service:"
               << service_config.service_class();
    return -1;
  }
  return 0;
}

uint64_t BrpcPsServer::start(const std::string &ip, uint32_t port) {
  std::unique_lock<std::mutex> lock(mutex_);

  std::string ip_port = ip + ":" + std::to_string(port);
  VLOG(3) << "server of rank " << _rank << " starts at " << ip_port;
  int num_threads = std::thread::hardware_concurrency();
  brpc::ServerOptions options;
  options.num_threads = num_threads;

  if (_server.Start(ip_port.c_str(), &options) != 0) {
    LOG(ERROR) << "BrpcPsServer start failed, ip_port=" << ip_port;
    return 0;
  }
  VLOG(0) << "BrpcPsServer::start registe_ps_server";
  _environment->registe_ps_server(ip, port, _rank);
  VLOG(0) << "BrpcPsServer::start wait";
  cv_.wait(lock, [&] { return stoped_; });

  PSHost host;
  host.ip = ip;
  host.port = port;
  host.rank = _rank;
  VLOG(0) << "BrpcPsServer::start return host.rank";
  return host.rank;
}

int32_t BrpcPsServer::port() { return _server.listen_address().port; }

int32_t BrpcPsService::initialize() {
  _is_initialize_shard_info = false;
  _service_handler_map[PS_STOP_SERVER] = &BrpcPsService::stop_server;
  _service_handler_map[PS_PULL_DENSE_TABLE] = &BrpcPsService::pull_dense;
  _service_handler_map[PS_PUSH_DENSE_TABLE] = &BrpcPsService::push_dense;
  _service_handler_map[PS_PULL_SPARSE_TABLE] = &BrpcPsService::pull_sparse;
  _service_handler_map[PS_PUSH_SPARSE_TABLE] = &BrpcPsService::push_sparse;
  _service_handler_map[PS_SAVE_ONE_TABLE] = &BrpcPsService::save_one_table;
  _service_handler_map[PS_SAVE_ALL_TABLE] = &BrpcPsService::save_all_table;
  _service_handler_map[PS_SHRINK_TABLE] = &BrpcPsService::shrink_table;
  _service_handler_map[PS_LOAD_ONE_TABLE] = &BrpcPsService::load_one_table;
  _service_handler_map[PS_LOAD_ALL_TABLE] = &BrpcPsService::load_all_table;
  _service_handler_map[PS_CLEAR_ONE_TABLE] = &BrpcPsService::clear_one_table;
  _service_handler_map[PS_CLEAR_ALL_TABLE] = &BrpcPsService::clear_all_table;
  _service_handler_map[PS_PUSH_DENSE_PARAM] = &BrpcPsService::push_dense_param;
  _service_handler_map[PS_PRINT_TABLE_STAT] = &BrpcPsService::print_table_stat;
  _service_handler_map[PS_PULL_GEO_PARAM] = &BrpcPsService::pull_geo_param;
  _service_handler_map[PS_PUSH_SPARSE_PARAM] =
      &BrpcPsService::push_sparse_param;
  _service_handler_map[PS_BARRIER] = &BrpcPsService::barrier;
  _service_handler_map[PS_START_PROFILER] = &BrpcPsService::start_profiler;
  _service_handler_map[PS_STOP_PROFILER] = &BrpcPsService::stop_profiler;
  _service_handler_map[PS_PUSH_GLOBAL_STEP] = &BrpcPsService::push_global_step;

  // shard初始化,server启动后才可从env获取到server_list的shard信息
  initialize_shard_info();

  return 0;
}

#define CHECK_TABLE_EXIST(table, request, response)        \
  if (table == NULL) {                                     \
    std::string err_msg("table not found with table_id:"); \
    err_msg.append(std::to_string(request.table_id()));    \
    set_response_code(response, -1, err_msg.c_str());      \
    return -1;                                             \
  }

int32_t BrpcPsService::initialize_shard_info() {
  if (!_is_initialize_shard_info) {
    std::lock_guard<std::mutex> guard(_initialize_shard_mutex);
    if (_is_initialize_shard_info) {
      return 0;
    }
    size_t shard_num = _server->environment()->get_ps_servers().size();
    auto &table_map = *(_server->table());
    for (auto itr : table_map) {
      itr.second->set_shard(_rank, shard_num);
    }
    _is_initialize_shard_info = true;
  }
  return 0;
}

void BrpcPsService::service(google::protobuf::RpcController *cntl_base,
                            const PsRequestMessage *request,
                            PsResponseMessage *response,
                            google::protobuf::Closure *done) {
  brpc::ClosureGuard done_guard(done);
  std::string log_label("ReceiveCmd-");
  if (!request->has_table_id()) {
    set_response_code(*response, -1, "PsRequestMessage.tabel_id is required");
    return;
  }

  response->set_err_code(0);
  response->set_err_msg("");
  auto *table = _server->table(request->table_id());
  brpc::Controller *cntl = static_cast<brpc::Controller *>(cntl_base);
  auto itr = _service_handler_map.find(request->cmd_id());
  if (itr == _service_handler_map.end()) {
    std::string err_msg(
        "undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
    err_msg.append(std::to_string(request->cmd_id()));
    set_response_code(*response, -1, err_msg.c_str());
    return;
  }
  serviceHandlerFunc handler_func = itr->second;
  int service_ret = (this->*handler_func)(table, *request, *response, cntl);
  if (service_ret != 0) {
    response->set_err_code(service_ret);
    response->set_err_msg("server internal error");
  }
}

int32_t BrpcPsService::pull_dense(Table *table, const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->pull_dense");
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 1) {
    set_response_code(
        response, -1,
        "PsRequestMessage.datas is requeired at least 1 for num of dense");
    return 0;
  }
  uint32_t num = *(const uint32_t *)request.params(0).c_str();
  if (num < 0) {
    set_response_code(response, -1,
                      "PsRequestMessage.datas[0] is invalid, num must >= 0");
    return 0;
  }

  std::vector<float> res_data;
  res_data.resize(num * table->value_accesor()->select_size() / sizeof(float));
  table->pull_dense(res_data.data(), num);

  cntl->response_attachment().append((char *)res_data.data(),
                                     res_data.size() * sizeof(float));

  return 0;
}

int32_t BrpcPsService::push_dense_param(Table *table,
                                        const PsRequestMessage &request,
                                        PsResponseMessage &response,
                                        brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->push_dense_param");
  CHECK_TABLE_EXIST(table, request, response)
  thread_local std::string push_buffer;
  auto &req_io_buffer = cntl->request_attachment();
  auto req_buffer_size = req_io_buffer.size();
  if (req_buffer_size < 1) {
    set_response_code(response, -1, "req attachment is empty");
    return 0;
  }
  push_buffer.resize(0);
  push_buffer.reserve(req_buffer_size);
  const char *data = (const char *)cntl->request_attachment().fetch(
      const_cast<char *>(push_buffer.data()), req_buffer_size);

  uint32_t num = *(const uint32_t *)data;

  const float *values = (const float *)(data + sizeof(uint32_t));
  if (table->push_dense_param(values, num) != 0) {
    set_response_code(response, -1, "push_dense_param failed");
  }
  return 0;
}

int32_t BrpcPsService::push_dense(Table *table, const PsRequestMessage &request,
                                  PsResponseMessage &response,
                                  brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->push_dense");
  CHECK_TABLE_EXIST(table, request, response)
  auto req_buffer_size = request.data().size();
  if (req_buffer_size < 1) {
    // set_response_code(response, 0, "push dense data is empty");
    return 0;
  }

  /*
  Push Content:
  |--num--|---valuesData---|
  |--4B---|----------------|
  */
  uint32_t num = *(const uint32_t *)(request.data().data());
  const float *values =
      (const float *)(request.data().data() + sizeof(uint32_t));
  if (table->push_dense(values, num) != 0) {
    set_response_code(response, -1, "push_dense failed");
  }

  return 0;
}

int32_t BrpcPsService::barrier(Table *table, const PsRequestMessage &request,
                               PsResponseMessage &response,
                               brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)

  if (request.params_size() < 1) {
    set_response_code(response, -1,
                      "PsRequestMessage.params is requeired at "
                      "least 1 for num of sparse_key");
    return 0;
  }

  auto trainer_id = request.client_id();
  auto barrier_type = request.params(0);
  table->barrier(trainer_id, barrier_type);
  return 0;
}

int32_t BrpcPsService::push_sparse_param(Table *table,
                                         const PsRequestMessage &request,
                                         PsResponseMessage &response,
                                         brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->push_sparse_param");
  CHECK_TABLE_EXIST(table, request, response)
  auto &push_data = request.data();
  if (push_data.size() < 1) {
    // set_response_code(response, 0, "push sparse data is empty");
    return 0;
  }
  if (request.params_size() < 1) {
    set_response_code(response, -1,
                      "PsRequestMessage.params is requeired at "
                      "least 1 for num of sparse_key");
    return 0;
  }
  uint32_t num = *(uint32_t *)(request.params(0).c_str());
  /*
  Push Content:
  |---keysData---|---valuesData---|
  |---8*{num}B---|----------------|
  */
  const uint64_t *keys = (const uint64_t *)push_data.data();
  const float *values =
      (const float *)(push_data.data() + sizeof(uint64_t) * num);
  if (table->push_sparse_param(keys, values, num) != 0) {
    set_response_code(response, -1, "push_sparse_param error");
  }
  return 0;
}

int32_t BrpcPsService::pull_geo_param(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->pull_geo_param");
  CHECK_TABLE_EXIST(table, request, response)
  thread_local std::string push_sparse_request_buffer;

  auto trainer_id = request.client_id();

  std::vector<float> values;
  std::vector<uint64_t> ids;
  table->pull_geo_param(trainer_id, &values, &ids);

  uint32_t num = ids.size();
  cntl->response_attachment().append((char *)(&num), sizeof(uint32_t));
  cntl->response_attachment().append((char *)ids.data(),
                                     ids.size() * sizeof(uint64_t));
  cntl->response_attachment().append((char *)values.data(),
                                     values.size() * sizeof(float));
  return 0;
}

int32_t BrpcPsService::pull_sparse(Table *table,
                                   const PsRequestMessage &request,
                                   PsResponseMessage &response,
                                   brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->pull_sparse");
  CHECK_TABLE_EXIST(table, request, response)
  thread_local std::string push_sparse_request_buffer;
  auto &req_io_buffer = cntl->request_attachment();
  auto req_buffer_size = req_io_buffer.size();
  if (req_buffer_size < 1) {
    set_response_code(response, -1, "req attachment is empty");
    return 0;
  }
  if (request.params_size() < 1) {
    set_response_code(response, -1,
                      "PsRequestMessage.params is requeired at "
                      "least 1 for num of sparse_key");
    return 0;
  }
  uint32_t num = *(uint32_t *)(request.params(0).c_str());
  push_sparse_request_buffer.resize(0);
  push_sparse_request_buffer.reserve(req_buffer_size);
  const char *data = (const char *)cntl->request_attachment().fetch(
      const_cast<char *>(push_sparse_request_buffer.data()), req_buffer_size);
  /*
  Attachment Content:
  |---keysData---|
  |---8*{num}B---|
  */
  const uint64_t *keys = (const uint64_t *)data;
  std::vector<float> res_data;
  res_data.resize(num * table->value_accesor()->select_size() / sizeof(float));
  table->pull_sparse(res_data.data(), keys, num);
  cntl->response_attachment().append((char *)res_data.data(),
                                     res_data.size() * sizeof(float));
  return 0;
}

int32_t BrpcPsService::push_sparse(Table *table,
                                   const PsRequestMessage &request,
                                   PsResponseMessage &response,
                                   brpc::Controller *cntl) {
  platform::RecordEvent record_event("PsService->push_sparse");
  CHECK_TABLE_EXIST(table, request, response)
  auto &push_data = request.data();
  if (push_data.size() < 1) {
    // set_response_code(response, 0, "push sparse data is empty");
    return 0;
  }
  if (request.params_size() < 1) {
    set_response_code(response, -1,
                      "PsRequestMessage.params is requeired at "
                      "least 1 for num of sparse_key");
    return 0;
  }
  uint32_t num = *(uint32_t *)(request.params(0).c_str());
  /*
  Push Content:
  |---keysData---|---valuesData---|
  |---8*{num}B---|----------------|
  */
  const uint64_t *keys = (const uint64_t *)push_data.data();
  const float *values =
      (const float *)(push_data.data() + sizeof(uint64_t) * num);
  if (table->push_sparse(keys, values, num) != 0) {
    set_response_code(response, -1, "push_sparse error");
  }
  return 0;
}

int32_t BrpcPsService::print_table_stat(Table *table,
                                        const PsRequestMessage &request,
                                        PsResponseMessage &response,
                                        brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  std::pair<int64_t, int64_t> ret = table->print_table_stat();
  paddle::framework::BinaryArchive ar;
  ar << ret.first << ret.second;
  std::string table_info(ar.Buffer(), ar.Length());
  response.set_data(table_info);

  return 0;
}

int32_t BrpcPsService::load_one_table(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response, -1,
        "PsRequestMessage.datas is requeired at least 2 for path & load_param");
    return -1;
  }
  if (table->load(request.params(0), request.params(1)) != 0) {
    set_response_code(response, -1, "table load failed");
    return -1;
  }
  return 0;
}

int32_t BrpcPsService::load_all_table(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  auto &table_map = *(_server->table());
  for (auto &itr : table_map) {
    if (load_one_table(itr.second.get(), request, response, cntl) != 0) {
      LOG(ERROR) << "load table[" << itr.first << "] failed";
      return -1;
    }
  }
  return 0;
}

int32_t BrpcPsService::save_one_table(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  if (request.params_size() < 2) {
    set_response_code(
        response, -1,
        "PsRequestMessage.datas is requeired at least 2, path&mode");
    return -1;
  }
  table->flush();

  int32_t feasign_size = 0;
  feasign_size = table->save(request.params(0), request.params(1));
  if (feasign_size < 0) {
    set_response_code(response, -1, "table save failed");
    return -1;
  }
  return feasign_size;
}

int32_t BrpcPsService::save_all_table(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  auto &table_map = *(_server->table());
  int32_t all_feasign_size = 0;
  int32_t feasign_size = 0;

  for (auto &itr : table_map) {
    feasign_size = save_one_table(itr.second.get(), request, response, cntl);
    if (feasign_size < 0) {
      LOG(ERROR) << "save table[" << itr.first << "] failed";
      return -1;
    }
  }
  return 0;
}

int32_t BrpcPsService::shrink_table(Table *table,
                                    const PsRequestMessage &request,
                                    PsResponseMessage &response,
                                    brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  table->flush();
  if (table->shrink() != 0) {
    set_response_code(response, -1, "table shrink failed");
  }
  return 0;
}

int32_t BrpcPsService::clear_one_table(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response)
  table->flush();
  table->clear();
  return 0;
}

int32_t BrpcPsService::clear_all_table(Table *table,
                                       const PsRequestMessage &request,
                                       PsResponseMessage &response,
                                       brpc::Controller *cntl) {
  auto &table_map = *(_server->table());
  for (auto &itr : table_map) {
    if (clear_one_table(itr.second.get(), request, response, cntl) != 0) {
      return -1;
    }
  }
  return 0;
}

int32_t BrpcPsService::stop_server(Table *table,
                                   const PsRequestMessage &request,
                                   PsResponseMessage &response,
                                   brpc::Controller *cntl) {
  auto *p_server = _server;
  std::thread t_stop([p_server]() {
    p_server->stop();
    LOG(INFO) << "Server Stoped";
  });
  t_stop.detach();
  return 0;
}

int32_t BrpcPsService::stop_profiler(Table *table,
                                     const PsRequestMessage &request,
                                     PsResponseMessage &response,
                                     brpc::Controller *cntl) {
  platform::DisableProfiler(platform::EventSortingKey::kDefault,
                            string::Sprintf("server_%s_profile", _rank));
  return 0;
}

int32_t BrpcPsService::start_profiler(Table *table,
                                      const PsRequestMessage &request,
                                      PsResponseMessage &response,
                                      brpc::Controller *cntl) {
  platform::EnableProfiler(platform::ProfilerState::kCPU);
  return 0;
}

int32_t BrpcPsService::push_global_step(Table *table,
                                        const PsRequestMessage &request,
                                        PsResponseMessage &response,
                                        brpc::Controller *cntl) {
  CHECK_TABLE_EXIST(table, request, response);
  auto req_buffer_size = request.data().size();
  if (req_buffer_size < 1) {
    set_response_code(response, 0, "run_program data is empty");
    return 0;
  }
  uint32_t num = *(const uint32_t *)(request.data().data());
  const int64_t *values =
      (const int64_t *)(request.data().data() + sizeof(uint32_t));
  auto trainer_id = request.client_id();
  if (table->push_dense(values, trainer_id) != 0) {
    set_response_code(response, -1, "run_program failed");
  }

  return 0;
}

}  // namespace distributed
}  // namespace paddle
