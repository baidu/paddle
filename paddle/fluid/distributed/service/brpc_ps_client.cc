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

#include <memory>
#include <sstream>
#include <string>

#include "paddle/fluid/distributed/service/brpc_ps_client.h"
#include "paddle/fluid/framework/archive.h"

const static int max_port = 65535;

DEFINE_int32(pserver_push_dense_merge_limit, 12,
             "limit max push_dense local merge requests");

DEFINE_int32(pserver_push_sparse_merge_limit, 12,
             "limit max push_sparse local merge requests");

DEFINE_int32(pserver_pull_dense_limit, 12,
             "limit max push_sparse local merge requests");

DEFINE_int32(pserver_async_push_dense_interval_ms, 10,
             "async push_dense to server interval");

DEFINE_int32(pserver_async_push_sparse_interval_ms, 10,
             "async push_sparse to server interval");

DEFINE_bool(pserver_scale_gradient_by_merge, false,
            "scale dense gradient when merged");

DEFINE_int32(pserver_communicate_compress_type, 0,
             "none:0 snappy:1 gzip:2 zlib:3 lz4:4");

DEFINE_int32(pserver_max_async_call_num, 13,
             "max task num in async_call_server");

DEFINE_int32(pserver_timeout_ms, 500000, "pserver request server timeout_ms");

DEFINE_int32(pserver_connect_timeout_ms, 10000,
             "pserver connect server timeout_ms");

DEFINE_int32(pserver_sparse_merge_thread, 1, "pserver sparse merge thread num");

namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace distributed {

inline size_t get_sparse_shard(uint32_t shard_num, uint32_t server_num,
                               uint64_t key) {
  size_t remind = shard_num % server_num;
  size_t local_shard_num =
      remind == 0 ? shard_num / server_num : shard_num / server_num + 1;
  return (key % shard_num) / local_shard_num;
}

void DownpourPsClientService::service(
    ::google::protobuf::RpcController *controller,
    const PsRequestMessage *request, PsResponseMessage *response,
    ::google::protobuf::Closure *done) {
  brpc::ClosureGuard done_guard(done);
  int ret = _client->handle_client2client_msg(
      request->cmd_id(), request->client_id(), request->data());
  response->set_err_code(0);
  response->set_err_msg("");
  if (ret != 0) {
    response->set_err_code(-1);
    response->set_err_msg("handle_client2client_msg failed");
  }
}

// 启动client端RpcService 用于数据互发等操作
int32_t BrpcPsClient::start_client_service() {
  if (_service.configure(this, _client_id) != 0) {
    LOG(ERROR)
        << "service initialize failed, service_name:DownpourPsClientService";
    return -1;
  }
  _server.AddService(&_service, brpc::SERVER_DOESNT_OWN_SERVICE);
  brpc::ServerOptions options;
  int start_port = 8500;
  options.num_threads = 24;

  if (_server.Start(butil::my_ip_cstr(), brpc::PortRange(start_port, max_port),
                    &options) != 0) {
    LOG(ERROR) << "BrpcPsServer start failed";
    return -1;
  }
  _env->registe_ps_client(butil::my_ip_cstr(), _server.listen_address().port,
                          _client_id);
  return 0;
}

int32_t BrpcPsClient::create_client2client_connection(
    int pserver_timeout_ms, int pserver_connect_timeout_ms, int max_retry) {
  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = pserver_timeout_ms;
  options.connection_type = "pooled";
  options.connect_timeout_ms = pserver_connect_timeout_ms;
  options.max_retry = max_retry;

  std::vector<PSHost> client_list = _env->get_ps_clients();
  _client_channels.resize(client_list.size());
  std::ostringstream os;
  std::string server_ip_port;
  for (size_t i = 0; i < client_list.size(); ++i) {
    server_ip_port.assign(client_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(client_list[i].port));
    _client_channels[i].reset(new brpc::Channel());
    if (_client_channels[i]->Init(server_ip_port.c_str(), "", &options) != 0) {
      VLOG(0) << "BrpcPSClient connect to Client:" << server_ip_port
              << " Failed! Try again.";
      std::string int_ip_port =
          GetIntTypeEndpoint(client_list[i].ip, client_list[i].port);
      if (_client_channels[i]->Init(int_ip_port.c_str(), "", &options) != 0) {
        LOG(ERROR) << "BrpcPSClient connect to Client:" << int_ip_port
                   << " Failed!";
        return -1;
      }
    }
    os << server_ip_port << ",";
  }
  LOG(INFO) << "Client connect success:" << os.str();
  return 0;
}

int32_t BrpcPsClient::initialize() {
  _async_call_num = 0;

  brpc::ChannelOptions options;
  options.protocol = "baidu_std";
  options.timeout_ms = FLAGS_pserver_timeout_ms;
  options.connection_type = "pooled";
  options.connect_timeout_ms = FLAGS_pserver_connect_timeout_ms;
  options.max_retry = 3;

  std::ostringstream os;
  std::string server_ip_port;
  std::string client_ip(butil::my_ip_cstr());

  // 获取server列表，并连接
  std::vector<PSHost> server_list = _env->get_ps_servers();
  _server_channels.resize(server_list.size());
  for (size_t i = 0; i < server_list.size(); ++i) {
    server_ip_port.assign(server_list[i].ip.c_str());
    server_ip_port.append(":");
    server_ip_port.append(std::to_string(server_list[i].port));
    for (size_t j = 0; j < _server_channels[i].size(); ++j) {
      _server_channels[i][j].reset(new brpc::Channel());
      if (_server_channels[i][j]->Init(server_ip_port.c_str(), "", &options) !=
          0) {
        VLOG(0) << "BrpcPSclient connect to Server:" << server_ip_port
                << " Failed! Try again.";
        std::string int_ip_port =
            GetIntTypeEndpoint(server_list[i].ip, server_list[i].port);
        if (_server_channels[i][j]->Init(int_ip_port.c_str(), "", &options) !=
            0) {
          LOG(ERROR) << "BrpcPSclient connect to Server:" << int_ip_port
                     << " Failed!";
          return -1;
        }
      }
    }
    os << server_ip_port << ",";
  }
  // 启动client探听接口, 并相互建立连接
  start_client_service();

  _running = true;
  _flushing = false;
  return 0;
}

int DownpourBrpcClosure::check_response(size_t request_idx, int cmd_id) {
  if (_cntls[request_idx]->Failed()) {
    LOG(ERROR) << "resquest cmd_id:" << cmd_id << " failed, "
                                                  "err:"
               << _cntls[request_idx]->ErrorText();
    return -1;
  }
  if (_responses[request_idx].err_code() != 0) {
    LOG(ERROR) << "response ret bad, server_idx:" << request_idx
               << "cmd_id:" << cmd_id
               << " err_code:" << _responses[request_idx].err_code()
               << " err_msg:" << _responses[request_idx].err_msg();
    return -1;
  }
  return 0;
}

int DownpourBrpcClosure::check_save_response(size_t request_idx, int cmd_id) {
  uint32_t feasign_size = 0;
  if (_cntls[request_idx]->Failed()) {
    LOG(ERROR) << "resquest cmd_id:" << cmd_id << " failed, "
                                                  "err:"
               << _cntls[request_idx]->ErrorText();
    return -1;
  }
  feasign_size = _responses[request_idx].err_code();
  if (feasign_size < 0) {
    LOG(ERROR) << "response ret bad, server_idx:" << request_idx
               << "cmd_id:" << cmd_id
               << " err_code:" << _responses[request_idx].err_code()
               << " err_msg:" << _responses[request_idx].err_msg();
    return -1;
  }
  return feasign_size;
}

std::string DownpourBrpcClosure::get_response(size_t request_idx, int cmd_id) {
  std::string data = _responses[request_idx].data();
  return data;
}

std::future<int32_t> BrpcPsClient::print_table_stat(uint32_t table_id) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, table_id](void *done) {
        int ret = 0;
        uint64_t feasign_size = 0;
        uint64_t mf_size = 0;
        paddle::framework::BinaryArchive ar;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PRINT_TABLE_STAT) != 0) {
            ret = -1;
            break;
          }
          std::string resp = closure->get_response(i, PS_PRINT_TABLE_STAT);
          ar.SetReadBuffer(const_cast<char *>(resp.c_str()), resp.length(),
                           nullptr);

          feasign_size += ar.Get<uint64_t>();
          mf_size += ar.Get<uint64_t>();
        }
        closure->set_promise_value(ret);
        std::cout << "table id: " << table_id
                  << ", feasign size: " << feasign_size
                  << ", mf size: " << mf_size << std::endl;
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PRINT_TABLE_STAT);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    PsService_Stub rpc_stub(get_cmd_channel(i));
    closure->cntl(i)->set_timeout_ms(
        10800000);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}
std::future<int32_t> BrpcPsClient::send_cmd(
    uint32_t table_id, int cmd_id, const std::vector<std::string> &params,
    int timeout_ms) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, cmd_id](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, cmd_id) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    for (const auto &param : params) {
      closure->request(i)->add_params(param);
    }
    PsService_Stub rpc_stub(get_cmd_channel(i));
    closure->cntl(i)->set_timeout_ms(
        timeout_ms);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::send_save_cmd(
    uint32_t table_id, int cmd_id, const std::vector<std::string> &params) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, cmd_id](void *done) {
        int ret = 0;
        uint32_t feasign_size = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_save_response(i, cmd_id) < 0) {
            ret = -1;
            break;
          }
          feasign_size += closure->check_save_response(i, cmd_id);
        }
        if (ret == 0) {
          closure->set_promise_value(feasign_size);
        } else {
          closure->set_promise_value(ret);
        }
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(cmd_id);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    for (const auto &param : params) {
      closure->request(i)->add_params(param);
    }
    PsService_Stub rpc_stub(get_cmd_channel(i));
    closure->cntl(i)->set_timeout_ms(
        10800000);  // cmd msg don't limit timeout for save/load
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::shrink(uint32_t table_id,
                                          const std::string threshold) {
  return send_cmd(table_id, PS_SHRINK_TABLE, {threshold});
}

std::future<int32_t> BrpcPsClient::load(const std::string &epoch,
                                        const std::string &mode) {
  return send_cmd(-1, PS_LOAD_ALL_TABLE, {epoch, mode});
}
std::future<int32_t> BrpcPsClient::load(uint32_t table_id,
                                        const std::string &epoch,
                                        const std::string &mode) {
  return send_cmd(table_id, PS_LOAD_ONE_TABLE, {epoch, mode});
}

std::future<int32_t> BrpcPsClient::save(const std::string &epoch,
                                        const std::string &mode) {
  return send_save_cmd(-1, PS_SAVE_ALL_TABLE, {epoch, mode});
}
std::future<int32_t> BrpcPsClient::save(uint32_t table_id,
                                        const std::string &epoch,
                                        const std::string &mode) {
  return send_save_cmd(table_id, PS_SAVE_ONE_TABLE, {epoch, mode});
}

std::future<int32_t> BrpcPsClient::clear() {
  return send_cmd(-1, PS_CLEAR_ALL_TABLE, {});
}
std::future<int32_t> BrpcPsClient::clear(uint32_t table_id) {
  return send_cmd(table_id, PS_CLEAR_ONE_TABLE, {});
}

std::future<int32_t> BrpcPsClient::flush() {
  _flushing = true;
  std::promise<int> promise;
  std::future<int32_t> fut = promise.get_future();
  do {
    VLOG(3) << "wait _async_call_num:" << _async_call_num;
    usleep(100000);  // sleep 100ms wait async end
  } while (_async_call_num > 0);
  promise.set_value(0);
  _flushing = false;
  return fut;
}

void BrpcPsClient::finalize_worker() {
  flush();
  _running = false;
  _server.Stop(1000);
  _server.Join();
}

std::future<int32_t> BrpcPsClient::stop_server() {
  return send_cmd(-1, PS_STOP_SERVER, {});
}

std::future<int32_t> BrpcPsClient::start_profiler() {
  return send_cmd(-1, PS_START_PROFILER, {});
}

std::future<int32_t> BrpcPsClient::stop_profiler() {
  return send_cmd(-1, PS_STOP_PROFILER, {});
}

std::future<int32_t> BrpcPsClient::barrier(size_t table_id,
                                           uint32_t barrier_type,
                                           int timeout_ms) {
  return send_cmd(table_id, PS_BARRIER, {std::to_string(barrier_type)},
                  timeout_ms);
}

std::future<int32_t> BrpcPsClient::pull_geo_param(size_t table_id,
                                                  std::vector<float> *values,
                                                  std::vector<uint64_t> *keys,
                                                  int pserver_idx) {
  auto *accessor = table_accessor(table_id);
  DownpourBrpcClosure *closure =
      new DownpourBrpcClosure(1, [keys, values, accessor](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        uint32_t shard_nums;
        if (closure->check_response(0, PS_PULL_GEO_PARAM) != 0) {
          ret = -1;
        }
        auto &res_io_buffer = closure->cntl(0)->response_attachment();
        butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
        io_buffer_itr.copy_and_forward((void *)(&shard_nums), sizeof(uint32_t));
        keys->resize(shard_nums);
        values->resize(shard_nums * accessor->update_dim());
        io_buffer_itr.copy_and_forward((void *)(keys->data()),
                                       sizeof(uint64_t) * shard_nums);
        io_buffer_itr.copy_and_forward((void *)(values->data()),
                                       shard_nums * accessor->update_size());
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  closure->request(0)->set_cmd_id(PS_PULL_GEO_PARAM);
  closure->request(0)->set_table_id(table_id);
  closure->request(0)->set_client_id(_client_id);
  PsService_Stub rpc_stub(get_cmd_channel(pserver_idx));
  closure->cntl(0)->set_log_id(butil::gettimeofday_ms());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}

std::future<int32_t> BrpcPsClient::push_sparse_param(
    size_t table_id, const uint64_t *keys, const float **update_values,
    size_t num, void *done) {
  auto *accessor = table_accessor(table_id);
  // 发送RPC请求
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  size_t request_call_num = _server_channels.size();
  std::vector<std::vector<uint64_t>> ids;
  std::vector<std::vector<const float *>> value_ptrs;
  ids.resize(request_call_num);
  value_ptrs.resize(request_call_num);
  for (size_t i = 0; i < num; ++i) {
    size_t pserver_idx = keys[i] % request_call_num;
    ids[pserver_idx].push_back(keys[i]);
    value_ptrs[pserver_idx].push_back(update_values[i]);
  }
  for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
    auto kvs = ids[shard_idx];
    auto value_ptr = value_ptrs[shard_idx];
    size_t kv_size = kvs.size();
    uint32_t value_size = accessor->update_size();
    // 发送RPC请求
    auto *push_request = closure->request(shard_idx);
    push_request->set_cmd_id(PS_PUSH_SPARSE_PARAM);
    push_request->set_table_id(table_id);
    push_request->set_client_id(_client_id);
    push_request->add_params((char *)&kv_size, sizeof(uint32_t));
    auto *push_data = push_request->mutable_data();
    push_data->resize(kv_size * (sizeof(uint64_t) + accessor->update_size()));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, kvs.data(), kv_size * sizeof(uint64_t));
    push_data_ptr += kv_size * sizeof(uint64_t);
    for (int i = 0; i < kv_size; ++i) {
      memcpy(push_data_ptr, value_ptr[i], accessor->update_size());
      push_data_ptr += accessor->update_size();
    }
    PsService_Stub rpc_stub(get_sparse_channel(shard_idx));
    closure->cntl(shard_idx)->set_request_compress_type(
        (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    rpc_stub.service(closure->cntl(shard_idx), closure->request(shard_idx),
                     closure->response(shard_idx), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::pull_dense(Region *regions,
                                              size_t region_num,
                                              size_t table_id) {
  auto *accessor = table_accessor(table_id);
  size_t request_call_num = _server_channels.size();
  uint32_t num_per_shard =
      dense_dim_per_shard(accessor->fea_dim(), request_call_num);
  // callback 将各shard结果，顺序填入region
  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [request_call_num, num_per_shard, regions, region_num,
                         accessor](void *done) {
        int ret = 0;
        size_t region_idx = 0;       // 当前填充的region偏移
        size_t region_data_idx = 0;  // 当前填充的region内data偏移
        auto *closure = (DownpourBrpcClosure *)done;
        size_t shard_data_size = num_per_shard * accessor->select_size();
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PULL_DENSE_TABLE) != 0) {
            ret = -1;
            break;
          }
          auto &res_io_buffer = closure->cntl(i)->response_attachment();

          butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
          size_t shard_buffer_remain = res_io_buffer.size();
          if (shard_buffer_remain != shard_data_size) {
            LOG(ERROR) << "expect res_size:" << shard_data_size
                       << ", but size:" << shard_buffer_remain
                       << ", ignore this response";
            ret = -1;
            break;
          }
          while (shard_buffer_remain > 0 && region_idx < region_num) {
            auto &region = regions[region_idx];
            if (region.size - region_data_idx >= shard_buffer_remain) {
              // region待填充空间 >= 分片buffer数据, 直接拷贝置入
              io_buffer_itr.copy_and_forward(
                  (void *)(region.data + region_data_idx), shard_buffer_remain);
              region_data_idx += shard_buffer_remain;
              shard_buffer_remain = 0;
            } else if (region.size - region_data_idx == 0) {
              // region填满，切换到下一个region
              ++region_idx;
              region_data_idx = 0;
            } else {
              // region不足以容纳所有数据，则能放多少 拷贝多少
              io_buffer_itr.copy_and_forward(
                  (void *)(region.data + region_data_idx),
                  region.size - region_data_idx);
              shard_buffer_remain -= (region.size - region_data_idx);
              ++region_idx;
              region_data_idx = 0;
            }
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PULL_DENSE_TABLE);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    closure->request(i)->add_params((char *)&num_per_shard,
                                    sizeof(num_per_shard));
    PsService_Stub rpc_stub(get_dense_channel(i));
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::push_dense_param(const Region *regions,
                                                    size_t region_num,
                                                    size_t table_id) {
  auto *accessor = table_accessor(table_id);
  size_t request_call_num = _server_channels.size();
  // 1.拆分Region数据到shard中，后续多shard并行拷贝数据
  std::vector<std::vector<Region>> regions_partition(request_call_num);
  uint32_t num_per_shard =
      dense_dim_per_shard(accessor->fea_dim(), request_call_num);
  size_t shard_data_size = num_per_shard * accessor->update_size();
  size_t current_region_idx = 0;
  size_t current_region_data_idx = 0;
  for (size_t i = 0; i < request_call_num; ++i) {
    size_t shard_data_remain_size = shard_data_size;
    while (shard_data_remain_size > 0 && current_region_idx < region_num) {
      const auto &region = regions[current_region_idx];
      size_t region_remain_size = region.size - current_region_data_idx;
      if (shard_data_remain_size >= region_remain_size) {
        regions_partition[i].push_back(
            Region(region.data + current_region_data_idx, region_remain_size));
        ++current_region_idx;
        current_region_data_idx = 0;
        shard_data_remain_size -= region_remain_size;
      } else {
        regions_partition[i].push_back(Region(
            region.data + current_region_data_idx, shard_data_remain_size));
        current_region_data_idx += shard_data_remain_size;
        shard_data_remain_size = 0;
      }
    }
  }

  DownpourBrpcClosure *closure =
      new DownpourBrpcClosure(request_call_num, [request_call_num](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < request_call_num; ++i) {
          if (closure->check_response(i, PS_PUSH_DENSE_PARAM) != 0) {
            ret = -1;
            break;
          }
        }
        closure->set_promise_value(ret);
      });
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  static const int REGION_ASSIGN_BUFFER_SIZE = 1024 * 10;
  static char region_assign_buffer[REGION_ASSIGN_BUFFER_SIZE];  //用于数据补齐
  //开始多shard并行拷贝&请求
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_DENSE_PARAM);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto &request_buffer = closure->cntl(i)->request_attachment();
    request_buffer.append((void *)&num_per_shard, sizeof(uint32_t));
    auto &region_list = regions_partition[i];
    size_t fill_remain_size = shard_data_size;
    for (auto &region : region_list) {
      fill_remain_size -= region.size;
      request_buffer.append((void *)region.data, region.size);
    }
    //保证各分片数据对齐
    while (fill_remain_size > 0) {
      size_t fill_num = fill_remain_size > REGION_ASSIGN_BUFFER_SIZE
                            ? REGION_ASSIGN_BUFFER_SIZE
                            : fill_remain_size;
      request_buffer.append((void *)region_assign_buffer, fill_num);
      fill_remain_size -= fill_num;
    }
    PsService_Stub rpc_stub(get_dense_channel(i));
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::push_sparse_raw_gradient(
    size_t table_id, const uint64_t *keys, const float **update_values,
    size_t num, void *done) {
  auto *accessor = table_accessor(table_id);
  //发送RPC请求
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  size_t request_call_num = _server_channels.size();
  std::vector<std::vector<uint64_t>> ids;
  std::vector<std::vector<const float *>> value_ptrs;
  ids.resize(request_call_num);
  value_ptrs.resize(request_call_num);

  for (size_t i = 0; i < num; ++i) {
    size_t pserver_idx = keys[i] % request_call_num;
    ids[pserver_idx].push_back(keys[i]);
    value_ptrs[pserver_idx].push_back(update_values[i]);
  }

  for (size_t shard_idx = 0; shard_idx < request_call_num; ++shard_idx) {
    auto kvs = ids[shard_idx];
    auto value_ptr = value_ptrs[shard_idx];

    size_t kv_size = kvs.size();
    uint32_t value_size = accessor->update_size();

    // 发送RPC请求
    auto *push_request = closure->request(shard_idx);
    push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
    push_request->set_table_id(table_id);
    push_request->set_client_id(_client_id);
    push_request->add_params((char *)&kv_size, sizeof(uint32_t));
    auto *push_data = push_request->mutable_data();
    push_data->resize(kv_size * (sizeof(uint64_t) + accessor->update_size()));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, kvs.data(), kv_size * sizeof(uint64_t));
    push_data_ptr += kv_size * sizeof(uint64_t);

    for (int i = 0; i < kv_size; ++i) {
      memcpy(push_data_ptr, value_ptr[i], accessor->update_size());
      push_data_ptr += accessor->update_size();
    }
    PsService_Stub rpc_stub(get_sparse_channel(shard_idx));
    closure->cntl(shard_idx)->set_request_compress_type(
        (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    rpc_stub.service(closure->cntl(shard_idx), closure->request(shard_idx),
                     closure->response(shard_idx), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::push_dense_raw_gradient(
    int table_id, float *total_send_data, size_t total_send_data_size,
    void *done) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  auto *accessor = table_accessor(table_id);
  uint32_t num_per_shard =
      dense_dim_per_shard(accessor->fea_dim(), request_call_num);
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_DENSE_TABLE);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto *push_data = closure->request(i)->mutable_data();
    push_data->clear();
    push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(float));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
    memcpy(push_data_ptr + sizeof(uint32_t),
           total_send_data + i * num_per_shard, num_per_shard * sizeof(float));
    VLOG(1) << "push_dense_raw_gradient finish memcpy";
    // closure->cntl(i)->set_request_compress_type(
    //     (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
    PsService_Stub rpc_stub(get_dense_channel(i));
    VLOG(1) << "push_dense_raw_gradient get_dense_channel " << i;
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
    VLOG(1) << "push_dense_raw_gradient async service " << i;
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::push_global_step(int table_id,
                                                    int64_t *total_send_data,
                                                    void *done) {
  size_t request_call_num = _server_channels.size();
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();
  for (size_t i = 0; i < request_call_num; ++i) {
    closure->request(i)->set_cmd_id(PS_PUSH_GLOBAL_STEP);
    closure->request(i)->set_table_id(table_id);
    closure->request(i)->set_client_id(_client_id);
    auto *push_data = closure->request(i)->mutable_data();
    push_data->clear();
    int32_t num_per_shard = 1;
    push_data->resize(sizeof(uint32_t) + num_per_shard * sizeof(int64_t));
    char *push_data_ptr = const_cast<char *>(push_data->data());
    memcpy(push_data_ptr, &num_per_shard, sizeof(uint32_t));
    memcpy(push_data_ptr + sizeof(uint32_t), total_send_data,
           num_per_shard * sizeof(int64_t));

    PsService_Stub rpc_stub(get_dense_channel(i));
    rpc_stub.service(closure->cntl(i), closure->request(i),
                     closure->response(i), closure);
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::pull_sparse(float **select_values,
                                               size_t table_id,
                                               const uint64_t *keys, size_t num,
                                               bool is_training) {
  size_t request_call_num = _server_channels.size();

  auto shard_sorted_kvs = std::make_shared<
      std::vector<std::vector<std::pair<uint64_t, float *>>>>();
  shard_sorted_kvs->resize(request_call_num);

  for (size_t i = 0; i < num; ++i) {
    size_t shard_id = keys[i] % request_call_num;
    shard_sorted_kvs->at(shard_id).push_back({keys[i], select_values[i]});
  }

  auto *accessor = table_accessor(table_id);
  size_t value_size = accessor->select_size();

  DownpourBrpcClosure *closure = new DownpourBrpcClosure(
      request_call_num, [shard_sorted_kvs, value_size](void *done) {
        int ret = 0;
        auto *closure = (DownpourBrpcClosure *)done;
        for (size_t i = 0; i < shard_sorted_kvs->size(); ++i) {
          if (closure->check_response(i, PS_PULL_SPARSE_TABLE) != 0) {
            ret = -1;
            break;
          }

          auto &request_kvs = shard_sorted_kvs->at(i);
          auto &res_io_buffer = closure->cntl(i)->response_attachment();
          butil::IOBufBytesIterator io_buffer_itr(res_io_buffer);
          uint64_t last_key = UINT64_MAX;
          float *last_value_data = NULL;

          for (size_t kv_idx = 0; kv_idx < request_kvs.size(); ++kv_idx) {
            auto *kv_pair = &(request_kvs[kv_idx]);
            if (kv_pair->first == last_key) {
              memcpy((void *)kv_pair->second, (void *)last_value_data,
                     value_size);
            } else {
              last_key = kv_pair->first;
              last_value_data = kv_pair->second;
              if (value_size !=
                  io_buffer_itr.copy_and_forward((void *)(last_value_data),
                                                 value_size)) {
                LOG(WARNING) << "res data is lack or not in format";
                ret = -1;
                break;
              }
            }
          }
        }
        closure->set_promise_value(ret);
      });

  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  for (size_t i = 0; i < request_call_num; ++i) {
    auto &sorted_kvs = shard_sorted_kvs->at(i);
    std::sort(sorted_kvs.begin(), sorted_kvs.end(),
              [](const std::pair<uint64_t, float *> &k1,
                 const std::pair<uint64_t, float *> &k2) {
                return k1.first < k2.first;
              });

    uint64_t last_key = UINT64_MAX;
    uint32_t kv_request_count = 0;
    size_t sorted_kv_size = sorted_kvs.size();
    auto &request_buffer = closure->cntl(i)->request_attachment();

    request_buffer.append((void *)&is_training, sizeof(bool));
    std::vector<uint32_t> keys_counter;
    keys_counter.reserve(sorted_kv_size);

    for (size_t kv_idx = 0; kv_idx < sorted_kv_size; ++kv_idx) {
      ++kv_request_count;
      uint32_t keys = 1;
      last_key = sorted_kvs[kv_idx].first;
      request_buffer.append((void *)&last_key, sizeof(uint64_t));
      while (kv_idx < sorted_kv_size - 1 &&
             last_key == sorted_kvs[kv_idx + 1].first) {
        ++kv_idx;
        ++keys;
      }
      keys_counter.push_back(keys);
    }

    request_buffer.append((void *)keys_counter.data(),
                          sizeof(uint32_t) * keys_counter.size());

    if (kv_request_count == 0) {
      closure->Run();
    } else {
      closure->request(i)->set_cmd_id(PS_PULL_SPARSE_TABLE);
      closure->request(i)->set_table_id(table_id);
      closure->request(i)->set_client_id(_client_id);
      closure->request(i)->add_params((char *)&kv_request_count,
                                      sizeof(uint32_t));
      PsService_Stub rpc_stub(get_cmd_channel(i));
      closure->cntl(i)->set_log_id(butil::gettimeofday_ms());
      rpc_stub.service(closure->cntl(i), closure->request(i),
                       closure->response(i), closure);
    }
  }
  return fut;
}

std::future<int32_t> BrpcPsClient::send_client2client_msg(
    int msg_type, int to_client_id, const std::string &msg) {
  auto promise = std::make_shared<std::promise<int32_t>>();
  std::future<int> fut = promise->get_future();
  if (to_client_id >= _client_channels.size()) {
    VLOG(0) << "to_client_id is out of range clients, which size is "
            << _client_channels.size();
    promise->set_value(-1);
    return fut;
  }
  auto *closure = new DownpourBrpcClosure(1, [msg_type](void *done) {
    auto *closure = (DownpourBrpcClosure *)done;
    int32_t ret = closure->check_response(0, msg_type + 1000);
    closure->set_promise_value(ret);
  });
  closure->add_promise(promise);
  closure->request(0)->set_cmd_id(msg_type);
  closure->request(0)->set_client_id(_client_id);
  closure->request(0)->set_data(msg);
  PsService_Stub rpc_stub(_client_channels[to_client_id].get());
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}

std::future<int32_t> BrpcPsClient::push_sparse_raw_gradient_partial(
    size_t table_id, const uint64_t *keys, const float **update_values,
    uint32_t num, void *done, int pserver_idx) {
  auto *accessor = table_accessor(table_id);
  size_t value_size = accessor->update_size();
  DownpourBrpcClosure *closure = reinterpret_cast<DownpourBrpcClosure *>(done);
  auto promise = std::make_shared<std::promise<int32_t>>();
  closure->add_promise(promise);
  std::future<int> fut = promise->get_future();

  // 发送RPC请求
  auto *push_request = closure->request(0);
  push_request->set_cmd_id(PS_PUSH_SPARSE_TABLE);
  push_request->set_table_id(table_id);
  push_request->set_client_id(_client_id);
  push_request->add_params((char *)&num, sizeof(uint32_t));
  auto *push_data = push_request->mutable_data();
  push_data->resize(num * (sizeof(uint64_t) + value_size));
  char *push_data_ptr = const_cast<char *>(push_data->data());
  memcpy(push_data_ptr, keys, num * sizeof(uint64_t));
  push_data_ptr += num * sizeof(uint64_t);
  for (int i = 0; i < num; ++i) {
    memcpy(push_data_ptr, update_values[i], value_size);
    push_data_ptr += value_size;
  }
  PsService_Stub rpc_stub(get_sparse_channel(pserver_idx));
  closure->cntl(0)->set_request_compress_type(
      (brpc::CompressType)FLAGS_pserver_communicate_compress_type);
  rpc_stub.service(closure->cntl(0), closure->request(0), closure->response(0),
                   closure);
  return fut;
}

int32_t BrpcPsClient::recv_and_save_table(const uint64_t table_id,
                                          const std::string &path) {
  // get var information
  std::string var_name = "";
  int64_t var_num = 0;
  int64_t var_shape = 0;
  const auto &worker_param = _config.worker_param().downpour_worker_param();
  for (size_t i = 0; i < worker_param.downpour_table_param_size(); ++i) {
    if (worker_param.downpour_table_param(i).table_id() == table_id) {
      var_name = worker_param.downpour_table_param(i).common().table_name();
      var_num = worker_param.downpour_table_param(i).accessor().fea_dim();
      var_shape = worker_param.downpour_table_param(i).accessor().embedx_dim();
      break;
    }
  }

  PADDLE_ENFORCE_NE(
      var_name, "",
      platform::errors::InvalidArgument(
          "Cannot find table id %d to save variables.", table_id));

  std::string var_store = string::Sprintf("%s", path);
  MkDirRecursively(var_store.c_str());

  // pull sparse from server
  std::vector<float> save_huge_vec(var_num * var_shape);
  std::vector<uint64_t> save_key(var_num);
  std::vector<float *> save_vec;
  for (size_t i = 0; i < save_key.size(); ++i) {
    save_key[i] = i;
    save_vec.push_back(save_huge_vec.data() + i * var_shape);
  }

  auto status = pull_sparse((float **)save_vec.data(), table_id,
                            save_key.data(), save_key.size(), true);
  status.wait();

  // create lod tensor
  std::shared_ptr<framework::Scope> scope;
  scope.reset(new framework::Scope());
  auto place = platform::CPUPlace();
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  framework::Variable *var = scope->Var(var_name);
  framework::LoDTensor *var_tensor = var->GetMutable<framework::LoDTensor>();

  std::vector<int64_t> vec_dim = {var_num, var_shape};
  var_tensor->Resize(framework::make_ddim(vec_dim));

  // copy and save
  float *tensor_data = var_tensor->mutable_data<float>(place);
  memcpy(tensor_data, save_huge_vec.data(),
         var_num * var_shape * sizeof(float));

  std::string file_name = string::Sprintf("%s/%s", var_store, var_name);
  std::ofstream fout(file_name, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                    platform::errors::Unavailable(
                        "Cannot open %s to save variables.", file_name));

  framework::SerializeToStream(fout, *var_tensor, dev_ctx);
  fout.close();

  return 0;
}

}  // namespace distributed
}  // namespace paddle
