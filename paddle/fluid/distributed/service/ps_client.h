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

#pragma once

#include <future>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/distributed/ps.pb.h"
#include "paddle/fluid/distributed/service/env.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/graph/graph_node.h"

namespace paddle {
namespace distributed {

using paddle::distributed::PsRequestMessage;
using paddle::distributed::PsResponseMessage;

typedef std::function<void(void *)> PSClientCallBack;
class PSClientClosure : public google::protobuf::Closure {
 public:
  PSClientClosure(PSClientCallBack callback) : _callback(callback) {}
  virtual ~PSClientClosure() {}
  virtual void set_promise_value(int value) {
    for (auto &promise : _promises) {
      promise->set_value(value);
    }
  }

  void add_promise(std::shared_ptr<std::promise<int32_t>> &promise) {
    _promises.push_back(promise);
  }

 protected:
  PSClientCallBack _callback;
  std::vector<std::shared_ptr<std::promise<int32_t>>> _promises;
};

class PSClient {
 public:
  PSClient() {}
  virtual ~PSClient() {}
  PSClient(PSClient &&) = delete;
  PSClient(const PSClient &) = delete;

  virtual int32_t configure(
      const PSParameter &config,
      const std::map<uint64_t, std::vector<paddle::distributed::Region>>
          &regions,
      PSEnvironment &_env, size_t client_id) final;

  virtual int32_t create_client2client_connection(
      int pserver_timeout_ms, int pserver_connect_timeout_ms,
      int max_retry) = 0;

  // 触发table数据退场
  virtual std::future<int32_t> shrink(uint32_t table_id,
                                      const std::string threshold) = 0;

  // 全量table进行数据load
  virtual std::future<int32_t> load(const std::string &epoch,
                                    const std::string &mode) = 0;
  // 指定table数据load
  virtual std::future<int32_t> load(uint32_t table_id, const std::string &epoch,
                                    const std::string &mode) = 0;
  // 全量table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> save(const std::string &epoch,
                                    const std::string &mode) = 0;
  // 指定table数据save  value_accessor根据mode，可能有不同的save条件
  virtual std::future<int32_t> save(uint32_t table_id, const std::string &epoch,
                                    const std::string &mode) = 0;

  //清空table数据
  virtual std::future<int32_t> clear() = 0;
  virtual std::future<int32_t> clear(uint32_t table_id) = 0;

  // pull dense的参数部分，并分块填充到本地网络参数中
  // start和num用于拉取部分参数
  // future结束前keys和values缓冲区不能再次使用
  // client将values按照区块拆包后送交多个sender
  // sender聚集同一区块的请求，累计多个填充buffer
  // server将参数区块中配置的某一维提取返回
  // 返回数据解包后填充到累计的多个buffer中
  virtual std::future<int32_t> pull_dense(Region *regions, size_t region_num,
                                          size_t table_id) = 0;  //保留

  // firstly push dense param for parameter server
  // this is neccessary because dense weight initialized in trainer on cold
  // start
  virtual std::future<int32_t> push_dense_param(const Region *regions,
                                                size_t region_num,
                                                size_t table_id) = 0;

  // 使用keys进行pull请求，结果填充values
  // keys和values的个数均为num个，每个value占用select_size空间
  // future结束前keys和values缓冲区不能再次使用
  // 整合多个线程请求的keys，聚集并分散发送到server
  // 返回结果后，遍历buffer并对values赋值
  // is_training 用于区分请求是训练/预测，server端对于特征和准入会有不同的处理.
  virtual std::future<int32_t> pull_sparse(float **select_values,
                                           size_t table_id,
                                           const uint64_t *keys, size_t num,
                                           bool is_training) = 0;

  virtual ::std::future<int32_t> pull_sparse_ptr(char **select_values,
                                                 size_t table_id,
                                                 const uint64_t *keys,
                                                 size_t num) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  virtual std::future<int32_t> print_table_stat(uint32_t table_id) = 0;

  // 确保所有积攒中的请求都发起发送
  virtual std::future<int32_t> flush() = 0;
  // server优雅退出
  virtual std::future<int32_t> stop_server() = 0;

  // server profilera
  virtual std::future<int32_t> start_profiler() = 0;
  virtual std::future<int32_t> stop_profiler() = 0;

  virtual std::future<int32_t> barrier(size_t table_id, uint32_t barrier_type,
                                       int timeout_ms) = 0;

  virtual std::future<int32_t> pull_geo_param(size_t table_id,
                                              std::vector<float> *values,
                                              std::vector<uint64_t> *keys,
                                              int pserver_idx) = 0;

  virtual std::future<int32_t> push_global_step(int table_id,
                                                int64_t *total_send_data,
                                                void *done) = 0;

  // recv table from server and save it in LodTensor
  virtual int32_t recv_and_save_table(const uint64_t table_id,
                                      const std::string &path) = 0;

  virtual void finalize_worker() = 0;
  // client to client, 消息发送
  virtual std::future<int32_t> send_client2client_msg(int msg_type,
                                                      int to_client_id,
                                                      const std::string &msg) {
    VLOG(0) << "Did not implement";
    std::promise<int32_t> promise;
    std::future<int> fut = promise.get_future();
    promise.set_value(-1);
    return fut;
  }

  // client2client消息处理，std::function<int32_t (int, int, const std::string&)
  // -> ret (msg_type, from_client_id, msg)
  typedef std::function<int32_t(int, int, const std::string &)> MsgHandlerFunc;
  virtual int registe_client2client_msg_handler(int msg_type,
                                                MsgHandlerFunc handler) {
    _msg_handler_map[msg_type] = handler;
    return 0;
  }
  virtual int handle_client2client_msg(int msg_type, int from_client_id,
                                       const std::string &msg) {
    auto itr = _msg_handler_map.find(msg_type);
    if (itr == _msg_handler_map.end()) {
      LOG(WARNING) << "unknown client2client_msg type:" << msg_type;
      return -1;
    }
    return itr->second(msg_type, from_client_id, msg);
  }

  virtual ValueAccessor *table_accessor(size_t table_id) {
    auto itr = _table_accessors.find(table_id);
    if (itr == _table_accessors.end()) {
      return NULL;
    }
    return itr->second.get();
  }

  virtual size_t get_server_nums() = 0;

  virtual std::future<int32_t> push_dense_raw_gradient(
      int table_id, float *total_send_data, size_t total_send_data_size,
      void *done) = 0;

  virtual std::future<int32_t> push_sparse_raw_gradient(
      size_t table_id, const uint64_t *keys, const float **update_values,
      size_t num, void *done) = 0;

  virtual std::future<int32_t> push_sparse_raw_gradient_partial(
      size_t table_id, const uint64_t *keys, const float **update_values,
      uint32_t num, void *done, int pserver_idx) = 0;

  virtual std::future<int32_t> push_sparse_param(size_t table_id,
                                                 const uint64_t *keys,
                                                 const float **update_values,
                                                 size_t num, void *done) = 0;

 protected:
  virtual int32_t initialize() = 0;
  size_t _client_id;
  PSParameter _config;
  std::map<uint64_t, std::vector<paddle::distributed::Region>>
      _dense_pull_regions;
  PSEnvironment *_env;
  std::unordered_map<uint32_t, std::shared_ptr<ValueAccessor>> _table_accessors;
  std::unordered_map<int32_t, MsgHandlerFunc>
      _msg_handler_map;  //处理client2client消息
};
REGISTER_PSCORE_REGISTERER(PSClient);

class PSClientFactory {
 public:
  static PSClient *create(const PSParameter &config);
};
}  // namespace distributed
}  // namespace paddle
