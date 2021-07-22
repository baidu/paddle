// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License 0//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/service/ps_client.h"

namespace paddle {
namespace distributed {

class Table;

class PsLocalClient : public PSClient {
 public:
  PsLocalClient() {}
  virtual ~PsLocalClient() { _running = false; }
  virtual int32_t create_client2client_connection(int pslib_timeout_ms,
                                                  int pslib_connect_timeout_ms,
                                                  int max_retry) {
    return 0;
  }

  virtual ::std::future<int32_t> shrink(uint32_t table_id,
                                        const std::string threshold) override;
  virtual ::std::future<int32_t> load(const std::string& epoch,
                                      const std::string& mode) override;
  virtual ::std::future<int32_t> load(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode) override;

  virtual ::std::future<int32_t> save(const std::string& epoch,
                                      const std::string& mode) override;
  virtual ::std::future<int32_t> save(uint32_t table_id,
                                      const std::string& epoch,
                                      const std::string& mode) override;

  virtual ::std::future<int32_t> clear() override;
  virtual ::std::future<int32_t> clear(uint32_t table_id) override;

  virtual ::std::future<int32_t> stop_server() override;

  virtual void finalize_worker() override {}
  virtual ::std::future<int32_t> pull_dense(Region* regions, size_t region_num,
                                            size_t table_id);

  virtual ::std::future<int32_t> push_dense(const Region* regions,
                                            size_t region_num, size_t table_id);

  virtual ::std::future<int32_t> push_dense_param(const Region* regions,
                                                  size_t region_num,
                                                  size_t table_id);

  virtual ::std::future<int32_t> pull_sparse(float** select_values,
                                             size_t table_id,
                                             const uint64_t* keys, size_t num,
                                             bool is_training) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual ::std::future<int32_t> pull_sparse_ptr(char** select_values,
                                                 size_t table_id,
                                                 const uint64_t* keys,
                                                 size_t num);

  virtual ::std::future<int32_t> print_table_stat(uint32_t table_id) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }
  virtual ::std::future<int32_t> push_sparse(size_t table_id,
                                             const uint64_t* keys,
                                             const float** update_values,
                                             size_t num);

  virtual ::std::future<int32_t> flush();
  // server profilera
  virtual std::future<int32_t> start_profiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  };

  virtual std::future<int32_t> stop_profiler() {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> barrier(size_t table_id, uint32_t barrier_type,
                                       int timeout_ms) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> pull_geo_param(size_t table_id,
                                              std::vector<float>* values,
                                              std::vector<uint64_t>* keys,
                                              int pserver_idx) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> push_global_step(int table_id,
                                                int64_t* total_send_data,
                                                void* done) {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  // recv table from server and save it in LodTensor
  virtual int32_t recv_and_save_table(const uint64_t table_id,
                                      const std::string& path) {
    return 0;
  }

  virtual ::std::future<int32_t> send_client2client_msg(
      int msg_type, int to_client_id, const std::string& msg) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }
  virtual size_t get_server_nums() { return 1; }

  virtual std::future<int32_t> push_dense_raw_gradient(
      int table_id, float* total_send_data, size_t total_send_data_size,
      void* callback) override;

  virtual std::future<int32_t> push_sparse_raw_gradient(
      size_t table_id, const uint64_t* keys, const float** update_values,
      size_t num, void* callback) override;

  virtual std::future<int32_t> push_sparse_raw_gradient_partial(
      size_t table_id, const uint64_t* keys, const float** update_values,
      uint32_t num, void* done, int pserver_idx) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

  virtual std::future<int32_t> push_sparse_param(size_t table_id,
                                                 const uint64_t* keys,
                                                 const float** update_values,
                                                 size_t num,
                                                 void* done) override {
    std::promise<int32_t> prom;
    std::future<int32_t> fut = prom.get_future();
    prom.set_value(0);

    return fut;
  }

 private:
  virtual int32_t initialize() override;

  std::future<int32_t> done() {
    std::shared_ptr<std::promise<int32_t>> prom =
        std::make_shared<std::promise<int32_t>>();
    std::future<int32_t> fut = prom->get_future();
    prom->set_value(0);
    return fut;
  }

  inline uint32_t dense_dim_per_shard(uint32_t dense_dim_total,
                                      uint32_t shard_num) {
    return dense_dim_total / shard_num + 1;
  }

  inline std::unordered_map<uint32_t, std::shared_ptr<Table>>* table() {
    return &_table_map;
  }

  inline Table* table(size_t table_id) {
    auto itr = _table_map.find(table_id);
    if (itr != _table_map.end()) {
      return itr->second.get();
    }
    LOG(ERROR) << "table not found " << table_id;
    return NULL;
  }

  std::unordered_map<uint32_t, std::shared_ptr<Table>> _table_map;

  bool _running = false;
  bool _flushing = false;

 private:
  float _mae = 0;
  float _mse = 0;
  uint16_t _push_times = 0;
};
}
}
