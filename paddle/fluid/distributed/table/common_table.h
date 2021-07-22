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

#include <algorithm>
#include <condition_variable>  // NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <set>

#include "paddle/fluid/distributed/common/sema.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/table.h"

namespace paddle {
namespace distributed {

template <typename T>
struct ReservoirValue {
  std::vector<T> values;
  uint32_t counter;
  uint32_t dim;

  ReservoirValue() {
    dim = 0;
    values.resize(dim);
    counter = 0;
  }

  ReservoirValue(uint32_t dim) {
    this->dim = dim;
    values.resize(dim);
    counter = 0;
  }

  void add(const T *value, int numel) {
    GetBlas<T>().VADD(numel, values.data(), value, values.data());
    counter++;
  }

  void add(T *value, int numel) {
    GetBlas<T>().VADD(numel, values.data(), value, values.data());
    counter++;
  }

  void avg() {
    if (counter == 0) return;
    auto scale = 1 / static_cast<T>(counter);
    GetBlas<T>().SCAL(values.size(), scale, values.data());
  }

  void reset() {
    std::fill(values.begin(), values.end(), 0);
    counter = 0;
  }
};

class SparseTable : public Table {
 public:
  SparseTable() {}
  virtual ~SparseTable() {}

  virtual void *get_shard(size_t shard_idx) { return 0; }

  int32_t pull_dense(float *values, size_t num) override { return 0; }

  int32_t push_dense(const float *values, size_t num) override { return 0; }

  static int32_t sparse_local_shard_num(uint32_t shard_num,
                                        uint32_t server_num) {
    if (shard_num % server_num == 0) {
      return shard_num / server_num;
    }
    size_t local_shard_num = shard_num / server_num + 1;
    return local_shard_num;
  }

  static size_t get_sparse_shard(uint32_t shard_num, uint32_t server_num,
                                 uint64_t key) {
    return (key % shard_num) / sparse_local_shard_num(shard_num, server_num);
  }
};

class DenseTable : public Table {
 public:
  DenseTable() {}
  virtual ~DenseTable() {}

  virtual void *get_shard(size_t shard_idx) { return 0; }
  int32_t pull_sparse(float *values,
                      const PullSparseValue &pull_value) override {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values,
                      size_t num) override {
    return 0;
  }
  int32_t push_dense_param(const float *values, size_t num) override {
    return 0;
  }
  int32_t shrink(const std::string &param) override { return 0; }
};

class BarrierTable : public Table {
 public:
  BarrierTable() {}
  virtual ~BarrierTable();

  virtual void *get_shard(size_t shard_idx) { return 0; }

  int32_t pull_dense(float *values, size_t num) override { return 0; }

  int32_t push_dense(const float *values, size_t num) override { return 0; }

  int32_t pull_sparse(float *values,
                      const PullSparseValue &pull_value) override {
    return 0;
  }
  int32_t push_sparse(const uint64_t *keys, const float *values,
                      size_t num) override {
    return 0;
  }
  int32_t push_dense_param(const float *values, size_t num) override {
    return 0;
  }
  int32_t shrink(const std::string &param) override { return 0; }
  virtual void clear() {}
  virtual int32_t flush() { return 0; }
  virtual int32_t load(const std::string &path, const std::string &param) {
    return 0;
  }
  virtual int32_t save(const std::string &path, const std::string &param) {
    return 0;
  }
  virtual int32_t initialize_shard() { return 0; }

  virtual int32_t initialize() override;
  // only for barrier
  // 0: send_barrier 1: recv_barrier 2: complete
  virtual int32_t barrier(const uint32_t trainer_id,
                          const std::string barrier_type) override;

  virtual int32_t set_table_map(
      std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map) override;

 private:
  void update_pour_thread();

  std::mutex mutex_;
  std::unique_ptr<std::thread> pour_update_thread_{nullptr};
  std::shared_ptr<LightweightSemaphore> sem_;
  std::shared_ptr<BlockingQueue<uint32_t>> trainer_ids_;
  std::set<uint32_t> trainer_all_;
  std::atomic<int> trigger_;
  bool exit_ = false;
  std::unordered_map<uint32_t, std::shared_ptr<Table>> *table_map_;
};
}  // namespace distributed
}  // namespace paddle
