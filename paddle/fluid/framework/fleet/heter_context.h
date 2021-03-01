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

#pragma once

#if (defined PADDLE_WITH_NCCL) && (defined PADDLE_WITH_PSLIB)

#include <algorithm>
#include <map>
#include <unordered_map>
#include <vector>

#include "common_value.h"  // NOLINT
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

class HeterContext {
 public:
  Scope* scope_{nullptr};
  std::vector<std::vector<FeatureKey>> feature_keys_;
  std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<FeatureValue>> feature_values_;
  std::vector<std::mutex*> mutex_lock_;
  uint32_t shard_num_ = 37;
  uint64_t size() {
    uint64_t total_size = 0;
    for (auto& keys : feature_keys_) {
      total_size += keys.size();
    }
    return total_size;
  }
  void SetShardNum(uint32_t shard_num) { shard_num_ = shard_num; }
  uint32_t ShardNum() { return shard_num_; }
  void init() { feature_keys_.resize(shard_num_); }
  void batch_add_keys(const std::vector<std::vector<uint64_t>>& thread_keys) {
    assert(thread_keys.size() == feature_keys_.size());

    for (uint32_t i = 0; i < shard_num_; i++) {
      int idx = 0;
      // mutex_lock_[i]->lock();
      idx = feature_keys_[i].size();
      feature_keys_[i].resize(feature_keys_[i].size() + thread_keys[i].size());
      for (uint64_t j = 0; j < thread_keys[i].size(); j++) {
        feature_keys_[i][idx + j] = thread_keys[i][j];
      }
      // mutex_lock_[i]->unlock();
    }
  }
  void UniqueKeys() {
    std::vector<std::thread> threads;
    auto unique_func = [this](int i) {
      auto& cur_keys = feature_keys_[i];
      std::sort(cur_keys.begin(), cur_keys.end());
      std::vector<FeatureKey>::iterator it;
      it = std::unique(cur_keys.begin(), cur_keys.end());
      cur_keys.resize(std::distance(cur_keys.begin(), it));
    };
    for (uint32_t i = 0; i < shard_num_; i++) {
      threads.push_back(std::thread(unique_func, i));
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
};

}  // end namespace framework
}  // end namespace paddle
#endif
