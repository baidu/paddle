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

#include "paddle/fluid/distributed/table/common_table.h"

namespace paddle {
namespace distributed {

BarrierTable::~BarrierTable() {
  exit_ = true;
  pour_update_thread_->join();
  pour_update_thread_.reset(nullptr);
}

int32_t BarrierTable::initialize() {
  auto trainers = _config.common().trainer_num();
  trigger_.store(trainers);

  for (int x = 0; x < trainers; ++x) {
    trainer_all_.insert(x);
  }

  trainer_ids_ = std::make_shared<BlockingQueue<uint32_t>>(trainers);
  sem_ = std::make_shared<LightweightSemaphore>(0);
  pour_update_thread_.reset(
      new std::thread(std::bind(&BarrierTable::update_pour_thread, this)));

  VLOG(1) << "BarrierTable init trigger: " << trigger_.load();
  return 0;
}

void BarrierTable::update_pour_thread() {
  VLOG(1) << "running update_pour_thread";
  while (!exit_) {
    std::set<uint32_t> trainer_ids;

    while (trainer_ids.size() < trigger_.load() && !exit_) {
      if (trainer_ids_->Size() != 0) {
        auto id = trainer_ids_->Pop();
        trainer_ids.insert(id);

        std::vector<uint32_t> diffs(trainer_all_.size());
        auto iter = std::set_difference(trainer_all_.begin(),
                                        trainer_all_.end(), trainer_ids.begin(),
                                        trainer_ids.end(), diffs.begin());
        diffs.resize(iter - diffs.begin());

        auto diff = to_string<uint32_t>(diffs);
        VLOG(1) << "receive trainer: " << id
                << ", still need trainers: " << diff;
      }
    }

    if (!exit_) {
      VLOG(1) << "barrier table optimize begin";
      for (auto& x : *table_map_) {
        auto table = x.second;
        table->pour();
      }
      VLOG(1) << "barrier table optimize done";
    }
    sem_->signal(trainer_ids.size());
  }
}

// 0: send_barrier 1: recv_barrier 2: complete
int32_t BarrierTable::barrier(const uint32_t trainer_id,
                              const std::string barrier_type) {
  if (barrier_type == "2") {
    trigger_.fetch_sub(1, std::memory_order::memory_order_relaxed);
    VLOG(3) << "trigger sub to : " << trigger_.load();
    return 0;
  }

  VLOG(3) << "barrier type: " << barrier_type
          << " add trainer id: " << trainer_id;
  trainer_ids_->Push(trainer_id);
  sem_->wait();
  return 0;
}

int32_t BarrierTable::set_table_map(
    std::unordered_map<uint32_t, std::shared_ptr<Table>>* table_map) {
  table_map_ = table_map;
  return 0;
}

}  // namespace distributed
}  // namespace paddle
