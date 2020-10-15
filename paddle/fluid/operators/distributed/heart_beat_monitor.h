// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <ThreadPool.h>
#include <gflags/gflags.h>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

enum WorkerStatus { UNINITED = 0, RUNNING, COMPLETED };

struct UnderMonitoredWorker {
  int id;
  WorkerStatus status;
  int timestamp;

  UnderMonitoredWorker() {}

  explicit UnderMonitoredWorker(int worker_id) {
    this->id = worker_id;
    this->status = UNINITED;
    this->timestamp = 0;
  }
};

class HeartBeatMonitor {
 public:
  explicit HeartBeatMonitor(int workers, bool is_chief,
                            std::string be_monitored_var)
      : workers_(workers),
        is_chief_(is_chief),
        be_monitored_var_(be_monitored_var),
        running_(true) {
    PADDLE_ENFORCE_GT(workers, 0, platform::errors::InvalidArgument(
                                      "workers must greater than 0."));

    for (auto worker_id = 0; worker_id < workers; worker_id++) {
      UnderMonitoredWorker worker(worker_id);
      worker_status_map_[worker_id] = std::move(worker);
    }

    // we define the No.0 pserver is the first parameter server
    // only No.0 will check the heartbeat of all trainers
    if (is_chief) {
      monitor_thread_.reset(new std::thread(
          std::bind(&HeartBeatMonitor::LostWorkerMonitor, this)));
    }
  }

  ~HeartBeatMonitor() {
    running_ = false;
    if (monitor_thread_) monitor_thread_->join();
  }

  static void Init(int workers, bool is_chief, std::string be_monitored_var) {
    std::call_once(init_flag_, &HeartBeatMonitor::InitImpl, workers, is_chief,
                   be_monitored_var);
  }

  static HeartBeatMonitor* GetInstance() { return monitor_.get(); }

  void Stop() {
    running_ = false;
    if (!monitor_) {
      VLOG(0) << "HeartBeatMonitor is not inited, do nothing";
    } else {
      if (monitor_thread_) {
        monitor_thread_->join();
        monitor_thread_.reset(nullptr);
      }
    }
  }

  void Update(const int worker_id, std::string be_monitored_var,
              WorkerStatus status);

  void LostWorkerMonitor();

 private:
  // Init is called by GetInstance.
  static void InitImpl(int workers, bool is_chief,
                       std::string be_monitored_var) {
    if (monitor_ == nullptr) {
      monitor_.reset(new HeartBeatMonitor(workers, is_chief, be_monitored_var));
    }
  }

  static std::once_flag init_flag_;
  static std::unique_ptr<HeartBeatMonitor> monitor_;

  int workers_;
  bool is_chief_;
  std::string be_monitored_var_;
  std::unordered_map<int, UnderMonitoredWorker> worker_status_map_;
  std::unique_ptr<std::thread> monitor_thread_{nullptr};
  std::mutex mutex_;
  bool running_ = false;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
