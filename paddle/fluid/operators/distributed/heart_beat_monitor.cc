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

#include "paddle/fluid/operators/distributed/heart_beat_monitor.h"

#include <ctime>

namespace paddle {
namespace operators {
namespace distributed {

DEFINE_int32(worker_update_interval_secs, 900,
             " the longest time interval between the worker update variables");

inline int GetCurrentUS() {
  // current date/time based on current system
  time_t t = std::time(0);
  int now = static_cast<int>(t);
  return now;
}

void HeartBeatMonitor::Update(const int worker_id, std::string be_monitored_var,
                              WorkerStatus status) {
  if (status == UNINITED) {
    LOG(WARNING) << "HeartBeatMonitor receive UNINITED status can not be used "
                    "in Update, something error";
  }

  if (!is_chief_) {
    return;
  }

  if ((be_monitored_var == be_monitored_var_ && status == RUNNING) ||
      status == COMPLETED) {
    auto timestamp = GetCurrentUS();
    UnderMonitoredWorker& worker = worker_status_map_.at(worker_id);

    if (worker.status != COMPLETED) {
      worker.status = status;
    }
    worker.timestamp = timestamp;
    return;
  }
}

void HeartBeatMonitor::LostWorkerMonitor() {
  VLOG(1) << "worker heartbeat monitor start at No.0 parameter server";
  while (running_) {
    for (int id = 0; id < workers_; ++id) {
      auto& worker = worker_status_map_.at(id);

      if (worker.status == UNINITED) {
        VLOG(4) << "worker " << worker.id << " is under UNINITED";
        continue;
      }
      if (worker.status == COMPLETED) {
        VLOG(4) << "worker " << worker.id << " is under COMPLETED";
        continue;
      }

      auto timestamp = GetCurrentUS();

      VLOG(4) << "worker " << worker.id << " status is " << worker.status
              << " timestamp is " << worker.timestamp << " the interval is "
              << timestamp - worker.timestamp;

      if (timestamp - worker.timestamp >= FLAGS_worker_update_interval_secs) {
        PADDLE_THROW(platform::errors::ExecutionTimeout(
            "the latest update of worker %d is %d secs ago, we doubt the "
            "the worker is not alive and this may have a bad effect on the "
            "fitting result, please check",
            worker.id, FLAGS_worker_update_interval_secs));
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10 * 1000));
  }
  VLOG(1) << "worker heartbeat monitor stopped, thread exit";
}

std::once_flag HeartBeatMonitor::init_flag_;
std::unique_ptr<HeartBeatMonitor> HeartBeatMonitor::monitor_(nullptr);

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
