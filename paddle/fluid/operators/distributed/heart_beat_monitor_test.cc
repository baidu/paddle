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

#include "gtest/gtest.h"

namespace paddle {
namespace operators {
namespace distributed {

void run(HeartBeatMonitor* monitor) { monitor->LostWorkerMonitor(); }

TEST(HeartBeatMonitor, All) {
  int trainers = 10;
  int pserver_id = 0;
  std::string var = "nce_w@GRAD.block0";
  std::string var2 = "nce_w@GRAD.block2";

  HeartBeatMonitor::Init(trainers, pserver_id == 0, var);

  auto* monitor = HeartBeatMonitor::GetInstance();

  std::vector<int> ids{1, 3, 5, 7};

  for (auto& id : ids) {
    monitor->Update(id, var, RUNNING);
  }

  monitor->Update(9, var2, RUNNING);
  monitor->Update(2, var, COMPLETED);

  std::thread t(run, monitor);
  t.detach();

  std::this_thread::sleep_for(std::chrono::milliseconds(15 * 1000));

  monitor->Stop();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
