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

#include <string>
#include <vector>

#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform

namespace framework {
class Variable;
}  // namespace framework

}  // namespace paddle

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int nranks_{1};
  int local_rank_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
  int nrings_{1};
};

class ParallelContext {
 public:
  explicit ParallelContext(const ParallelStrategy& strategy,
                           const platform::Place& place)
      : strategy_(strategy), place_(place) {}

  virtual ~ParallelContext() = default;

  virtual void Init() = 0;

  virtual void InitWithRingID(int ring_id) = 0;

  virtual void AllReduceByStream(const framework::Variable& src,
                                 framework::Variable* dst, int ring_id,
                                 bool use_calc_stream) = 0;

  virtual paddle::platform::DeviceContext* GetDeviceContext(int ring_id) = 0;

  // comm_stream[ring_id] wait compute_stream.
  // if CPU, should do nothing.
  virtual void WaitCompute(int ring_id) = 0;

  // compute_stream wait comm_stream[ring_id]
  // if CPU, should do nothing.
  virtual void WaitComm(int ring_id) = 0;

  // synchorize compute stream
  virtual void SynchronizeCompute() = 0;

  inline int GetNRings() const { return ring_ids_.size(); }

  inline int64_t GetNRanks() const { return strategy_.nranks_; }

 protected:
  ParallelStrategy strategy_;
  platform::Place place_;
  std::vector<int> ring_ids_;
};

}  //  namespace imperative
}  //  namespace paddle
