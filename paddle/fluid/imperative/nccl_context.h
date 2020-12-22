//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

// network header files
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <sys/socket.h>
#endif

#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/imperative/all_reduce.h"
#include "paddle/fluid/platform/dynload/nccl.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace imperative {

struct ParallelStrategy {
  int nranks_{1};
  int local_rank_{0};
  std::vector<std::string> trainer_endpoints_{};
  std::string current_endpoint_{""};
  // TODO(shenliang03): support multi stream communication
  int nrings_{1};
};

class ParallelContext {
 public:
  explicit ParallelContext(const ParallelStrategy& strategy,
                           const platform::Place& place)
      : strategy_(strategy), place_(place) {}

  virtual ~ParallelContext() {}

  virtual void Init() = 0;

  virtual void AllReduceByStream(const framework::Variable& src,
                                 framework::Variable* dst, int ring_id = 0,
                                 bool use_calc_stream = false) = 0;
#if defined(PADDLE_WITH_NCCL)
  virtual paddle::platform::CUDADeviceContext* GetDeviceContext(
      int ring_id) = 0;
#endif

  inline int GetNRings() { return strategy_.nrings_; }

 protected:
  ParallelStrategy strategy_;
  platform::Place place_;
};

#if defined(PADDLE_WITH_NCCL)
class NCCLParallelContext : public ParallelContext {
 public:
  explicit NCCLParallelContext(const ParallelStrategy& strategy,
                               const platform::Place& place)
      : ParallelContext(strategy, place) {}

  ~NCCLParallelContext() {}

  void BcastNCCLId(std::vector<ncclUniqueId>& nccl_ids, int root);  // NOLINT

  void Init() override;

  void AllReduceByStream(const framework::Variable& src,
                         framework::Variable* dst, int ring_id,
                         bool use_calc_stream) override;

  paddle::platform::CUDADeviceContext* GetDeviceContext(int ring_id) override;

 protected:
  void RecvNCCLID(const std::string& endpoint,
                  std::vector<ncclUniqueId>& nccl_ids);  // NOLINT

  void SendNCCLID(const std::string& endpoint,
                  const std::vector<ncclUniqueId>& nccl_ids);
};
#endif

}  //  namespace imperative
}  //  namespace paddle
