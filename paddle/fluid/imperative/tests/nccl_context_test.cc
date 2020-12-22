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

#include "paddle/fluid/imperative/nccl_context.h"

#include "gtest/gtest.h"

namespace imperative = paddle::imperative;
namespace platform = paddle::platform;

int nrings = 2;
imperative::ParallelStrategy GetStrategy(int local_rank) {
  std::vector<std::string> eps = {"127.0.0.1:9866", "localhost:9867"};
  imperative::ParallelStrategy strategy;
  strategy.trainer_endpoints_ = eps;
  strategy.current_endpoint_ = eps[local_rank];
  strategy.nranks_ = 2;
  strategy.local_rank_ = local_rank;
  strategy.nrings_ = nrings;
  return strategy;
}

#if defined(PADDLE_WITH_NCCL)
void BcastNCCLId(int local_rank, std::vector<ncclUniqueId>* nccl_ids) {
  auto strategy = GetStrategy(local_rank);
  platform::CUDAPlace gpu(local_rank);
  imperative::NCCLParallelContext ctx(strategy, gpu);
  ctx.BcastNCCLId(*nccl_ids, 0);
}

TEST(BcastNCCLId, Run) {
  std::vector<ncclUniqueId> nccl_ids;
  nccl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    platform::dynload::ncclGetUniqueId(&nccl_ids[i]);
  }

  std::thread t(BcastNCCLId, 0, &nccl_ids);

  std::vector<ncclUniqueId> recv_nccl_ids;
  recv_nccl_ids.resize(nrings);
  for (int i = 0; i < nrings; ++i) {
    platform::dynload::ncclGetUniqueId(&recv_nccl_ids[i]);
  }
  BcastNCCLId(1, &recv_nccl_ids);

  t.join();
  for (int i = 0; i < nrings; ++i) {
    EXPECT_EQ(0, std::memcmp(nccl_ids[i].internal, recv_nccl_ids[i].internal,
                             NCCL_UNIQUE_ID_BYTES));
  }
}
#endif
