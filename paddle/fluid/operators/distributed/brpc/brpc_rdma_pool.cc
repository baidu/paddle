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

#ifdef PADDLE_WITH_BRPC_RDMA

#include "paddle/fluid/operators/distributed/brpc/brpc_rdma_pool.h"
#include "brpc/channel.h"
#include "brpc/rdma/rdma_helper.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

RdmaMemPool& RdmaMemPool::Instance() {
  static RdmaMemPool* g_rdma_mem_pool = new RdmaMemPool();
  return *g_rdma_mem_pool;
}

void* RdmaMemPool::Find(const std::string& varname, int64_t size) {
  pthread_rwlock_rdlock(&access_);
  auto it = pool_.find(varname);
  if (it == pool_.end()) {
    pthread_rwlock_unlock(&access_);
    return nullptr;
  }

  auto info = it->second;
  if (info.data_size != size) {
    pthread_rwlock_unlock(&access_);
    PADDLE_THROW(platform::errors::InvalidArgument(
        "var:%s size:%ld != %ld", varname, size, info.data_size));
    return nullptr;
  }

  pthread_rwlock_unlock(&access_);
  return info.data;
}

void RdmaMemPool::Register(const std::string& varname, void* data,
                           int64_t data_size) {
  void* old = Find(varname, data_size);
  if (old != nullptr) {
    PADDLE_ENFORCE_EQ(
        data, old, platform::errors::InvalidArgument("var:%s data:%ld != %ld",
                                                     varname, data, old));
    VLOG(7) << "Find on rdma:" << varname << " data:" << data
            << " data_size:" << data_size;
    return;
  }

  VarInfo info;
  info.data = data;
  info.data_size = data_size;

  pthread_rwlock_wrlock(&access_);
  pool_[varname] = info;
  pthread_rwlock_unlock(&access_);

  if (brpc::rdma::RegisterMemoryForRdma(data, data_size)) {
    PADDLE_THROW(platform::errors::Unavailable(
        "Register memory for RDMA failed. Register %s data: %s data size %d "
        "error.",
        varname, data, data_size));
  }

  VLOG(4) << "register on rdma:" << varname << " data:" << data
          << " data_size:" << data_size;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle

#endif
