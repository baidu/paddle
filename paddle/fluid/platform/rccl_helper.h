//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_NCCL
#pragma once

#include <stdio.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/dynload/rccl.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#define NCCL_ID_VARNAME "NCCLID"

namespace paddle {
namespace platform {

inline ncclDataType_t ToNCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return ncclFloat;
  } else if (type == framework::proto::VarType::FP64) {
    return ncclDouble;
  } else if (type == framework::proto::VarType::INT32) {
    return ncclInt;
  } else if (type == framework::proto::VarType::INT64) {
    return ncclInt64;
  } else if (type == framework::proto::VarType::FP16) {
    return ncclFloat16;
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in rccl is not supported."));
  }
}

// NOTE(minqiyang): according to the ncclGroupEnd documentations:
// https://docs.nvidia.com/deeplearning/sdk/nccl-api/ncclapidoc.html,
// ncclGroupEnd will wait for all communicators to be initialized, which will
// cause blocking problem when a runtime_error was thrown, so try only guard
// NCCL actions when use it.
class NCCLGroupGuard {
 public:
  static std::mutex &NCCLMutex() {
    static std::mutex mtx;
    return mtx;
  }

  inline NCCLGroupGuard() {
    NCCLMutex().lock();
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupStart());
  }

  inline ~NCCLGroupGuard() PADDLE_MAY_THROW {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupEnd());
    NCCLMutex().unlock();
  }
};

struct NCCLContext {
  std::unique_ptr<CUDADeviceContext> ctx_;
  ncclComm_t comm_;

  explicit NCCLContext(int dev_id)
      : ctx_(new CUDADeviceContext(CUDAPlace(dev_id))), comm_{nullptr} {}

  hipStream_t stream() const { return ctx_->stream(); }
  ncclComm_t comm() const { return comm_; }

  int device_id() const {
    return BOOST_GET_CONST(platform::CUDAPlace, ctx_->GetPlace()).device;
  }
};

struct NCCLContextMap {
  std::unordered_map<int, NCCLContext> contexts_;
  std::vector<int> order_;

  explicit NCCLContextMap(const std::vector<platform::Place> &places,
                          ncclUniqueId *nccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0) {
    PADDLE_ENFORCE_EQ(!places.empty(), true,
                      platform::errors::InvalidArgument(
                          "The RCCL place should not be empty."));
    order_.reserve(places.size());
    for (auto &p : places) {
      int dev_id = BOOST_GET_CONST(CUDAPlace, p).device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, NCCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(), contexts_.size(),
        platform::errors::Unavailable("RCCL Context Map does not support "
                                      "contain two or more same device."));

    std::unique_ptr<ncclComm_t[]> comms(new ncclComm_t[order_.size()]);
    // if num_trainers == 1, should create a new nccl id for local comms.
    if (num_trainers == 1 && nccl_id == nullptr) {
      std::lock_guard<std::mutex> guard(NCCLGroupGuard::NCCLMutex());
      PADDLE_RETRY_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
          comms.get(), static_cast<int>(order_.size()), order_.data()));
    } else {
      PADDLE_ENFORCE_NOT_NULL(nccl_id, platform::errors::InvalidArgument(
                                           "The RCCL id should not be null."));
      {
        int nranks = num_trainers * order_.size();
        NCCLGroupGuard gurad;
        for (size_t i = 0; i < order_.size(); ++i) {
          int gpu_id = order_[i];
          int rank;
          if (order_.size() > 1) {
            rank = trainer_id * order_.size() + i;
          } else {
            rank = trainer_id;
          }
          VLOG(1) << "init rccl rank:" << rank << ", nranks:" << nranks
                  << ", gpu_id:" << gpu_id << ", dev_id:" << order_[i];
          PADDLE_RETRY_CUDA_SUCCESS(hipSetDevice(gpu_id));
          PADDLE_RETRY_CUDA_SUCCESS(platform::dynload::ncclCommInitRank(
              comms.get() + i, nranks, *nccl_id, rank));
        }
      }
    }
    int i = 0;
    for (auto &dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
  }

  NCCLContextMap(const NCCLContextMap &other) = delete;
  NCCLContextMap &operator=(const NCCLContextMap &other) = delete;

  CUDADeviceContext *DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  CUDADeviceContext *DevCtx(platform::Place p) const {
    return DevCtx(BOOST_GET_CONST(CUDAPlace, p).device);
  }

  const NCCLContext &at(platform::Place p) const {
    return this->at(BOOST_GET_CONST(CUDAPlace, p).device);
  }

  const NCCLContext &at(int dev_id) const { return contexts_.at(dev_id); }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second.ctx_->Wait();
    }
  }
};

inline std::string GetFlatNCCLVarName(size_t pos) {
  if (pos == 0) {
    return NCCL_ID_VARNAME;
  }
  return string::Sprintf("%s_%d", NCCL_ID_VARNAME, static_cast<int>(pos));
}

inline std::string GetHierarchicalExterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_exter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}
inline std::string GetHierarchicalInterNCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_inter_%s_%d", NCCL_ID_VARNAME,
                         static_cast<int>(pos));
}

class NCCLCommunicator {
 public:
  NCCLCommunicator() {}
  virtual ~NCCLCommunicator() PADDLE_MAY_THROW {}

  NCCLContextMap *DefaultFlatCtx() const {
    if (flat_ctxs_.size() == 0) {
      return nullptr;
    }

    return flat_ctxs_[0].get();
  }

  std::vector<std::unique_ptr<NCCLContextMap>> *GetFlatCtxs() {
    return &flat_ctxs_;
  }

  NCCLContextMap *GetFlatCtx(size_t run_order) const {
    return flat_ctxs_[run_order % flat_ctxs_.size()].get();
  }

  NCCLContextMap *GetRunEnvNCCLCtx(size_t run_order,
                                   bool use_hierarchical_allreduce) const {
    if (!use_hierarchical_allreduce) {
      return GetFlatCtx(run_order);
    }

    return GetHierarchicalInterCtx(run_order);
  }

  /*
   *When nccl inits nccl comm using ncclCommInitAll, it meets error when
   *allreduce ophandle and sync_batch_norm_op use ncclallreduce parallelly. So
   *create a new nccl comm for sync_batch_norm_op. And these codes should be
   *polished with a unified nccl management.
  */
  NCCLContextMap *GetSyncBatchNormCtx(
      framework::Scope *scope, const std::vector<platform::Place> &places) {
    auto *nccl_id_var = scope->FindVar(NCCL_ID_VARNAME);
    if (nccl_id_var != nullptr) {
      return DefaultFlatCtx();
    }

    if (sync_batch_norm_ctx_.get() == nullptr) {
      sync_batch_norm_ctx_.reset(new NCCLContextMap(places));
    }
    return sync_batch_norm_ctx_.get();
  }

  void InitFlatCtxs(const std::vector<platform::Place> &places,
                    const std::vector<ncclUniqueId *> &nccl_ids,
                    size_t trainers_num, size_t trainer_id) {
    if (nccl_ids.size() == 0) {
      auto ptr = new platform::NCCLContextMap(places);
      VLOG(1) << "init local trainer";
      flat_ctxs_.emplace_back(ptr);
    } else {
      for (size_t i = 0; i < nccl_ids.size(); i++) {
        auto ptr = new platform::NCCLContextMap(places, nccl_ids[i],
                                                trainers_num, trainer_id);
        VLOG(1) << "init trainer_id:" << trainer_id << ", comm no:" << i;
        flat_ctxs_.emplace_back(ptr);
      }
    }

    // as Executor have no way to use ncclComm created by ParallelExecutor,
    // we assign all flatten contexts to NCCLCommContext to fix.
    int nranks = static_cast<int>(trainers_num * places.size());
    int nrings = static_cast<int>(flat_ctxs_.size());
    for (int ring_id = 0; ring_id < nrings; ++ring_id) {
      for (size_t p = 0; p < places.size(); ++p) {
        int rank = trainer_id * places.size() + p;
        int dev_id = BOOST_GET_CONST(CUDAPlace, places[p]).device;
        auto &ctx = flat_ctxs_[ring_id]->contexts_.at(dev_id);
        NCCLCommContext::Instance().AssignNCCLComm(ctx.comm_, nranks, rank,
                                                   dev_id, ring_id);
      }
    }
  }

  void InitHierarchicalCtxs(const std::vector<platform::Place> &places,
                            const std::vector<ncclUniqueId *> &inter_nccl_ids,
                            const std::vector<ncclUniqueId *> &exter_nccl_ids,
                            size_t trainers_num, size_t trainer_id,
                            size_t inter_trainers_num,
                            size_t exter_trainers_num) {
    PADDLE_ENFORCE_EQ(
        trainers_num, inter_trainers_num * exter_trainers_num,
        platform::errors::InvalidArgument(
            "trainers_num:%llu != inter_trainers_num:%llu * "
            "exter_trainers_num:%llu",
            trainers_num, inter_trainers_num, exter_trainers_num));

    PADDLE_ENFORCE_GT(
        inter_trainers_num, 1,
        platform::errors::InvalidArgument(
            "The inter_trainers_num:%llu should be larger than 1.",
            inter_trainers_num));

    int inter_trainer_id = trainer_id % inter_trainers_num;
    for (size_t i = 0; i < inter_nccl_ids.size(); i++) {
      VLOG(1) << "init inter_trainer_id:" << inter_trainer_id
              << ", comm no:" << i;
      auto local = new NCCLContextMap(places, inter_nccl_ids[i],
                                      inter_trainers_num, inter_trainer_id);

      h_inter_ctxs_.emplace_back(local);
    }

    int exter_trainer_id = -1;
    if (trainer_id % inter_trainers_num == 0) {
      exter_trainer_id = trainer_id / inter_trainers_num;
    }

    if (exter_trainer_id >= 0) {
      for (size_t i = 0; i < exter_nccl_ids.size(); i++) {
        auto ex = new NCCLContextMap(places, exter_nccl_ids[i],
                                     exter_trainers_num, exter_trainer_id);
        VLOG(1) << "init exter_trainer_id:" << exter_trainer_id
                << ", comm no:" << i;
        h_exter_ctxs_.emplace_back(ex);
      }
    }
  }

  bool NeedExterAllReduce() const { return h_exter_ctxs_.size() > 0; }

  NCCLContextMap *GetHierarchicalInterCtx(size_t run_order) const {
    PADDLE_ENFORCE_GT(h_inter_ctxs_.size(), 0,
                      platform::errors::InvalidArgument(
                          "Hierarchical ctxs should be initialized firstly!"));
    return h_inter_ctxs_[run_order % h_inter_ctxs_.size()].get();
  }

  NCCLContextMap *GetHierarchicalExterCtx(size_t run_order) const {
    PADDLE_ENFORCE_GT(h_exter_ctxs_.size(), 0,
                      platform::errors::InvalidArgument(
                          "Hierarchical ctxs should be initialized firstly!"));
    return h_exter_ctxs_[run_order % h_exter_ctxs_.size()].get();
  }

  std::vector<std::unique_ptr<NCCLContextMap>> *GetHierarchicalInterCtxs() {
    return &h_inter_ctxs_;
  }

  std::vector<std::unique_ptr<NCCLContextMap>> *GetHierarchicalExterCtxs() {
    return &h_exter_ctxs_;
  }

 protected:
  // Support multi nccl comm on default nccl ring while NCCLContextMap can't.
  std::vector<std::unique_ptr<NCCLContextMap>> flat_ctxs_;

  // h_inter_ctxs_ and h_exter_ctxs_ are for 2d allreduce.
  // And h_exter_ctxs_ can support multi comm too.
  std::vector<std::unique_ptr<NCCLContextMap>> h_inter_ctxs_;
  std::vector<std::unique_ptr<NCCLContextMap>> h_exter_ctxs_;

  // just used for sync_batch_norm op.
  std::unique_ptr<NCCLContextMap> sync_batch_norm_ctx_;
};

}  // namespace platform
}  // namespace paddle
#endif
