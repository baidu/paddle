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

#pragma once

#if defined(PADDLE_WITH_HCCL) || defined(PADDLE_WITH_RCCL) || defined(PADDLE_WITH_ASCEND_CL)

#include <stdio.h>
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <typeindex>
#include <unordered_map>
#include <vector>


#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/dynload/hccl.h"
#endif

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

#define HCCL_ID_VARNAME "HCCLID"

namespace paddle {
namespace platform {

inline HcclDataType ToHCCLDataType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP32) {
    return HCCL_DATA_TYPE_FP32;
  } else if (type == framework::proto::VarType::FP16) {
    return HCCL_DATA_TYPE_FP16;
  }else if (type == framework::proto::VarType::INT32) {
    return HCCL_DATA_TYPE_INT32;
  } else if (type == framework::proto::VarType::INT8) {
    return HCCL_DATA_TYPE_INT8;
  }
  // else if (type == framework::proto::VarType::FP64) {
  //   return HCCL_DATA_TYPE_FP32;
  // }
  else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "This datatype in hccl is not supported."));
  }
}

// NOTE(minqiyang): according to the ncclGroupEnd documentations:
// https://docs.nvidia.com/deeplearning/sdk/nccl-api/ncclapidoc.html,
// ncclGroupEnd will wait for all communicators to be initialized, which will
// cause blocking problem when a runtime_error was thrown, so try only guard
// HCCL actions when use it.

// class HCCLGroupGuard {
//  public:
//   static std::mutex &HCCLMutex() {
//     static std::mutex mtx;
//     return mtx;
//   }

//   inline HCCLGroupGuard() {
//     HCCLMutex().lock();
//     PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupStart());
//   }

//   inline ~HCCLGroupGuard() PADDLE_MAY_THROW {
//     PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupEnd());
//     HCCLMutex().unlock();
//   }
// };

struct HCCLContext {
  std::unique_ptr<NPUDeviceContext> ctx_;
  HcclComm comm_;

  explicit HCCLContext(int dev_id)
      : ctx_(new NPUDeviceContext(NPUPlace(dev_id))), comm_{nullptr} {}

  aclrtStream stream() const { return ctx_->stream(); }
  HcclComm comm() const { return comm_; }

  int device_id() const {
    return BOOST_GET_CONST(platform::NPUPlace, ctx_->GetPlace()).device;
  }
};

struct HCCLContextMap {
  std::unordered_map<int, HCCLContext> contexts_;
  std::vector<int> order_;

  explicit HCCLContextMap(const std::vector<platform::Place> &places,
                          HcclRootInfo *hccl_id = nullptr,
                          size_t num_trainers = 1, size_t trainer_id = 0) {
    PADDLE_ENFORCE_EQ(!places.empty(), true,
                      platform::errors::InvalidArgument(
                          "The HCCL place should not be empty."));
    order_.reserve(places.size());
    for (auto &p : places) {
      int dev_id = BOOST_GET_CONST(NPUPlace, p).device;
      order_.emplace_back(dev_id);
      contexts_.emplace(dev_id, HCCLContext(dev_id));
    }
    PADDLE_ENFORCE_EQ(
        order_.size(), contexts_.size(),
        platform::errors::Unavailable("HCCL Context Map does not support "
                                      "contain two or more same device."));

    std::unique_ptr<HcclComm[]> comms(new HcclComm[order_.size()]);
    // if num_trainers == 1, should create a new nccl id for local comms.
    if (num_trainers == 1 && hccl_id == nullptr) {
      // we do not know how to tackle this situation under hccl
      // std::lock_guard<std::mutex> guard(HCCLGroupGuard::HCCLMutex());
      // PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::ncclCommInitAll(
      //     comms.get(), static_cast<int>(order_.size()), order_.data()));
    } else {
      PADDLE_ENFORCE_NOT_NULL(hccl_id, platform::errors::InvalidArgument(
                                           "The HCCL id should not be null."));
      {
        int nranks = num_trainers * order_.size();
        // HCCLGroupGuard gurad;
        for (size_t i = 0; i < order_.size(); ++i) {
          int gpu_id = order_[i];
          int rank;
          if (order_.size() > 1) {
            rank = trainer_id * order_.size() + i;
          } else {
            rank = trainer_id;
          }
          VLOG(1) << "init hccl rank:" << rank << ", nranks:" << nranks
                  << ", gpu_id:" << gpu_id << ", dev_id:" << order_[i];
          aclrtSetDevice(gpu_id);
          PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclCommInitRootInfo(
              nranks, hccl_id, rank, comms.get() + i));
        }
      }
    }
    int i = 0;
    for (auto &dev_id : order_) {
      contexts_.at(dev_id).comm_ = comms[i++];
    }
  }

  HCCLContextMap(const HCCLContextMap &other) = delete;
  HCCLContextMap &operator=(const HCCLContextMap &other) = delete;

  NPUDeviceContext *DevCtx(int dev_id) const { return at(dev_id).ctx_.get(); }

  NPUDeviceContext *DevCtx(platform::Place p) const {
    return DevCtx(BOOST_GET_CONST(NPUPlace, p).device);
  }

  const HCCLContext &at(platform::Place p) const {
    return this->at(BOOST_GET_CONST(NPUPlace, p).device);
  }

  const HCCLContext &at(int dev_id) const { return contexts_.at(dev_id); }

  void WaitAll() {
    for (auto &p : contexts_) {
      p.second.ctx_->Wait();
    }
  }
};

inline std::string GetFlatHCCLVarName(size_t pos) {
  if (pos == 0) {
    return HCCL_ID_VARNAME;
  }
  return string::Sprintf("%s_%d", HCCL_ID_VARNAME, static_cast<int>(pos));
}

inline std::string GetHierarchicalExterHCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_exter_%s_%d", HCCL_ID_VARNAME,
                         static_cast<int>(pos));
}
inline std::string GetHierarchicalInterHCCLVarName(size_t pos) {
  return string::Sprintf("Hierarchical_inter_%s_%d", HCCL_ID_VARNAME,
                         static_cast<int>(pos));
}

class HCCLCommunicator {
 public:
  HCCLCommunicator() {}
  virtual ~HCCLCommunicator() PADDLE_MAY_THROW {}

  HCCLContextMap *DefaultFlatCtx() const {
    if (flat_ctxs_.size() == 0) {
      return nullptr;
    }

    return flat_ctxs_[0].get();
  }

  std::vector<std::unique_ptr<HCCLContextMap>> *GetFlatCtxs() {
    return &flat_ctxs_;
  }

  HCCLContextMap *GetFlatCtx(size_t run_order) const {
    return flat_ctxs_[run_order % flat_ctxs_.size()].get();
  }

  HCCLContextMap *GetRunEnvHCCLCtx(size_t run_order,
                                   bool use_hierarchical_allreduce) const {
    if (!use_hierarchical_allreduce) {
      return GetFlatCtx(run_order);
    }

    return GetHierarchicalInterCtx(run_order);
  }


   /*
    When nccl inits nccl comm using ncclCommInitAll, it meets error when
    allreduce ophandle and sync_batch_norm_op use ncclallreduce parallelly. So
    create a new nccl comm for sync_batch_norm_op. And these codes should be
    polished with a unified nccl management.
   */

  HCCLContextMap *GetSyncBatchNormCtx(framework::Scope* scope, const std::vector<platform::Place> &places) {
    auto *hccl_id_var = scope->FindVar(HCCL_ID_VARNAME);
    if (hccl_id_var != nullptr) {
      return DefaultFlatCtx();
    }

    if (sync_batch_norm_ctx_.get() == nullptr) {
      sync_batch_norm_ctx_.reset(new HCCLContextMap(places));
    }
    return sync_batch_norm_ctx_.get();
  }

  void InitFlatCtxs(const std::vector<platform::Place> &places,
                    const std::vector<HcclRootInfo *> &hccl_ids,
                    size_t trainers_num, size_t trainer_id) {
    if (hccl_ids.size() == 0) {
      auto ptr = new platform::HCCLContextMap(places);
      VLOG(1) << "init local trainer";
      flat_ctxs_.emplace_back(ptr);
    } else {
      for (size_t i = 0; i < hccl_ids.size(); i++) {
        auto ptr = new platform::HCCLContextMap(places, hccl_ids[i],
                                                trainers_num, trainer_id);
        VLOG(1) << "init trainer_id:" << trainer_id << ", comm no:" << i;
        flat_ctxs_.emplace_back(ptr);
      }
    }

    // as Executor have no way to use ncclComm created by ParallelExecutor,
    // we assign all flatten contexts to HCCLCommContext to fix.
    int nranks = static_cast<int>(trainers_num * places.size());
    int nrings = static_cast<int>(flat_ctxs_.size());
    for (int ring_id = 0; ring_id < nrings; ++ring_id) {
      for (size_t p = 0; p < places.size(); ++p) {
        int rank = trainer_id * places.size() + p;
        int dev_id = BOOST_GET_CONST(NPUPlace, places[p]).device;
        auto &ctx = flat_ctxs_[ring_id]->contexts_.at(dev_id);
        HCCLCommContext::Instance().AssignHCCLComm(ctx.comm_, nranks, rank,
                                                   dev_id, ring_id);
      }
    }
  }

  void InitHierarchicalCtxs(const std::vector<platform::Place> &places,
                            const std::vector<HcclRootInfo *> &inter_hccl_ids,
                            const std::vector<HcclRootInfo *> &exter_hccl_ids,
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
    for (size_t i = 0; i < inter_hccl_ids.size(); i++) {
      VLOG(1) << "init inter_trainer_id:" << inter_trainer_id
              << ", comm no:" << i;
      auto local = new HCCLContextMap(places, inter_hccl_ids[i],
                                      inter_trainers_num, inter_trainer_id);

      h_inter_ctxs_.emplace_back(local);
    }

    int exter_trainer_id = -1;
    if (trainer_id % inter_trainers_num == 0) {
      exter_trainer_id = trainer_id / inter_trainers_num;
    }

    if (exter_trainer_id >= 0) {
      for (size_t i = 0; i < exter_hccl_ids.size(); i++) {
        auto ex = new HCCLContextMap(places, exter_hccl_ids[i],
                                     exter_trainers_num, exter_trainer_id);
        VLOG(1) << "init exter_trainer_id:" << exter_trainer_id
                << ", comm no:" << i;
        h_exter_ctxs_.emplace_back(ex);
      }
    }
  }

  bool NeedExterAllReduce() const { return h_exter_ctxs_.size() > 0; }

  HCCLContextMap *GetHierarchicalInterCtx(size_t run_order) const {
    PADDLE_ENFORCE_GT(h_inter_ctxs_.size(), 0,
                      platform::errors::InvalidArgument(
                          "Hierarchical ctxs should be initialized firstly!"));
    return h_inter_ctxs_[run_order % h_inter_ctxs_.size()].get();
  }

  HCCLContextMap *GetHierarchicalExterCtx(size_t run_order) const {
    PADDLE_ENFORCE_GT(h_exter_ctxs_.size(), 0,
                      platform::errors::InvalidArgument(
                          "Hierarchical ctxs should be initialized firstly!"));
    return h_exter_ctxs_[run_order % h_exter_ctxs_.size()].get();
  }

  std::vector<std::unique_ptr<HCCLContextMap>> *GetHierarchicalInterCtxs() {
    return &h_inter_ctxs_;
  }

  std::vector<std::unique_ptr<HCCLContextMap>> *GetHierarchicalExterCtxs() {
    return &h_exter_ctxs_;
  }

 protected:
  // Support multi nccl comm on default nccl ring while HCCLContextMap can't.
  std::vector<std::unique_ptr<HCCLContextMap>> flat_ctxs_;

  // h_inter_ctxs_ and h_exter_ctxs_ are for 2d allreduce.
  // And h_exter_ctxs_ can support multi comm too.
  std::vector<std::unique_ptr<HCCLContextMap>> h_inter_ctxs_;
  std::vector<std::unique_ptr<HCCLContextMap>> h_exter_ctxs_;

  // just used for sync_batch_norm op.
  std::unique_ptr<HCCLContextMap> sync_batch_norm_ctx_;
};

}  // namespace platform
}  // namespace paddle
#endif
