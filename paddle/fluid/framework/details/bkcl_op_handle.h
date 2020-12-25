//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "xpu/bkcl.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/bkcl_helper.h"

DECLARE_bool(sync_bkcl_allreduce);

namespace paddle {
namespace framework {
namespace details {

class BKCLOpHandleBase : public OpHandleBase {
 public:
  BKCLOpHandleBase(ir::Node* node, const std::vector<platform::Place>& places,
                   const platform::BKCLCommunicator* bkcl_ctxs)
      : OpHandleBase(node), places_(places), bkcl_ctxs_(bkcl_ctxs) {
    if (bkcl_ctxs == nullptr) {
      return;
    }
    // init device context
    auto default_bkcl_ctxs = bkcl_ctxs_->DefaultFlatCtx();
    for (auto& p : places_) {
      this->SetDeviceContext(p, default_bkcl_ctxs->DevCtx(p));
    }
  }

  virtual ~BKCLOpHandleBase() {}

  void SetRunEnv(int run_order, bool use_hierarchical_allreduce) {
    PADDLE_ENFORCE_GE(run_order, 0, platform::errors::PreconditionNotMet(
                                        "run_order must >= 0"));
    PADDLE_ENFORCE_NE(use_hierarchical_allreduce, true,
                      platform::errors::Unimplemented(
                          "xpu doesn't support hierarchical_allreduce"));

    run_order_ = run_order;
    use_hierarchical_allreduce_ = use_hierarchical_allreduce;

    VLOG(10) << "SetRunEnv "
             << " run_order:" << run_order
             << ", use_hierarchical_allreduce:" << use_hierarchical_allreduce;

    if (bkcl_ctxs_ == nullptr) {
      return;
    }

    if (!use_hierarchical_allreduce_) {
      auto ctxs = bkcl_ctxs_->GetFlatCtx(run_order);
      for (auto& p : places_) {
        this->SetDeviceContext(p, ctxs->DevCtx(p));
      }
      return;
    }
  }

  void FlatBKCLAllReduce(platform::Place place, const void* sendbuff,
                         void* recvbuff, size_t count, BKCLDataType datatype,
                         BKCLOp op) {
    PADDLE_ENFORCE_GE(run_order_, 0, platform::errors::PreconditionNotMet(
                                         "run_order must > 0"));
    auto flat_bkcl_ctxs = bkcl_ctxs_->GetFlatCtx(run_order_);
    int dev_id = BOOST_GET_CONST(platform::XPUPlace, place).device;
    auto& bkcl_ctx = flat_bkcl_ctxs->at(dev_id);
    auto comm = bkcl_ctx.comm_;

    VLOG(10) << "before all reduce buffer:" << sendbuff << ", numel:" << count
             << ", dev_id:" << dev_id << ", dtype:" << datatype
             << ", place:" << place;

    PADDLE_ENFORCE_EQ(
        bkcl_all_reduce(comm, sendbuff, recvbuff, count, datatype, op, NULL),
        BKCL_SUCCESS,
        platform::errors::PreconditionNotMet("bckl all reduce failed"));
  }

  void BKCLAllReduce(platform::Place place, const void* sendbuff,
                     void* recvbuff, size_t count, BKCLDataType datatype,
                     BKCLOp op) {
    PADDLE_ENFORCE_GE(run_order_, 0, platform::errors::PreconditionNotMet(
                                         "run_order must > 0"));
    PADDLE_ENFORCE_EQ(use_hierarchical_allreduce_, false,
                      platform::errors::Unimplemented(
                          "xpu doesn't support hierarchical all reduce"));
    if (!use_hierarchical_allreduce_) {
      FlatBKCLAllReduce(place, sendbuff, recvbuff, count, datatype, op);
      return;
    }
  }

 protected:
  std::vector<platform::Place> places_;
  const platform::BKCLCommunicator* bkcl_ctxs_{nullptr};
  // When multi trainer call collective function, they need run the same order.
  // Or the program will hang.So we use allreduce_deps_pass to set this
  // run_order_.
  int run_order_{0};
  // Use 2d allreduce or not.
  bool use_hierarchical_allreduce_{false};
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
