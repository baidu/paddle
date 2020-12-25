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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/broadcast_op_handle.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace details {

struct DummyVarHandle;
struct VarHandle;

namespace f = paddle::framework;
namespace p = paddle::platform;

using DeviceType = paddle::platform::DeviceType;

// test data amount
const f::DDim kDims = {20, 20};

struct TestBroadcastOpHandle {
  std::vector<std::unique_ptr<p::DeviceContext>> ctxs_;
  std::vector<Scope*> local_scopes_;
  std::vector<Scope*> param_scopes_;
  Scope g_scope_;
  OpHandleBase* op_handle_;
  std::vector<VarHandleBase*> vars_;
  std::vector<std::unique_ptr<ir::Node>> nodes_;
  std::vector<p::Place> place_list_;
  DeviceType use_device_;
#if defined(PADDLE_WITH_NCCL)
  std::unique_ptr<platform::NCCLContextMap> nccl_ctxs_;
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
  std::unique_ptr<platform::BKCLContextMap> bkcl_ctxs_;
#endif

  void WaitAll() {
    for (size_t j = 0; j < ctxs_.size(); ++j) {
      ctxs_[j]->Wait();
    }
#if defined(PADDLE_WITH_NCCL)
    if (nccl_ctxs_) {
      nccl_ctxs_->WaitAll();
    }
#endif
#if defined(PADDLE_WITH_XPU_BKCL)
    if (bkcl_ctxs_) {
      bkcl_ctxs_->WaitAll();
    }
#endif
  }

  void InitCtxOnDevice(DeviceType use_device) {
    use_device_ = use_device;
    if (use_device_ == p::kXPU) {
#if defined(PADDLE_WITH_XPU_BKCL)
      int count = p::GetXPUDeviceCount();
      if (count <= 1) {
        LOG(WARNING) << "Cannot test multi-xpu Broadcast, because the XPU "
                        "device count is "
                     << count;
        exit(0);
      }
      for (int i = 0; i < count; ++i) {
        auto p = p::XPUPlace(i);
        place_list_.push_back(p);
        ctxs_.emplace_back(new p::XPUDeviceContext(p));
      }
      bkcl_ctxs_.reset(new platform::BKCLContextMap(place_list_));
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with BKCL."));
#endif
    } else if (use_device_ == p::kCUDA) {
#if defined(PADDLE_WITH_NCCL)
      int count = p::GetCUDADeviceCount();
      if (count <= 1) {
        LOG(WARNING) << "Cannot test multi-gpu Broadcast, because the CUDA "
                        "device count is "
                     << count;
        exit(0);
      }
      for (int i = 0; i < count; ++i) {
        auto p = p::CUDAPlace(i);
        place_list_.push_back(p);
        ctxs_.emplace_back(new p::CUDADeviceContext(p));
      }
      nccl_ctxs_.reset(new platform::NCCLContextMap(place_list_));
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with NCLL."));
#endif
    } else {
      int count = 8;
      for (int i = 0; i < count; ++i) {
        auto p = p::CPUPlace();
        place_list_.push_back(p);
        ctxs_.emplace_back(new p::CPUDeviceContext(p));
      }
#if defined(PADDLE_WITH_XPU_BKCL)
      bkcl_ctxs_.reset(nullptr);
#endif
#if defined(PADDLE_WITH_NCCL)
      nccl_ctxs_.reset(nullptr);
#endif
    }
  }

  void InitBroadcastOp(size_t input_scope_idx) {
    nodes_.clear();
    std::unordered_map<Scope*, Scope*> scope_map;
    for (size_t j = 0; j < place_list_.size(); ++j) {
      local_scopes_.push_back(&(g_scope_.NewScope()));
      Scope& local_scope = local_scopes_.back()->NewScope();
      local_scope.Var("out");
      param_scopes_.emplace_back(&local_scope);
      scope_map.emplace(local_scopes_.back(), param_scopes_.back());
    }
    param_scopes_[input_scope_idx]->Var("input");

    nodes_.emplace_back(
        ir::CreateNodeForTest("node0", ir::Node::Type::kOperation));
    if (use_device_ == p::kCUDA) {
#if defined(PADDLE_WITH_NCCL)
      op_handle_ = new BroadcastOpHandle(nodes_.back().get(), local_scopes_,
                                         place_list_, nccl_ctxs_.get());
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with NCCL."));
#endif
    } else if (use_device_ == p::kXPU) {
#if defined(PADDLE_WITH_XPU_BKCL)
      op_handle_ = new BroadcastOpHandle(nodes_.back().get(), local_scopes_,
                                         place_list_, bkcl_ctxs_.get());
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with BKCL."));
#endif
    } else {
      op_handle_ = new BroadcastOpHandle(nodes_.back().get(), local_scopes_,
                                         place_list_);
    }

    op_handle_->SetLocalExecScopes(scope_map);

    nodes_.emplace_back(
        ir::CreateNodeForTest("node1", ir::Node::Type::kVariable));
    auto* in_var_handle = new VarHandle(nodes_.back().get(), 1, input_scope_idx,
                                        "input", place_list_[input_scope_idx]);
    vars_.emplace_back(in_var_handle);
    op_handle_->AddInput(in_var_handle);

    // add dummy var

    nodes_.emplace_back(
        ir::CreateNodeForTest("node2", ir::Node::Type::kVariable));
    vars_.emplace_back(new DummyVarHandle(nodes_.back().get()));
    DummyVarHandle* dummy_var_handle =
        static_cast<DummyVarHandle*>(vars_.back());
    dummy_var_handle->ClearGeneratedOp();
    op_handle_->AddInput(dummy_var_handle);

    for (size_t j = 0; j < place_list_.size(); ++j) {
      if (use_device_ != p::kCUDA) {
        op_handle_->SetDeviceContext(place_list_[j], ctxs_[j].get());
      }
      nodes_.emplace_back(
          ir::CreateNodeForTest("node3", ir::Node::Type::kVariable));
      VarHandle* out_var_handle =
          new VarHandle(nodes_.back().get(), 2, j, "out", place_list_[j]);
      vars_.emplace_back(out_var_handle);
      op_handle_->AddOutput(out_var_handle);
    }

    // add dummy var
    nodes_.emplace_back(
        ir::CreateNodeForTest("node4", ir::Node::Type::kVariable));
    vars_.emplace_back(new DummyVarHandle(nodes_.back().get()));
    DummyVarHandle* out_dummy_var_handle =
        static_cast<DummyVarHandle*>(vars_.back());
    out_dummy_var_handle->ClearGeneratedOp();
    op_handle_->AddOutput(out_dummy_var_handle);
  }

  std::vector<float> InitLoDTensor(const std::string& varname,
                                   size_t input_scope_idx, const f::LoD& lod,
                                   float val_scalar = 0.0) {
    auto var = param_scopes_[input_scope_idx]->FindVar(varname);

    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable %s is not found in scope.",
                                        varname));
    auto lod_tensor = var->GetMutable<f::LoDTensor>();
    std::vector<float> send_vector(static_cast<size_t>(f::product(kDims)));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k + val_scalar;
    }
    paddle::framework::TensorFromVector<float>(
        send_vector, *(ctxs_[input_scope_idx]), lod_tensor);
    lod_tensor->set_lod(lod);
    lod_tensor->Resize(kDims);
    return send_vector;
  }

  std::vector<float> InitSelectedRows(const std::string& varname,
                                      size_t input_scope_idx,
                                      const std::vector<int64_t>& rows,
                                      int height, float value_scalar = 0.0) {
    std::vector<float> send_vector(static_cast<size_t>(f::product(kDims)));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k + value_scalar;
    }

    auto var = param_scopes_[input_scope_idx]->FindVar(varname);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable %s is not found in scope.",
                                        varname));
    auto selected_rows = var->GetMutable<f::SelectedRows>();
    auto value = selected_rows->mutable_value();
    value->mutable_data<float>(kDims, place_list_[input_scope_idx]);
    selected_rows->set_height(height);
    selected_rows->set_rows(rows);

    paddle::framework::TensorFromVector<float>(
        send_vector, *(ctxs_[input_scope_idx]), value);

    return send_vector;
  }

  void SelectedRowsEqual(const std::string& varname, int input_scope_idx,
                         const std::vector<float>& send_vector,
                         const std::vector<int64_t>& rows, int height) {
    auto var = param_scopes_[input_scope_idx]->FindVar(varname);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable %s is not found in scope.",
                                        varname));
    auto& selected_rows = var->Get<f::SelectedRows>();
    auto rt = selected_rows.value();
    PADDLE_ENFORCE_EQ(selected_rows.height(), height,
                      platform::errors::InvalidArgument(
                          "The height of SelectedRows is not equal to "
                          "the expected, expect %d, but got %ld.",
                          height, selected_rows.height()));

    for (size_t k = 0; k < selected_rows.rows().size(); ++k) {
      PADDLE_ENFORCE_EQ(
          selected_rows.rows()[k], rows[k],
          platform::errors::InvalidArgument(
              "The item at position %zu of rows of SelectedRows "
              "is not equal to the expected, expect %ld, but got %ld.",
              k, rows[k], selected_rows.rows()[k]));
    }

    p::CPUPlace cpu_place;
    f::Tensor result_tensor;
    f::TensorCopySync(rt, cpu_place, &result_tensor);
    float* ct = result_tensor.data<float>();

    for (int64_t i = 0; i < f::product(kDims); ++i) {
      ASSERT_NEAR(ct[i], send_vector[i], 1e-5);
    }
  }

  void LoDTensorEqual(const std::string& varname,
                      const std::vector<float>& send_vec, const f::LoD& lod,
                      framework::Scope* scope) {
    p::CPUPlace cpu_place;
    auto var = scope->FindVar(varname);
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::NotFound("Variable %s is not found in scope.",
                                        varname));
    auto tensor = var->Get<f::LoDTensor>();
    PADDLE_ENFORCE_EQ(tensor.lod(), lod,
                      platform::errors::InvalidArgument(
                          "The LoD of tensor is not equal to "
                          "the expected, expect %s, but got %s.",
                          lod, tensor.lod()));
    f::Tensor result_tensor;
    f::TensorCopySync(tensor, cpu_place, &result_tensor);
    float* ct = result_tensor.mutable_data<float>(cpu_place);
    for (int64_t k = 0; k < f::product(kDims); ++k) {
      ASSERT_NEAR(ct[k], send_vec[k], 1e-5);
    }
  }

  void TestBroadcastLodTensor(size_t input_scope_idx) {
    f::LoD lod{{0, 10, 20}};
    auto send_vector = InitLoDTensor("input", input_scope_idx, lod);

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();
    for (size_t j = 0; j < place_list_.size(); ++j) {
      LoDTensorEqual("out", send_vector, lod, param_scopes_[j]);
    }
  }

  void TestBroadcastSelectedRows(size_t input_scope_idx) {
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    int height = static_cast<int>(kDims[0] * 2);
    auto send_vector = InitSelectedRows("input", input_scope_idx, rows, height);

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();
    for (size_t j = 0; j < place_list_.size(); ++j) {
      SelectedRowsEqual("out", input_scope_idx, send_vector, rows, height);
    }
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
