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

#include "paddle/fluid/framework/details/gather_op_handle.h"

#include <memory>
#include <unordered_map>

#include "gtest/gtest.h"

namespace paddle {
namespace framework {
namespace details {
struct DummyVarHandle;

namespace f = paddle::framework;
namespace p = paddle::platform;

using DeviceType = paddle::platform::DeviceType;

// test data amount
const f::DDim kDims = {20, 20};

struct TestGatherOpHandle {
  std::vector<std::unique_ptr<p::DeviceContext>> ctxs_;
  std::vector<Scope*> local_scopes_;
  std::vector<Scope*> param_scopes_;
  Scope g_scope_;
  OpHandleBase* op_handle_;
  std::vector<VarHandleBase*> vars_;
  std::vector<p::Place> gpu_list_;
  std::vector<std::unique_ptr<ir::Node>> nodes_;

  void WaitAll() {
    for (size_t j = 0; j < ctxs_.size(); ++j) {
      ctxs_[j]->Wait();
    }
  }

  void InitCtxOnGpu(bool use_gpu) {
    if (use_gpu) {
#ifdef PADDLE_WITH_CUDA
      int count = p::GetCUDADeviceCount();
      if (count <= 1) {
        LOG(WARNING) << "Cannot test multi-gpu Broadcast, because the CUDA "
                        "device count is "
                     << count;
        exit(0);
      }
      for (int i = 0; i < count; ++i) {
        auto p = p::CUDAPlace(i);
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CUDADeviceContext(p));
      }
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
    } else {
      int count = 8;
      for (int i = 0; i < count; ++i) {
        auto p = p::CPUPlace();
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CPUDeviceContext(p));
      }
    }
  }

  void InitGatherOp(size_t input_scope_idx) {
    nodes_.clear();
    std::unordered_map<Scope*, Scope*> scope_map;
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      local_scopes_.push_back(&(g_scope_.NewScope()));
      Scope& local_scope = local_scopes_.back()->NewScope();
      local_scope.Var("input");
      param_scopes_.emplace_back(&local_scope);
      scope_map.emplace(local_scopes_.back(), param_scopes_.back());
    }
    param_scopes_[input_scope_idx]->Var("out");

    nodes_.emplace_back(
        ir::CreateNodeForTest("node", ir::Node::Type::kOperation).release());
    op_handle_ =
        new GatherOpHandle(nodes_.back().get(), local_scopes_, gpu_list_);

    op_handle_->SetLocalExecScopes(scope_map);

    // add input
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      op_handle_->SetDeviceContext(gpu_list_[j], ctxs_[j].get());
      nodes_.emplace_back(
          ir::CreateNodeForTest("node1", ir::Node::Type::kVariable).release());
      auto* in_var_handle =
          new VarHandle(nodes_.back().get(), 1, j, "input", gpu_list_[j]);
      vars_.emplace_back(in_var_handle);
      op_handle_->AddInput(in_var_handle);
    }

    // add dummy var
    nodes_.emplace_back(
        ir::CreateNodeForTest("node2", ir::Node::Type::kVariable).release());
    vars_.emplace_back(new DummyVarHandle(nodes_.back().get()));
    DummyVarHandle* in_dummy_var_handle =
        static_cast<DummyVarHandle*>(vars_.back());
    in_dummy_var_handle->ClearGeneratedOp();
    op_handle_->AddInput(in_dummy_var_handle);

    // add output
    nodes_.emplace_back(
        ir::CreateNodeForTest("node3", ir::Node::Type::kVariable).release());
    auto* out_var_handle =
        new VarHandle(nodes_.back().get(), 2, input_scope_idx, "out",
                      gpu_list_[input_scope_idx]);
    vars_.emplace_back(out_var_handle);
    op_handle_->AddOutput(out_var_handle);

    // add dummy var
    nodes_.emplace_back(
        ir::CreateNodeForTest("node4", ir::Node::Type::kVariable).release());
    vars_.emplace_back(new DummyVarHandle(nodes_.back().get()));
    DummyVarHandle* dummy_var_handle =
        static_cast<DummyVarHandle*>(vars_.back());
    op_handle_->AddOutput(dummy_var_handle);
  }

  void TestGatherSelectedRows(size_t output_scope_idx) {
    int height = kDims[0] * 2;
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    std::vector<float> send_vector(f::product(kDims));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k;
    }

    for (size_t input_scope_idx = 0; input_scope_idx < gpu_list_.size();
         ++input_scope_idx) {
      auto in_var = param_scopes_.at(input_scope_idx)->FindVar("input");
      PADDLE_ENFORCE_NOT_NULL(
          in_var, platform::errors::NotFound(
                      "The variable '%s' is not found in the scope.", "input"));
      auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();
      auto value = in_selected_rows->mutable_value();
      value->mutable_data<float>(kDims, gpu_list_[input_scope_idx]);

      in_selected_rows->set_height(height);
      in_selected_rows->set_rows(rows);

      paddle::framework::TensorFromVector<float>(
          send_vector, *(ctxs_[input_scope_idx]), value);
      value->Resize(kDims);
    }

    auto out_var = param_scopes_.at(output_scope_idx)->FindVar("out");
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound(
                     "The variable '%s' is not found in the scope.", "out"));
    auto out_selected_rows = out_var->GetMutable<f::SelectedRows>();

    auto in_var = param_scopes_.at(output_scope_idx)->FindVar("input");
    auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();

    out_selected_rows->mutable_value()->ShareDataWith(
        in_selected_rows->value());

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();

    p::CPUPlace cpu_place;

    auto& out_select_rows = out_var->Get<f::SelectedRows>();
    auto rt = out_select_rows.value();

    PADDLE_ENFORCE_EQ(out_select_rows.height(), height,
                      platform::errors::InvalidArgument(
                          "The height of SelectedRows is not equal to "
                          "the expected, expect %d, but got %d.",
                          height, out_select_rows.height()));

    for (size_t k = 0; k < out_select_rows.rows().size(); ++k) {
      PADDLE_ENFORCE_EQ(
          out_select_rows.rows()[k], rows[k % rows.size()],
          platform::errors::InvalidArgument(
              "The item at position %d of rows of SelectedRows is not equal to "
              "the expected, expect %d, but got %d.",
              k, rows[k % rows.size()], out_select_rows.rows()[k]));
    }

    f::Tensor result_tensor;
    f::TensorCopy(rt, cpu_place, *(ctxs_[output_scope_idx]), &result_tensor);
    float* ct = result_tensor.data<float>();

    for (int64_t j = 0;
         j < f::product(kDims) * static_cast<int64_t>(gpu_list_.size()); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j % send_vector.size()], 1e-5);
    }
  }
};

TEST(GatherTester, TestCPUGatherTestSelectedRows) {
  TestGatherOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitGatherOp(input_scope_idx);
  test_op.TestGatherSelectedRows(input_scope_idx);
}

#ifdef PADDLE_WITH_CUDA

TEST(GatherTester, TestGPUGatherTestSelectedRows) {
  TestGatherOpHandle test_op;
  size_t input_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitGatherOp(input_scope_idx);
  test_op.TestGatherSelectedRows(input_scope_idx);
}
#endif

}  // namespace details
}  // namespace framework
}  // namespace paddle
