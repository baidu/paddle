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

#include "paddle/fluid/framework/details/fused_broadcast_op_handle.h"

#include <memory>
#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/details/broadcast_op_handle_test.h"
#include "paddle/fluid/framework/details/op_handle_base.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace details {

struct VarHandle;
using DeviceType = paddle::platform::DeviceType;

struct TestFusedBroadcastOpHandle : TestBroadcastOpHandle {
  std::vector<std::string> out_varnames_;
  std::vector<std::unique_ptr<ir::Node>> nodes_;

  void InitFusedBroadcastOp(std::vector<size_t> input_scope_idxes) {
    nodes_.clear();
    // initialize scope and var
    std::unordered_map<Scope*, Scope*> scope_map;
    for (size_t i = 0; i < place_list_.size(); ++i) {
      local_scopes_.push_back(&(g_scope_.NewScope()));
      Scope& local_scope = local_scopes_.back()->NewScope();
      for (size_t j = 0; j < input_scope_idxes.size(); ++j) {
        local_scope.Var("out_var" + std::to_string(j));
        if (i == j) local_scope.Var("in_var" + std::to_string(j));
      }
      param_scopes_.emplace_back(&local_scope);
      scope_map.emplace(local_scopes_.back(), param_scopes_.back());
    }

    // create op handle node
    nodes_.emplace_back(
        ir::CreateNodeForTest("fused_broadcast", ir::Node::Type::kOperation));
    if (use_device_ == p::kCUDA) {
#if defined(PADDLE_WITH_NCCL)
      op_handle_ = new FusedBroadcastOpHandle(
          nodes_.back().get(), local_scopes_, place_list_, nccl_ctxs_.get());
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with CUDA."));
#endif
    } else if (use_device_ == p::kXPU) {
#if defined(PADDLE_WITH_XPU_BKCL)
      op_handle_ = new FusedBroadcastOpHandle(
          nodes_.back().get(), local_scopes_, place_list_, bkcl_ctxs_.get());
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with XPU."));
#endif
    } else {
      op_handle_ = new FusedBroadcastOpHandle(nodes_.back().get(),
                                              local_scopes_, place_list_);
    }

    op_handle_->SetLocalExecScopes(scope_map);

    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      // add input var handle
      nodes_.emplace_back(ir::CreateNodeForTest("in_node" + std::to_string(i),
                                                ir::Node::Type::kVariable));
      VarHandle* in_var_handle = new VarHandle(
          nodes_.back().get(), 1, input_scope_idxes[i],
          "in_var" + std::to_string(i), place_list_[input_scope_idxes[i]]);
      vars_.emplace_back(in_var_handle);
      op_handle_->AddInput(in_var_handle);

      // add output var handle
      for (size_t j = 0; j < place_list_.size(); ++j) {
        nodes_.emplace_back(ir::CreateNodeForTest(
            "out_node" + std::to_string(i), ir::Node::Type::kVariable));
        VarHandle* out_var_handle =
            new VarHandle(nodes_.back().get(), 2, j,
                          "out_var" + std::to_string(i), place_list_[j]);
        vars_.emplace_back(out_var_handle);
        op_handle_->AddOutput(out_var_handle);
      }
    }
  }

  void TestFusedBroadcastLoDTensor(std::vector<size_t> input_scope_idxes) {
    std::vector<std::vector<float>> send_vec;
    f::LoD lod{{0, 10, 20}};
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string varname("in_var" + std::to_string(i));
      float val_scalar = static_cast<float>(i);
      send_vec.push_back(
          InitLoDTensor(varname, input_scope_idxes[i], lod, val_scalar));
    }

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string& varname("out_var" + std::to_string(i));
      for (size_t j = 0; j < place_list_.size(); ++j) {
        LoDTensorEqual(varname, send_vec[i], lod, param_scopes_[j]);
      }
    }
  }

  void TestFusedBroadcastSelectedRows(std::vector<size_t> input_scope_idxes) {
    std::vector<std::vector<float>> send_vector;
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    int height = static_cast<int>(kDims[0] * 2);
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string varname("in_var" + std::to_string(i));
      float val_scalar = static_cast<float>(i);
      send_vector.push_back(InitSelectedRows(varname, input_scope_idxes[i],
                                             rows, height, val_scalar));
    }

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();
    for (size_t i = 0; i < input_scope_idxes.size(); ++i) {
      const std::string& varname("out_var" + std::to_string(i));
      for (size_t j = 0; j < place_list_.size(); ++j) {
        SelectedRowsEqual(varname, input_scope_idxes[i], send_vector[i], rows,
                          height);
      }
    }
  }
};

TEST(FusedBroadcastTester, CPULodTensor) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnDevice(p::kCPU);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastLoDTensor(input_scope_idxes);
}

TEST(FusedBroadcastTester, CPUSelectedRows) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnDevice(p::kCPU);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastSelectedRows(input_scope_idxes);
}

#if defined(PADDLE_WITH_CUDA) && defined(PADDLE_WITH_NCCL)
TEST(FusedBroadcastTester, GPULodTensor) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnDevice(p::kCUDA);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastLoDTensor(input_scope_idxes);
}

TEST(FusedBroadcastTester, GPUSelectedRows) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnDevice(p::kCUDA);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastSelectedRows(input_scope_idxes);
}
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
TEST(FusedBroadcastTester, XPULodTensor) {
  TestFusedBroadcastOpHandle test_op;
  std::vector<size_t> input_scope_idxes = {0, 1};
  test_op.InitCtxOnDevice(p::kXPU);
  test_op.InitFusedBroadcastOp(input_scope_idxes);
  test_op.TestFusedBroadcastLoDTensor(input_scope_idxes);
}
#endif

}  // namespace details
}  // namespace framework
}  // namespace paddle
