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

#include "paddle/fluid/framework/details/reduce_op_handle.h"

#include <unordered_map>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
namespace details {
namespace f = paddle::framework;
namespace p = paddle::platform;

using DeviceType = paddle::platform::DeviceType;

// test data amount
const f::DDim kDims = {20, 20};

struct TestReduceOpHandle {
  bool use_gpu_;
  Scope g_scope_;
  std::vector<Scope *> local_scopes_;
  std::vector<Scope *> param_scopes_;
  OpHandleBase *op_handle_;
  std::vector<VarHandleBase *> vars_;
  std::vector<p::Place> gpu_list_;
  std::vector<std::unique_ptr<p::DeviceContext>> ctxs_;

#if defined(PADDLE_WITH_NCCL)
  std::unique_ptr<platform::NCCLContextMap> nccl_ctxs_;
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
  }

  void InitCtxOnGpu(bool use_gpu) {
    use_gpu_ = use_gpu;
    if (use_gpu) {
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
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CUDADeviceContext(p));
      }
      nccl_ctxs_.reset(new platform::NCCLContextMap(gpu_list_));
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with NCLL."));
#endif
    } else {
      int count = 8;
      for (int i = 0; i < count; ++i) {
        auto p = p::CPUPlace();
        gpu_list_.push_back(p);
        ctxs_.emplace_back(new p::CPUDeviceContext(p));
      }
#if defined(PADDLE_WITH_NCCL)
      nccl_ctxs_.reset(nullptr);
#endif
    }
  }

  void InitReduceOp(size_t out_scope_idx) {
    std::vector<std::unique_ptr<ir::Node>> nodes;
    // init scope
    std::unordered_map<Scope *, Scope *> scope_map;
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      local_scopes_.push_back(&(g_scope_.NewScope()));
      Scope &local_scope = local_scopes_.back()->NewScope();
      local_scope.Var("input");
      param_scopes_.emplace_back(&local_scope);
      scope_map.emplace(local_scopes_.back(), param_scopes_.back());
    }
    param_scopes_[out_scope_idx]->Var("out");

    nodes.emplace_back(new ir::Node("node"));
    if (use_gpu_) {
#if defined(PADDLE_WITH_NCCL)
      op_handle_.reset(new ReduceOpHandle(nodes.back().get(), local_scopes_,
                                          gpu_list_, nccl_ctxs_.get()));
#else
      PADDLE_THROW(
          platform::errors::PreconditionNotMet("Not compiled with NCLL."));
#endif
    } else {
#if defined(PADDLE_WITH_NCCL)
      op_handle_.reset(new ReduceOpHandle(nodes.back().get(), local_scopes_,
                                          gpu_list_, nccl_ctxs_.get()));
#else
      op_handle_.reset(
          new ReduceOpHandle(nodes.back().get(), local_scopes_, gpu_list_));
#endif
    }

    op_handle_->SetLocalExecScopes(scope_map);

    // init op handle
    // add input
    for (size_t j = 0; j < gpu_list_.size(); ++j) {
      if (!use_gpu_) {
        op_handle_->SetDeviceContext(gpu_list_[j], ctxs_[j].get());
      }
      nodes.emplace_back(new ir::Node("node1"));
      auto *in_var_handle =
          new VarHandle(nodes.back().get(), 1, j, "input", gpu_list_[j]);
      in_var_handle->ClearGeneratedOp();
      vars_.emplace_back(in_var_handle);
      op_handle_->AddInput(in_var_handle);
    }

    // add dummy var
    vars_.emplace_back(new DummyVarHandle());
    DummyVarHandle *in_dummy_var_handle =
        static_cast<DummyVarHandle *>(vars_.back().get());
    in_dummy_var_handle->ClearGeneratedOp();
    op_handle_->AddInput(in_dummy_var_handle);

    // add output
    nodes.emplace_back(new ir::Node("node2"));
    auto *out_var_handle = new VarHandle(nodes.back().get(), 2, out_scope_idx,
                                         "out", gpu_list_[out_scope_idx]);
    vars_.emplace_back(out_var_handle);
    op_handle_->AddOutput(out_var_handle);

    // add dummy var
    vars_.emplace_back(new DummyVarHandle());
    DummyVarHandle *dummy_var_handle =
        static_cast<DummyVarHandle *>(vars_.back().get());
    op_handle_->AddOutput(dummy_var_handle);
  }

  void TestReduceSelectedRows(size_t output_scope_idx) {
    int height = kDims[0] * 2;
    std::vector<int64_t> rows{0, 1, 2, 3, 3, 0, 14, 7, 3, 1,
                              2, 4, 6, 3, 1, 1, 1,  1, 3, 7};
    std::vector<float> send_vector(f::product(kDims));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k;
    }

    for (size_t input_scope_idx = 0; input_scope_idx < gpu_list_.size();
         ++input_scope_idx) {
      auto in_var = param_scopes_[input_scope_idx]->FindVar("input");

      PADDLE_ENFORCE_NOT_NULL(
          in_var, platform::errors::NotFound(
                      "Variable %s is not found in scope.", "input"));
      auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();
      auto value = in_selected_rows->mutable_value();
      value->mutable_data<float>(kDims, gpu_list_[input_scope_idx]);

      in_selected_rows->set_height(height);
      in_selected_rows->set_rows(rows);

      paddle::framework::TensorFromVector<float>(
          send_vector, *(ctxs_[input_scope_idx]), value);
      value->Resize(kDims);
    }

    auto out_var = param_scopes_[output_scope_idx]->FindVar("out");
    PADDLE_ENFORCE_NOT_NULL(out_var,
                            platform::errors::NotFound(
                                "Variable %s is not found in scope.", "out"));
    auto out_selected_rows = out_var->GetMutable<f::SelectedRows>();

    auto in_var = param_scopes_[output_scope_idx]->FindVar("input");
    auto in_selected_rows = in_var->GetMutable<f::SelectedRows>();

    out_selected_rows->mutable_value()->ShareDataWith(
        in_selected_rows->value());

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();

    p::CPUPlace cpu_place;

    auto &out_select_rows = out_var->Get<f::SelectedRows>();
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
    f::TensorCopySync(rt, cpu_place, &result_tensor);
    float *ct = result_tensor.data<float>();

    for (int64_t j = 0; j < f::product(result_tensor.dims()); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j % send_vector.size()], 1e-5);
    }
  }  // namespace details

  void TestReduceLodTensors(size_t output_scope_idx) {
    std::vector<float> send_vector(static_cast<size_t>(f::product(kDims)));
    for (size_t k = 0; k < send_vector.size(); ++k) {
      send_vector[k] = k;
    }
    f::LoD lod{{0, 10, 20}};

    for (size_t input_scope_idx = 0; input_scope_idx < gpu_list_.size();
         ++input_scope_idx) {
      auto in_var = param_scopes_[input_scope_idx]->FindVar("input");
      PADDLE_ENFORCE_NOT_NULL(
          in_var, platform::errors::NotFound(
                      "Variable %s is not found in scope.", "input"));
      auto in_lod_tensor = in_var->GetMutable<f::LoDTensor>();
      in_lod_tensor->mutable_data<float>(kDims, gpu_list_[input_scope_idx]);
      in_lod_tensor->set_lod(lod);

      paddle::framework::TensorFromVector<float>(
          send_vector, *(ctxs_[input_scope_idx]), in_lod_tensor);
    }

    auto out_var = param_scopes_[output_scope_idx]->FindVar("out");
    PADDLE_ENFORCE_NOT_NULL(out_var,
                            platform::errors::NotFound(
                                "Variable %s is not found in scope.", "out"));
    auto out_lodtensor = out_var->GetMutable<f::LoDTensor>();

    auto in_var = param_scopes_[output_scope_idx]->FindVar("input");
    auto in_lodtensor = in_var->Get<f::LoDTensor>();

    out_lodtensor->ShareDataWith(in_lodtensor);

    DeviceType use_device = p::kCPU;
    op_handle_->Run(use_device);

    WaitAll();

    p::CPUPlace cpu_place;

    auto &rt = out_var->Get<f::LoDTensor>();

    f::Tensor result_tensor;
    f::TensorCopySync(rt, cpu_place, &result_tensor);
    float *ct = result_tensor.data<float>();

    for (int64_t j = 0; j < f::product(result_tensor.dims()); ++j) {
      ASSERT_NEAR(ct[j], send_vector[j] * gpu_list_.size(), 1e-5);
    }
  }
};  // namespace details

TEST(ReduceTester, TestCPUReduceTestSelectedRows) {
  TestReduceOpHandle test_op;
  size_t out_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitReduceOp(out_scope_idx);
  test_op.TestReduceSelectedRows(out_scope_idx);
}
TEST(ReduceTester, TestCPUReduceTestLodTensor) {
  TestReduceOpHandle test_op;
  size_t out_scope_idx = 0;
  test_op.InitCtxOnGpu(false);
  test_op.InitReduceOp(out_scope_idx);
  test_op.TestReduceLodTensors(out_scope_idx);
}
#ifdef PADDLE_WITH_CUDA

TEST(ReduceTester, TestGPUReduceTestSelectedRows) {
  TestReduceOpHandle test_op;
  size_t out_scope_idx = 0;
  test_op.InitCtxOnGpu(true);
  test_op.InitReduceOp(out_scope_idx);
  test_op.TestReduceSelectedRows(out_scope_idx);
}

TEST(ReduceTester, TestGPUReduceTestLodTensor) {
  TestReduceOpHandle test_op;
  size_t out_scope_idx = 0;
  test_op.InitCtxOnGpu(true);
  test_op.InitReduceOp(out_scope_idx);
  test_op.TestReduceLodTensors(out_scope_idx);
}
#endif

}  // namespace details
}  // namespace framework
}  // namespace paddle
