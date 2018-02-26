/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/init.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

USE_NO_KERNEL_OP(ncclInit);
USE_CUDA_ONLY_OP(ncclAllReduce);
USE_CUDA_ONLY_OP(ncclReduce);
USE_CUDA_ONLY_OP(ncclBcast);

namespace f = paddle::framework;
namespace p = paddle::platform;

static std::vector<int> gpu_list;

// test data amount
const f::DDim kDims = {100, 100};

// nccl op common tester, init communicator.
class NCCLTester : public ::testing::Test {
 public:
  virtual void SetUp() override {
    paddle::platform::CPUPlace cpu_place;
    for (size_t i = 0; i < gpu_list.size(); ++i) {
      p::CUDAPlace place(i);
      dev_ctxs.emplace_back(new p::CUDADeviceContext(place));
    }

    NCCLInitOp();
  }

  virtual void TearDown() override {
    for (auto &device_context : dev_ctxs) {
      delete device_context;
    }
  }

  void NCCLInitOp() {
    paddle::platform::CPUPlace cpu_place;
    std::unique_ptr<f::OpDesc> op1(new f::OpDesc);

    op1->SetType("ncclInit");
    op1->SetInput("parallel_scopes", {"p_scopes"});
    op1->SetOutput("Communicator", {"comm"});

    auto *var = g_scope.Var("comm");
    var->GetMutable<p::Communicator>();

    auto *scope_var = g_scope.Var("p_scopes");
    auto *p_scopes = scope_var->GetMutable<std::vector<f::Scope *>>();
    (*p_scopes).resize(gpu_list.size());

    auto op = f::OpRegistry::CreateOp(*op1);
    VLOG(1) << "invoke NCCLInitOp.";
    op->Run(g_scope, cpu_place);
    VLOG(1) << "NCCLInitOp finished.";
  }

  template <class T>
  void PerThreadProgram(int gpu_id, const f::OpDesc &op_desc, f::Scope *scope) {
    std::unique_lock<std::mutex> lk(mu);
    const f::OpDesc *op1 = &op_desc;

    p::CUDAPlace place(gpu_id);
    auto &ctx = dev_ctxs.at(gpu_id);

    auto *send_tensor = scope->Var("st")->GetMutable<f::LoDTensor>();
    auto *recv_tensor = scope->Var("rt")->GetMutable<f::LoDTensor>();

    if (!send_tensor->numel()) {
      send_tensor->Resize(kDims);
      send_tensor->mutable_data<T>(kDims, place);

      std::vector<T> send_vector(f::product(kDims), gpu_id);
      paddle::framework::TensorFromVector<T>(send_vector, *ctx, send_tensor);
      ctx->Wait();
      VLOG(1) << "Send Tensor filled with elements " << send_tensor->numel();
    }

    lk.unlock();

    PADDLE_ENFORCE(send_tensor->numel() == f::product(kDims),
                   "Tensor numel not match!");

    auto op = f::OpRegistry::CreateOp(*op1);

    VLOG(1) << "Device : " << gpu_id << " invoke " << op_desc.Type();
    VLOG(1) << " send_tensor : " << send_tensor->numel()
            << " recv_tensor : " << recv_tensor->numel();
    op->Run(*scope, place);
    VLOG(1) << "Device : " << gpu_id << " finished " << op_desc.Type();
  }

 public:
  std::vector<p::DeviceContext *> dev_ctxs;
  f::Scope g_scope;
  std::mutex mu;
};

// ncclInitOp with desc
TEST_F(NCCLTester, ncclInitOp) {}

// ncclAllReduceOp with desc
TEST_F(NCCLTester, ncclAllReduceOp) {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  op2->SetType("ncclAllReduce");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    dev_scopes.emplace_back(&g_scope.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>, this, gpu_list[i],
                   *op2.get(), dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    ths[i].join();
  }

  // check results
  float result = std::accumulate(gpu_list.begin(), gpu_list.end(), 0);

  for (size_t i = 0; i < dev_scopes.size(); ++i) {
    p::CPUPlace cpu_place;
    p::CUDAPlace gpu_place(gpu_list[i]);

    auto &recv_tensor = dev_scopes[i]->FindVar("rt")->Get<f::LoDTensor>();
    auto *rt = recv_tensor.data<float>();
    auto *result_tensor = dev_scopes[i]->Var("ct")->GetMutable<f::LoDTensor>();
    result_tensor->Resize(kDims);
    auto *ct = result_tensor->mutable_data<float>(cpu_place);

    paddle::memory::Copy(
        cpu_place, ct, p::CUDAPlace(gpu_list[i]), rt,
        recv_tensor.numel() * sizeof(float),
        static_cast<p::CUDADeviceContext *>(dev_ctxs[i])->stream());

    for (int64_t j = 0; j < f::product(kDims); ++j) {
      ASSERT_NEAR(ct[j], result, 1e-5);
    }
  }
}

// ncclReduceOp with desc
TEST_F(NCCLTester, ncclReduceOp) {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  const int kRoot = 0;
  op2->SetType("ncclReduce");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});
  op2->SetAttr("root", kRoot);

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    dev_scopes.emplace_back(&g_scope.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>, this, gpu_list[i],
                   *op2.get(), dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    ths[i].join();
  }

  // check results on
  float result = std::accumulate(gpu_list.begin(), gpu_list.end(), 0);

  p::CPUPlace cpu_place;
  p::CUDAPlace gpu_place(gpu_list[kRoot]);

  auto &recv_tensor = dev_scopes[kRoot]->FindVar("rt")->Get<f::LoDTensor>();
  auto *rt = recv_tensor.data<float>();
  auto *result_tensor =
      dev_scopes[kRoot]->Var("ct")->GetMutable<f::LoDTensor>();
  result_tensor->Resize(kDims);
  auto *ct = result_tensor->mutable_data<float>(cpu_place);

  paddle::memory::Copy(
      cpu_place, ct, p::CUDAPlace(gpu_list[kRoot]), rt,
      recv_tensor.numel() * sizeof(float),
      static_cast<p::CUDADeviceContext *>(dev_ctxs[kRoot])->stream());

  for (int64_t j = 0; j < f::product(kDims); ++j) {
    ASSERT_NEAR(ct[j], result, 1e-5);
  }
}

// ncclBcastOp with desc
TEST_F(NCCLTester, ncclBcastOp) {
  std::unique_ptr<f::OpDesc> op2(new f::OpDesc);
  const int kRoot = 0;
  op2->SetType("ncclBcast");
  op2->SetInput("X", {"st"});
  op2->SetInput("Communicator", {"comm"});
  op2->SetOutput("Out", {"rt"});
  op2->SetAttr("root", kRoot);

  std::vector<f::Scope *> dev_scopes;

  std::vector<std::thread> ths;

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    dev_scopes.emplace_back(&g_scope.NewScope());
    std::thread th(&NCCLTester::PerThreadProgram<float>, this, gpu_list[i],
                   *op2.get(), dev_scopes[i]);
    ths.emplace_back(std::move(th));
  }

  for (size_t i = 0; i < gpu_list.size(); ++i) {
    ths[i].join();
  }

  const int idx = 1;
  // check results on
  float result = kRoot;

  p::CPUPlace cpu_place;
  p::CUDAPlace gpu_place(gpu_list[idx]);

  auto &recv_tensor = dev_scopes[idx]->FindVar("rt")->Get<f::LoDTensor>();
  auto *rt = recv_tensor.data<float>();
  auto *result_tensor = dev_scopes[idx]->Var("ct")->GetMutable<f::LoDTensor>();
  result_tensor->Resize(kDims);
  auto *ct = result_tensor->mutable_data<float>(cpu_place);

  paddle::memory::Copy(
      cpu_place, ct, p::CUDAPlace(gpu_list[idx]), rt,
      recv_tensor.numel() * sizeof(float),
      static_cast<p::CUDADeviceContext *>(dev_ctxs[idx])->stream());

  for (int64_t j = 0; j < f::product(kDims); ++j) {
    ASSERT_NEAR(ct[j], result, 1e-5);
  }
}

int main(int argc, char **argv) {
  const int dev_count = p::GetCUDADeviceCount();
  if (dev_count <= 1) {
    LOG(WARNING)
        << "Cannot test multi-gpu nccl, because the CUDA device count is "
        << dev_count;
    return 0;
  }

  std::vector<paddle::platform::Place> places;

  places.emplace_back(paddle::platform::CPUPlace());
  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    places.emplace_back(paddle::platform::CUDAPlace(i));
    gpu_list.emplace_back(i);
  }

  VLOG(0) << " DeviceCount " << count;
  paddle::platform::DeviceContextPool::Init(places);

  testing::InitGoogleTest(&argc, argv);

  // device context should be release before scope.
  // otherwise driver will down.
  return RUN_ALL_TESTS();
}
