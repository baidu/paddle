/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef _WIN32
#include <unistd.h>
#endif

#include <stdio.h>

#include <string>
#include <thread>  // NOLINT
#include <vector>

#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/collective/c_allreduce_op.h"
#include "paddle/fluid/operators/collective/c_broadcast_op.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_broadcast);
USE_OP(c_allreduce_sum);
USE_NO_KERNEL_OP(c_comm_init_hccl);
USE_NO_KERNEL_OP(c_create_group);
USE_OP_DEVICE_KERNEL(c_broadcast, NPU);
USE_OP_DEVICE_KERNEL(c_allreduce_sum, NPU);

void Prepare(f::Scope* scope, const p::DeviceContext& ctx) {
  std::string rank_table_file = getenv("RANK_TABLE_FILE");
  int rank_id = atoi(getenv("RANK_ID"));
  int device_id = atoi(getenv("DEVICE_ID"));

  printf("rank_table_file: %s, rank_id = %d, device_id = %d\n",
         rank_table_file.c_str(), rank_id, device_id);

  f::AttributeMap attrs;
  attrs["rank_table_file"] = rank_table_file;
  attrs["rank_id"] = rank_id;
  attrs["device_id"] = device_id;
  auto comm_init_op =
      f::OpRegistry::CreateOp("c_comm_init_hccl", {}, {}, attrs);
  auto place = ctx.GetPlace();
  comm_init_op->Run(*scope, place);
  ctx.Wait();

  f::AttributeMap create_attrs;
  create_attrs["group_name"] = HCOM_GROUP_PREFIX + std::to_string(0);
  create_attrs["nranks"] = 2;
  std::vector<int> rank_ids{0, 1};
  create_attrs["rank_ids"] = rank_ids;
  auto create_group_op =
      f::OpRegistry::CreateOp("c_create_group", {}, {}, create_attrs);
  create_group_op->Run(*scope, place);
  ctx.Wait();
}
void TestHCCLBroadcastOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  int num = 2;
  std::vector<float> init;
  int rank_id = atoi(getenv("RANK_ID"));
  std::cout << "rank_id:" << rank_id << std::endl;
  for (int64_t i = 0; i < num * num; ++i) {
    init.push_back(1.0 + rank_id);
    std::cout << init[0];
  }
  std::cout << std::endl;

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num, num});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num, num});
  tensor_out->mutable_data<float>(place);  // allocate

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"] = std::string("tagx");
  attrs["root"] = 0;
  attrs["ring_id"] = 0;

  auto op = f::OpRegistry::CreateOp("c_broadcast", {{"X", {"X"}}},
                                    {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 1.0);
  }
}

void TestHCCLAllReduceOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<float> init;
  int rank_id = atoi(getenv("RANK_ID"));
  std::cout << "rank_id:" << rank_id << std::endl;

  int num1 = 1;
  int num2 = 4;

  for (int64_t i = 0; i < num1 * num2; ++i) {
    init.push_back(1.0);
    // init.push_back(1.0 + rank_id * 3);
    std::cout << init[0];
  }
  std::cout << std::endl;

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({num1, num2});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num1, num2});
  tensor_out->mutable_data<float>(place);  // allocate

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"] = std::string("tagx");
  attrs["ring_id"] = 0;

  auto op = f::OpRegistry::CreateOp("c_allreduce_sum", {{"X", {"X"}}},
                                    {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 2.0);
  }
}
TEST(c_broadcast, NPU) {
  f::Scope scope;
  char* npu_id = getenv("FLAGS_selected_npus");

  p::NPUDeviceContext ctx(p::NPUPlace(atoi(npu_id)));

  Prepare(&scope, ctx);
  // TestHCCLBroadcastOp(&scope, ctx);
  TestHCCLAllReduceOp(&scope, ctx);
}
