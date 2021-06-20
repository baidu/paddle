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
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

#include "paddle/fluid/operators/collective/c_broadcast_op.h"
#include "paddle/fluid/operators/collective/c_ascend_collective_test_help.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(c_broadcast);
USE_OP_DEVICE_KERNEL(c_sync_comm_stream, NPU);
USE_OP_DEVICE_KERNEL(c_broadcast, NPU);

DECLARE_string(selected_npus);

void TestHCCLBroadcastOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;
  // init
  auto x = scope->Var("Data");
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
  auto out = scope->Var("OutData");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num, num});
  tensor_out->mutable_data<float>(place);  // allocate

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs["tag"] = std::string("tagx");
  attrs["root"] = 0;
  attrs["ring_id"] = 0;

  auto op = f::OpRegistry::CreateOp("c_broadcast", {{"X", {"Data"}}},
                                    {{"Out", {"OutData"}}}, attrs);

  op->Run(*scope, place);

  // comm sync

  auto sync_op = f::OpRegistry::CreateOp(
      "c_sync_comm_stream", {{"X", {"Data"}}}, {{"Out", {"OutData"}}}, attrs);
  sync_op->Run(*scope, place);

  // ctx.Wait();

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  EXPECT_EQ(out_vec.size(), init.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], 1.0);
  }
}

TEST(c_sync_comm_stream_op, NPU) {
  f::Scope scope;
   PaddleEcclCommGroupIdType group_name = "test_group_1";

  // only support one device, if more than one device, use first default
  p::NPUDeviceContext ctx(p::NPUPlace(atoi(FLAGS_selected_npus.c_str())));

  PrepareUniqueId(&scope, ctx, group_name);
  Prepare(&scope, ctx, group_name);
  TestHCCLBroadcastOp(&scope, ctx);
}
