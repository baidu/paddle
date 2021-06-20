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

#include "paddle/fluid/operators/collective/recv_v2_op.h"
#include "paddle/fluid/operators/collective/c_ascend_collective_test_help.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(recv_v2);
USE_OP_DEVICE_KERNEL(recv_v2, NPU);

void TestHcomRecvOp(f::Scope* scope, const p::DeviceContext& ctx) {
  std::cout << "BEGIN TEST:" << __FUNCTION__ << std::endl;

  int num = atoi(getenv("DATA_SIZE"));
  EXPECT_GT(num, 0);
  EXPECT_LT(num, 1 << 15);
  int rank_id = atoi(getenv("RANK_ID"));
  VLOG(3) << "rank_id:" << rank_id << std::endl;

  ctx.Wait();
  auto place = ctx.GetPlace();
  auto out = scope->Var("Data");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({num, num});
  tensor_out->mutable_data<float>(place);  // allocate

  ctx.Wait();

  f::AttributeMap attrs;
  attrs["tag"] = std::string("srtest");
  attrs["peer"] = atoi(getenv("SRC_RANK"));
  attrs["ring_id"] = 0;
  attrs["srTag"] = 0;
  std::vector<int> out_shape;
  out_shape.push_back(num);
  out_shape.push_back(num);
  attrs["out_shape"] = out_shape;

  auto op = f::OpRegistry::CreateOp("recv_v2", {}, {{"Out", {"Data"}}}, attrs);
  VLOG(3) << "CreateOp recv_v2";

  for (int i = 0; i < 10; i++) {
    op->Run(*scope, place);
  }
  VLOG(3) << "Run op recv_v2";
  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);
  ctx.Wait();
  std::vector<float> init(num * num, 1.0 * atoi(getenv("DEST_RANK")));
  EXPECT_EQ(out_vec == init, true);
}

TEST(recv_v2, NPU) {
  f::Scope scope;
  PaddleEcclCommGroupIdType group_name = "test_group_1";

  char* npu_id = getenv("FLAGS_selected_npus");
  VLOG(3) << "Select npu:" << npu_id;
  p::NPUDeviceContext ctx(p::NPUPlace(atoi(npu_id)));

  PrepareUniqueId(&scope, ctx, group_name);
  Prepare(&scope, ctx, group_name);
  TestHcomRecvOp(&scope, ctx);
}
