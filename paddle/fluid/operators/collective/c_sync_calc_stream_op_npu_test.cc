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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, NPU);
USE_NO_KERNEL_OP(c_sync_calc_stream);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto y = scope->Var("Y");
  auto tensor_y = y->GetMutable<f::LoDTensor>();

  std::vector<T> init_x;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_x.push_back(static_cast<T>(1.0));
  }

  std::vector<T> init_y;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_y.push_back(static_cast<T>(2.0));
  }

  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({10, 10});
  TensorFromVector(init_y, ctx, tensor_y);
  tensor_y->Resize({10, 10});

  f::AttributeMap attrs;
  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // sync data
  auto sync_op0 = f::OpRegistry::CreateOp("c_sync_calc_stream", {{"X", {"X"}}},
                                          {{"Out", {"Out"}}}, attrs);
  sync_op0->Run(*scope, place);

  // run

  auto op =
      f::OpRegistry::CreateOp("elementwise_add", {{"X", {"X"}}, {"Y", {"Y"}}},
                              {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  // sync op run
  auto sync_op = f::OpRegistry::CreateOp("c_sync_calc_stream", {{"X", {"X"}}},
                                         {{"Out", {"Out"}}}, attrs);
  sync_op->Run(*scope, place);

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  // sync op copy
  auto sync_op2 = f::OpRegistry::CreateOp("c_sync_calc_stream", {{"X", {"X"}}},
                                          {{"Out", {"Out"}}}, attrs);
  sync_op2->Run(*scope, place);

  float expected = 3.0;

  EXPECT_EQ(out_vec.size(), init_x.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], static_cast<T>(expected));
  }
}

TEST(c_sync_calc_stream, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}
