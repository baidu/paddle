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

#include <iostream>
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

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(expand);
USE_OP_DEVICE_KERNEL(expand, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto in = scope->Var("X");
  auto expand_times = scope->Var("ExpandTimes");
  auto out = scope->Var("Out");
  auto in_t = in->GetMutable<f::LoDTensor>();
  auto out_t = out->GetMutable<f::LoDTensor>();
  auto expand_times_t = expand_times->GetMutable<f::LoDTensor>();

  auto place = ctx.GetPlace();
  TensorFromVector(std::vector<T>(3 * 1 * 7, 1), ctx, in_t);
  TensorFromVector(std::vector<int>({1, 10, 1}), ctx, expand_times_t);

  in_t->Resize(f::make_ddim({3, 1, 7}));
  expand_times_t->Resize(f::make_ddim({3}));
  out_t->Resize(f::make_ddim({3, 10, 7}));
  out_t->mutable_data<T>(place);

  f::AttributeMap attrs = {{}};
  auto op = f::OpRegistry::CreateOp(
      "expand", {{"X", {"X"}}, {"ExpandTimes", {"ExpandTimes"}}},
      {{"Out", {"Out"}}}, attrs);
  op->Run(*scope, place);
  ctx.Wait();

  auto out_dim = out_t->dims();
  EXPECT_EQ(out_dim.at(0), 3);
  EXPECT_EQ(out_dim.at(1), 10);
  EXPECT_EQ(out_dim.at(2), 7);
}

TEST(expand, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}
