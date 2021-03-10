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

USE_OP(range);
USE_OP_DEVICE_KERNEL(range, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto start = scope->Var("Start");
  auto tensor_start = start->GetMutable<f::LoDTensor>();
  std::vector<T> init_start;
  init_start.push_back(static_cast<T>(1));
  TensorFromVector(init_start, ctx, tensor_start);
  tensor_start->Resize({1});

  auto end = scope->Var("End");
  auto tensor_end = end->GetMutable<f::LoDTensor>();
  std::vector<T> init_end;
  init_end.push_back(static_cast<T>(10));
  TensorFromVector(init_end, ctx, tensor_end);
  tensor_end->Resize({1});

  auto step = scope->Var("Step");
  auto tensor_step = step->GetMutable<f::LoDTensor>();
  std::vector<T> init_step;
  init_step.push_back(static_cast<T>(2));
  TensorFromVector(init_step, ctx, tensor_step);
  tensor_step->Resize({1});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  auto op = f::OpRegistry::CreateOp(op_type, {{"Start", {"Start"}},
                                              {"End", {"End"}},
                                              {"Step", {"Step"}}},
                                             {{"Out", {"Out"}}}, {});

  op->Run(*scope, place);

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);
  ctx.Wait();

  EXPECT_EQ(static_cast<T>(out_vec.size()), static_cast<T>(5));
  EXPECT_EQ(static_cast<T>(out_vec[0]), static_cast<T>(1.0));
  EXPECT_EQ(static_cast<T>(out_vec[1]), static_cast<T>(3.0));
  EXPECT_EQ(static_cast<T>(out_vec[2]), static_cast<T>(5.0));
  EXPECT_EQ(static_cast<T>(out_vec[3]), static_cast<T>(7.0));
  EXPECT_EQ(static_cast<T>(out_vec[4]), static_cast<T>(9.0));
}


TEST(range, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<int>(&scope, ctx, "range");
}

