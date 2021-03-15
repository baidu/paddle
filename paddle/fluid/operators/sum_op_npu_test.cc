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

USE_OP(sum);
USE_OP_DEVICE_KERNEL(sum, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x1 = scope->Var("X1");
  auto tensor_x1 = x1->GetMutable<f::LoDTensor>();

  auto x2 = scope->Var("X2");
  auto tensor_x2 = x2->GetMutable<f::LoDTensor>();

  std::cout << "1111111111111" << std::endl;

  std::vector<T> init_x1;
  for (int64_t i = 0; i < 10; ++i) {
    init_x1.push_back(static_cast<T>(1.0));
  }

  std::vector<T> init_x2;
  for (int64_t i = 0; i < 10; ++i) {
    init_x2.push_back(static_cast<T>(1.0));
  }

  std::cout << "22222222222" << std::endl;

  TensorFromVector(init_x1, ctx, tensor_x1);
  tensor_x1->Resize({1, 10});

  TensorFromVector(init_x2, ctx, tensor_x2);
  tensor_x2->Resize({1, 10});

  std::cout << "3333333333" << std::endl;
  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  //f::AttributeMap attrs = {{"axis", 0}};
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp("sum", {{"X", {"X1", "X2"}}},
                                    {{"Out", {"Out"}}}, attrs);

  std::cout << "444444444444444444444" << std::endl;
  op->Run(*scope, place);
  ctx.Wait();

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  float expected;
  expected = 3.0;

  std::cout << "555555555555555" << std::endl;
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], static_cast<T>(expected));
  }
}



TEST(sum, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

