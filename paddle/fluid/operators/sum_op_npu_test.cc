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

  std::vector<T> init_x1;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_x1.push_back(static_cast<T>(1.0));
  }

  TensorFromVector(init_x1, ctx, tensor_x1);
  tensor_x1->Resize({10, 10});

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  f::AttributeMap attrs;

  ctx.Wait();

  // run
  auto place = ctx.GetPlace();

  auto op = f::OpRegistry::CreateOp("sum", {{"X", {"X1"}}},
                                    {{"Out", {"Out"}}}, attrs);

//  auto op = f::OpRegistry::CreateOp("sum", {{"X", {init_input}}},
 //                                   {{"Out", {"Out"}}}, attrs);


  op->Run(*scope, place);

  ctx.Wait();

  // eval time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for(int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }

  ctx.Wait();

  gettimeofday(&end, NULL);
  int micros = (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  float expected = 6.0;
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], static_cast<T>(expected));
  }
}

TEST(sum, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx);
}

/*
TEST(sum, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx);
}
*/

