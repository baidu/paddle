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

USE_OP(gelu);
USE_OP_DEVICE_KERNEL(gelu, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<T> init_x;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_x.push_back(static_cast<T>(1.0));
  }

  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({10, 10});

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  f::AttributeMap attrs;

  ctx.Wait();

  // run
  auto place = ctx.GetPlace();

  auto op = f::OpRegistry::CreateOp("gelu", {{"X", {"X"}}}, {{"Out", {"Out"}}},
                                    attrs);
  op->Run(*scope, place);

  ctx.Wait();

  // eval time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }

  ctx.Wait();

  gettimeofday(&end, NULL);
  int micros =
      (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  float expected = 0.841192;
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_FLOAT_EQ(out_vec[i], static_cast<T>(expected));
  }
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx) {
  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<f::LoDTensor>();

  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<T> init_dout;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_dout.push_back(static_cast<T>(1.0));
  }

  std::vector<T> init_x;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_x.push_back(static_cast<T>(1.0));
  }

  TensorFromVector(init_dout, ctx, tensor_dout);
  tensor_dout->Resize({10, 10});
  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({10, 10});

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();

  f::AttributeMap attrs;

  ctx.Wait();

  // run
  auto place = ctx.GetPlace();

  auto op = f::OpRegistry::CreateOp("gelu_grad",
                                    {{"Out@GRAD", {"DOut"}}, {"X", {"X"}}},
                                    {{"X@GRAD", {"DX"}}}, attrs);
  op->Run(*scope, place);

  ctx.Wait();

  // eval time
  struct timeval start, end;
  gettimeofday(&start, NULL);

  for (int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }

  ctx.Wait();

  gettimeofday(&end, NULL);
  int micros =
      (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> dx_vec;
  TensorToVector(*tensor_dx, ctx, &dx_vec);

  float expected = 1.082964;
  for (uint32_t i = 0; i < dx_vec.size(); i++) {
    EXPECT_FLOAT_EQ(dx_vec[i], static_cast<T>(expected));
  }
}

TEST(gelu, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

TEST(gelu_grad, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx);
}
