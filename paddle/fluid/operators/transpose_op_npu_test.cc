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

#include <cmath>
#include <iostream>
#include <numeric>
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

USE_OP(transpose2);
USE_OP_DEVICE_KERNEL(transpose2, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto out = scope->Var("Out");
  auto xshape = scope->Var("XShape");
  auto* x_t = x->GetMutable<f::LoDTensor>();
  auto* out_t = out->GetMutable<f::LoDTensor>();
  auto* xshape_t = xshape->GetMutable<f::LoDTensor>();
  auto place = ctx.GetPlace();

  int dim0 = 2;
  int dim1 = 3;
  TensorFromVector(std::vector<T>({0, 1, 2, 3, 4, 5}), ctx, x_t);
  ctx.Wait();
  x_t->Resize({dim0, dim1});
  out_t->Resize({dim0, dim1});
  ctx.Wait();
  out_t->mutable_data<T>(place);
  ctx.Wait();
  xshape_t->Resize({dim0, dim1});
  xshape_t->mutable_data<T>(place);
  f::AttributeMap attrs = {{"axis", std::vector<int>({1, 0})},
                           {"data_format", std::string("AnyLayout")}};
  auto op = f::OpRegistry::CreateOp("transpose2", {{"X", {"X"}}},
                                    {{"Out", {"Out"}}, {"XShape", {"XShape"}}},
                                    attrs);
  ctx.Wait();
  op->Run(*scope, place);
  ctx.Wait();
  std::vector<T> out_v;
  TensorToVector(*out_t, ctx, &out_v);
  ctx.Wait();

  EXPECT_EQ(out_t->numel(), dim0 * dim1);
  EXPECT_EQ(out_v[0], 0);
  EXPECT_EQ(out_v[1], 3);
  EXPECT_EQ(out_v[2], 1);
  EXPECT_EQ(out_v[3], 4);
  EXPECT_EQ(out_v[4], 2);
  EXPECT_EQ(out_v[5], 5);
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto xshape = scope->Var("XShape");
  auto x_grad = scope->Var("X@GRAD");
  auto out_grad = scope->Var("Out@GRAD");

  auto* x_grad_t = x_grad->GetMutable<f::LoDTensor>();
  auto* xshape_t = xshape->GetMutable<f::LoDTensor>();
  auto* out_grad_t = out_grad->GetMutable<f::LoDTensor>();

  int dim0 = 2;
  int dim1 = 3;
  auto place = ctx.GetPlace();

  TensorFromVector(std::vector<T>({0, 1, 2, 3, 4, 5}), ctx, out_grad_t);
  ctx.Wait();

  x_grad_t->Resize({dim0, dim1});
  xshape_t->Resize(
      {0, dim0,
       dim1});  // NOTE(zhiqiu): 0 is needed, see its infershape function
  out_grad_t->Resize({dim0, dim1});

  f::AttributeMap attrs = {{"axis", std::vector<int>({1, 0})},
                           {"data_format", std::string("AnyLayout")}};

  auto op = f::OpRegistry::CreateOp(
      "transpose2_grad", {{"Out@GRAD", {"Out@GRAD"}}, {"XShape", {"XShape"}}},
      {{"X@GRAD", {"X@GRAD"}}}, attrs);

  op->Run(*scope, place);
  ctx.Wait();
  std::vector<T> out_v;
  TensorToVector(*x_grad_t, ctx, &out_v);
  ctx.Wait();

  EXPECT_EQ(x_grad_t->numel(), dim0 * dim1);
  EXPECT_EQ(out_v[0], 0);
  EXPECT_EQ(out_v[1], 3);
  EXPECT_EQ(out_v[2], 1);
  EXPECT_EQ(out_v[3], 4);
  EXPECT_EQ(out_v[4], 2);
  EXPECT_EQ(out_v[5], 5);
}

TEST(transpose2, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

TEST(transpose2_grad, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx);
}
