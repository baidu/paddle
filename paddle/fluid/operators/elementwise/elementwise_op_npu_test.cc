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

USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, NPU);
USE_OP(elementwise_sub);
USE_OP_DEVICE_KERNEL(elementwise_sub, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
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

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(op_type, {{"X", {"X"}}, {"Y", {"Y"}}},
                                    {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);
  ctx.Wait();

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();
  float expected;
  if (op_type == "elementwise_add") {
    expected = 3.0;
  } else if (op_type == "elementwise_sub") {
    expected = -1.0;
  }
  EXPECT_EQ(out_vec.size(), init_x.size());
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], static_cast<T>(expected));
  }
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx,
                 std::string op_type) {
  // init
  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<f::LoDTensor>();
  tensor_dout->Resize({2, 3, 5});

  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  tensor_x->Resize({2, 3, 5});

  auto y = scope->Var("Y");
  auto tensor_y = y->GetMutable<f::LoDTensor>();
  tensor_y->Resize({1, 5});

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();

  auto dy = scope->Var("DY");
  auto tensor_dy = dy->GetMutable<f::LoDTensor>();

  std::vector<T> init_dout;
  for (int64_t i = 0; i < tensor_dout->numel(); ++i) {
    init_dout.push_back(static_cast<T>(1.0));
  }

  TensorFromVector(init_dout, ctx, tensor_dout);
  tensor_dout->Resize({2, 3, 5});

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(
      op_type, {{"Out@GRAD", {"DOut"}}, {"X", {"X"}}, {"Y", {"Y"}}},
      {{"X@GRAD", {"DX"}}, {"Y@GRAD", {"DY"}}}, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);
  ctx.Wait();

  std::vector<T> dx_vec;
  TensorToVector(*tensor_dx, ctx, &dx_vec);

  std::vector<T> dy_vec;
  TensorToVector(*tensor_dy, ctx, &dy_vec);

  ctx.Wait();
  float expected_x, expected_y;
  if (op_type == "elementwise_add_grad") {
    expected_x = 1.0;
    expected_y = 6.0;
  } else if (op_type == "elementwise_sub_grad") {
    expected_x = 1.0;
    expected_y = -6.0;
  }

  for (uint32_t i = 0; i < dx_vec.size(); i++) {
    EXPECT_EQ(dx_vec[i], static_cast<T>(expected_x));
  }
  for (uint32_t i = 0; i < dy_vec.size(); i++) {
    EXPECT_EQ(dy_vec[i], static_cast<T>(expected_y));
  }
}

TEST(elementwise_add, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx, "elementwise_add");
}

TEST(elementwise_sub, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx, "elementwise_sub");
}

TEST(elementwise_sub, NPU_fp16) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<p::float16>(&scope, ctx, "elementwise_sub");
}

TEST(elementwise_sub_grad, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx, "elementwise_sub_grad");
}

TEST(elementwise_add_grad, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx, "elementwise_add_grad");
}
