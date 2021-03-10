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

USE_OP(mean);
USE_OP_DEVICE_KERNEL(mean, NPU);
USE_OP(mean_grad);
USE_OP_DEVICE_KERNEL(mean_grad, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<T> init;
  init.push_back(static_cast<T>(1.0));
  init.push_back(static_cast<T>(2.0));
  init.push_back(static_cast<T>(3.0));
  init.push_back(static_cast<T>(4.0));

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({4});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  auto op = f::OpRegistry::CreateOp(op_type,
                           {{"X", {"X"}}},
                           {{"Out", {"Out"}}},
                           {});

  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)1);
  EXPECT_EQ((float)out_vec[0], (float)2.5);
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx,
                 std::string op_type) {
  // init
  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<f::LoDTensor>();
  float dvalue = 2.0;
  tensor_dout->Resize({1});
  std::vector<T> init_dout;
  init_dout.push_back(static_cast<T>(dvalue));
  TensorFromVector(init_dout, ctx, tensor_dout);
  ctx.Wait();

  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();
  tensor_x->Resize({4});

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();
  tensor_dx->Resize({4});

  ctx.Wait();

  auto op = f::OpRegistry::CreateOp(op_type, 
                                    {{"Out@GRAD", {"DOut"}},
                                     {"X", {"X"}}},
                                    {{"X@GRAD", {"DX"}}},
                                    {});

  auto place = ctx.GetPlace();
  op->Run(*scope, place);

  std::vector<float> out_vec;
  TensorToVector(*tensor_dx, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)4);
  EXPECT_EQ((float)out_vec[0], (float)1.0/dvalue);
  EXPECT_EQ((float)out_vec[1], (float)1.0/dvalue);
  EXPECT_EQ((float)out_vec[2], (float)1.0/dvalue);
  EXPECT_EQ((float)out_vec[3], (float)1.0/dvalue);
}

TEST(mean, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "mean");
}


TEST(mean_grad, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    CompareGrad<float>(&scope, ctx, "mean_grad");
}

