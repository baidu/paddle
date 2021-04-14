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
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/string/printf.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(gather);
USE_OP_DEVICE_KERNEL(gather, NPU);
USE_OP(gather_grad);
USE_OP_DEVICE_KERNEL(gather_grad, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx,
             std::string op_type) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto index = scope->Var("Index");
  auto tensor_index = index->GetMutable<f::LoDTensor>();

  std::vector<T> init_x;
  for (int64_t i = 1; i < 7; ++i) {
    // 1,2,3,4,5,6
    init_x.push_back(static_cast<T>(i));
  }

  // [[1, 2],[3, 4],[5, 6]]
  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize(paddle::framework::make_ddim({3, 2}));

  std::vector<int> init_index = {1, 2};
  paddle::framework::TensorFromVector<int>(init_index, ctx, tensor_index);
  tensor_index->Resize(paddle::framework::make_ddim({2}));

  ctx.Wait();

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs = {{"validate_indices", true}};
  auto op = f::OpRegistry::CreateOp(
      op_type, {{"X", {"X"}}, {"Index", {"Index"}}}, {{"Out", {"Out"}}}, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  // ref:https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/api/paddle/tensor/manipulation/gather_cn.html#gather
  for (int i = 0; i < static_cast<int>(out_vec.size()); ++i) {
    VLOG(3) << "out_vec[" << i << "] : " << out_vec[i];
  }
  uint32_t expected_size = 4;
  EXPECT_EQ((uint32_t)out_vec.size(), expected_size);

  // {3, 4, 5, 6}
  std::vector<T> expected_out_vec;
  for (int64_t i = 3; i < 7; ++i) {
    expected_out_vec.push_back(static_cast<T>(i));
  }
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], expected_out_vec[i]);
  }
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx,
                 std::string op_type) {
  // init
  auto index = scope->Var("Index");
  auto tensor_index = index->GetMutable<f::LoDTensor>();

  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<f::LoDTensor>();

  std::vector<int> init_index = {0, 1};
  paddle::framework::TensorFromVector<int>(init_index, ctx, tensor_index);
  tensor_index->Resize(paddle::framework::make_ddim({2}));

  std::vector<T> init_x = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize(paddle::framework::make_ddim({3, 2}));

  std::vector<T> init_dout = {5.0, 10.0, 2.0, 3.0};
  TensorFromVector(init_dout, ctx, tensor_dout);
  tensor_dout->Resize(paddle::framework::make_ddim({2, 2}));

  ctx.Wait();

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(
      op_type, {{"X", {"X"}}, {"Index", {"Index"}}, {"Out@GRAD", {"DOut"}}},
      {{"X@GRAD", {"DX"}}}, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);

  std::vector<T> dx_vec;
  TensorToVector(*tensor_dx, ctx, &dx_vec);

  ctx.Wait();

  uint32_t expected_size = 3 * 2;
  EXPECT_EQ((uint32_t)dx_vec.size(), expected_size);

  std::vector<T> expected_dx_vec = {5.0, 10.0, 2.0, 3.0, 0.0, 0.0};
  for (uint32_t i = 0; i < dx_vec.size(); i++) {
    VLOG(3) << "dx_vec[i]=" << dx_vec[i];
    EXPECT_EQ(dx_vec[i], expected_dx_vec[i]);
  }
}

TEST(gather, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx, "gather");
}

TEST(gather, NPU_fp16) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<p::float16>(&scope, ctx, "gather");
}

TEST(gather_grad, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx, "gather_grad");
}
