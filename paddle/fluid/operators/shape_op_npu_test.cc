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

USE_OP(shape);
USE_OP_DEVICE_KERNEL(shape, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx, int size) {
  // init
  auto x = scope->Var("Input");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  int dim1=10;
  int dim2=size;

  std::vector<T> init;
  for (int64_t i = 0; i < dim1 * dim2; ++i) {
    init.push_back(static_cast<T>(0.1));
  }

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({dim1, dim2});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs;
  auto op =
      f::OpRegistry::CreateOp("shape", {{"Input", {"Input"}}},
                              {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)2);
  EXPECT_EQ(out_vec[0], dim1);
  EXPECT_EQ(out_vec[1], size);
}


TEST(shape, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  int size=2;
  Compare<float>(&scope, ctx, size);
}


