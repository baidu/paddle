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

USE_OP(mul);
USE_OP_DEVICE_KERNEL(mul, NPU);

void Compare(f::Scope* scope, const p::DeviceContext& ctx, int size) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto y = scope->Var("Y");
  auto tensor_y = y->GetMutable<f::LoDTensor>();
  int dim1 = 1024;
  int dim2 = size;

  //  std::vector<paddle::platform::float16> init;
  std::vector<float> init;
  for (int64_t i = 0; i < dim1 * dim2; ++i) {
    // init.push_back(paddle::platform::float16(1.0));
    init.push_back(1.0);
  }

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({dim1, dim2});
  TensorFromVector(init, ctx, tensor_y);
  tensor_y->Resize({dim2, dim1});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({dim1, dim1});
  // tensor_out->mutable_data<paddle::platform::float16>(place);  // allocate
  tensor_out->mutable_data<float>(place);  // allocate

  // run
  auto op = f::OpRegistry::CreateOp("mul", {{"X", {"X"}}, {"Y", {"Y"}}},
                                    {{"Out", {"Out"}}}, {});

  op->Run(*scope, place);
  ctx.Wait();

  struct timeval start, end;
  gettimeofday(&start, NULL);
  for (int i = 0; i < 100; i++) {
    op->Run(*scope, place);
  }
  ctx.Wait();
  gettimeofday(&end, NULL);
  int micros =
      (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  // printf("idx:%d, time:%d\n", i, micros/100);
  printf("size:%d time:%d\n", size, micros / 100);

  std::vector<float> out_vec;
  // std::vector<paddle::platform::float16> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(dim1 * dim1));
  for (uint32_t i = 0; i < out_vec.size(); i++) {
    EXPECT_EQ(out_vec[i], size);
  }
}

TEST(mul, NPU) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  int size = 1;
  for (int i = 0; i < 18; i++) {
    Compare(&scope, ctx, size);
    size *= 2;
  }
}