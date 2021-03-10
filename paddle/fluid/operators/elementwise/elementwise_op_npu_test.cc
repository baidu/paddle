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
USE_OP(elementwise_mul);
USE_OP_DEVICE_KERNEL(elementwise_mul, NPU);
USE_OP(elementwise_div);
USE_OP_DEVICE_KERNEL(elementwise_div, NPU);
USE_OP(elementwise_max);
USE_OP_DEVICE_KERNEL(elementwise_max, NPU);
USE_OP(elementwise_min);
USE_OP_DEVICE_KERNEL(elementwise_min, NPU);
USE_OP(elementwise_pow);
USE_OP_DEVICE_KERNEL(elementwise_pow, NPU);

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

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  f::AttributeMap attrs;

  ctx.Wait();

  // run
  auto place = ctx.GetPlace();

  auto op = f::OpRegistry::CreateOp(op_type, {{"X", {"X"}}, {"Y", {"Y"}}},
                                    {{"Out", {"Out"}}}, attrs);
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
  int micros = (((end.tv_sec - start.tv_sec) * 1000000) +
                  end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  float expected;
  if (op_type == "elementwise_add") {
    expected = 3.0;
  } else if (op_type == "elementwise_sub") {
    expected = -1.0;
  } else if (op_type == "elementwise_mul") {
    expected = 2.0;
  } else if (op_type == "elementwise_div") {
    expected = 0.5;
  } else if (op_type == "elementwise_max") {
    expected = 2.0;
  } else if (op_type == "elementwise_min") {
    expected = 1.0;
  } else if (op_type == "elementwise_pow") {
    expected = 1.0;
  }

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

  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto y = scope->Var("Y");
  auto tensor_y = y->GetMutable<f::LoDTensor>();

  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  std::vector<T> init_dout;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_dout.push_back(static_cast<T>(1.0));
  }

  std::vector<T> init_x;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_x.push_back(static_cast<T>(2.0));
  }

  std::vector<T> init_y;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    init_y.push_back(static_cast<T>(3.0));
  }

  std::vector<T> init_out;
  for (int64_t i = 0; i < 10 * 10; ++i) {
    if (op_type == "elementwise_add_grad") {
      init_out.push_back(static_cast<T>(5.0));
    } else if (op_type == "elementwise_sub_grad") {
      init_out.push_back(static_cast<T>(-1.0));
    } else if (op_type == "elementwise_mul_grad") {
      init_out.push_back(static_cast<T>(6.0));
    } else if (op_type == "elementwise_div_grad") {
      init_out.push_back(static_cast<T>(2.0/3.0));
    }
  }

  TensorFromVector(init_dout, ctx, tensor_dout);
  tensor_dout->Resize({10, 10});
  TensorFromVector(init_x, ctx, tensor_x);
  tensor_x->Resize({10, 10});
  TensorFromVector(init_y, ctx, tensor_y);
  tensor_y->Resize({10, 10});
  TensorFromVector(init_out, ctx, tensor_out);
  tensor_out->Resize({10, 10});

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();

  auto dy = scope->Var("DY");
  auto tensor_dy = dy->GetMutable<f::LoDTensor>();

  f::AttributeMap attrs;

  ctx.Wait();

  // run
  auto place = ctx.GetPlace();

  auto op = f::OpRegistry::CreateOp(op_type,
    {{"Out@GRAD", {"DOut"}}, {"X", {"X"}}, {"Y", {"Y"}}, {"Out", {"Out"}}},
    {{"X@GRAD", {"DX"}}, {"Y@GRAD", {"DY"}}}, attrs);
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
  int micros = (((end.tv_sec - start.tv_sec) * 1000000) +
                  end.tv_usec) - (start.tv_usec);
  printf("used time: %d\n", micros / 100);

  // eval value
  std::vector<T> dx_vec;
  TensorToVector(*tensor_dx, ctx, &dx_vec);

  std::vector<T> dy_vec;
  TensorToVector(*tensor_dy, ctx, &dy_vec);

  float expected_dx;
  float expected_dy;
  if (op_type == "elementwise_add_grad") {
    expected_dx = 1.0;
    expected_dy = 1.0;
  } else if (op_type == "elementwise_sub_grad") {
    expected_dx = 1.0;
    expected_dy = -1.0;
  } else if (op_type == "elementwise_mul_grad") {
    expected_dx = 3.0;
    expected_dy = 2.0;
  } else if (op_type == "elementwise_div_grad") {
    expected_dx = 1.0/3.0;
    expected_dy = 2.0/9.0;
  }

  for (uint32_t i = 0; i < dx_vec.size(); i++) {
    EXPECT_FLOAT_EQ(dx_vec[i], static_cast<T>(expected_dx));
  }
  for (uint32_t i = 0; i < dy_vec.size(); i++) {
    EXPECT_FLOAT_EQ(dy_vec[i], static_cast<T>(expected_dy));
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

TEST(elementwise_mul, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "elementwise_mul");
}

TEST(elementwise_div, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "elementwise_div");
}

TEST(elementwise_max, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "elementwise_max");
}

TEST(elementwise_min, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "elementwise_min");
}

TEST(elementwise_pow, NPU_fp32) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<float>(&scope, ctx, "elementwise_pow");
}

TEST(elementwise_sub, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_sub");
}

TEST(elementwise_mul, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_mul");
}

TEST(elementwise_div, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_div");
}

TEST(elementwise_max, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_max");
}

TEST(elementwise_min, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_min");
}

TEST(elementwise_pow, NPU_fp16) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    Compare<p::float16>(&scope, ctx, "elementwise_pow");
}

TEST(elementwise_sub_grad, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    CompareGrad<float>(&scope, ctx, "elementwise_sub_grad");
}

TEST(elementwise_mul_grad, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    CompareGrad<float>(&scope, ctx, "elementwise_mul_grad");
}

TEST(elementwise_div_grad, NPU) {
    f::Scope scope;
    p::NPUDeviceContext ctx(p::NPUPlace(0));
    CompareGrad<float>(&scope, ctx, "elementwise_div_grad");
}
