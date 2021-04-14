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

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/enforce.h"

namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

using Tensor = paddle::framework::Tensor;

USE_OP(check_finite_and_unscale);
USE_OP_DEVICE_KERNEL(check_finite_and_unscale, NPU);

struct InputVars {
  std::string name;
  f::LoDTensor *tensor;
};

template <typename T>
void Compare(f::Scope *scope, const p::DeviceContext &ctx) {
  const f::DDim dims = f::make_ddim({2, 2});
  auto place = ctx.GetPlace();

  // init input
  std::vector<InputVars> input_names = {
      {"x", scope->Var("x")->GetMutable<f::LoDTensor>()},
      {"x1", scope->Var("x1")->GetMutable<f::LoDTensor>()}};

  auto *scale = scope->Var("scale")->GetMutable<f::LoDTensor>();

  // init output
  auto *out = scope->Var("out")->GetMutable<f::LoDTensor>();
  auto *out1 = scope->Var("out1")->GetMutable<f::LoDTensor>();
  auto *found_inf = scope->Var("found_inf")->GetMutable<f::LoDTensor>();

  // Initialize input data
  const int num_inputs = input_names.size();
  size_t numel = static_cast<size_t>(f::product(dims));

  for (int i = 0; i < num_inputs; ++i) {
    std::vector<T> init_xs;
    for (size_t j = 0; j < numel; ++j) {
      if (j == 0) {
        init_xs.push_back(static_cast<T>(NAN));
      } else {
        init_xs.push_back(static_cast<T>(j + 1));
      }
    }
    f::TensorFromVector(init_xs, ctx, input_names[i].tensor);
    input_names[i].tensor->Resize(dims);
  }

  f::TensorFromVector(std::vector<T>{static_cast<T>(0.5)}, ctx, scale);

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  auto op = f::OpRegistry::CreateOp(
      "check_finite_and_unscale", {{"X", {"x", "x1"}}, {"Scale", {"scale"}}},
      {{"Out", {"out", "out1"}}, {"FoundInfinite", {"found_inf"}}}, attrs);
  op->Run(*scope, place);
  ctx.Wait();

  // out0
  std::vector<T> out_vec;
  f::TensorToVector(*out, ctx, &out_vec);
  EXPECT_EQ(out_vec.size(), static_cast<size_t>(4));
  for (size_t j = 0; j < out_vec.size(); ++j) {
    VLOG(3) << "out_vec[" << j << "]:" << out_vec[j];
  }

  ctx.Wait();

  // out0
  std::vector<T> out1_vec;
  f::TensorToVector(*out1, ctx, &out1_vec);
  EXPECT_EQ(out1_vec.size(), static_cast<size_t>(4));
  for (size_t j = 0; j < out1_vec.size(); ++j) {
    VLOG(3) << "out1_vec[" << j << "]:" << out1_vec[j];
  }

  ctx.Wait();

  // out found_inf
  Tensor found_inf_tensor;
  found_inf_tensor.Resize({1});
  bool *is_finite_data =
      found_inf_tensor.mutable_data<bool>(paddle::platform::CPUPlace());
  f::TensorCopy(*found_inf, place, &found_inf_tensor);
  EXPECT_FALSE(*is_finite_data);

  ctx.Wait();
}

TEST(check_finite_and_unscale, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

TEST(check_finite_and_unscale, NPU_fp16) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<p::float16>(&scope, ctx);
}
