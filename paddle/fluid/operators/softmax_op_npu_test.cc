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
#include "paddle/fluid/framework/tensor_util.h"


namespace f = paddle::framework;
namespace p = paddle::platform;
namespace m = paddle::operators::math;

USE_OP(softmax);
USE_OP_DEVICE_KERNEL(softmax, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  std::vector<T> init;
  for (int i = 3; i < 9; ++i) {
    init.push_back(static_cast<T>(i));
  }

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({2, 3});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({2, 3});
  tensor_out->mutable_data<T>(place);  // allocate

  // run
  int axis = 1;
  f::AttributeMap attrs = {
            {"axis", axis},
            {"use_cudnn", false},
            {"use_mkldnn", false},
            {"mkldnn_data_type", std::string("float32")},
            {"is_test", false}, };

  auto op =
      f::OpRegistry::CreateOp("softmax", {{"X", {"X"}}},
                              {{"Out", {"Out"}}}, attrs);

  op->Run(*scope, place);
  ctx.Wait();

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  for (int i = 0; i < static_cast<int>(out_vec.size()); ++i) {
       VLOG(3) << "out_vec[" << i << "] : "<< out_vec[i];
  }

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(6));
}


template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  std::vector<T> out_init;

  out_init.push_back(static_cast<T>(0.6670));
  out_init.push_back(static_cast<T>(0.5888));
  out_init.push_back(static_cast<T>(0.4543));
  out_init.push_back(static_cast<T>(0.3330));
  out_init.push_back(static_cast<T>(0.4112));
  out_init.push_back(static_cast<T>(0.5457));

  TensorFromVector(out_init, ctx, tensor_out);
  tensor_out->Resize({2, 3});

  ctx.Wait();

  auto dout = scope->Var("DOut");
  auto tensor_dout = dout->GetMutable<f::LoDTensor>();

  std::vector<T> dout_init;
  for (int i = 0; i < 6; ++i) {
    dout_init.push_back(static_cast<T>(1.0));
  }

  TensorFromVector(dout_init, ctx, tensor_dout);
  tensor_dout->Resize({2, 3});

  ctx.Wait();

  auto dx = scope->Var("DX");
  auto tensor_dx = dx->GetMutable<f::LoDTensor>();

  ctx.Wait();

  // run
  f::AttributeMap attrs;
  attrs = {
      {"name", std::string("softmax_grad")},
      {"axis", static_cast<int>(1)},
            {"use_cudnn", false},
            {"use_mkldnn", false},
            {"mkldnn_data_type", std::string("float32")},
            {"is_test", false},
            {"data_format", std::string("AnyLayout")}, };
  auto op =
      f::OpRegistry::CreateOp("softmax_grad",
                              {{"Out", {"Out"}},
                               {"Out@GRAD", {"DOut"}}},
                              {{"X@GRAD", {"DX"}}}, attrs);

  auto place = ctx.GetPlace();
  op->Run(*scope, place);
  ctx.Wait();

  EXPECT_EQ((uint32_t)tensor_dx->dims()[0], (uint32_t)(2));
  EXPECT_EQ((uint32_t)tensor_dx->dims()[1], (uint32_t)(3));

  ctx.Wait();

  std::vector<float> out_vec;
  TensorToVector(*tensor_dx, ctx, &out_vec);

  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(6));
  EXPECT_NEAR((float)out_vec[0], (float)(-0.4737), 0.1);
  EXPECT_NEAR((float)out_vec[1], (float)(-0.4181), 0.1);
  EXPECT_NEAR((float)out_vec[2], (float)(-0.3226), 0.1);
  EXPECT_NEAR((float)out_vec[3], (float)(-0.0965), 0.1);
  EXPECT_NEAR((float)out_vec[4], (float)(-0.1192), 0.1);
  EXPECT_NEAR((float)out_vec[5], (float)(-0.1582), 0.1);
}

TEST(softmax, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

TEST(softmax_grad, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx);
}
