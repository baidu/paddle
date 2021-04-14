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

USE_OP(lookup_table_v2);
USE_OP_DEVICE_KERNEL(lookup_table_v2, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto ids = scope->Var("Ids");
  auto out = scope->Var("Out");
  auto w = scope->Var("W");

  auto ids_t = ids->GetMutable<f::LoDTensor>();
  auto out_t = out->GetMutable<f::LoDTensor>();
  auto w_t = w->GetMutable<f::LoDTensor>();
  int bsz = 10;
  int dim = 32;
  int seqlen = 8;
  int vocab_size = 100;
  TensorFromVector(std::vector<int64_t>(bsz * seqlen, 3), ctx, ids_t);
  std::vector<T> val(vocab_size * dim, 10.);
  TensorFromVector(val, ctx, w_t);
  ids_t->Resize({bsz, seqlen});
  w_t->Resize({vocab_size, dim});
  out_t->Resize({bsz, seqlen, dim});
  ctx.Wait();

  auto place = ctx.GetPlace();
  out_t->mutable_data<T>(place);
  f::AttributeMap attrs = {{}};
  auto op = f::OpRegistry::CreateOp("lookup_table_v2",
                                    {{"W", {"W"}}, {"Ids", {"Ids"}}},
                                    {{"Out", {"Out"}}}, attrs);
  op->Run(*scope, place);
  std::vector<T> out_v;
  TensorToVector(*out_t, ctx, &out_v);
  ctx.Wait();
  EXPECT_EQ(out_t->numel(), bsz * seqlen * dim);
  T res = std::accumulate(out_v.begin(), out_v.end(), 0.);
  float eps = 1.e-6;
  EXPECT_LT(fabs(res - bsz * seqlen * dim * 10.), eps);
}

template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto w = scope->Var("W");
  auto ids = scope->Var("Ids");
  auto out = scope->Var("DOut");
  auto dw = scope->Var("DW");

  auto w_t = w->GetMutable<f::LoDTensor>();
  auto ids_t = ids->GetMutable<f::LoDTensor>();
  auto out_t = out->GetMutable<f::LoDTensor>();
  auto dw_t = dw->GetMutable<f::LoDTensor>();

  int bsz = 2;
  int dim = 2;
  int seqlen = 2;
  int vocab_size = 4;

  std::vector<int64_t> val_int(bsz * seqlen, 3);
  std::vector<T> val(vocab_size * dim, 0.);
  std::vector<T> val_out(bsz * seqlen * dim, 1.);

  TensorFromVector(val_int, ctx, ids_t);
  TensorFromVector(val, ctx, w_t);
  TensorFromVector(val, ctx, dw_t);
  TensorFromVector(val_out, ctx, out_t);

  w_t->Resize({vocab_size, dim});
  ids_t->Resize({bsz, seqlen});
  out_t->Resize({bsz, seqlen, dim});
  dw_t->Resize({vocab_size, dim});

  ctx.Wait();

  auto place = ctx.GetPlace();
  out_t->mutable_data<T>(place);
  w_t->mutable_data<T>(place);
  dw_t->mutable_data<T>(place);
  f::AttributeMap attrs = {{}};
  auto op = f::OpRegistry::CreateOp(
      "lookup_table_v2_grad",
      {{"Ids", {"Ids"}}, {"W", {"W"}}, {"Out@GRAD", {"DOut"}}},
      {{"W@GRAD", {"DW"}}}, attrs);
  op->Run(*scope, place);
  ctx.Wait();
  std::vector<T> w_v;
  TensorToVector(*dw_t, ctx, &w_v);
  ctx.Wait();
  EXPECT_EQ(dw_t->numel(), vocab_size * dim);
  T res = std::accumulate(w_v.begin(), w_v.end(), 0.);
  float eps = 1.e-6;
  EXPECT_LT(fabs(res - bsz * seqlen * dim), eps);
}

TEST(lookup_table_v2, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}

TEST(lookup_table_v2_grad, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  CompareGrad<float>(&scope, ctx);
}
