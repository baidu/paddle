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

USE_OP(top_k);
USE_OP_DEVICE_KERNEL(top_k, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto x = scope->Var("X");
  auto tensor_x = x->GetMutable<f::LoDTensor>();

  auto k = scope->Var("K");
  auto tensor_k = k->GetMutable<f::LoDTensor>();

  int dim0 = 100;
  int top_num = 5;

  std::vector<T> init;
  for (int64_t i = 0; i < dim0; ++i) {
    init.push_back(static_cast<T>(0.01 * i));
  }
  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({dim0});
  ctx.Wait();

  std::vector<int> init_k;
  for (int i = 0; i < 1; i++ ) {
      init_k.push_back(top_num);
  }
  TensorFromVector(init_k, ctx, tensor_k);
  tensor_k->Resize({1});
  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  auto indices = scope->Var("Indices");
  auto tensor_indices = indices->GetMutable<f::LoDTensor>();

  // run
  auto op =
      f::OpRegistry::CreateOp("top_k", {{"X", {"X"}}, {"K", {"K"}}},
                              {{"Out", {"Out"}}, {"Indices", {"Indices"}}}, {});

  op->Run(*scope, place);
  ctx.Wait();

  /*
  struct timeval start, end;
  gettimeofday(&start, NULL);
  for(int i=0; i<100; i++){
    op->Run(*scope, place);
  }
  ctx.Wait();
  gettimeofday(&end, NULL);
  int micros = (((end.tv_sec - start.tv_sec) * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("time:%d\n" , micros/100);
  */

  for (auto i = 0; i < tensor_out->dims().size(); ++i){
      VLOG(3) << "dim:" <<  i << " " << tensor_out->dims()[i];
  }

  f::Tensor cpu_tensor;
  TensorCopySync(*tensor_out, p::CPUPlace(), &cpu_tensor);
  auto data = cpu_tensor.data<T>();
  auto vec_data = std::vector<T>(data, data + tensor_out->numel());
  for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << "top_k vec_data_out["<< i << "] = " << vec_data[i];
  }

  f::Tensor cpu_tensor1;
  TensorCopySync(*tensor_indices, p::CPUPlace(), &cpu_tensor1);
  auto data1 = cpu_tensor1.data<T>();
  auto vec_data1 = std::vector<T>(data1, data1 + tensor_indices->numel());
  for(int i=0; i<static_cast<int>(vec_data1.size()); ++i){
    VLOG(3) << "topk index_out["<< i << "] = " << vec_data1[i];
  }


  ctx.Wait();

  // EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(dim0 * dim1 * dim2));
};


TEST(top_k, NPU_fp16) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<p::float16>(&scope, ctx);
}

