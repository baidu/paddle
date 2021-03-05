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
  for (int i = 0; i < 6; ++i) {
    init.push_back(static_cast<T>(i));
  }

  TensorFromVector(init, ctx, tensor_x);
  tensor_x->Resize({2, 3});

  ctx.Wait();

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();
  tensor_out->Resize({2, 3});
  tensor_out->mutable_data<T>(place); // allocate

  // run
  int axis = 1;
  f::AttributeMap attrs = {{"axis", axis}};
  
  auto op =
      f::OpRegistry::CreateOp("softmax", {{"X", {"X"}}},
                              {{"Out", {"Out"}}}, attrs);

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
  //printf("time:%d\n" , micros/100);
  VLOG(3) << "time: " << micros/100;
  */

  std::vector<T> out_vec;
  TensorToVector(*tensor_out, ctx, &out_vec);

  for (int i = 0; i < static_cast<int>(out_vec.size()); ++i){
       VLOG(3) << "out_vec[" << i << "] : "<< out_vec[i];
  }
  
  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(6));
};


template <typename T>
void CompareGrad(f::Scope* scope, const p::DeviceContext& ctx) {
  // init
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  std::vector<T> out_init;
  out_init.push_back(static_cast<T>(0.2));
  out_init.push_back(static_cast<T>(0.3));
  out_init.push_back(static_cast<T>(0.5));
  out_init.push_back(static_cast<T>(0.1));
  out_init.push_back(static_cast<T>(0.3));
  out_init.push_back(static_cast<T>(0.6));

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
  auto op =
      f::OpRegistry::CreateOp("softmax_grad", {{"Out", {"Out"}}, {"Out@GRAD", {"DOut"}}},
                              {{"X@GRAD", {"DX"}}}, attrs);

  auto place = ctx.GetPlace();
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
  //printf("time:%d\n" , micros/100);
  VLOG(3) << "time: " << micros/100;

  */
  for (auto i = 0; i < tensor_dx->dims().size(); ++i){
      VLOG(3) << "softmax grad dim: " << i << " " << tensor_dx->dims()[i];
  }

  ctx.Wait();
  /*
  f::LoDTensor cpu_tensor;
  TensorCopySync(*tensor_dx, p::CPUPlace(), &cpu_tensor);
  auto data = cpu_tensor.data<T>();
  auto vec_data = std::vector<T>(data, data + tensor_dx->numel());
  for(int i=0; i <static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << "softmax vec_data_out["<< i << "] = " << vec_data[i];
  }

  std::vector<T> out_vec;
  TensorToVector(*tensor_dx, ctx, &out_vec);

  for (int i = 0; i < static_cast<int>(out_vec.size()); ++i){
       VLOG(3) << "softmax out_vec[" << i << "] : "<< out_vec[i];
  }
  
  ctx.Wait();

  EXPECT_EQ((uint32_t)out_vec.size(), (uint32_t)(6));
  */
};

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

