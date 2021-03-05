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

USE_OP(uniform_random);
USE_OP_DEVICE_KERNEL(uniform_random, NPU);

template <typename T>
void Compare(f::Scope* scope, const p::DeviceContext& ctx) {
  // init

  std::vector<T> shape;
  for (int64_t i = 1; i < 3; ++i) {
    shape.push_back(i);
  }

  auto min = static_cast<float>(0.1);
  auto max = static_cast<float>(0.9);
  auto seed = static_cast<int>(2021);

  auto place = ctx.GetPlace();
  auto out = scope->Var("Out");
  auto tensor_out = out->GetMutable<f::LoDTensor>();

  // run
  f::AttributeMap attrs = {{"shape", shape}, 
                           {"min", min}, 
                           {"max", max},
                           {"seed", seed}};
  
  auto op =
      f::OpRegistry::CreateOp("uniform_random", {},
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

  /*
  f::Tensor cpu_tensor;
  TensorCopySync(*tensor_out, p::CPUPlace(), &cpu_tensor);
  auto data = cpu_tensor.data<T>();
  auto vec_data = std::vector<T>(data, data + tensor_out->numel());
  for(int i=0; i<static_cast<int>(vec_data.size()); ++i){
    VLOG(3) << "uniform_random vec_data_out["<< i << "] = " << vec_data[i];
  }

  */

  for (auto i = 0; i < tensor_out->dims().size(); ++i){
      //printf("dim%d: %ld", i, tensor_out->dims()[i]);
      VLOG(3) << "dim" << i << ": " << tensor_out->dims()[i];
  }
  
  ctx.Wait();

};


TEST(uniform_random, NPU_fp32) {
  f::Scope scope;
  p::NPUDeviceContext ctx(p::NPUPlace(0));
  Compare<float>(&scope, ctx);
}
