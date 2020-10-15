// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <random>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"

USE_OP(pool2d);
USE_OP_DEVICE_KERNEL(pool2d, MKLDNN);
USE_OP(transpose);
USE_OP_DEVICE_KERNEL(transpose, MKLDNN);

namespace paddle {
namespace operators {

struct InputVars {
  std::string name;
  framework::LoDTensor *tensor;
};

TEST(test_pool2d_transpose_nhwc, cpu_place) {
  framework::DDim dims({1, 4, 8, 512});           // NHWC shape
  framework::DDim expected_dims({1, 7, 512, 3});  // NHWC expected shape
  platform::CPUPlace p;
  framework::Scope scope;

  InputVars input_name = {"x",
                          scope.Var("x")->GetMutable<framework::LoDTensor>()};
  // Initialize input data
  std::uniform_real_distribution<float> dist(static_cast<float>(10.0),
                                             static_cast<float>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(framework::product(dims));
  input_name.tensor->Resize(dims);
  auto data_ptr = input_name.tensor->mutable_data<float>(p);
  for (size_t i = 0; i < numel; ++i) {
    data_ptr[i] = dist(engine);
  }

  scope.Var("y")->GetMutable<framework::LoDTensor>();
  auto *z = scope.Var("z")->GetMutable<framework::LoDTensor>();

  auto &pool = platform::DeviceContextPool::Instance();

  // Make pool2d followed by transpose

  auto ksize = std::vector<int>(2, 2);
  auto op_pool = framework::OpRegistry::CreateOp(
      "pool2d", {{"X", {"x"}}}, {{"Out", {"y"}}},
      {{"pooling_type", {std::string("max")}},
       {"ksize", {ksize}},
       {"data_format", {std::string("NHWC")}},
       {"use_mkldnn", {true}}});

  auto axis = std::vector<int>(4, 0);
  axis[1] = 2;
  axis[2] = 3;
  axis[3] = 1;
  auto op_transpose = framework::OpRegistry::CreateOp(
      "transpose", {{"X", {"y"}}}, {{"Out", {"z"}}},
      {{"axis", {axis}}, {"use_mkldnn", {true}}});

  op_pool->Run(scope, p);
  op_transpose->Run(scope, p);
  pool.Get(p)->Wait();

  // Verify shape of output
  PADDLE_ENFORCE_EQ(z->dims(), expected_dims,
                    platform::errors::InvalidArgument(
                        "Computed shape does not match expected shape"));
}

}  // namespace operators
}  // namespace paddle
