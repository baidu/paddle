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

USE_OP(elementwise_add);
USE_OP_DEVICE_KERNEL(elementwise_add, MKLDNN);
USE_OP(relu);
USE_OP_DEVICE_KERNEL(relu, MKLDNN);
USE_OP(softmax);
USE_OP_DEVICE_KERNEL(softmax, MKLDNN);

namespace paddle {
namespace operators {

struct InputVars {
  std::string name;
  framework::LoDTensor *tensor;
};

template <typename T>
bool TestMain(const platform::Place &place, const std::string &op_type,
              const framework::DDim &dims, const int num_inputs) {
  framework::Scope scope;

  std::vector<InputVars> input_names = {
      {"x", scope.Var("x")->GetMutable<framework::LoDTensor>()},
      {"x1", num_inputs > 1
                 ? scope.Var("x1")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x2", num_inputs > 2
                 ? scope.Var("x2")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x3", num_inputs > 3
                 ? scope.Var("x3")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x4", num_inputs > 4
                 ? scope.Var("x4")->GetMutable<framework::LoDTensor>()
                 : nullptr}};
  auto *y = scope.Var("y")->GetMutable<framework::LoDTensor>();

  // Initialize input data
  std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                         static_cast<T>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(framework::product(dims));
  for (int i = 0; i < num_inputs; ++i) {
    input_names[i].tensor->Resize(dims);
    auto data_ptr = input_names[i].tensor->mutable_data<T>(place);
    for (size_t i = 0; i < numel; ++i) {
      data_ptr[i] = dist(engine);
    }
  }

  // Initialize output
  y->Resize(dims);
  auto y_ptr = y->mutable_data<T>(place);
  for (size_t i = 0; i < numel; ++i) {
    y_ptr[i] = static_cast<T>(0);
  }

  auto &pool = platform::DeviceContextPool::Instance();

  // Out of place (reference) computation
  auto op_ref = num_inputs > 1 ? framework::OpRegistry::CreateOp(
                                     op_type, {{"X", {"x"}}, {"Y", {"x1"}}},
                                     {{"Out", {"y"}}}, {{"use_mkldnn", {true}}})
                               : framework::OpRegistry::CreateOp(
                                     op_type, {{"X", {"x"}}}, {{"Out", {"y"}}},
                                     {{"use_mkldnn", {true}}});

  op_ref->Run(scope, place);
  pool.Get(place)->Wait();

  // Get reference (out of place) result
  auto &ref_tensor = scope.FindVar("y")->Get<framework::LoDTensor>();

  // In-place (to be tested) computation
  auto op = num_inputs > 1 ? framework::OpRegistry::CreateOp(
                                 op_type, {{"X", {"x"}}, {"Y", {"x1"}}},
                                 {{"Out", {"x"}}}, {{"use_mkldnn", {true}}})
                           : framework::OpRegistry::CreateOp(
                                 op_type, {{"X", {"x"}}}, {{"Out", {"x"}}},
                                 {{"use_mkldnn", {true}}});

  op->Run(scope, place);
  platform::DeviceContextPool::Instance().Get(place)->Wait();

  // Get in-place result
  auto &out_tensor = scope.FindVar("x")->Get<framework::LoDTensor>();
  PADDLE_ENFORCE_EQ(
      &out_tensor, input_names[0].tensor,
      platform::errors::InvalidArgument(
          "Input and output vars should share tensor for In-place test"));

  // compare results
  auto *ref_ptr = ref_tensor.data<T>();
  auto *out_ptr = out_tensor.data<T>();
  bool is_equal = std::equal(out_ptr, out_ptr + numel, ref_ptr);
  return is_equal;
}

TEST(test_softmax_inplace, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  ASSERT_TRUE(TestMain<float>(p, "softmax", dims, 1));
}

TEST(test_relu_inplace, cpu_place) {
  framework::DDim dims({1, 12, 20, 20});
  platform::CPUPlace p;
  ASSERT_TRUE(TestMain<float>(p, "relu", dims, 1));
}

}  // namespace operators
}  // namespace paddle
