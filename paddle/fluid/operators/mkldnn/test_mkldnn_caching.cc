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
#include <map>
#include <random>
#include <string>
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
USE_OP(elementwise_mul);
USE_OP_DEVICE_KERNEL(elementwise_mul, MKLDNN);
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

class CacheTester {
 public:
  CacheTester() {
    // Clear oneDNN cache
    auto &pool = platform::DeviceContextPool::Instance();
    platform::CPUPlace place;
    onednn_dev_ctx_ =
        dynamic_cast<platform::MKLDNNDeviceContext *>(pool.Get(place));
    onednn_dev_ctx_->ResetBlobMap(nullptr);
  }

  bool Analyze(unsigned short int num_entries) {
    //  Number of created objects in cache should be as expected (num_entries)
    return onednn_dev_ctx_->GetCachedObjectsNumber() == num_entries;
  }

 private:
  platform::MKLDNNDeviceContext *onednn_dev_ctx_;
};

template <typename T>
void RunOperator(const platform::Place &place, const std::string &op_type,
                 const framework::DDim &dims, const std::string &output_name,
                 bool inplace = false) {
  framework::Scope scope;

  std::map<const std::string, int> num_inputs = {{"softmax", 1},
                                                 {"relu", 1},
                                                 {"elementwise_add", 2},
                                                 {"elementwise_mul", 2}};

  std::string first_input = inplace == true ? output_name : "x";

  std::vector<InputVars> input_names = {
      {first_input, scope.Var(first_input)->GetMutable<framework::LoDTensor>()},
      {"x1", num_inputs[op_type] > 1
                 ? scope.Var("x1")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x2", num_inputs[op_type] > 2
                 ? scope.Var("x2")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x3", num_inputs[op_type] > 3
                 ? scope.Var("x3")->GetMutable<framework::LoDTensor>()
                 : nullptr},
      {"x4", num_inputs[op_type] > 4
                 ? scope.Var("x4")->GetMutable<framework::LoDTensor>()
                 : nullptr}};
  auto *y = scope.Var(output_name)->GetMutable<framework::LoDTensor>();

  // Initialize input data
  std::uniform_real_distribution<T> dist(static_cast<T>(10.0),
                                         static_cast<T>(20.0));
  std::mt19937 engine;
  size_t numel = static_cast<size_t>(framework::product(dims));
  for (int i = 0; i < num_inputs[op_type]; ++i) {
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

  auto op = num_inputs[op_type] > 1
                ? framework::OpRegistry::CreateOp(
                      op_type, {{"X", {first_input}}, {"Y", {"x1"}}},
                      {{"Out", {output_name}}}, {{"use_mkldnn", {true}}})
                : framework::OpRegistry::CreateOp(
                      op_type, {{"X", {first_input}}}, {{"Out", {output_name}}},
                      {{"use_mkldnn", {true}}});

  op->Run(scope, place);
  pool.Get(place)->Wait();
}

TEST(test_softmax_reuse_cache, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "softmax", dims, "softmax_out");
  RunOperator<float>(p, "softmax", dims, "softmax_out");
  PADDLE_ENFORCE_EQ(ct.Analyze(4), true,
                    platform::errors::InvalidArgument(
                        "Wrong number of cached oneDNN objects"));
}

TEST(test_softmax_noreuse_cache, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "softmax", dims, "softmax_out");
  RunOperator<float>(p, "softmax", dims, "softmax_out2");
  PADDLE_ENFORCE_EQ(ct.Analyze(8), true,
                    platform::errors::InvalidArgument(
                        "Wrong number of cached oneDNN objects"));
}

TEST(test_softmax_inplace_cache, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "softmax", dims, "softmax_out");
  RunOperator<float>(p, "softmax", dims, "softmax_out", true);
  PADDLE_ENFORCE_EQ(ct.Analyze(7), true,
                    platform::errors::InvalidArgument(
                        "Wrong number of cached oneDNN objects"));
}

TEST(test_relu_inplace_cache, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "relu", dims, "relu_out");
  RunOperator<float>(p, "relu", dims, "relu_out", true);
  PADDLE_ENFORCE_EQ(ct.Analyze(7), true,
                    platform::errors::InvalidArgument(
                        "Wrong number of cached oneDNN objects"));
}

TEST(test_elementwise_add_reuse_cache, cpu_place) {
  framework::DDim dims({32, 64});
  platform::CPUPlace p;
  CacheTester ct;
  RunOperator<float>(p, "elementwise_add", dims, "elementwise_add_out");
  RunOperator<float>(p, "relu", dims, "elementwise_add_out", true);
  PADDLE_ENFORCE_EQ(ct.Analyze(8), true,
                    platform::errors::InvalidArgument(
                        "Wrong number of cached oneDNN objects"));
}

}  // namespace operators
}  // namespace paddle
