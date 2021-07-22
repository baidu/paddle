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

#include <thread>  // NOLINT
#include <utility>
#include "gtest/gtest.h"

#include "paddle/fluid/framework/async_variable.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace framework {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

void set_async_variable(AsyncVariable* async_var, float value) {
  Tensor* t = async_var->GetMutable<Tensor>();
  t->Resize({2, 4});
  auto* data = t->mutable_data<float>(platform::CPUPlace());
  for (int i = 0; i < t->numel(); ++i) {
    data[i] = value;
  }
}

TEST(AsyncVariableTest, EmplaceGetSameThread) {
  AsyncVariable async_var;

  async_var.Emplace<Tensor>();
  // var is available after Emplace
  EXPECT_TRUE(async_var.isAvailable());
  float value = 57;
  set_async_variable(&async_var, value);

  Tensor t = async_var.Get<Tensor>();
  auto* data = t.data<float>();
  for (int i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(data[i], value);
  }
}

TEST(AsyncVariableTest, ThreadJoin) {
  AsyncVariable async_var;
  async_var.Emplace<Tensor>();

  float value = 42;
  std::thread check_thread = std::thread(set_async_variable, &async_var, value);
  check_thread.join();

  Tensor t = async_var.Get<Tensor>();
  auto* data = t.data<float>();
  for (int i = 0; i < t.numel(); ++i) {
    EXPECT_EQ(data[i], value);
  }
}

}  // namespace framework
}  // namespace paddle
