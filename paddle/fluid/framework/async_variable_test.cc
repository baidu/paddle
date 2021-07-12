/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include <thread>  // NOLINT
#include <utility>

#include "paddle/fluid/framework/async_variable.h"

#include "gtest/gtest.h"

namespace paddle {
namespace framework {

TEST(AsyncVariableTest, EmplaceGetSameThread) {
  AsyncVariable async_variable;
  int origin = 3;
  async_variable.Emplace<int>(std::forward<int>(origin));
  int value = async_variable.Get<int>();
  EXPECT_EQ(value, 3);

  int* mutable_value = async_variable.GetMutable<int>();
  EXPECT_EQ(*mutable_value, 3);

  *mutable_value = 5;
  EXPECT_EQ(value, 3);
  EXPECT_EQ(async_variable.Get<int>(), 5);
}

void emplace_async_variable(AsyncVariable* async_variable) {
  async_variable->Emplace<float>(std::forward<float>(57.0f));
}

TEST(AsyncVariableTest, ThreadJoinEmplace) {
  AsyncVariable async_variable;
  std::thread check_thread =
      std::thread(emplace_async_variable, &async_variable);
  check_thread.join();
  float value = async_variable.Get<float>();
  EXPECT_EQ(value, 57.0f);
}

void mutable_async_variable(AsyncVariable* async_variable) {
  int* mutable_value = async_variable->GetMutable<int>();
  *mutable_value = 14;
}

TEST(AsyncVariableTest, ThreadJoinMutable) {
  AsyncVariable async_variable;
  std::thread check_thread =
      std::thread(mutable_async_variable, &async_variable);
  check_thread.join();
  int value = async_variable.Get<int>();
  EXPECT_EQ(value, 14);
}

}  // namespace framework
}  // namespace paddle
