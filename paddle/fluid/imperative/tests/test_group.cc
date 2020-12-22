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

#include <memory>
#include <ostream>
#include <sstream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/imperative/reducer.h"
#endif

namespace paddle {
namespace imperative {

#if defined(PADDLE_WITH_NCCL)
TEST(TestGroup, TestPrintGroupMessage) {
  Group group;
  std::stringstream stream1, stream2;
  stream1 << group;
  ASSERT_STREQ(stream1.str().c_str(),
               "numel: 0 ;is_sparse: 0 ;var number: 0\n[]\n");

  std::vector<size_t> vars;
  size_t vars_num = 102;
  for (size_t i = 0; i < vars_num; ++i) {
    vars.push_back(i);
  }
  group.variable_indices_ = vars;
  group.all_length_ = 102;
  group.is_sparse_ = false;

  std::string head = "numel: 102 ;is_sparse: 0 ;var number: 102\n";
  head = head + "[";
  auto begin = vars.begin();
  auto end = vars.end();
  for (int i = 0; begin != end && i < 100; ++i, ++begin) {
    if (i > 0) head += ' ';
    head += std::to_string(*begin);
  }
  if (begin != end) {
    head += " ...";
  }
  head += "]\n";
  stream2 << group;
  ASSERT_STREQ(stream2.str().c_str(), head.c_str());
}

#endif

}  // namespace imperative
}  // namespace paddle
