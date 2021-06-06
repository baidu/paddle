// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "paddle/fluid/imperative/tracer.h"

namespace paddle {
namespace pybind {

uint64_t time_delete_varbase = 0;

class tmpclass2 {
 public:
  tmpclass2() {}
  ~tmpclass2() {
    std::cout << "time_delete_varbase " << time_delete_varbase << std::endl;
  }
  int a;
};
tmpclass2 aa;
int run_times = 0;

void VarBaseDeleter(imperative::VarBase* ptr) {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  auto start = (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
  delete ptr;
  gettimeofday(&tv, nullptr);
  auto end = (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
  time_delete_varbase = end - start + time_delete_varbase;
}
}
}
