/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_PSLIB

#include <map>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"
#include "paddle/fluid/framework/scope.h"
#include "common_value.h"

namespace paddle {
namespace framework {

class GpuTask {
 public:
  Scope* scope_{nullptr};
  std::vector<std::vector<FeatureKey>> feature_keys_;
  std::vector<std::vector<paddle::ps::DownpourFixedFeatureValue*>> value_ptr_;
  std::vector<std::vector<FeatureValue>> feature_values_;
  uint64_t size() {
    uint64_t total_size = 0;
    for (auto& keys : feature_keys_) {
      total_size += keys.size();
    }
    return total_size;
  }
};

}  // end namespace framework
}  // end namespace paddle
#endif
