// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/details/computation_op_handle.h"
#include "paddle/fluid/framework/details/var_handle.h"
#include "paddle/fluid/framework/garbage_collector.h"

namespace paddle {
namespace framework {

class VarDesc;

namespace ir {

using GarbageCollectorMap =
    std::map<platform::Place, std::unique_ptr<GarbageCollector>>;

const char kMemOptVarInfoMapList[] = "mem_opt_var_info_map_list";
const char kGarbageCollector[] = "garbage_collector";
const char kAllPlaces[] = "all_places";
const char kUseCuda[] = "use_cuda";

class LastLiveOpOfVarInfo {
 public:
  details::VarHandle *var() const { return var_; }

  void set_var(details::VarHandle *var) { var_ = var; }

  const std::unordered_set<details::ComputationOpHandle *> &ops() const {
    return ops_;
  }

  std::unordered_set<details::ComputationOpHandle *> *mutable_ops() {
    return &ops_;
  }

 private:
  details::VarHandle *var_{nullptr};
  std::unordered_set<details::ComputationOpHandle *> ops_;
};

using LastLiveOpsOfVars = std::unordered_map<std::string, LastLiveOpOfVarInfo>;
const char kLastLiveOpsOfVars[] = "last_live_ops_of_var";

VarDesc *TryGetLatestVarDesc(const std::vector<details::VarHandle *> &vars);

using SkipReuseVars = std::unordered_set<std::string>;
const char kSkipReuseVars[] = "skip_reuse_var";

}  // namespace ir
}  // namespace framework
}  // namespace paddle
