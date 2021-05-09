// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class MemOptVarInfo {
 public:
  MemOptVarInfo(const std::string &name, size_t ref_cnt) : name_(name) {
    SetRefCnt(ref_cnt);
  }

  bool DecreaseRefCnt() {
    return ref_cnt_ == 1 || (runtime_ref_cnt_.fetch_sub(1) == 1);
  }

  void ResetRuntimeRefCnt() {
    if (ref_cnt_ != 1) {
      runtime_ref_cnt_ = ref_cnt_;
    }
  }

  void SetRefCnt(size_t ref_cnt) {
    PADDLE_ENFORCE_GE(
        ref_cnt, 1,
        platform::errors::InvalidArgument(
            "Reference count(%d) must be larger than or equal to 1.", ref_cnt));
    ref_cnt_ = ref_cnt;
    runtime_ref_cnt_ = ref_cnt;
  }

  // Skip all memory optimization, including memory reuse and garbage collection
  void SetSkipAllMemoryOptimization(bool is_skipped) {
    skip_all_memory_optimization_ = is_skipped;
  }

  bool IsSkippedAllMemoryOptimization() const {
    return skip_all_memory_optimization_;
  }

  // Skip all memory reuse, including inplace and cross op memory reuse
  void SetSkipMemoryReuse(bool is_skipped) { skip_memory_reuse_ = is_skipped; }

  bool IsSkippedMemoryReuse() const {
    return skip_memory_reuse_ || skip_all_memory_optimization_;
  }

  const std::string &Name() const { return name_; }

 private:
  std::string name_;

  /**
   * ref_cnt_ is the total number of last-lived ops of variable. It would not
   * be changed during iterations.
   *
   * runtime_ref_cnt_ is the runtime reference count of variable, which would
   * decrease 1 when each EagerDeletionOpHandle runs. As a result, it should
   * be reset to ref_cnt_ after each iteration ends. Since operators are
   * scheduled in many threads inside ParallelExecutor, runtime_ref_cnt_
   * must be an atomic integer to guarantee the thread safety and visibility.
   *
   * Speciallly, if ref_cnt_ is 1, we do not need to reset runtime_ref_cnt_
   * after iteration ends.
   */
  size_t ref_cnt_;
  std::atomic<size_t> runtime_ref_cnt_;
  bool skip_memory_reuse_{false};
  bool skip_all_memory_optimization_{false};
};

using MemOptVarInfoMapList = std::vector<
    std::unordered_map<std::string, std::shared_ptr<MemOptVarInfo>>>;

class SkipMemOptVarsGuard {
 public:
  SkipMemOptVarsGuard(MemOptVarInfoMapList *list,
                      const std::vector<std::string> &vars,
                      bool need_reset_ref_cnt)
      : list_(list), need_reset_ref_cnt_(need_reset_ref_cnt) {
    if (!list_) return;

    skip_vars_.reserve(vars.size() * list->size());
    for (auto &var : vars) {
      for (auto &map : *list_) {
        auto iter = map.find(var);
        if (iter != map.end() &&
            !iter->second->IsSkippedAllMemoryOptimization()) {
          VLOG(3) << "Set skip memory optimzation of " << var;
          iter->second->SetSkipAllMemoryOptimization(true);
          skip_vars_.emplace_back(iter->second.get());
        }
      }
    }
  }

  ~SkipMemOptVarsGuard() {
    for (auto *var : skip_vars_) {
      var->SetSkipAllMemoryOptimization(false);
    }

    if (list_ && need_reset_ref_cnt_) {
      for (auto &map : *list_) {
        for (auto &pair : map) {
          pair.second->ResetRuntimeRefCnt();
        }
      }
    }
  }

 private:
  MemOptVarInfoMapList *list_;
  bool need_reset_ref_cnt_;
  std::vector<MemOptVarInfo *> skip_vars_;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
