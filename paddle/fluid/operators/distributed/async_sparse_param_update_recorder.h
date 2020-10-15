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

#include <ThreadPool.h>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {
namespace distributed {

class ConcurrentSet {
 public:
  ConcurrentSet() : pool_(new ::ThreadPool(1)) {}
  ~ConcurrentSet() {}

  std::future<void> Update(const std::vector<int64_t>& rows) {
    auto task = [this, rows] {
      if (VLOG_IS_ON(3)) {
        std::ostringstream sstream;
        sstream << "[";
        for (auto& id : rows) {
          sstream << id << ", ";
        }
        sstream << "]";
        VLOG(3) << "update ids -> " << sstream.str();
      }
      for (auto row : rows) {
        set_.insert(row);
      }
    };
    return pool_->enqueue(std::move(task));
  }

  std::future<void> GetAndClear(std::vector<int64_t>* result) {
    auto task = [this, &result] {
      result->clear();
      for (auto& id : set_) {
        result->push_back(id);
      }
      if (VLOG_IS_ON(3)) {
        std::ostringstream sstream;
        sstream << "[";
        for (auto& id : *result) {
          sstream << id << ", ";
        }
        sstream << "]";
        VLOG(3) << "result ids size: " << result->size() << " "
                << sstream.str();
      }
      set_.clear();
    };
    return pool_->enqueue(std::move(task));
  }

 private:
  std::unordered_set<int64_t> set_;
  std::unique_ptr<::ThreadPool> pool_{nullptr};
};

class AsyncSparseParamUpdateRecorder {
  using TrainerToRows = std::vector<std::unique_ptr<ConcurrentSet>>;

 public:
  AsyncSparseParamUpdateRecorder(
      int trainer_num,
      const std::unordered_map<std::string, std::string>& grad_to_param)
      : trainer_num_(trainer_num), grad_to_param_(grad_to_param) {
    if (VLOG_IS_ON(3)) {
      std::ostringstream sstream;
      sstream << "[";
      for (auto& item : grad_to_param) {
        sstream << item.first << ":" << item.second << ", ";
      }
      sstream << "]";
      VLOG(3) << "trainer_num: " << trainer_num
              << " grad_to_param_: " << sstream.str();
    }
    for (auto& iter : grad_to_param) {
      param_to_grad_[iter.second] = iter.first;
      auto& param_name = iter.second;
      param_to_updated_rows_[param_name] = TrainerToRows();
      auto& trainer_to_rows = param_to_updated_rows_[param_name];
      for (auto i = 0; i < trainer_num; ++i) {
        trainer_to_rows.emplace_back(new ConcurrentSet());
      }
    }
  }

  ~AsyncSparseParamUpdateRecorder() = default;

  void Update(const std::string& grad_name,
              const std::vector<int64_t>& update_rows) {
    VLOG(3) << "update grad: " << grad_name
            << " row size: " << update_rows.size();
    auto& param_name = grad_to_param_.at(grad_name);
    auto& trainer_to_rows = param_to_updated_rows_.at(param_name);

    std::vector<std::future<void>> fs;
    for (auto& set : trainer_to_rows) {
      fs.push_back(set->Update(update_rows));
    }
    for (auto& f : fs) {
      f.wait();
    }
  }

  void GetAndClear(const std::string& param_name, int trainer_id,
                   std::vector<int64_t>* result) {
    VLOG(3) << "GetAndClear param: " << param_name
            << " for trainer: " << trainer_id;
    PADDLE_ENFORCE_LT(
        trainer_id, trainer_num_,
        platform::errors::InvalidArgument(
            "The value of trainer_id: %s should less than trainer_num: %s.",
            trainer_id, trainer_num_));
    param_to_updated_rows_.at(param_name)[trainer_id]
        ->GetAndClear(result)
        .wait();
  }

  bool HasParam(const std::string& param_name) {
    return param_to_grad_.find(param_name) != param_to_grad_.end();
  }

  bool HasGrad(const std::string& grad_name) {
    return grad_to_param_.find(grad_name) != grad_to_param_.end();
  }

 private:
  const int trainer_num_;
  std::unordered_map<std::string, std::string> grad_to_param_;
  std::unordered_map<std::string, std::string> param_to_grad_;
  std::unordered_map<std::string, TrainerToRows> param_to_updated_rows_;

  // init recorder
 public:
  static void Init(
      int trainer_num,
      const std::unordered_map<std::string, std::string>& grad_to_param) {
    InitImpl(trainer_num, grad_to_param);
  }

  static AsyncSparseParamUpdateRecorder* GetInstance() {
    return recorder_.get();
  }

 private:
  // Init is called by GetInstance.
  static void InitImpl(
      int trainer_num,
      const std::unordered_map<std::string, std::string>& grad_to_param) {
    if (recorder_ == nullptr) {
      recorder_.reset(
          new AsyncSparseParamUpdateRecorder(trainer_num, grad_to_param));
    }
  }

  static std::once_flag init_flag_;
  static std::unique_ptr<AsyncSparseParamUpdateRecorder> recorder_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
