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

#pragma once

#include <ThreadPool.h>
#include <assert.h>
#include <pthread.h>
#include <string>
#include "Eigen/Dense"
#include "paddle/fluid/distributed/table/accessor.h"
#include "paddle/fluid/distributed/table/common_table.h"
#include "paddle/fluid/distributed/table/depends/dense.h"
#include "paddle/fluid/distributed/table/depends/initializers.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace distributed {

class DenseOptimizer;

class CommonDenseTable : public DenseTable {
 public:
  explicit CommonDenseTable() {}
  virtual ~CommonDenseTable() {}
  virtual int32_t initialize() override;
  virtual int32_t initialize_shard() override { return 0; }
  virtual void create_initializer(const std::string& attr,
                                  const std::string& name);
  virtual int32_t initialize_value();
  virtual int32_t initialize_optimizer();
  virtual int32_t pull_dense(float* pull_values, size_t num) override;
  virtual int32_t push_dense_param(const float* values, size_t num) override;
  virtual int32_t push_dense(const float* values, size_t num) override;
  virtual int32_t pour() override;
  virtual int32_t set_global_lr(float* lr) override;

  int32_t load(const std::string& path, const std::string& param) override {
    VLOG(0) << "Dense table may load by "
               "paddle.distributed.fleet.init_server";
    return 0;
  }

  int32_t save(const std::string& path, const std::string& param) override {
    VLOG(0)
        << "Dense table may be saved by "
           "paddle.distributed.fleet.save_persistables/save_inference_model";
    return 0;
  }

  virtual int32_t flush() override { return 0; }
  virtual int32_t shrink(const std::string& param) override { return 0; }
  virtual void clear() override { return; }

  virtual int32_t set_program_env(
      framework::Scope* scope, platform::Place place,
      const std::vector<framework::ProgramDesc>* sub_program) override {
    sub_program_ = sub_program;
    return 0;
  }

 protected:
  int32_t _push_dense(const float* values, size_t num);

 private:
  const int task_pool_size_ = 1;
  bool sync = true;
  std::vector<std::shared_ptr<::ThreadPool>> _shards_task_pool;
  int param_dim_ = 0;
  int param_idx_ = 0;
  std::shared_ptr<DenseOptimizer> optimizer_ = nullptr;
  std::vector<std::vector<float>> values_;
  ReservoirValue<float> pull_reservoir_;
  std::unordered_map<std::string, Initializer*> initializers_;
  std::unordered_map<std::string, int> names_index_;

  const std::vector<framework::ProgramDesc>* sub_program_;
};

}  // namespace distributed
}  // namespace paddle
