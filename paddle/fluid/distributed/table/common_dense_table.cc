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

#include "paddle/fluid/distributed/table/common_dense_table.h"

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

void CommonDenseTable::create_initializer(const std::string& attr,
                                          const std::string& name) {
  auto slices = string::split_string<std::string>(attr, "&");

  if (slices[0] == "gaussian_random") {
    initializers_[name] = new GaussianInitializer(slices);
  } else if (slices[0] == "fill_constant") {
    initializers_[name] = new FillConstantInitializer(slices);
  } else if (slices[0] == "uniform_random") {
    initializers_[name] = new UniformInitializer(slices);
  } else if (slices[0] == "truncated_gaussian_random") {
    initializers_[name] = new TruncatedGaussianInitializer(slices);
  } else {
    PADDLE_THROW(
        platform::errors::InvalidArgument("%s can not be supported", name));
  }
}

int32_t CommonDenseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  sync = _config.common().sync();
  VLOG(1) << "table " << _config.common().table_name() << " is sync: " << sync;
  _global_lr = new float(1.0);

  initialize_value();
  initialize_optimizer();
  return 0;
}

int32_t CommonDenseTable::initialize_value() {
  auto common = _config.common();
  int size = static_cast<int>(common.params().size());
  values_.resize(size);
  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];
    if (varname == "Param") {
      param_dim_ = dim;
      param_idx_ = x;
    }
    auto& initializer = common.initializers()[x];

    create_initializer(initializer, varname);
    values_[x].resize(dim);
    names_index_[varname] = x;

    for (int y = 0; y < dim; ++y) {
      values_[x][y] = initializers_[varname]->GetValue();
    }
  }

  pull_reservoir_ = ReservoirValue<float>(param_dim_);
  return 0;
}

int32_t CommonDenseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();
  auto attrs = common.attributes();

  if (name == "sgd") {
    optimizer_ = std::make_shared<DSGD>(common, &values_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<DAdam>(common, &values_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "sum") {
    optimizer_ = std::make_shared<DSUM>(common, &values_);
  } else if (name == "adagrad") {
    optimizer_ = std::make_shared<DAdagrad>(common, &values_);
    optimizer_->set_global_lr(_global_lr);
  } else {
    VLOG(0) << "init optimizer failed";
  }
  VLOG(3) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonDenseTable::set_global_lr(float* lr) {
  _global_lr = lr;
  optimizer_->set_global_lr(_global_lr);
  return 0;
}

int32_t CommonDenseTable::pull_dense(float* pull_values, size_t num) {
  std::copy(values_[param_idx_].begin(), values_[param_idx_].end(),
            pull_values);
  return 0;
}

int32_t CommonDenseTable::push_dense_param(const float* values, size_t num) {
  PADDLE_ENFORCE_GE(
      num, param_dim_,
      paddle::platform::errors::InvalidArgument(
          "update desne param numel expected %d, but got %d", param_dim_, num));
  std::copy_n(values, param_dim_, values_[param_idx_].begin());
  return 0;
}

int32_t CommonDenseTable::pour() {
  pull_reservoir_.avg();
  _push_dense(pull_reservoir_.values.data(), pull_reservoir_.values.size());
  pull_reservoir_.reset();
  return 0;
}

int32_t CommonDenseTable::push_dense(const float* values, size_t num) {
  if (sync) {
    std::future<int> task =
        _shards_task_pool[0]->enqueue([this, &values]() -> int {
          pull_reservoir_.add(values, param_dim_);
          return 0;
        });
    task.wait();
  } else {
    _push_dense(values, num);
  }
  return 0;
}

int32_t CommonDenseTable::_push_dense(const float* values, size_t num) {
  PADDLE_ENFORCE_GE(
      num, param_dim_,
      paddle::platform::errors::InvalidArgument(
          "update desne numel expected %d, but got %d", param_dim_, num));

  std::vector<int> buckets = bucket(param_dim_, task_pool_size_);
  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &buckets, &values]() -> int {
          auto begin = buckets[shard_id];
          auto end = buckets[shard_id + 1];
          optimizer_->update(values, param_dim_, begin, end);
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

}  // namespace distributed
}  // namespace paddle
