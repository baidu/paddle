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

#include "paddle/fluid/distributed/table/common_sparse_table.h"
#include <algorithm>
#include <sstream>
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

#define PSERVER_SAVE_SUFFIX "_txt"
namespace paddle {
namespace distributed {

struct Meta {
  std::string param;
  int shard_id;
  std::vector<std::string> names;
  std::vector<int> dims;
  uint64_t count;
  std::unordered_map<std::string, int> dims_map;

  explicit Meta(const std::string& metapath) {
    std::ifstream file(metapath);
    std::string line;
    int num_lines = 0;
    while (std::getline(file, line)) {
      if (StartWith(line, "#")) {
        continue;
      }
      auto pairs = paddle::string::split_string<std::string>(line, "=");
      PADDLE_ENFORCE_EQ(
          pairs.size(), 2,
          paddle::platform::errors::InvalidArgument(
              "info in %s except k=v, but got %s", metapath, line));

      if (pairs[0] == "param") {
        param = pairs[1];
      }
      if (pairs[0] == "shard_id") {
        shard_id = std::stoi(pairs[1]);
      }
      if (pairs[0] == "row_names") {
        names = paddle::string::split_string<std::string>(pairs[1], ",");
      }
      if (pairs[0] == "row_dims") {
        auto dims_strs =
            paddle::string::split_string<std::string>(pairs[1], ",");
        for (auto& str : dims_strs) {
          dims.push_back(std::stoi(str));
        }
      }
      if (pairs[0] == "count") {
        count = std::stoull(pairs[1]);
      }
    }
    for (int x = 0; x < names.size(); ++x) {
      dims_map[names[x]] = dims[x];
    }
  }

  Meta(std::string param, int shard_id, std::vector<std::string> row_names,
       std::vector<int> dims, uint64_t count) {
    this->param = param;
    this->shard_id = shard_id;
    this->names = row_names;
    this->dims = dims;
    this->count = count;
  }

  std::string ToString() {
    std::stringstream ss;
    ss << "param=" << param << "\n";
    ss << "shard_id=" << shard_id << "\n";
    ss << "row_names=" << paddle::string::join_strings(names, ',') << "\n";
    ss << "row_dims=" << paddle::string::join_strings(dims, ',') << "\n";
    ss << "count=" << count << "\n";
    return ss.str();
  }
};

void ProcessALine(const std::vector<std::string>& columns, const Meta& meta,
                  std::vector<std::vector<float>>* values) {
  PADDLE_ENFORCE_EQ(columns.size(), 2,
                    paddle::platform::errors::InvalidArgument(
                        "The data format does not meet the requirements. It "
                        "should look like feasign_id \t params."));

  auto load_values = paddle::string::split_string<std::string>(columns[1], ",");
  values->reserve(meta.names.size());

  int offset = 0;
  for (int x = 0; x < meta.names.size(); ++x) {
    std::vector<float> val;
    auto start = load_values.begin() + offset;
    auto end = load_values.begin() + offset + meta.dims[x];
    PADDLE_ENFORCE_LE(offset + meta.dims[x], load_values.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The data format in txt does not meet the field "
                          "requirements defined in meta"));

    std::transform(start, end, std::back_inserter(val),
                   [](std::string va) { return std::stof(va); });
    values->push_back(val);
    offset += meta.dims[x];
  }
}

int64_t SaveToText(std::ostream* os, std::shared_ptr<ValueBlock> block,
                   const int mode) {
  for (auto value : block->values_) {
    auto* vs = value.second->data_.data();
    std::stringstream ss;
    auto id = value.first;
    ss << id << "\t";

    for (int i = 0; i < block->value_length_; i++) {
      ss << vs[i];
      ss << ",";
    }

    ss << "\n";

    os->write(ss.str().c_str(), sizeof(char) * ss.str().size());
  }

  return block->values_.size();
}

int64_t LoadFromText(const std::string& valuepath, const std::string& metapath,
                     const int pserver_id, const int pserver_num,
                     const int local_shard_num,
                     std::vector<std::shared_ptr<ValueBlock>>* blocks) {
  Meta meta = Meta(metapath);

  int num_lines = 0;
  std::ifstream file(valuepath);
  std::string line;

  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    auto id = std::stoull(values[0]);

    if (id % pserver_num != pserver_id) {
      VLOG(0) << "will not load " << values[0] << " from " << valuepath
              << ", please check id distribution";
      continue;
    }

    auto shard_id = id % local_shard_num;
    auto block = blocks->at(shard_id);

    std::vector<std::vector<float>> kvalues;
    ProcessALine(values, meta, &kvalues);
    // warning: need fix
    block->Init(id);
  }

  return 0;
}

int32_t CommonSparseTable::initialize() {
  _shards_task_pool.resize(task_pool_size_);
  for (int i = 0; i < _shards_task_pool.size(); ++i) {
    _shards_task_pool[i].reset(new ::ThreadPool(1));
  }

  sync = _config.common().sync();
  VLOG(1) << "table " << _config.common().table_name() << " is sync: " << sync;

  _global_lr = new float(1.0);

  auto common = _config.common();
  int size = static_cast<int>(common.params().size());

  size_t offset = 0;
  for (int x = 0; x < size; ++x) {
    auto& varname = common.params()[x];
    auto& dim = common.dims()[x];

    value_idx_[varname] = x;
    value_names_.push_back(varname);
    value_dims_.push_back(dim);
    value_offsets_.push_back(offset);
    initializer_attrs_.push_back(common.initializers()[x]);

    if (varname == "Param") {
      param_dim_ = dim;
      param_offset_ = offset;
    }

    offset += dim;
  }

  initialize_value();
  initialize_optimizer();
  initialize_recorder();
  return 0;
}

int32_t CommonSparseTable::initialize_recorder() { return 0; }

int32_t CommonSparseTable::initialize_value() {
  shard_values_.reserve(task_pool_size_);

  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard =
        std::make_shared<ValueBlock>(value_names_, value_dims_, value_offsets_,
                                     value_idx_, initializer_attrs_, "none");

    shard_values_.emplace_back(shard);
  }

  auto accessor = _config.accessor();

  std::vector<uint64_t> feasigns;

  for (size_t x = 0; x < accessor.fea_dim(); ++x) {
    if (x % _shard_num == _shard_idx) {
      feasigns.push_back(x);
    }
  }

  VLOG(0) << "has " << feasigns.size() << " ids need to be pre inited";

  auto buckets = bucket(feasigns.size(), 10);
  for (int x = 0; x < 10; ++x) {
    auto bucket_feasigns = buckets[x + 1] - buckets[x];
    std::vector<uint64_t> ids(bucket_feasigns);
    std::copy(feasigns.begin() + buckets[x], feasigns.begin() + buckets[x + 1],
              ids.begin());
    std::vector<float> pulls;
    pulls.resize(bucket_feasigns * param_dim_);
    pull_sparse(pulls.data(), ids.data(), bucket_feasigns);
  }

  return 0;
}

int32_t CommonSparseTable::initialize_optimizer() {
  auto common = _config.common();
  auto name = common.name();

  if (name == "sgd") {
    optimizer_ = std::make_shared<SSGD>(value_names_, value_dims_,
                                        value_offsets_, value_idx_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "adam") {
    optimizer_ = std::make_shared<SAdam>(value_names_, value_dims_,
                                         value_offsets_, value_idx_);
    optimizer_->set_global_lr(_global_lr);
  } else if (name == "sum") {
    optimizer_ = std::make_shared<SSUM>(value_names_, value_dims_,
                                        value_offsets_, value_idx_);
  } else {
    VLOG(0) << "init optimizer failed";
  }

  VLOG(0) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonSparseTable::set_global_lr(float* lr) {
  _global_lr = lr;
  optimizer_->set_global_lr(_global_lr);
  return 0;
}

int32_t CommonSparseTable::load(const std::string& path,
                                const std::string& param) {
  rwlock_->WRLock();
  VLOG(0) << "sparse table load with " << path << " with meta " << param;
  LoadFromText(path, param, _shard_idx, _shard_num, task_pool_size_,
               &shard_values_);
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::save(const std::string& dirname,
                                const std::string& param) {
  rwlock_->WRLock();
  int mode = std::stoi(param);
  VLOG(0) << "sparse table save: " << dirname << " mode: " << mode;

  auto varname = _config.common().table_name();
  std::string var_store =
      string::Sprintf("%s/%s%s", dirname, varname, PSERVER_SAVE_SUFFIX);
  MkDirRecursively(var_store.c_str());

  VLOG(3) << "save " << varname << " in dir: " << var_store << " begin";
  std::vector<std::string> params(_config.common().params().begin(),
                                  _config.common().params().end());
  std::string shard_var_pre =
      string::Sprintf("%s.block%d", varname, _shard_idx);

  std::string value_ = string::Sprintf("%s/%s.txt", var_store, shard_var_pre);

  std::unique_ptr<std::ofstream> value_out(new std::ofstream(value_));

  int64_t total_ins = 0;
  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // save values
    total_ins += SaveToText(value_out.get(), shard_values_[shard_id], mode);
  }
  value_out->close();

  // save meta
  std::stringstream stream;
  stream << "param=" << _config.common().table_name() << "\n";
  stream << "shard_id=" << _shard_idx << "\n";
  stream << "row_names="
         << paddle::string::join_strings(_config.common().params(), ',')
         << "\n";
  stream << "row_dims="
         << paddle::string::join_strings(_config.common().dims(), ',') << "\n";
  stream << "count=" << total_ins << "\n";
  std::string meta_ = string::Sprintf("%s/%s.meta", var_store, shard_var_pre);
  std::unique_ptr<std::ofstream> meta_out(new std::ofstream(meta_));
  meta_out->write(stream.str().c_str(), sizeof(char) * stream.str().size());
  meta_out->close();
  VLOG(3) << "save " << varname << " in dir: " << var_store << " done";
  rwlock_->UNLock();
  return 0;
}

std::pair<int64_t, int64_t> CommonSparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& value : shard_values_) {
    feasign_size += value->values_.size();
  }

  return {feasign_size, mf_size};
}

int32_t CommonSparseTable::pour() {
  rwlock_->RDLock();

  std::vector<float> values;
  std::vector<uint64_t> keys;

  keys.reserve(pull_reservoir_.size());
  values.reserve(pull_reservoir_.size() * param_dim_);

  for (auto& val : pull_reservoir_) {
    keys.push_back(val.first);
    auto& reservoir = val.second;
    reservoir.avg();
    std::copy(reservoir.values.begin(), reservoir.values.end(),
              std::back_inserter(values));
  }
  _push_sparse(keys.data(), values.data(), pull_reservoir_.size());

  pull_reservoir_.clear();
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::pull_sparse(float* pull_values, const uint64_t* keys,
                                       size_t num) {
  rwlock_->RDLock();

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &pull_values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            auto* value = block->InitFromInitializer(id);
            std::copy_n(value + param_offset_, param_dim_,
                        pull_values + param_dim_ * offset);
          }

          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::_push_sparse(const uint64_t* keys,
                                        const float* values, size_t num) {
  rwlock_->RDLock();
  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &values, num, &offset_bucket]() -> int {
          auto& offsets = offset_bucket[shard_id];
          optimizer_->update(keys, values, num, offsets,
                             shard_values_[shard_id].get());
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::push_sparse(const uint64_t* keys,
                                       const float* values, size_t num) {
  if (sync) {
    std::future<int> task =
        _shards_task_pool[0]->enqueue([this, &keys, &values, num]() -> int {
          for (int x = 0; x < num; ++x) {
            auto id = keys[x];
            auto has = pull_reservoir_.find(id);

            if (has == pull_reservoir_.end()) {
              pull_reservoir_[id] = ReservoirValue<float>(param_dim_);
            }

            auto& reservoir = pull_reservoir_[id];
            reservoir.add(values + x * param_dim_, param_dim_);
          }
          return 0;
        });
    task.wait();
  } else {
    _push_sparse(keys, values, num);
  }

  return 0;
}

int32_t CommonSparseTable::push_sparse_param(const uint64_t* keys,
                                             const float* values, size_t num) {
  rwlock_->RDLock();

  std::vector<std::vector<uint64_t>> offset_bucket;
  offset_bucket.resize(task_pool_size_);

  for (int x = 0; x < num; ++x) {
    auto y = keys[x] % task_pool_size_;
    offset_bucket[y].push_back(x);
  }

  std::vector<std::future<int>> tasks(task_pool_size_);

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, &keys, &offset_bucket, &values]() -> int {
          auto& block = shard_values_[shard_id];
          auto& offsets = offset_bucket[shard_id];

          for (int i = 0; i < offsets.size(); ++i) {
            auto offset = offsets[i];
            auto id = keys[offset];
            auto* value = block->InitFromInitializer(id);
            std::copy_n(values + param_dim_ * offset, param_dim_,
                        value + param_offset_);
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  rwlock_->UNLock();
  return 0;
}

int32_t CommonSparseTable::flush() { return 0; }

int32_t CommonSparseTable::shrink() {
  VLOG(0) << "shrink coming soon";
  return 0;
}
void CommonSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
