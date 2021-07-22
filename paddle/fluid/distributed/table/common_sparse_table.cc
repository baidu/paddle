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
#include <sstream>

#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
class ValueBlock;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace distributed {

void CommonSparseTable::ProcessALine(const std::vector<std::string>& columns,
                                     const Meta& meta, const int64_t id,
                                     std::vector<std::vector<float>>* values) {
  auto colunmn_size = columns.size();
  auto load_values =
      paddle::string::split_string<std::string>(columns[colunmn_size - 1], ",");
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

    std::transform(start, end, std::back_inserter(val), [id](std::string va) {
      float v = 0.0;

      try {
        v = lexical_cast<float>(va);
      } catch (boost::bad_lexical_cast& e) {
        VLOG(0) << "id: " << id << " get unexpected value: " << va
                << " and be reset to: 0.0";
      }
      return v;
    });

    values->push_back(val);
    offset += meta.dims[x];
  }
}

void CommonSparseTable::SaveMetaToText(std::ostream* os,
                                       const CommonAccessorParameter& common,
                                       const size_t shard_idx,
                                       const int64_t total) {
  // save meta
  std::stringstream stream;
  stream << "param=" << common.table_name() << "\n";
  stream << "shard_id=" << shard_idx << "\n";
  stream << "row_names=" << paddle::string::join_strings(common.params(), ',')
         << "\n";
  stream << "row_dims=" << paddle::string::join_strings(common.dims(), ',')
         << "\n";
  stream << "count=" << total << "\n";
  os->write(stream.str().c_str(), sizeof(char) * stream.str().size());
}

int64_t CommonSparseTable::SaveValueToText(std::ostream* os,
                                           std::shared_ptr<ValueBlock> block,
                                           std::shared_ptr<::ThreadPool> pool,
                                           const int mode, int shard_id) {
  int64_t save_num = 0;
  for (auto& table : block->values_) {
    for (auto& value : table) {
      if (mode == SaveMode::delta && !value.second->need_save_) {
        continue;
      }

      ++save_num;

      std::stringstream ss;
      auto* vs = value.second->data_.data();

      auto id = value.first;

      ss << id << "\t" << value.second->count_ << "\t"
         << value.second->unseen_days_ << "\t" << value.second->is_entry_
         << "\t";

      for (int i = 0; i < block->value_length_ - 1; i++) {
        ss << std::to_string(vs[i]) << ",";
      }

      ss << std::to_string(vs[block->value_length_ - 1]);
      ss << "\n";

      os->write(ss.str().c_str(), sizeof(char) * ss.str().size());

      if (mode == SaveMode::base || mode == SaveMode::delta) {
        value.second->need_save_ = false;
      }
    }
  }

  return save_num;
}

int64_t CommonSparseTable::LoadFromText(
    const std::string& valuepath, const std::string& metapath,
    const int pserver_id, const int pserver_num, const int local_shard_num,
    std::vector<std::shared_ptr<ValueBlock>>* blocks) {
  Meta meta = Meta(metapath);

  int num_lines = 0;
  std::ifstream file(valuepath);
  std::string line;

  while (std::getline(file, line)) {
    auto values = paddle::string::split_string<std::string>(line, "\t");
    auto id = lexical_cast<uint64_t>(values[0]);

    if (id % pserver_num != pserver_id) {
      VLOG(3) << "will not load " << values[0] << " from " << valuepath
              << ", please check id distribution";
      continue;
    }

    auto shard_id = id % local_shard_num;
    auto block = blocks->at(shard_id);

    std::vector<std::vector<float>> kvalues;
    ProcessALine(values, meta, id, &kvalues);

    block->Init(id, false);

    VALUE* value_instant = block->GetValue(id);

    if (values.size() == 5) {
      value_instant->count_ = lexical_cast<int>(values[1]);
      value_instant->unseen_days_ = lexical_cast<int>(values[2]);
      value_instant->is_entry_ =
          static_cast<bool>(lexical_cast<int>(values[3]));
    }

    std::vector<float*> block_values = block->Get(id, meta.names, meta.dims);
    auto blas = GetBlas<float>();
    for (int x = 0; x < meta.names.size(); ++x) {
      blas.VCOPY(meta.dims[x], kvalues[x].data(), block_values[x]);
    }
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
  auto common = _config.common();
  shard_values_.reserve(task_pool_size_);

  for (int x = 0; x < task_pool_size_; ++x) {
    auto shard = std::make_shared<ValueBlock>(
        value_names_, value_dims_, value_offsets_, value_idx_,
        initializer_attrs_, common.entry());

    shard_values_.emplace_back(shard);
  }

  auto accessor = _config.accessor();
  std::vector<uint64_t> feasigns;

  for (size_t x = 0; x < accessor.fea_dim(); ++x) {
    if (x % _shard_num == _shard_idx) {
      feasigns.push_back(x);
    }
  }

  VLOG(3) << "has " << feasigns.size() << " ids need to be pre inited";

  auto buckets = bucket(feasigns.size(), 10);
  for (int x = 0; x < 10; ++x) {
    auto bucket_feasigns = buckets[x + 1] - buckets[x];
    std::vector<uint64_t> ids(bucket_feasigns);
    std::copy(feasigns.begin() + buckets[x], feasigns.begin() + buckets[x + 1],
              ids.begin());

    std::vector<uint32_t> fres;
    fres.resize(ids.size(), 1);

    auto pull_value = PullSparseValue(ids, fres, param_dim_);
    std::vector<float> pulls;
    pulls.resize(bucket_feasigns * param_dim_);
    pull_sparse(pulls.data(), pull_value);
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
  } else if (name == "adagrad") {
    optimizer_ = std::make_shared<SAdagrad>(value_names_, value_dims_,
                                            value_offsets_, value_idx_);
    optimizer_->set_global_lr(_global_lr);
  } else {
    VLOG(3) << "init optimizer failed";
  }

  VLOG(3) << "init optimizer " << name << " done";
  return 0;
}

int32_t CommonSparseTable::set_global_lr(float* lr) {
  _global_lr = lr;
  optimizer_->set_global_lr(_global_lr);
  return 0;
}

int32_t CommonSparseTable::load(const std::string& path,
                                const std::string& param) {
  auto begin = GetCurrentUS();
  rwlock_->WRLock();
  LoadFromText(path, param, _shard_idx, _shard_num, task_pool_size_,
               &shard_values_);
  rwlock_->UNLock();
  auto end = GetCurrentUS();

  auto varname = _config.common().table_name();
  VLOG(0) << "load " << varname << " with value: " << path
          << " , meta: " << param
          << " using: " << std::to_string((end - begin) / 1e+6) << " seconds";

  return 0;
}

int32_t CommonSparseTable::save(const std::string& dirname,
                                const std::string& param) {
  auto begin = GetCurrentUS();
  rwlock_->WRLock();
  int mode = std::stoi(param);
  VLOG(3) << "sparse table save: " << dirname << " mode: " << mode;

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

  std::unique_ptr<std::ofstream> vs(new std::ofstream(value_));

  int64_t total_ins = 0;
  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // save values
    auto shard_save_num =
        SaveValueToText(vs.get(), shard_values_[shard_id],
                        _shards_task_pool[shard_id], mode, shard_id);
    total_ins += shard_save_num;
  }
  vs->close();

  std::string meta_ = string::Sprintf("%s/%s.meta", var_store, shard_var_pre);
  std::unique_ptr<std::ofstream> ms(new std::ofstream(meta_));
  SaveMetaToText(ms.get(), _config.common(), _shard_idx, total_ins);
  ms->close();

  auto end = GetCurrentUS();
  rwlock_->UNLock();
  VLOG(0) << "save " << varname << " with path: " << value_
          << " using: " << std::to_string((end - begin) / 1e+6) << " seconds";

  return 0;
}

std::pair<int64_t, int64_t> CommonSparseTable::print_table_stat() {
  int64_t feasign_size = 0;
  int64_t mf_size = 0;

  for (auto& shard : shard_values_) {
    for (auto& table : shard->values_) {
      feasign_size += table.size();
    }
  }

  return {feasign_size, mf_size};
}

int32_t CommonSparseTable::pour() {
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
  return 0;
}

int32_t CommonSparseTable::pull_sparse(float* pull_values,
                                       const PullSparseValue& pull_value) {
  auto shard_num = task_pool_size_;
  std::vector<std::future<int>> tasks(shard_num);

  for (int shard_id = 0; shard_id < shard_num; ++shard_id) {
    tasks[shard_id] = _shards_task_pool[shard_id]->enqueue(
        [this, shard_id, shard_num, &pull_value, &pull_values]() -> int {
          auto& block = shard_values_[shard_id];

          std::vector<int> offsets;
          pull_value.Fission(shard_id, shard_num, &offsets);

          if (pull_value.is_training_) {
            for (auto& offset : offsets) {
              auto feasign = pull_value.feasigns_[offset];
              auto frequencie = pull_value.frequencies_[offset];
              auto* value = block->Init(feasign, true, frequencie);
              std::copy_n(value + param_offset_, param_dim_,
                          pull_values + param_dim_ * offset);
            }
          } else {
            for (auto& offset : offsets) {
              auto feasign = pull_value.feasigns_[offset];
              auto* value = block->Init(feasign, false);
              std::copy_n(value + param_offset_, param_dim_,
                          pull_values + param_dim_ * offset);
            }
          }

          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t CommonSparseTable::pull_sparse_ptr(char** pull_values,
                                           const uint64_t* keys, size_t num) {
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
            auto* value = block->InitGet(id);
            // std::copy_n(value + param_offset_, param_dim_,
            //            pull_values + param_dim_ * offset);
            pull_values[offset] = reinterpret_cast<char*>(value);
          }

          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t CommonSparseTable::_push_sparse(const uint64_t* keys,
                                        const float* values, size_t num) {
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

int32_t CommonSparseTable::push_sparse(const uint64_t* keys,
                                       const float** values, size_t num) {
  _push_sparse(keys, values, num);
  return 0;
}

int32_t CommonSparseTable::_push_sparse(const uint64_t* keys,
                                        const float** values, size_t num) {
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
          for (size_t i = 0; i < offsets.size(); ++i) {
            std::vector<uint64_t> tmp_off = {0};
            optimizer_->update(keys + offsets[i], values[offsets[i]], num,
                               tmp_off, shard_values_[shard_id].get());
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t CommonSparseTable::push_sparse_param(const uint64_t* keys,
                                             const float* values, size_t num) {
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
            auto* value = block->Init(id, false);
            std::copy_n(values + param_dim_ * offset, param_dim_,
                        value + param_offset_);
            block->SetEntry(id, true);
          }
          return 0;
        });
  }

  for (size_t shard_id = 0; shard_id < tasks.size(); ++shard_id) {
    tasks[shard_id].wait();
  }
  return 0;
}

int32_t CommonSparseTable::flush() { return 0; }

int32_t CommonSparseTable::shrink(const std::string& param) {
  int threshold = std::stoi(param);
  VLOG(3) << "sparse table shrink: " << threshold;

  for (int shard_id = 0; shard_id < task_pool_size_; ++shard_id) {
    // shrink
    VLOG(4) << shard_id << " " << task_pool_size_ << " begin shrink";
    shard_values_[shard_id]->Shrink(threshold);
  }
  return 0;
}

void CommonSparseTable::clear() { VLOG(0) << "clear coming soon"; }

}  // namespace distributed
}  // namespace paddle
