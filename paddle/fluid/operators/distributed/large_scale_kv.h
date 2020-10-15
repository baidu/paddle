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
#include <gflags/gflags.h>
#include <functional>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/rw_lock.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/printf.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace operators {
namespace distributed {

enum Mode { training, infer };
enum InitType { uniform_random, fill_constant, gaussian_random };

inline std::vector<int> bucket(const int v_size, const int b_size) {
  int remainder = v_size % b_size;
  int bucket = v_size / b_size;
  std::vector<int> ret_vec(b_size, bucket);
  for (int i = 0; i < remainder; ++i) {
    ret_vec[i] = ret_vec[i] + 1;
  }
  int cur_bucket = 0;
  for (int &j : ret_vec) {
    int tmp = j;
    j = cur_bucket;
    cur_bucket += tmp;
  }
  ret_vec.push_back(cur_bucket);
  return ret_vec;
}

class Initializer {
 public:
  Initializer() {}

  explicit Initializer(const std::vector<std::string> &attrs) {}

  virtual float GetValue() = 0;

  virtual ~Initializer() {}

 protected:
  std::string name_;
  unsigned int seed_;
};

class UniformInitializer : public Initializer {
 public:
  explicit UniformInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    min_ = std::stof(attrs[2]);
    max_ = std::stof(attrs[3]);

    dist_ = std::uniform_real_distribution<float>(min_, max_);
    random_engine_ = framework::GetCPURandomEngine(seed_);
  }

  float GetValue() override { return dist_(*random_engine_); }

 private:
  float min_;
  float max_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::uniform_real_distribution<float> dist_;
};

template <typename T>
inline bool entry(const int count, const T threshold);

template <>
inline bool entry<std::string>(const int count, const std::string threshold) {
  return true;
}

template <>
inline bool entry<int>(const int count, const int threshold) {
  return count >= threshold;
}

template <>
inline bool entry<float>(const int count, const float threshold) {
  UniformInitializer uniform = UniformInitializer({"0", "0", "1"});
  return uniform.GetValue() >= threshold;
}

class GaussianInitializer : public Initializer {
 public:
  explicit GaussianInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    seed_ = static_cast<unsigned int>(std::stoi(attrs[1]));
    mean_ = std::stof(attrs[2]);
    std_ = std::stof(attrs[3]);

    random_engine_ = framework::GetCPURandomEngine(seed_);

    dist_ = std::normal_distribution<float>(mean_, std_);
  }

  float GetValue() override { return dist_(*random_engine_); }

 private:
  float std_;
  float mean_;

  std::shared_ptr<std::mt19937_64> random_engine_;
  std::normal_distribution<float> dist_;
};

class FillConstantInitializer : public Initializer {
 public:
  explicit FillConstantInitializer(const std::vector<std::string> &attrs) {
    name_ = attrs[0];
    value_ = std::stof(attrs[1]);
  }

  float GetValue() override { return value_; }

 private:
  float value_;
};

struct SparseMeta {
  std::string name;
  std::string grad_name;
  std::vector<std::string> value_names;
  std::vector<int> value_dims;
  std::vector<std::string> cached_varnames;
  std::vector<std::string> initializer_attrs;
  std::string entry;
  Mode mode;

  std::string ToString() {
    std::stringstream ss;
    ss << "name: " << name << " ";
    ss << "mode: " << mode << " ";

    for (int i = 0; i < static_cast<int>(value_names.size()); i++) {
      ss << "value_name: " << value_names[i] << " dim: " << value_dims[i]
         << " ";
    }

    ss << " grad var: " << grad_name;

    ss << " cached varnames: ";
    for (int i = 0; i < static_cast<int>(cached_varnames.size()); i++) {
      ss << cached_varnames[i] << " ";
    }

    ss << " initializer attrs: ";
    for (int i = 0; i < static_cast<int>(initializer_attrs.size()); i++) {
      ss << initializer_attrs[i] << " ";
    }

    ss << " entry attrs: " << entry;

    return ss.str();
  }
};

struct VALUE {
  explicit VALUE(const std::vector<std::string> &names)
      : names_(names), count_(0), unseen_days_(0) {
    values_.resize(names.size());
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      places[names[i]] = i;
    }
  }

  void set(std::vector<std::vector<float>> *values) {
    values_ = std::move(*values);
  }

  void set(const std::vector<std::string> &names,
           const std::vector<std::vector<float>> &values) {
    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      auto idx = places[names[i]];
      auto value = values[i];
      values_[idx].assign(value.begin(), value.end());
    }
  }

  std::vector<std::vector<float> *> get() {
    auto pts = std::vector<std::vector<float> *>();
    pts.reserve(values_.size());

    for (auto &value : values_) {
      pts.push_back(&value);
    }
    return pts;
  }

  int fetch_count() { return ++count_; }
  void reset_unseen_days() { unseen_days_ = 0; }

  void set_entry(bool is_entry) { is_entry_ = is_entry; }

  bool get_entry() { return is_entry_; }

  std::vector<std::vector<float> *> get(const std::vector<std::string> names) {
    auto pts = std::vector<std::vector<float> *>();
    pts.reserve(values_.size());

    for (int i = 0; i < static_cast<int>(names.size()); i++) {
      pts.push_back(&(values_[places[names[i]]]));
    }
    return pts;
  }

  std::vector<std::string> names_;
  int count_;
  bool seen_after_last_save_;
  int unseen_days_;
  bool is_entry_;
  std::vector<std::vector<float>> values_;
  std::unordered_map<std::string, int> places;
};

class ValueBlock {
 public:
  explicit ValueBlock(const std::vector<std::string> value_names,
                      const std::vector<int> value_dims, const Mode &mode,
                      const std::vector<std::string> &init_attrs,
                      const std::string &entry_attr)
      : value_names_(value_names), value_dims_(value_dims), mode_(mode) {
    // for Initializer
    for (size_t i = 0; i < value_names.size(); i++) {
      auto name = value_names[i];
      auto slices = string::split_string<std::string>(init_attrs[i], "&");

      if (slices[0] == "gaussian_random") {
        initializers_[name] = new GaussianInitializer(slices);
      } else if (slices[0] == "fill_constant") {
        initializers_[name] = new FillConstantInitializer(slices);
      } else if (slices[0] == "uniform_random") {
        initializers_[name] = new UniformInitializer(slices);
      } else {
        PADDLE_THROW(
            platform::errors::InvalidArgument("%s can not be supported", name));
      }
    }

    // for Entry
    {
      if (entry_attr == "none") {
        entry_func_ =
            std::bind(entry<std::string>, std::placeholders::_1, "none");
      } else {
        auto slices = string::split_string<std::string>(entry_attr, "&");
        if (slices[0] == "count_filter") {
          int threshold = std::stoi(slices[1]);
          entry_func_ = std::bind(entry<int>, std::placeholders::_1, threshold);
        } else if (slices[0] == "probability") {
          float threshold = std::stof(slices[1]);
          entry_func_ =
              std::bind(entry<float>, std::placeholders::_1, threshold);
        }
      }
    }

    rwlock_.reset(new framework::RWLock);
  }

  ~ValueBlock() {
    //    for (auto init : initializers_) {
    //      delete init.second;
    //      initializers_.erase(init.first);
    //    }
    //
    //    for (auto value : values_) {
    //      delete value.second;
    //      values_.erase(value.first);
    //    }
  }

  void Init(const int64_t &id, std::vector<std::vector<float>> *values,
            int count) {
    if (Has(id)) {
      PADDLE_THROW(platform::errors::AlreadyExists("id already exist, error"));
    }

    if (values->size() != value_names_.size()) {
      PADDLE_THROW(
          platform::errors::AlreadyExists("values can not match, error"));
    }

    auto value = new VALUE(value_names_);
    value->set(values);
    value->seen_after_last_save_ = true;
    value->count_ = count;
    values_[id] = value;
  }

  std::vector<std::vector<float> *> Get(
      const int64_t &id, const std::vector<std::string> &value_names) {
    rwlock_->RDLock();
    auto ret_values = values_.at(id)->get(value_names);
    rwlock_->UNLock();
    return ret_values;
  }

  void InitFromInitializer(const int64_t &id,
                           const std::vector<std::string> &value_names) {
    rwlock_->WRLock();

    if (Has(id)) {
      Update(id);
      rwlock_->UNLock();
      return;
    }

    auto rets = std::vector<std::vector<float>>();
    rets.resize(value_names_.size());

    for (int i = 0; i < static_cast<int>(value_names_.size()); i++) {
      auto name = value_names_[i];
      auto *init = initializers_.at(name);

      auto dim = value_dims_[i];
      rets[i].resize(dim);

      for (int j = 0; j < static_cast<int>(dim); j++) {
        rets[i][j] = init->GetValue();
      }
    }

    Init(id, &rets, 0);
    Update(id);
    rwlock_->UNLock();
  }

  bool GetEntry(const int64_t &id) {
    rwlock_->RDLock();
    auto value = values_.at(id);
    auto entry = value->get_entry();
    rwlock_->UNLock();
    return entry;
  }

  void Set(const int64_t &id, const std::vector<std::string> &value_names,
           const std::vector<std::vector<float>> &values) {
    rwlock_->WRLock();
    auto value = values_.at(id);
    value->set(value_names, values);
    rwlock_->UNLock();
  }

  void Update(const int64_t id) {
    auto *value = values_.at(id);
    value->reset_unseen_days();
    auto count = value->fetch_count();

    if (!value->get_entry()) {
      value->set_entry(entry_func_(count));
    }
  }

 private:
  bool Has(const int64_t id) {
    auto got = values_.find(id);
    if (got == values_.end()) {
      return false;
    } else {
      return true;
    }
  }

 public:
  std::unordered_map<int64_t, VALUE *> values_;

 private:
  std::vector<std::string> value_names_;
  std::vector<int> value_dims_;
  Mode mode_;
  std::function<bool(int64_t)> entry_func_;
  std::unordered_map<std::string, Initializer *> initializers_;
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};
};

class SparseVariable {
 public:
  explicit SparseVariable(const SparseMeta &meta) {
    meta_.name = meta.name;
    meta_.mode = meta.mode;
    meta_.value_names = meta.value_names;
    meta_.value_dims = meta.value_dims;
    meta_.grad_name = meta.grad_name;
    meta_.cached_varnames = meta.cached_varnames;
    meta_.initializer_attrs = meta.initializer_attrs;
    meta_.entry = meta.entry;

    for (int i = 0; i < static_cast<int>(meta_.value_names.size()); i++) {
      values_dims_[meta_.value_names[i]] = meta_.value_dims[i];
    }

    for (size_t i = 0; i < shard_num_; i++) {
      auto block = std::make_shared<ValueBlock>(
          meta.value_names, meta.value_dims, meta.mode, meta.initializer_attrs,
          meta.entry);
      shard_blocks_.emplace_back(block);
    }

    rwlock_.reset(new framework::RWLock);
  }

  void Init(const std::vector<int64_t> &ids) {
    rwlock_->RDLock();
    for (auto &id : ids) {
      auto *block = GetShard(id);
      block->InitFromInitializer(id, meta_.value_names);
    }
    rwlock_->UNLock();
  }

  void Get(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           std::vector<std::vector<std::vector<float> *>> *values) {
    values->resize(ids.size());

    auto buckets = bucket(ids.size(), 8);
    std::vector<std::future<void>> fs;

    for (int j = 0; j < 8; ++j) {
      auto begin = buckets[j];
      auto end = buckets[j + 1];

      fs.push_back(
          framework::Async([begin, end, &values, &ids, &value_names, this]() {
            for (int x = begin; x < end; x++) {
              auto id = ids[x];
              auto *block = GetShard(id);
              auto id_values = block->Get(id, value_names);
              (*values)[x] = id_values;
            }
          }));
    }

    for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
  }

  void GetEntry(const std::vector<int64_t> &ids, std::vector<int64_t> *values) {
    auto buckets = bucket(ids.size(), 8);
    std::vector<std::future<void>> fs;

    for (int j = 0; j < 8; ++j) {
      auto begin = buckets[j];
      auto end = buckets[j + 1];

      fs.push_back(framework::Async([begin, end, &values, &ids, this]() {
        for (int x = begin; x < end; x++) {
          auto id = ids[x];
          auto *block = GetShard(id);
          auto is_entry = block->GetEntry(id);

          if (!is_entry) {
            values->push_back(id);
          }
        }
      }));
    }
    for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();
  }

  void Set(const std::vector<int64_t> &ids,
           const std::vector<std::string> &value_names,
           const std::vector<std::vector<std::vector<float>>> &values) {
    for (int i = 0; i < static_cast<int>(ids.size()); i++) {
      GetShard(ids[i])->Set(ids[i], value_names, values[i]);
    }
  }

  void Dims(std::vector<std::string> value_names, std::vector<int64_t> *dims) {
    for (auto &name : value_names) {
      dims->push_back(values_dims_.at(name));
    }
  }

  std::vector<std::string> CachedVarnames() const {
    return meta_.cached_varnames;
  }

  void Load(const std::string &dirname) {
    rwlock_->WRLock();
    VLOG(1) << "load " << meta_.name << " from dir: " << dirname << " begin";

    std::vector<std::string> filenames;
    for (auto &value_name : meta_.value_names) {
      auto filename = string::Sprintf("%s/%s", dirname, value_name);
      filenames.push_back(filename);
    }

    LoadFromSelectedRows(filenames, meta_.value_names);
    VLOG(1) << "load " << meta_.name << " in dir: " << dirname << " done";
    rwlock_->UNLock();
  }

  void LoadFromSelectedRows(const std::vector<std::string> &filenames,
                            const std::vector<std::string> &valuenames) {
    std::vector<std::shared_ptr<framework::Variable>> variables;
    auto place = platform::CPUPlace();

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto var = std::make_shared<framework::Variable>();
      variables.push_back(var);
      auto &filename = filenames[i];
      std::ifstream fin(filename, std::ios::binary);
      auto *selectedRows = var->GetMutable<framework::SelectedRows>();

      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);

      framework::DeserializeFromStream(fin, selectedRows, dev_ctx);
      selectedRows->SyncIndex();
    }

    std::vector<const float *> tensors;

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto &slr = variables[i]->Get<framework::SelectedRows>();
      auto src_t = slr.value();
      const auto *value = src_t.data<float>();
      tensors.push_back(value);
    }

    for (int i = 1; i < static_cast<int>(filenames.size()); i++) {
      auto rows_0 = variables[0]->Get<framework::SelectedRows>().rows();
      auto rows_i = variables[i]->Get<framework::SelectedRows>().rows();

      bool is_equal = std::equal(rows_0.begin(), rows_0.end(), rows_i.begin());

      if (!is_equal) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "%s and %s are not equal, can not be load rightly", filenames[0],
            filenames[i]));
      }
    }

    auto rows = variables[0]->Get<framework::SelectedRows>().rows();

    for (auto i = 0; i < static_cast<int64_t>(rows.size()); i++) {
      auto id = rows[i];
      std::vector<std::vector<float>> values;
      values.resize(filenames.size());

      for (int j = 0; j < static_cast<int>(filenames.size()); ++j) {
        values[j].resize(meta_.value_dims[j]);
        std::memcpy(values[j].data(), tensors[j] + i * meta_.value_dims[j],
                    sizeof(float) * meta_.value_dims[j]);
      }

      auto *block = GetShard(id);
      block->Init(id, &values, 0);
      block->Update(id);
    }
  }

  void Save(const std::string &dirname, const int mode = 0) {
    rwlock_->WRLock();
    VLOG(3) << "save " << meta_.name << " in dir: " << dirname << " begin";

    MkDirRecursively(dirname.c_str());

    std::vector<std::string> filenames;
    for (auto &value_name : meta_.value_names) {
      auto filename = string::Sprintf("%s/%s", dirname, value_name);
      filenames.push_back(filename);
    }

    SaveToSelectedRows(filenames, meta_.value_names, mode);
    VLOG(3) << "save " << meta_.name << " in dir: " << dirname << " done";
    rwlock_->UNLock();
  }

  void SaveToSelectedRows(const std::vector<std::string> &filenames,
                          const std::vector<std::string> &valuenames,
                          const int mode) {
    for (auto &value_name : valuenames) {
      auto it = std::find(meta_.value_names.begin(), meta_.value_names.end(),
                          value_name);
      if (it == meta_.value_names.end()) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "[%s] is invalid param for [%s]", value_name, meta_.name));
      }
    }

    auto place = platform::CPUPlace();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    std::vector<int64_t> ids;

    for (auto &block : shard_blocks_) {
      for (auto value : block->values_) {
        if (mode == 0) {
          ids.push_back(value.first);
        } else {
          bool id_need_save = false;
          // save all params
          if (mode == 1) {
            id_need_save = true;
          } else {
            id_need_save = value.second->seen_after_last_save_;
          }

          if (id_need_save) {
            ids.push_back(value.first);
          }
          value.second->seen_after_last_save_ = false;
        }
      }
    }

    VLOG(3) << "save " << ids.size() << " feasigns for " << meta_.name
            << " with mode: " << mode;

    std::vector<std::shared_ptr<framework::Variable>> variables;
    std::vector<float *> tensors;
    std::vector<int64_t> dims;

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto dim = values_dims_.at(valuenames[i]);
      auto var = std::make_shared<framework::Variable>();
      auto *slr = var->GetMutable<framework::SelectedRows>();
      auto *src_t = slr->mutable_value();

      src_t->Resize({static_cast<int64_t>(ids.size()), dim});
      auto *value = src_t->mutable_data<float>(place);

      dims.push_back(dim);
      variables.push_back(var);
      tensors.push_back(value);
    }

    std::vector<std::vector<std::vector<float> *>> values;
    Get(ids, valuenames, &values);

    int64_t offset = 0;
    for (auto &vss : values) {
      for (int i = 0; i < static_cast<int>(vss.size()); i++) {
        auto &vs = vss[i];
        std::memcpy(tensors[i] + offset * dims[i], vs->data(),
                    sizeof(float) * dims[i]);
      }
      offset += 1;
    }

    for (auto &var : variables) {
      auto *slr = var->GetMutable<framework::SelectedRows>();
      slr->set_rows(ids);
      slr->set_height(ids.size());
    }

    for (int i = 0; i < static_cast<int>(filenames.size()); i++) {
      auto &filename = filenames[i];
      auto &selectedRows = variables[i]->Get<framework::SelectedRows>();

      std::ofstream fout(filename, std::ios::binary);
      PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                        platform::errors::Unavailable(
                            "Cannot open %s to save variables.", filename));

      framework::SerializeToStream(fout, selectedRows, dev_ctx);
      fout.close();
    }
  }

  void SaveToText(const std::vector<std::string> &filenames,
                  const std::vector<std::string> &valuenames) {
    for (auto &value_name : valuenames) {
      auto it = std::find(meta_.value_names.begin(), meta_.value_names.end(),
                          value_name);
      if (it == meta_.value_names.end()) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "[%s] is invalid param for [%s]", value_name, meta_.name));
      }
    }

    std::vector<std::unique_ptr<std::ofstream>> fouts;

    for (auto filename : filenames) {
      std::unique_ptr<std::ofstream> fout(new std::ofstream(filename));
      fouts.push_back(std::move(fout));
    }

    for (auto &block : shard_blocks_) {
      for (auto value : block->values_) {
        std::vector<std::vector<float> *> vss = value.second->get(valuenames);

        auto id = value.first;

        for (int i = 0; i < static_cast<int>(vss.size()); i++) {
          auto &vs = vss[i];
          std::stringstream ss;
          ss << id << "\t";
          ss << vs->size() << "\t";
          for (auto v : (*vs)) {
            ss << v << " ";
          }
          ss << "\n";

          fouts[i]->write(ss.str().c_str(), sizeof(char) * ss.str().size());
        }
      }
    }

    for (int i = 0; i < static_cast<int>(fouts.size()); i++) {
      fouts[i]->close();
    }
  }

  int64_t Size() {
    int64_t cnt = 0;

    for (auto &block : shard_blocks_) {
      cnt += block->values_.size();
    }
    return cnt;
  }

  ValueBlock *GetShard(const int64_t id) {
    return shard_blocks_[id & shard_mask_].get();
  }

  SparseMeta *GetMeta() { return &meta_; }

 private:
  std::unique_ptr<framework::RWLock> rwlock_{nullptr};

  SparseMeta meta_;
  std::unordered_map<std::string, int64_t> values_dims_;
  const size_t shard_mask_ = 127;
  const size_t shard_num_ = 128;
  std::vector<std::shared_ptr<ValueBlock>> shard_blocks_;
};

class LargeScaleKV {
 public:
  LargeScaleKV() {}

  explicit LargeScaleKV(const std::vector<SparseMeta> &table_metas) {
    for (auto &sparse_meta : table_metas) {
      auto table_name = sparse_meta.name;
      auto meta = std::shared_ptr<SparseVariable>(
          new SparseVariable(std::move(sparse_meta)));
      sparse_variables[table_name] = meta;
      grad_to_variables[sparse_meta.grad_name] = table_name;
      grad_names_.push_back(sparse_meta.grad_name);
    }
  }

  ~LargeScaleKV() {}

  static std::shared_ptr<LargeScaleKV> GetInstantcePtr() { return scale_kv_; }

  static LargeScaleKV *GetInstance() { return scale_kv_.get(); }

  static LargeScaleKV *InitInstance(
      const std::vector<SparseMeta> &table_metas) {
    std::call_once(init_flag_, &LargeScaleKV::Init, table_metas);
    return scale_kv_.get();
  }

  static void Init(const std::vector<SparseMeta> &table_metas) {
    if (scale_kv_.get() == nullptr) {
      scale_kv_.reset(new LargeScaleKV(table_metas));
    }
  }

  SparseVariable *Get(const std::string &name) {
    auto variable = sparse_variables.at(name);
    return variable.get();
  }

  bool ParamInLargeScale(const std::string &name) {
    auto got = sparse_variables.find(name);

    if (got == sparse_variables.end()) {
      return false;
    }

    return true;
  }

  bool GradInLargeScale(const std::string &name) {
    auto got = grad_to_variables.find(name);

    if (got == grad_to_variables.end()) {
      return false;
    }

    return true;
  }

  SparseVariable *GetByGrad(const std::string &name) {
    return Get(grad_to_variables[name]);
  }

  const std::vector<std::string> &GetAllGrads() { return grad_names_; }

 private:
  std::unordered_map<std::string, std::shared_ptr<SparseVariable>>
      sparse_variables;
  std::unordered_map<std::string, std::string> grad_to_variables;
  std::vector<std::string> grad_names_;
  static std::shared_ptr<LargeScaleKV> scale_kv_;
  static std::once_flag init_flag_;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
