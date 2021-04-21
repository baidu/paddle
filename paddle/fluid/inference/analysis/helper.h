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

#include <sys/stat.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <typeindex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/port.h"

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#define GCC_ATTRIBUTE(attr__)
#define MKDIR(path) _mkdir(path)
#else
#include <unistd.h>
#define GCC_ATTRIBUTE(attr__) __attribute__((attr__));
#define MKDIR(path) mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH)
#endif
#define __SHOULD_USE_RESULT__ GCC_ATTRIBUTE(warn_unused_result)

namespace paddle {
namespace inference {
namespace analysis {

template <typename T>
void SetAttr(framework::proto::OpDesc *op, const std::string &name,
             const T &data);

template <typename Vec>
int AccuDims(Vec &&vec, int size) {
  int res = 1;
  for (int i = 0; i < size; i++) {
    res *= std::forward<Vec>(vec)[i];
  }
  return res;
}

#define SET_TYPE(type__) dic_[std::type_index(typeid(type__))] = #type__;
/*
 * Map typeid to representation.
 */
struct DataTypeNamer {
  static const DataTypeNamer &Global() {
    static auto *x = new DataTypeNamer();
    return *x;
  }

  template <typename T>
  const std::string &repr() const {
    auto x = std::type_index(typeid(T));
    PADDLE_ENFORCE_GT(dic_.count(x), 0, platform::errors::PreconditionNotMet(
                                            "unknown type for representation"));
    return dic_.at(x);
  }

  const std::string &repr(const std::type_index &type) const {  // NOLINT
    PADDLE_ENFORCE_GT(dic_.count(type), 0,
                      platform::errors::PreconditionNotMet(
                          "unknown type for representation"));
    return dic_.at(type);
  }

 private:
  DataTypeNamer() {
    SET_TYPE(int);
    SET_TYPE(bool);
    SET_TYPE(float);
    SET_TYPE(void *);
  }

  std::unordered_map<std::type_index, std::string> dic_;
};
#undef SET_TYPE

template <typename IteratorT>
class iterator_range {
  IteratorT begin_, end_;

 public:
  template <typename Container>
  explicit iterator_range(Container &&c) : begin_(c.begin()), end_(c.end()) {}

  iterator_range(const IteratorT &begin, const IteratorT &end)
      : begin_(begin), end_(end) {}

  const IteratorT &begin() const { return begin_; }
  const IteratorT &end() const { return end_; }
};

/*
 * An registry helper class, with its records keeps the order they registers.
 */
template <typename T>
class OrderedRegistry {
 public:
  T *Register(const std::string &name, T *x) {
    PADDLE_ENFORCE_EQ(dic_.count(name), 0,
                      platform::errors::PreconditionNotMet(
                          "There exists duplicate key [%s]", name));
    dic_[name] = elements_.size();
    elements_.emplace_back(std::unique_ptr<T>(x));
    return elements_.back().get();
  }

  T *Lookup(const std::string &name) {
    auto it = dic_.find(name);
    if (it == dic_.end()) return nullptr;
    return elements_[it->second].get();
  }

 protected:
  std::unordered_map<std::string, int> dic_;
  std::vector<std::unique_ptr<T>> elements_;
};

template <typename T>
T &GetFromScope(const framework::Scope &scope, const std::string &name) {
  framework::Variable *var = scope.FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var, platform::errors::PreconditionNotMet(
               "The var which name is %s should not be nullptr.", name));
  return *var->GetMutable<T>();
}

static framework::proto::ProgramDesc LoadProgramDesc(
    const std::string &model_path) {
  std::ifstream fin(model_path, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      fin.is_open(), true,
      platform::errors::NotFound(
          "Cannot open file %s, please confirm whether the file exists",
          model_path));
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();
  framework::proto::ProgramDesc program_desc;
  program_desc.ParseFromString(buffer);
  return program_desc;
}

static bool FileExists(const std::string &filepath) {
  std::ifstream file(filepath);
  bool exists = file.is_open();
  file.close();
  return exists;
}

static bool PathExists(const std::string &path) {
  struct stat statbuf;
  if (stat(path.c_str(), &statbuf) != -1) {
    if (S_ISDIR(statbuf.st_mode)) {
      return true;
    }
  }
  return false;
}

static std::string GetDirRoot(const std::string &path) {
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  size_t i = path.rfind(sep, path.length());
  if (i != std::string::npos) {
    return (path.substr(0, i));
  }
  return path;
}

static std::string GetOrCreateModelOptCacheDir(const std::string &model_root) {
  std::string opt_cache_dir = model_root + "/_opt_cache/";
  if (!PathExists(opt_cache_dir)) {
    PADDLE_ENFORCE_NE(
        MKDIR(opt_cache_dir.c_str()), -1,
        platform::errors::PreconditionNotMet(
            "Can not create optimize cache directory: %s, Make sure you "
            "have permission to write",
            opt_cache_dir));
  }
  return opt_cache_dir;
}

static std::string GetTrtCalibPath(const std::string &model_root,
                                   const std::string &engine_key) {
  return model_root + "/trt_calib_" + engine_key;
}

// If there is no calib table data file in model_opt_cache_dir, return "".
static std::string GetTrtCalibTableData(const std::string &model_opt_cache_dir,
                                        const std::string &engine_key,
                                        bool enable_int8) {
  std::string trt_calib_table_path =
      GetTrtCalibPath(model_opt_cache_dir, engine_key);
  if (enable_int8 && FileExists(trt_calib_table_path)) {
    VLOG(3) << "Calibration table file: " << trt_calib_table_path
            << "is found here";
    std::ifstream infile(trt_calib_table_path, std::ios::in);
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string calibration_data(buffer.str());
    return calibration_data;
  }
  return "";
}

static std::string GetTrtEngineSerializedPath(const std::string &model_root,
                                              const std::string &engine_key) {
  return model_root + "/trt_serialized_" + engine_key;
}

static std::string GetTrtEngineSerializedData(
    const std::string &model_opt_cache_dir, const std::string &engine_key) {
  std::string trt_serialized_path =
      GetTrtEngineSerializedPath(model_opt_cache_dir, engine_key);
  if (FileExists(trt_serialized_path)) {
    std::cout << "Found TRT serialized file: " << trt_serialized_path << "\n";
    std::ifstream infile(trt_serialized_path, std::ios::binary);
    std::stringstream buffer;
    buffer << infile.rdbuf();
    std::string trt_engine_serialized_data(buffer.str());
    return trt_engine_serialized_data;
  }
  std::cout << "TRT serialized file not found.\n";
  return "";
}

static void SaveTrtEngineSerializedDataToFile(
    const std::string &trt_serialized_path,
    const std::string &engine_serialized_data) {
  std::ofstream outfile(trt_serialized_path, std::ios::binary);
  outfile << engine_serialized_data;
  outfile.close();
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

#define PADDLE_DISALLOW_COPY_AND_ASSIGN(type__) \
  type__(const type__ &) = delete;              \
  void operator=(const type__ &) = delete;
