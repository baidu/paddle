/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/any.hpp>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/op_version_proto.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace compatible {

struct OpUpdateRecord {
  enum class Type {
    kInvalid = 0,
    kModifyAttr,
    kNewAttr,
    kNewInput,
    kNewOutput,
    kBugfixWithBehaviorChanged,
  };
  Type type_;
  std::string remark_;
};

struct ModifyAttr : OpUpdateRecord {
  ModifyAttr(const std::string& name, const std::string& remark,
             const boost::any& default_value)
      : OpUpdateRecord({Type::kModifyAttr, remark}),
        name_(name),
        default_value_(default_value) {
    // TODO(Shixiaowei02): Check the data type with proto::OpDesc.
  }

 private:
  std::string name_;
  boost::any default_value_;
};

struct NewAttr : OpUpdateRecord {
  NewAttr(const std::string& name, const std::string& remark,
          const boost::any& default_value)
      : OpUpdateRecord({Type::kNewAttr, remark}),
        name_(name),
        default_value_(default_value) {}

 private:
  std::string name_;
  boost::any default_value_;
};

struct NewInput : OpUpdateRecord {
  NewInput(const std::string& name, const std::string& remark)
      : OpUpdateRecord({Type::kNewInput, remark}), name_(name) {}

 private:
  std::string name_;
};

struct NewOutput : OpUpdateRecord {
  NewOutput(const std::string& name, const std::string& remark)
      : OpUpdateRecord({Type::kNewOutput, remark}), name_(name) {}

 private:
  std::string name_;
};

struct BugfixWithBehaviorChanged : OpUpdateRecord {
  explicit BugfixWithBehaviorChanged(const std::string& remark)
      : OpUpdateRecord({Type::kBugfixWithBehaviorChanged, remark}) {}
};

class OpVersionDesc {
 public:
  OpVersionDesc& ModifyAttr(const std::string& name, const std::string& remark,
                            boost::any default_value) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::ModifyAttr(name, remark, default_value)));
    return *this;
  }

  OpVersionDesc& NewAttr(const std::string& name, const std::string& remark,
                         boost::any default_value) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::NewAttr(name, remark, default_value)));
    return *this;
  }

  OpVersionDesc& NewInput(const std::string& name, const std::string& remark) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::NewInput(name, remark)));
    return *this;
  }

  OpVersionDesc& NewOutput(const std::string& name, const std::string& remark) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::NewOutput(name, remark)));
    return *this;
  }

  OpVersionDesc& BugfixWithBehaviorChanged(const std::string& remark) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::BugfixWithBehaviorChanged(remark)));
    return *this;
  }

 private:
  std::vector<std::shared_ptr<OpUpdateRecord>> infos_;
};

class OpVersion {
 public:
  OpVersion& AddCheckpoint(const std::string& note,
                           const OpVersionDesc& op_version_desc) {
    checkpoints_.push_back(Checkpoint({note, op_version_desc}));
    return *this;
  }
  uint32_t GetVersionID() const {
    return static_cast<uint32_t>(checkpoints_.size());
  }

 private:
  struct Checkpoint {
    std::string note_;
    OpVersionDesc op_version_desc_;
  };
  std::vector<Checkpoint> checkpoints_;
};

class OpVersionRegistrar {
 public:
  static OpVersionRegistrar& GetInstance() {
    static OpVersionRegistrar instance;
    return instance;
  }
  OpVersion& Register(const std::string& op_type) {
    PADDLE_ENFORCE_EQ(
        op_version_map_.find(op_type), op_version_map_.end(),
        platform::errors::AlreadyExists(
            "'%s' is registered in operator version more than once.", op_type));
    op_version_map_.insert({op_type, OpVersion()});
    return op_version_map_[op_type];
  }
  const std::unordered_map<std::string, OpVersion>& GetVersionMap() {
    return op_version_map_;
  }
  uint32_t GetVersionID(const std::string& op_type) const {
    auto it = op_version_map_.find(op_type);
    if (it == op_version_map_.end()) {
      return 0;
    }
    return it->second.GetVersionID();
  }

 private:
  std::unordered_map<std::string, OpVersion> op_version_map_;

  OpVersionRegistrar() = default;
  OpVersionRegistrar& operator=(const OpVersionRegistrar&) = delete;
};

inline void SaveOpVersions(
    const std::unordered_map<std::string, OpVersion>& src,
    pb::OpVersionMap* dst) {
  for (const auto& pair : src) {
    (*dst)[pair.first].SetVersionID(pair.second.GetVersionID());
  }
}

class OpVersionComparator {
 public:
  virtual bool operator()() = 0;
  virtual ~OpVersionComparator() = default;
};

#define ADD_OP_VERSION_COMPARATOR(cmp_name, cmp_math)                   \
  class OpVersion##cmp_name##Comparator : public OpVersionComparator {  \
   public:                                                              \
    explicit OpVersion##cmp_name##Comparator(const std::string op_name, \
                                             uint32_t target_version)   \
        : op_name_(op_name), target_version_(target_version) {}         \
    virtual bool operator()() {                                         \
      return OpVersionRegistrar::GetInstance().GetVersionID(op_name_)   \
          cmp_math target_version_;                                     \
    }                                                                   \
    virtual ~OpVersion##cmp_name##Comparator() {}                       \
                                                                        \
   private:                                                             \
    std::string op_name_;                                               \
    uint32_t target_version_;                                           \
  };

ADD_OP_VERSION_COMPARATOR(LE, <=);
ADD_OP_VERSION_COMPARATOR(EQ, ==);
ADD_OP_VERSION_COMPARATOR(GE, >=);
ADD_OP_VERSION_COMPARATOR(NE, !=);

class OpVersionComparatorCombination {
 public:
  OpVersionComparatorCombination() {}

  OpVersionComparatorCombination& LE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionLEComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& EQ(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionEQComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& GE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionGEComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& NE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionNEComparator(op_name, target_version)));
    return *this;
  }

  bool IsMatched() const {
    for (const auto& cmp : op_version_comparators_) {
      if (!(*cmp)()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<OpVersionComparator>> op_version_comparators_;
};

class PassVersionCheckers {
 public:
  PassVersionCheckers& AddCombination(
      const OpVersionComparatorCombination& combinations) {
    pass_version_checkers_.push_back(combinations);
    return *this;
  }
  bool IsPassCompatible() const {
    if (pass_version_checkers_.empty()) {
      return true;
    }
    for (const auto& checker : pass_version_checkers_) {
      if (checker.IsMatched()) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<OpVersionComparatorCombination> pass_version_checkers_;
};

class PassVersionCheckerRegistrar {
 public:
  static PassVersionCheckerRegistrar& GetInstance() {
    static PassVersionCheckerRegistrar instance;
    return instance;
  }
  PassVersionCheckers& Register(const std::string& pass_name) {
    return pass_version_checkers_map_[pass_name];
  }
  bool IsPassCompatible(const std::string& fuse_pass_name) const {
    auto iter = pass_version_checkers_map_.find(fuse_pass_name);
    if (iter == pass_version_checkers_map_.end()) {
      return true;
    }
    return iter->second.IsPassCompatible();
  }

 private:
  std::unordered_map<std::string, PassVersionCheckers>
      pass_version_checkers_map_;

  PassVersionCheckerRegistrar() = default;
  PassVersionCheckerRegistrar& operator=(const PassVersionCheckerRegistrar&) =
      delete;
};

}  // namespace compatible
}  // namespace framework
}  // namespace paddle

#define REGISTER_OP_VERSION(op_type)                                       \
  static paddle::framework::compatible::OpVersion                          \
      RegisterOpVersion__##op_type =                                       \
          paddle::framework::compatible::OpVersionRegistrar::GetInstance() \
              .Register(#op_type)

#define REGISTER_PASS_CAPABILITY(pass_name)                        \
  static auto RegisterOpPassVersionChecker__##pass_name =          \
      paddle::framework::compatible::PassVersionCheckerRegistrar:: \
          GetInstance()                                            \
              .Register(#pass_name)
