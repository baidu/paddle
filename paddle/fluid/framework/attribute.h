/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <stdint.h>
#include <functional>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "boost/variant/get.hpp"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {

template <typename T>
struct ExtractAttribute {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  T* operator()(Attribute& attr) const {
    T* attr_value = nullptr;
    try {
      attr_value = &boost::get<T>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type %s, its type is %s.", attr_name_,
          paddle::platform::demangle(typeid(T).name()),
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

// special handle bool
// FIXME(yuyang18): Currently we cast bool into int in python binding. It is
// hard to change the logic there. In another way, we should correct handle
// if the user set `some_flag=1`.
//
// FIX ME anytime if there is a better solution.
template <>
struct ExtractAttribute<bool> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  bool* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = BOOST_GET_CONST(int, attr);
      attr = static_cast<bool>(val);
    } else if (attr.type() == typeid(float)) {  // NOLINT
      float val = BOOST_GET_CONST(float, attr);
      attr = static_cast<bool>(val);
    }
    bool* attr_value = nullptr;
    try {
      attr_value = &boost::get<bool>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type bool, its type is %s.", attr_name_,
          paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<int64_t> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  int64_t* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = BOOST_GET_CONST(int, attr);
      attr = static_cast<int64_t>(val);
    } else if (attr.type() == typeid(float)) {  // NOLINT
      int val = BOOST_GET_CONST(float, attr);
      attr = static_cast<int64_t>(val);
    }
    int64_t* attr_value = nullptr;
    try {
      attr_value = &boost::get<int64_t>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type int64_t, its type is %s.",
          attr_name_, paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<std::vector<int64_t>> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  std::vector<int64_t>* operator()(Attribute& attr) const {
    if (attr.type() == typeid(std::vector<int>)) {  // NOLINT
      std::vector<int> val = BOOST_GET_CONST(std::vector<int>, attr);
      std::vector<int64_t> vec(val.begin(), val.end());
      attr = vec;
    } else if (attr.type() == typeid(std::vector<float>)) {  // NOLINT
      std::vector<float> val = BOOST_GET_CONST(std::vector<float>, attr);
      std::vector<int64_t> vec(val.begin(), val.end());
      attr = vec;
    }
    std::vector<int64_t>* attr_value = nullptr;
    try {
      attr_value = &boost::get<std::vector<int64_t>>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type std::vector<int64_t>, its type is "
          "%s.",
          attr_name_, paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<float> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  float* operator()(Attribute& attr) const {
    if (attr.type() == typeid(int)) {  // NOLINT
      int val = BOOST_GET_CONST(int, attr);
      attr = static_cast<float>(val);
    } else if (attr.type() == typeid(int64_t)) {  // NOLINT
      int64_t val = BOOST_GET_CONST(int64_t, attr);
      attr = static_cast<float>(val);
    }
    float* attr_value = nullptr;
    try {
      attr_value = &boost::get<float>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type float, its type is %s.",
          attr_name_, paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};

template <>
struct ExtractAttribute<std::vector<double>> {
  explicit ExtractAttribute(const std::string& attr_name)
      : attr_name_(attr_name) {}

  std::vector<double>* operator()(Attribute& attr) const {
    if (attr.type() == typeid(std::vector<int>)) {  // NOLINT
      std::vector<int> val = BOOST_GET_CONST(std::vector<int>, attr);
      std::vector<double> vec(val.begin(), val.end());
      attr = vec;
    } else if (attr.type() == typeid(std::vector<float>)) {  // NOLINT
      std::vector<float> val = BOOST_GET_CONST(std::vector<float>, attr);
      std::vector<double> vec(val.begin(), val.end());
      attr = vec;
    }
    std::vector<double>* attr_value = nullptr;
    try {
      attr_value = &boost::get<std::vector<double>>(attr);
    } catch (boost::bad_get& bad_get) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Cannot get attribute (%s) by type std::vector<double>, its type is "
          "%s.",
          attr_name_, paddle::platform::demangle(attr.type().name())));
    }
    return attr_value;
  }

  const std::string& attr_name_;
};
template <typename T>
inline proto::AttrType AttrTypeID() {
  Attribute tmp = T();
  return static_cast<proto::AttrType>(tmp.which() - 1);
}

Attribute GetAttrValue(const proto::OpDesc::Attr& attr_desc);

class AttrReader {
 public:
  explicit AttrReader(const AttributeMap& attrs, const AttributeMap& default_attrs = {} ) 
      : attrs_(attrs), default_attrs_(default_attrs) {}

  template <typename T>
  inline const T& Get(const std::string& name) const {
    PADDLE_ENFORCE_NE(attrs_.count(name), 0,
                      platform::errors::NotFound(
                          "Attribute (%s) should be in AttributeMap.", name));

    Attribute& attr = const_cast<Attribute&>(attrs_.at(name));
    ExtractAttribute<T> extract_attr(name);
    T* attr_value = extract_attr(attr);
    return *attr_value;
  }

 private:
  const AttributeMap& attrs_;
  const AttributeMap& default_attrs_;
};

// check whether a value(attribute) fit a certain limit
template <typename T>
class GreaterThanChecker {
 public:
  explicit GreaterThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE_GT(
        value, lower_bound_,
        platform::errors::OutOfRange(
            "Check for attribute value greater than a certain value failed."));
  }

 private:
  T lower_bound_;
};

template <typename T>
class EqualGreaterThanChecker {
 public:
  explicit EqualGreaterThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE_GE(
        value, lower_bound_,
        platform::errors::OutOfRange("Check for attribute valur equal or "
                                     "greater than a certain value failed."));
  }

 private:
  T lower_bound_;
};

// we can provide users more common Checker, like 'LessThanChecker',
// 'BetweenChecker'...

template <typename T>
class DefaultValueSetter {
 public:
  explicit DefaultValueSetter(T default_value)
      : default_value_(default_value) {}
  const T& operator()() const { return default_value_; }

 private:
  T default_value_;
};

template <typename T>
class EnumInContainer {
 public:
  explicit EnumInContainer(const std::unordered_set<T>& c) : container_(c) {}
  void operator()(const T& val) const {
    PADDLE_ENFORCE_NE(
        container_.find(val), container_.end(),
        platform::errors::NotFound("Value %s is not in enum container %s.", val,
                                   ContainerDebugString()));
  }

 private:
  std::string ContainerDebugString() const {
    std::ostringstream sout;
    sout << "[";
    size_t cnt = 0;
    for (auto& v : container_) {
      sout << v;
      ++cnt;
      if (cnt != container_.size()) {
        sout << " ,";
      }
    }
    sout << "]";
    return sout.str();
  }

  std::unordered_set<T> container_;
};

// check whether a certain attribute fit its limits
// an attribute can have more than one limits
template <typename T>
class TypedAttrChecker {
  typedef std::function<const T&()> DefaultValueChecker;
  typedef std::function<void(const T&)> ValueChecker;

 public:
  explicit TypedAttrChecker(const std::string& attr_name)
      : attr_name_(attr_name) {}

  TypedAttrChecker& InEnum(const std::unordered_set<T>& range) {
    value_checkers_.push_back(EnumInContainer<T>(range));
    return *this;
  }

  TypedAttrChecker& GreaterThan(const T& lower_bound) {
    value_checkers_.push_back(GreaterThanChecker<T>(lower_bound));
    return *this;
  }

  TypedAttrChecker& EqualGreaterThan(const T& lower_bound) {
    value_checkers_.push_back(EqualGreaterThanChecker<T>(lower_bound));
    return *this;
  }

  // we can add more common limits, like LessThan(), Between()...

  TypedAttrChecker& SetDefault(const T& default_value) {
    PADDLE_ENFORCE_EQ(
        default_value_setter_.empty(), true,
        platform::errors::AlreadyExists(
            "Attribute (%s) has a default value and cannot be set repeatedly.",
            attr_name_));
    default_value_setter_.push_back(DefaultValueSetter<T>(default_value));
    return *this;
  }

  // allow users provide their own checker
  TypedAttrChecker& AddCustomChecker(const ValueChecker& checker) {
    value_checkers_.push_back(checker);
    return *this;
  }

  void operator()(AttributeMap* attr_map,
                  bool get_default_value_only = false) const {
    if (get_default_value_only) {
      if (!default_value_setter_.empty()) {
        attr_map->emplace(attr_name_, default_value_setter_[0]());
      }
      return;
    }

    auto it = attr_map->find(attr_name_);
    if (it == attr_map->end()) {
      // user do not set this attr
      PADDLE_ENFORCE_EQ(
          default_value_setter_.empty(), false,
          platform::errors::InvalidArgument(
              "Attribute (%s) is not set correctly.", attr_name_));
      // default_value_setter_ has no more than one element
      attr_map->emplace(attr_name_, default_value_setter_[0]());
    }
    it = attr_map->find(attr_name_);
    ExtractAttribute<T> extract_attr(attr_name_);
    T* attr_value = extract_attr(it->second);
    for (const auto& checker : value_checkers_) {
      checker(*attr_value);
    }
  }

 private:
  std::string attr_name_;
  std::vector<ValueChecker> value_checkers_;
  std::vector<DefaultValueChecker> default_value_setter_;
};

// check whether op's all attributes fit their own limits
class OpAttrChecker {
  typedef std::function<void(AttributeMap*, bool)> AttrChecker;

 public:
  template <typename T>
  TypedAttrChecker<T>& AddAttrChecker(const std::string& attr_name) {
    attr_checkers_.push_back(TypedAttrChecker<T>(attr_name));
    AttrChecker& checker = attr_checkers_.back();
    return *(checker.target<TypedAttrChecker<T>>());
  }

  void Check(AttributeMap* attr_map, bool explicit_only = false) const {
    auto checker_num = attr_checkers_.size();
    if (explicit_only) checker_num = explicit_checker_num_;
    for (size_t i = 0; i < checker_num; ++i) {
      attr_checkers_[i](attr_map, false);
    }
  }

  AttributeMap GetAttrsDefaultValuesMap() const {
    AttributeMap default_values_map;
    for (const auto& checker : attr_checkers_) {
      checker(&default_values_map, true);
    }
    return default_values_map;
  }

  void InitDefaultMap() {
    for (const auto& checker : attr_checkers_) {
      checker(&default_values_map_, true);
    }
  }

  const AttributeMap& default_attr_map() const { return default_values_map_; }

  void RecordExplicitCheckerNum() {
    explicit_checker_num_ = attr_checkers_.size();
  }

 private:
  std::vector<AttrChecker> attr_checkers_;

  AttributeMap default_values_map_;

  // in order to improve the efficiency of dynamic graph mode,
  // we divede the attribute into explicit type and implicit type.
  // for explicit attribute, we mean the attribute added in the customized
  // op makers, usually it's defined in the overloaded Make method.
  // for implicit attribute, we mean the attribute added outside of the Make
  // method like "op_role", "op_role_var", and they are useless in dynamic graph
  // mode
  size_t explicit_checker_num_;
};

}  // namespace framework
}  // namespace paddle
