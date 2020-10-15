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

#include <map>
#include <string>

#include "paddle/fluid/framework/program_desc.h"

#pragma once

namespace paddle {
namespace framework {

namespace proto {
class OpCompatibleMap;
}  // namespace proto

enum class OpCompatibleType {
  compatible = 0,       //   support previous version
  DEFIN_NOT = 1,        //   definitely can't support previous version
  possible = 2,         //   possible can support previous version, not sure
  bug_fix = 3,          //   bug fix, can't support previous version
  precision_change = 4  //   precision change, may cause difference
};

struct CompatibleInfo {
  CompatibleInfo(std::string required_version, OpCompatibleType compatible_type)
      : required_version_(required_version),
        compatible_type_(compatible_type) {}
  CompatibleInfo() {}

  // op required version, previous version not support
  std::string required_version_;
  OpCompatibleType compatible_type_;
};

class OpCompatibleMap {
 public:
  OpCompatibleMap() : default_required_version_("1.5.0") {}
  void InitOpCompatibleMap();

  CompatibleInfo GetOpCompatibleInfo(std::string op_name) const;

  /* IsRequireMiniVersion
   *  return type OpCompatibleType */

  OpCompatibleType IsRequireMiniVersion(std::string op_name,
                                        std::string current_version) const;

  const std::string& GetDefaultRequiredVersion() const {
    return default_required_version_;
  }

 private:
  std::map<std::string, CompatibleInfo> op_compatible_map_;
  std::string default_required_version_;
};

}  // namespace framework
}  // namespace paddle
