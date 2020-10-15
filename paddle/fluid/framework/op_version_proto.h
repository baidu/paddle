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

#include <string>
#include "paddle/fluid/framework/framework.pb.h"

namespace paddle {
namespace framework {
namespace compatible {
namespace pb {

class OpVersion {
 public:
  explicit OpVersion(proto::OpVersion* desc) : desc_{desc} {}
  void SetVersionID(uint32_t version) { desc_->set_version(version); }

 private:
  proto::OpVersion* desc_;
};

class OpVersionMap {
 public:
  explicit OpVersionMap(proto::OpVersionMap* desc) : desc_{desc} {}
  OpVersion operator[](const std::string& key) {
    for (int i = 0; i < desc_->pair_size(); ++i) {
      if (desc_->pair(i).op_name() == key) {
        return OpVersion(desc_->mutable_pair(i)->mutable_op_version());
      }
    }
    auto* pair = desc_->add_pair();
    pair->set_op_name(key);
    return OpVersion(pair->mutable_op_version());
  }

 private:
  proto::OpVersionMap* desc_;
};

}  // namespace pb
}  // namespace compatible
}  // namespace framework
}  // namespace paddle
