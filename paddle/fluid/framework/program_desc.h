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

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/proto_desc.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace framework {

class BlockDesc;

class ProgramDesc {
 public:
  ProgramDesc();

  explicit ProgramDesc(const proto::ProgramDesc &desc);

  ProgramDesc(const ProgramDesc &o);

  explicit ProgramDesc(const std::string &binary_str);

  BlockDesc *AppendBlock(const BlockDesc &parent);

  BlockDesc *MutableBlock(size_t idx) {
    if (idx == static_cast<size_t>(kNoneBlockIndex)) {
      return nullptr;
    } else {
      return blocks_[idx].get();
    }
  }

  const BlockDesc &Block(size_t idx) const { return *blocks_[idx]; }

  size_t Size() const { return blocks_.size(); }

  void Flush();

  void CopyFrom(const proto::ProgramDesc &desc);

  proto::ProgramDesc *Proto();

  proto::OpVersionMap *OpVersionMap();

  int64_t Version() const;

  void SetVersion(const int64_t version);

  // The output variable of feed_op is referenced as feed_target.
  // This function is used to collect the output variable's name of all
  // feed_ops.
  const std::vector<std::string> GetFeedTargetNames();

  // The input variable of fetch_op is referenced as fetch_target.
  // This function is used to collect the input variable's name of all
  // fetch_ops.
  const std::vector<std::string> GetFetchTargetNames();

  // The input variable of feed_op that holds input Tensor provided by users is
  // referenced as feed_holder.
  // This function is used to change or unify the feed_holder variables' name.
  void SetFeedHolderName(const std::string &feed_holder_name);

  // The output variable of fetch_op that holds output Tensor needed by users is
  // referenced as fetch_holder.
  // This function is used to change or unify the fetch_holder variables' name.
  void SetFetchHolderName(const std::string &fetch_holder_name);

 private:
  void InitFromProto();

  proto::ProgramDesc desc_;

  std::vector<std::unique_ptr<BlockDesc>> blocks_;
};
}  // namespace framework
}  // namespace paddle
