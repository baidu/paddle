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

#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/version.h"

namespace paddle {
namespace framework {

BlockDesc *ProgramDesc::AppendBlock(const BlockDesc &parent) {
  auto *b = desc_.add_blocks();
  b->set_parent_idx(parent.ID());
  b->set_idx(desc_.blocks_size() - 1);
  blocks_.emplace_back(new BlockDesc(this, b));
  return blocks_.back().get();
}

void ProgramDesc::Flush() {
  for (auto &block : blocks_) {
    block->Flush();
  }
}

proto::ProgramDesc *ProgramDesc::Proto() {
  Flush();
  return &desc_;
}

proto::OpVersionMap *ProgramDesc::OpVersionMap() {
  return desc_.mutable_op_version_map();
}

int64_t ProgramDesc::Version() const { return desc_.version().version(); }

void ProgramDesc::SetVersion(const int64_t version) {
  desc_.mutable_version()->set_version(version);
}

ProgramDesc::ProgramDesc() {
  SetVersion(kCurProgramVersion);
  auto *block = desc_.mutable_blocks()->Add();
  block->set_idx(kRootBlockIndex);
  block->set_parent_idx(kNoneBlockIndex);
  blocks_.emplace_back(new BlockDesc(this, block));
}

ProgramDesc::ProgramDesc(const ProgramDesc &o) {
  desc_ = o.desc_;
  for (int i = 0; i < desc_.blocks_size(); ++i) {
    auto *block = desc_.mutable_blocks(i);
    blocks_.emplace_back(new BlockDesc(*o.blocks_[i], block, this));
  }
  for (size_t block_id = 0; block_id < blocks_.size(); ++block_id) {
    auto all_ops = blocks_[block_id]->AllOps();
    for (size_t op_id = 0; op_id < all_ops.size(); ++op_id) {
      auto &op = all_ops[op_id];

      for (const std::string &attr_name : op->AttrNames()) {
        if (op->GetAttrType(attr_name) == proto::AttrType::BLOCK) {
          int sub_block_id =
              o.Block(block_id).Op(op_id)->GetBlockAttrId(attr_name);
          op->SetBlockAttr(attr_name, MutableBlock(sub_block_id));
        } else if (op->GetAttrType(attr_name) == proto::AttrType::BLOCKS) {
          std::vector<int> sub_block_ids =
              o.Block(block_id).Op(op_id)->GetBlocksAttrIds(attr_name);
          std::vector<BlockDesc *> block_descs;
          for (int block_id : sub_block_ids) {
            block_descs.push_back(MutableBlock(block_id));
          }
          op->SetBlocksAttr(attr_name, block_descs);
        }
      }
    }
  }
}

ProgramDesc::ProgramDesc(const proto::ProgramDesc &desc) {
  desc_ = desc;
  InitFromProto();
}

void ProgramDesc::CopyFrom(const proto::ProgramDesc &desc) {
  blocks_.clear();
  desc_ = desc;
  InitFromProto();
}

ProgramDesc::ProgramDesc(const std::string &binary_str) {
  PADDLE_ENFORCE_EQ(desc_.ParseFromString(binary_str), true,
                    platform::errors::InvalidArgument(
                        "Failed to parse program_desc from binary string."));
  InitFromProto();
}

void ProgramDesc::InitFromProto() {
  for (auto &block_desc : *desc_.mutable_blocks()) {
    blocks_.emplace_back(new BlockDesc(this, &block_desc));
  }
  for (auto &block : blocks_) {
    for (auto *op : block->AllOps()) {
      for (const auto &attr : op->Proto()->attrs()) {
        if (attr.type() == proto::AttrType::BLOCK) {
          size_t blk_idx = attr.block_idx();
          op->SetBlockAttr(attr.name(), this->MutableBlock(blk_idx));
        } else if (attr.type() == proto::AttrType::BLOCKS) {
          auto blks_idx = attr.blocks_idx();
          std::vector<BlockDesc *> block_descs;
          for (int blk_idx : blks_idx) {
            block_descs.push_back(this->MutableBlock(blk_idx));
          }
          op->SetBlocksAttr(attr.name(), block_descs);
        }
      }
    }
  }
}

const std::vector<std::string> ProgramDesc::GetFeedTargetNames() {
  auto &global_block = Block(0);
  // The order of feed_target_names must follow the index specified in `col`.
  // since feed operator's order doesn't necessary follow 'col'.
  std::vector<std::string> feed_target_names;
  for (auto *op : global_block.AllOps()) {
    if (op->Type() == kFeedOpType) {
      size_t col = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (col >= feed_target_names.size()) {
        feed_target_names.resize(col + 1);
      }
      feed_target_names[col] = op->Output("Out")[0];
    }
  }
  return feed_target_names;
}

const std::vector<std::string> ProgramDesc::GetFetchTargetNames() {
  auto &global_block = Block(0);
  // The order of fetch_target_names must follow the index specified in `col`.
  // since fetch operator's order doesn't necessary follow 'col'.
  std::vector<std::string> fetch_target_names;
  for (auto *op : global_block.AllOps()) {
    if (op->Type() == kFetchOpType) {
      size_t col = BOOST_GET_CONST(int, op->GetAttr("col"));
      if (col >= fetch_target_names.size()) {
        fetch_target_names.resize(col + 1);
      }
      fetch_target_names[col] = op->Input("X")[0];
    }
  }
  return fetch_target_names;
}

void ProgramDesc::SetFeedHolderName(const std::string &feed_holder_name) {
  auto *global_block = MutableBlock(0);
  int index = 0;
  for (auto *op : global_block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      // Unify the input's name of all feed_ops to feed_holder_name
      global_block->RemoveVar(op->Input("X")[0]);
      op->SetInput("X", {feed_holder_name});
      op->SetAttr("col", {index});
      op->CheckAttrs();
      index++;
    }
  }

  auto *feed_holder = global_block->Var(feed_holder_name);
  feed_holder->SetType(proto::VarType::FEED_MINIBATCH);
  feed_holder->SetPersistable(true);
}

void ProgramDesc::SetFetchHolderName(const std::string &fetch_holder_name) {
  auto *global_block = MutableBlock(0);
  int index = 0;
  for (auto *op : global_block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      // Unify the output's name of all fetch_ops to fetch_holder_name
      global_block->RemoveVar(op->Output("Out")[0]);
      op->SetOutput("Out", {fetch_holder_name});
      op->SetAttr("col", {index});
      op->CheckAttrs();
      index++;
    }
  }

  auto *fetch_holder = global_block->Var(fetch_holder_name);
  fetch_holder->SetType(proto::VarType::FETCH_LIST);
  fetch_holder->SetPersistable(true);
}

}  // namespace framework
}  // namespace paddle
