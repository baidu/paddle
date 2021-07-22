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

#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/io/fs.h"

#include <boost/lexical_cast.hpp>
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"

namespace paddle {
namespace distributed {

std::shared_ptr<IndexWrapper> IndexWrapper::s_instance_(nullptr);

int TreeIndex::Load(const std::string filename) {
  int err_no;
  auto fp = paddle::framework::fs_open_read(filename, &err_no, "");
  PADDLE_ENFORCE_NE(
      fp, nullptr,
      platform::errors::InvalidArgument(
          "Open file %s failed. Please check whether the file exists.",
          filename));

  int num = 0;
  max_id_ = 0;
  fake_node_.set_id(0);
  fake_node_.set_is_leaf(false);
  fake_node_.set_probability(0.0);
  max_code_ = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp.get());
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    size_t read_num =
        fread(const_cast<char*>(content.data()), 1, num, fp.get());
    PADDLE_ENFORCE_EQ(
        read_num, static_cast<size_t>(num),
        platform::errors::InvalidArgument(
            "Read from file: %s failed. Valid Format is "
            "an integer representing the length of the following string, "
            "and the string itself.We got an iteger[% d], "
            "but the following string's length is [%d].",
            filename, num, read_num));

    KVItem item;
    PADDLE_ENFORCE_EQ(
        item.ParseFromString(content), true,
        platform::errors::InvalidArgument("Parse from file: %s failed. It's "
                                          "content can't be parsed by KVItem.",
                                          filename));

    if (item.key() == ".tree_meta") {
      meta_.ParseFromString(item.value());
    } else {
      auto code = boost::lexical_cast<uint64_t>(item.key());
      IndexNode node;
      node.ParseFromString(item.value());
      PADDLE_ENFORCE_NE(node.id(), 0,
                        platform::errors::InvalidArgument(
                            "Node'id should not be equel to zero."));
      if (node.is_leaf()) {
        id_codes_map_[node.id()] = code;
      }
      data_[code] = node;
      if (node.id() > max_id_) {
        max_id_ = node.id();
      }
      if (code > max_code_) {
        max_code_ = code;
      }
    }
    ret = fread(&num, sizeof(num), 1, fp.get());
  }
  total_nodes_num_ = data_.size();
  max_code_ += 1;
  return 0;
}

std::vector<IndexNode> TreeIndex::GetNodes(const std::vector<uint64_t>& codes) {
  std::vector<IndexNode> nodes;
  nodes.reserve(codes.size());
  for (size_t i = 0; i < codes.size(); i++) {
    if (CheckIsValid(codes[i])) {
      nodes.push_back(data_.at(codes[i]));
    } else {
      nodes.push_back(fake_node_);
    }
  }
  return nodes;
}

std::vector<uint64_t> TreeIndex::GetLayerCodes(int level) {
  uint64_t level_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  uint64_t level_offset = level_num - 1;

  std::vector<uint64_t> res;
  res.reserve(level_num);
  for (uint64_t i = 0; i < level_num; i++) {
    auto code = level_offset + i;
    if (CheckIsValid(code)) {
      res.push_back(code);
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::GetAncestorCodes(
    const std::vector<uint64_t>& ids, int level) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());

  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
    if (id_codes_map_.find(ids[i]) == id_codes_map_.end()) {
      res.push_back(max_code_);
    } else {
      auto code = id_codes_map_.at(ids[i]);
      cur_level = meta_.height() - 1;

      while (level >= 0 && cur_level > level) {
        code = (code - 1) / meta_.branch();
        cur_level--;
      }
      res.push_back(code);
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::GetChildrenCodes(uint64_t ancestor,
                                                  int level) {
  auto level_code_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  auto code_min = level_code_num - 1;
  auto code_max = meta_.branch() * level_code_num - 1;

  std::vector<uint64_t> parent;
  parent.push_back(ancestor);
  std::vector<uint64_t> res;
  size_t p_idx = 0;
  while (true) {
    size_t p_size = parent.size();
    for (; p_idx < p_size; p_idx++) {
      for (int i = 0; i < meta_.branch(); i++) {
        auto code = parent[p_idx] * meta_.branch() + i + 1;
        if (data_.find(code) != data_.end()) parent.push_back(code);
      }
    }
    if ((code_min <= parent[p_idx]) && (parent[p_idx] < code_max)) {
      break;
    }
  }

  return std::vector<uint64_t>(parent.begin() + p_idx, parent.end());
}

std::vector<uint64_t> TreeIndex::GetTravelCodes(uint64_t id, int start_level) {
  std::vector<uint64_t> res;
  PADDLE_ENFORCE_NE(id_codes_map_.find(id), id_codes_map_.end(),
                    paddle::platform::errors::InvalidArgument(
                        "id = %d doesn't exist in Tree.", id));
  auto code = id_codes_map_.at(id);
  int level = meta_.height() - 1;

  while (level >= start_level) {
    res.push_back(code);
    code = (code - 1) / meta_.branch();
    level--;
  }
  return res;
}

std::vector<IndexNode> TreeIndex::GetAllLeafs() {
  std::vector<IndexNode> res;
  res.reserve(id_codes_map_.size());
  for (auto& ite : id_codes_map_) {
    auto code = ite.second;
    res.push_back(data_.at(code));
  }
  return res;
}

}  // end namespace distributed
}  // end namespace paddle
