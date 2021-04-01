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
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <thread>
#include <stdio.h>

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/distributed/service/communicator.h"
#include "paddle/fluid/framework/fleet/index_wrapper.h"

namespace paddle {
namespace framework {

using paddle::distributed::Communicator;

std::shared_ptr<IndexWrapper> IndexWrapper::s_instance_(nullptr);

int TreeIndex::load(std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }

  int num = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    if (fread(const_cast<char*>(content.data()), 1, num, fp)
        != static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
    KVItem item;
    if (!item.ParseFromString(content)) {
      fprintf(stderr, "Parse from file: %s failed.\n", filename.c_str());
      break;
    }
    if (item.key() == ".tree_meta") {
      meta_.ParseFromString(item.value());
    } else {
      auto code = boost::lexical_cast<uint64_t>(item.key());
      Node node;
      node.ParseFromString(item.value());
      if (node.is_leaf()) {
        id_codes_map_[node.id()] = code;
      }
      data_[code] = node;
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }
  fclose(fp);
  total_nodes_num_ = data_.size();
  return 0;
}

std::vector<uint64_t> TreeIndex::get_nodes_given_level(int level, bool ret_code) {
  uint64_t level_num = static_cast<uint64_t>(std::pow(meta_.branch(), level));
  uint64_t level_offset = level_num - 1;

  std::vector<uint64_t> res;
  res.reserve(level_num);
  for (uint64_t i = 0; i < level_num; i++) {
    auto code = level_offset + i;
    if (data_.find(code) != data_.end()) {
      if (ret_code) {
        res.push_back(code);
      } else {
        res.push_back(data_[code].id());
      }
    }
  }
  
  return res;
}

std::vector<std::vector<uint64_t>> TreeIndex::get_parent_path(std::vector<uint64_t>& ids, int start_level, bool ret_code) {
  std::vector<std::vector<uint64_t>> res(ids.size(), std::vector<uint64_t>());
  for (size_t i = 0; i < ids.size(); i++) {
    auto code = id_codes_map_[ids[i]];
    int level = meta_.height() - 1;

    while (level >= start_level) {
      if (ret_code) {
        res[i].push_back(code);
      } else {
        res[i].push_back(data_[code].id());
      }
      code = (code - 1) / meta_.branch();
      level --;
    }
  }
  return res;
}

std::vector<uint64_t> TreeIndex::get_ancestor_given_level(std::vector<uint64_t>& ids, int level, bool ret_code) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());
  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
    auto code = id_codes_map_[ids[i]];
    cur_level = meta_.height() - 1;
    while (cur_level > level) {
      code = (code - 1) / meta_.branch();
      cur_level --;
    }
    if (ret_code) {
      res.push_back(code);
    } else {
      res.push_back(data_[code].id());
    }
  }
  return res;
}


int GraphIndex::load(std::string filename) {
  FILE* fp = fopen(filename.c_str(), "rb");
  if (fp == NULL) {
    fprintf(stderr, "Can not open file: %s\n", filename.c_str());
    return -1;
  }

  int num = 0;
  size_t ret = fread(&num, sizeof(num), 1, fp);
  while (ret == 1 && num > 0) {
    std::string content(num, '\0');
    if (fread(const_cast<char*>(content.data()), 1, num, fp)
        != static_cast<size_t>(num)) {
      fprintf(stderr, "Read from file: %s failed, invalid format.\n",
              filename.c_str());
      break;
    }
    KVItem item;
    if (!item.ParseFromString(content)) {
      fprintf(stderr, "Parse from file: %s failed.\n", filename.c_str());
      break;
    }

    //下面解析 待更改 proto文件中 graph的数据格式是 
   /*
message GraphNode {
    required uint64 node_id = 1;
    repeated pair<string node_id, float probility> =2;
}

message Path2Item{
    required string path_code = 1;
    repeated pair<uint64 item_id, float probility> = 2;
}

message Item2Path{
    required uint64 item_id=1;
    repeated pair<string path_code, float probility> = 2;
}

message GraphMeta {
   required int32 width = 1;
   required int32 height = 2; 
}

message KVItem {
    required bytes key = 1;
    required bytes value = 2;
}
   */

    if (item.key() == ".graph_meta") {
      meta_.ParseFromString(item.value());
    } else {
      auto code = boost::lexical_cast<uint64_t>(item.key());
      Node node;
      node.ParseFromString(item.value());
      //  替换为最后一列
      if (node.is_leaf()) {
        id_codes_map_[node.id()] = code;
      }
      data_[code] = node;
    }
    ret = fread(&num, sizeof(num), 1, fp);
  }
  fclose(fp);
  total_nodes_num_ = data_.size();
  return 0;
}


//map<item,[pair<path,pro>]>
//map<path,[pair<item,pro>]>

std::vector<uint64_t> GraphIndex::item_to_path(int item) {
  std::vector<uint64_t> res;
  res.reserve(ids.size());
  int cur_level;
  for (size_t i = 0; i < ids.size(); i++) {
    auto code = id_codes_map_[ids[i]];
    cur_level = meta_.height() - 1;
    while (cur_level > level) {
      code = (code - 1) / meta_.branch();
      cur_level --;
    }
    if (ret_code) {
      res.push_back(code);
    } else {
      res.push_back(data_[code].id());
    }
  }
  return res;
}

std::vector<uint64_t> GraphIndex::path_to_item(){
  std::vector<>
}

}  // end namespace framework
}  // end namespace paddle