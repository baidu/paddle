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
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/index_dataset.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

class Index {
 public:
  Index() {}
  ~Index() {}
};

class TreeIndex : public Index {
 public:
  TreeIndex() {}
  ~TreeIndex() {}

  int height() { return meta_.height(); }
  int branch() { return meta_.branch(); }
  uint64_t total_node_nums() { return total_nodes_num_; }

  std::vector<uint64_t> get_nodes_given_level(int level, bool ret_code = false);
  std::vector<std::vector<uint64_t>> get_parent_path(std::vector<uint64_t>& ids,
                                                     int start_level = 0,
                                                     bool ret_code = false);
  std::vector<uint64_t> get_ancestor_given_level(std::vector<uint64_t>& ids,
                                                 int level,
                                                 bool ret_code = false);

  std::vector<uint64_t> get_all_items() {
    std::vector<uint64_t> ids;
    ids.reserve(id_codes_map_.size());
    for (auto& ite : id_codes_map_) {
      ids.push_back(ite.first);
    }
    return ids;
  }

  std::unordered_map<uint64_t, uint64_t> get_relation(
      int level, std::vector<uint64_t>& ids) {
    std::unordered_map<uint64_t, uint64_t> pi_new;

    for (auto& id : ids) {
      auto code = id_codes_map_[id];
      auto cur_level = meta_.height() - 1;
      while (cur_level > level) {
        code = (code - 1) / meta_.branch();
        cur_level--;
      }
      pi_new[id] = code;
    }
    return pi_new;
  }

  std::vector<uint64_t> get_children_given_ancestor_and_level(uint64_t ancestor,
                                                              int level) {
    auto level_code_num =
        static_cast<uint64_t>(std::pow(meta_.branch(), level));
    auto code_min = level_code_num - 1;
    auto code_max = level * level_code_num - 1;

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

  std::vector<uint64_t> get_travel_path(uint64_t child, uint64_t ancestor) {
    std::vector<uint64_t> res;
    while (child > ancestor) {
      res.push_back(data_[child].id());
      child = (child - 1) / meta_.branch();
    }
    return res;
  }

  uint64_t tree_max_node() { return max_id_; }

  int load(const std::string path);
  std::unordered_map<uint64_t, Node> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;

  uint64_t total_nodes_num_;
  TreeMeta meta_;
  uint64_t max_id_;
};

class GraphIndex : public Index {
 public:
  GraphIndex() {}
  ~GraphIndex() {}

  int height() { return meta_.height(); }
  int width() { return meta_.width(); }

  int load(std::string path);

  std::vector<std::vector<int64_t>> get_path_of_item(
      std::vector<uint64_t>& items);
  std::vector<std::vector<uint64_t>> get_item_of_path(
      std::vector<int64_t>& paths);

 private:
  GraphMeta meta_;
  std::unordered_map<uint64_t, std::vector<int64_t>> item_path_dict_;
  std::unordered_map<int64_t, std::unordered_set<uint64_t>> path_item_set_dict_;
};

using TreePtr = std::shared_ptr<TreeIndex>;
using GraphPtr = std::shared_ptr<GraphIndex>;

class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  void clear_tree() { tree_map.clear(); }

  TreePtr GetTreeIndex(const std::string name) {
    PADDLE_ENFORCE_NE(tree_map.find(name), tree_map.end(), "");
    return tree_map[name];
  }

  void insert_tree_index(const std::string name, const std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    int ret = tree->load(tree_path);
    if (ret != 0) return;
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  GraphPtr GetGraphIndex(const std::string name) {
    PADDLE_ENFORCE_NE(graph_map.find(name), graph_map.end(), "");
    return graph_map[name];
  }

  void insert_graph_index(std::string name, std::string graph_path) {
    if (graph_map.find(name) != graph_map.end()) {
      return;
    }
    GraphPtr graph = std::make_shared<GraphIndex>();
    int ret = graph->load(graph_path);
    if (ret != 0) return;
    graph_map.insert(std::pair<std::string, GraphPtr>{name, graph});
  }

  // IndexWrapper singleton
  static std::shared_ptr<IndexWrapper> GetInstancePtr() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::IndexWrapper());
    }
    return s_instance_;
  }

  // IndexWrapper singleton
  static IndexWrapper* GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::IndexWrapper());
    }
    return s_instance_.get();
  }

 private:
  static std::shared_ptr<IndexWrapper> s_instance_;
  std::unordered_map<std::string, TreePtr> tree_map;
  std::unordered_map<std::string, GraphPtr> graph_map;
};

}  // end namespace framework
}  // end namespace paddle