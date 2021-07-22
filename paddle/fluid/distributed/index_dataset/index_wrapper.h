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
#include "paddle/fluid/distributed/index_dataset/index_dataset.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {

class Index {
 public:
  Index() {}
  ~Index() {}
};

class TreeIndex : public Index {
 public:
  TreeIndex() {}
  ~TreeIndex() {}

  int Height() { return meta_.height(); }
  int Branch() { return meta_.branch(); }
  uint64_t TotalNodeNums() { return total_nodes_num_; }
  uint64_t EmbSize() { return max_id_ + 1; }
  int Load(const std::string path);

  inline bool CheckIsValid(int code) {
    if (data_.find(code) != data_.end()) {
      return true;
    } else {
      return false;
    }
  }

  std::vector<IndexNode> GetNodes(const std::vector<uint64_t>& codes);
  std::vector<uint64_t> GetLayerCodes(int level);
  std::vector<uint64_t> GetAncestorCodes(const std::vector<uint64_t>& ids,
                                         int level);
  std::vector<uint64_t> GetChildrenCodes(uint64_t ancestor, int level);
  std::vector<uint64_t> GetTravelCodes(uint64_t id, int start_level);
  std::vector<IndexNode> GetAllLeafs();

  std::unordered_map<uint64_t, IndexNode> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  uint64_t total_nodes_num_;
  TreeMeta meta_;
  uint64_t max_id_;
  uint64_t max_code_;
  IndexNode fake_node_;
};

class GraphIndex : public Index {
 public:
  GraphIndex() {}
  ~GraphIndex() {}

  uint32_t height() { return meta_.height(); }
  uint32_t width() { return meta_.width(); }
  uint32_t item_path_nums() { return meta_.item_path_nums(); }

  void set_height(uint32_t height) { meta_.set_height(height); }

  void set_width(uint32_t weight) { meta_.set_width(weight); }

  void set_item_path_nums(uint32_t num) { meta_.set_item_path_nums(num); }

  void reset_mapping() {
    item_path_dict_.clear();
    path_item_set_dict_.clear();
  }

  std::vector<uint32_t> create_path(uint64_t item_id);
  std::vector<uint32_t> generate_random_path();
  void add_item(uint64_t item_id, std::vector<uint32_t> vec);

  std::unordered_map<uint64_t, std::vector<uint32_t>> get_item_path_dict() {
    return item_path_dict_;
  }
  int load(std::string path);

  int save(std::string filename);
  int writeToFile(FILE* fp, KVItem& item);
  std::vector<std::vector<uint32_t>> get_path_of_item(
      std::vector<uint64_t>& items);
  std::vector<std::vector<uint64_t>> get_item_of_path(
      std::vector<uint32_t>& paths);
  std::vector<uint64_t> gather_unique_items_of_paths(
      std::vector<uint32_t>& paths);
  int update_Jpath_of_item(
      std::unordered_map<uint64_t, std::vector<uint32_t>>& candidate_list,
      std::unordered_map<uint64_t, std::vector<double>>& candidate_score,
      const int T, const double lamb, const int polynomial_order);

 private:
  GraphMeta meta_;

  std::unordered_map<uint64_t, std::vector<uint32_t>> item_path_dict_;
  std::unordered_map<uint32_t, std::unordered_set<uint64_t>>
      path_item_set_dict_;
};

using TreePtr = std::shared_ptr<TreeIndex>;
using GraphPtr = std::shared_ptr<GraphIndex>;

class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  void clear_tree() { tree_map.clear(); }

  TreePtr get_tree_index(const std::string name) {
    PADDLE_ENFORCE_NE(tree_map.find(name), tree_map.end(),
                      paddle::platform::errors::InvalidArgument(
                          "tree [%s] doesn't exist. Please insert it firstly "
                          "by API[\' insert_tree_index \'].",
                          name));
    return tree_map[name];
  }

  void insert_tree_index(const std::string name, const std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      VLOG(0) << "Tree " << name << " has already existed.";
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    int ret = tree->Load(tree_path);
    PADDLE_ENFORCE_EQ(ret, 0, paddle::platform::errors::InvalidArgument(
                                  "Load tree[%s] from path[%s] failed. Please "
                                  "check whether the file exists.",
                                  name, tree_path));
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  GraphPtr GetGraphIndex(const std::string name) {
    PADDLE_ENFORCE_NE(graph_map.find(name), graph_map.end(), "");
    return graph_map[name];
  }

  void insert_graph_index(std::string name, std::string graph_path) {
    GraphPtr graph = std::make_shared<GraphIndex>();
    int ret = graph->load(graph_path);
    if (ret != 0) return;
    graph_map.insert(std::pair<std::string, GraphPtr>{name, graph});
  }

  void insert_graph_index_by_meta_info(std::string name, int height, int width,
                                       int path_volume) {
    GraphPtr graph = std::make_shared<GraphIndex>();
    graph->set_height(height);
    graph->set_width(width);
    graph->set_item_path_nums(path_volume);
    graph_map.insert(std::pair<std::string, GraphPtr>{name, graph});
  }

  void save_graph_index_to_file(std::string name, std::string graph_path) {
    if (graph_map.find(name) != graph_map.end()) {
      VLOG(0) << "graph index " << name << " is not found";
      return;
    }
    graph_map[name]->save(graph_path);
  }

  static std::shared_ptr<IndexWrapper> GetInstancePtr() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::IndexWrapper());
    }
    return s_instance_;
  }

  static IndexWrapper* GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::distributed::IndexWrapper());
    }
    return s_instance_.get();
  }

 private:
  static std::shared_ptr<IndexWrapper> s_instance_;
  std::unordered_map<std::string, TreePtr> tree_map;
  std::unordered_map<std::string, GraphPtr> graph_map;
};

}  // end namespace distributed
}  // end namespace paddle
