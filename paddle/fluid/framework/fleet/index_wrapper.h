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

  int height() {return meta_.height();}
  int branch() {return meta_.branch();}
  uint64_t total_node_nums() {return total_nodes_num_;}
  
  std::vector<uint64_t> get_nodes_given_level(int level, bool ret_code=false);
  std::vector<std::vector<uint64_t>> get_parent_path(std::vector<uint64_t>& ids, int start_level=0, bool ret_code=false);
  std::vector<uint64_t> get_ancestor_given_level(std::vector<uint64_t>& ids, int level, bool ret_code=false);

  int load(std::string path);
  std::unordered_map<uint64_t, Node> data_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  
  uint64_t total_nodes_num_;
  TreeMeta meta_;
};

using TreePtr = std::shared_ptr<TreeIndex>;



class GraphIndex : public Index {
 public:
  GraphIndex() {}
  ~GraphIndex() {}

  int depth() {return meta_.depth();}
  int width() {return meta_.width();}
  int k_;  //数据扩展的倍数
  
 // std::vector<float> node_pro;//map()
  std::unordered_map<uint64 code, float probability> next_node_pro;
  
  int load(std::string path);
  
  std::unordered_map<uint64_t, Node> dat_;
  std::unordered_map<uint64_t, uint64_t> id_codes_map_;
  std::unordered_map<itemId, codepath>
  
  uint64_t k_;
  GraphMeta meta_;
};

using GraphPtr = std::shared_ptr<GraphIndex>;


class IndexWrapper {
 public:
  virtual ~IndexWrapper() {}
  IndexWrapper() {}

  // tree
  void clear_tree() { tree_map.clear(); }

  TreeIndex* GetTreeIndex(const std::string name) { 
    PADDLE_ENFORCE_NE(tree_map.find(name), tree_map.end(), "");
    return tree_map[name].get();
  }

  void insert_tree_index(std::string name, std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = std::make_shared<TreeIndex>();
    int ret = tree->load(tree_path);
    if (ret != 0) return;
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }


  // graph
  void clear_graph() { graph_map.clear(); }

  TreeIndex* GetGraphIndex(const std::string name) { 
    PADDLE_ENFORCE_NE(graph_map.find(name), graph_map.end(), "");
    return graph_map[name].get();
  }

  void insert_graph_index(std::string name, std::string graph_path) {
    if (graph_map.find(name) != graph_map.end()) {
      return;
    }
    GrahPtr graph = std::make_shared<GraphIndex>();
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


  
 private:
  static std::shared_ptr<IndexWrapper> s_instance_;
  std::unordered_map<std::string, TreePtr> tree_map;
  std::unordered_map<std::string, GraphPtr> graph_map;
};

}  // end namespace framework
}  // end namespace paddle