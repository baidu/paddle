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

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

struct Node {
  Node() : parent_node(NULL), id(0), height(0) {}
  ~Node() {}
  std::vector<Node*> sub_nodes;
  // uint32_t sub_node_num;
  Node* parent_node;
  uint64_t id;
  std::vector<float> embedding;
  int16_t height;  //层级
};

class Tree {
 public:
  Tree() : _nodes(NULL), _head(NULL) {}
  ~Tree() {
    if (_nodes) {
      delete[] _nodes;
      _nodes = NULL;
    }
  }

  void print_tree();
  int dump_tree(const uint64_t table_id, int fea_value_dim,
                const std::string tree_path);
  bool trace_back(uint64_t id, std::vector<std::pair<uint64_t, uint32_t>>* ids);
  int load(std::string path);
  Node* get_node();
  size_t get_total_node_num();

 private:
  // tree data info
  Node* _nodes{nullptr};
  // head pointer
  Node* _head{nullptr};
  // total number of nodes
  size_t _total_node_num{0};
  // leaf node map
  std::unordered_map<uint64_t, Node*> _leaf_node_map;
  // version
  std::string _version{""};
  //树的高度
  int16_t _tree_height{0};
};

using TreePtr = std::shared_ptr<Tree>;
class TreeWrapper {
 public:
  virtual ~TreeWrapper() {}
  TreeWrapper() {}

  // TreeWrapper singleton
  static std::shared_ptr<TreeWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::TreeWrapper());
    }
    return s_instance_;
  }

  void clear() { tree_map.clear(); }

  void insert(std::string name, std::string tree_path) {
    if (tree_map.find(name) != tree_map.end()) {
      return;
    }
    TreePtr tree = std::make_shared<Tree>();
    tree->load(tree_path);
    tree_map.insert(std::pair<std::string, TreePtr>{name, tree});
  }

  void dump(std::string name, const uint64_t table_id, int fea_value_dim,
            const std::string tree_path) {
    if (tree_map.find(name) == tree_map.end()) {
      return;
    }
    tree_map.at(name)->dump_tree(table_id, fea_value_dim, tree_path);
  }

  void sample(const uint16_t sample_slot, const uint64_t type_slot,
              std::vector<Record>* src_datas,
              std::vector<Record>* sample_results, const uint64_t start_h) {
    sample_results->clear();
    for (auto& data : *src_datas) {
      VLOG(1) << "src record";
      //data.Print();
      uint64_t start_idx = sample_results->size();
      VLOG(1) << "before sample, sample_results.size = " << start_idx;
      uint64_t sample_feasign_idx = -1, type_feasign_idx = -1;
      bool sample_sign = false, type_sign = false; 
      for (uint64_t i = 0; i < data.uint64_feasigns_.size(); i++) {
        if (data.uint64_feasigns_[i].slot() == sample_slot) {
          sample_sign = true;
          sample_feasign_idx = i;
        }
        if (data.uint64_feasigns_[i].slot() == type_slot) {
          type_sign = true;
          type_feasign_idx = i;
        }
        if (sample_sign && type_sign)
          break;

      }
      if (!type_sign){
        VLOG(1) << "none vertical tyep";
      }

      VLOG(1) << "sample_feasign_idx: " << sample_feasign_idx
              << "; type_feasign_idx: " << type_feasign_idx;
      // why > 0?
      if (sample_sign) {
        std::vector<std::pair<uint64_t, uint32_t>> trace_ids;
        for (std::unordered_map<std::string, TreePtr>::iterator ite =
                 tree_map.begin();
             ite != tree_map.end(); ite++) {
          bool in_tree = ite->second->trace_back(
              data.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_,
              &trace_ids);
          if (in_tree) {
            break;
          } else {
            PADDLE_ENFORCE_EQ(trace_ids.size(), 0, "");
          }
        }
        for (uint64_t i = 0; i < trace_ids.size(); i++) {
          Record instance(data);
          instance.uint64_feasigns_[sample_feasign_idx].sign().uint64_feasign_ =
              trace_ids[i].first;
          // for auc, fake node vertical id
          if (trace_ids[i].second > start_h){
            instance.uint64_feasigns_[type_feasign_idx].sign().uint64_feasign_ =
                (instance.uint64_feasigns_[type_feasign_idx]
                     .sign()
                     .uint64_feasign_ +
                 1) *
                    100 +
                trace_ids[i].second;
            sample_results->push_back(instance);
          }
        }
      }
      //for (auto i = start_idx; i < sample_results->size(); i++) {
      //  sample_results->at(i).Print();
      //}
    }
    return;
  }

 public:
  std::unordered_map<std::string, TreePtr> tree_map;

 private:
  static std::shared_ptr<TreeWrapper> s_instance_;
};

}  // end namespace framework
}  // end namespace paddle
