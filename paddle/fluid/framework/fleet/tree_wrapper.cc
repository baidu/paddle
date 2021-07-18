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

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/tree_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"

namespace paddle {
namespace framework {

std::shared_ptr<TreeWrapper> TreeWrapper::s_instance_(nullptr);

int Tree::load(std::string path) {
  uint64_t linenum = 0;
  size_t idx = 0;
  std::vector<std::string> lines;
  std::vector<std::string> strs;
  std::vector<std::string> items;

  int err_no;
  std::shared_ptr<FILE> fp_ = fs_open_read(path, &err_no, "");
  string::LineFileReader reader;
  while (reader.getline(&*(fp_.get()))) {
    auto line = std::string(reader.get());
    strs.clear();
    boost::split(strs, line, boost::is_any_of("\t"));
    if (0 == linenum) {
      _total_node_num = boost::lexical_cast<size_t>(strs[0]);
      _nodes = new Node[_total_node_num];
      if (strs.size() > 1) {
        _tree_height = boost::lexical_cast<int16_t>(strs[1]);
      }
      ++linenum;
      continue;
    }
    if (strs.size() < 4) {
      LOG(WARNING) << "each line must has more than field";
      return -1;
    }
    Node& node = _nodes[idx];
    // id
    node.id = boost::lexical_cast<uint64_t>(strs[0]);
    // embedding
    items.clear();
    if (!strs[1].empty()) {
      boost::split(items, strs[1], boost::is_any_of(" "));
      for (size_t i = 0; i != items.size(); ++i) {
        node.embedding.emplace_back(boost::lexical_cast<float>(items[i]));
      }
    }
    // parent
    items.clear();
    if (!strs[2].empty()) {
      node.parent_node = _nodes + boost::lexical_cast<int>(strs[2]);
    }
    // child
    items.clear();
    if (!strs[3].empty()) {
      boost::split(items, strs[3], boost::is_any_of(" "));
      // node.sub_nodes = new Node*[items.size()];
      for (size_t i = 0; i != items.size(); ++i) {
        node.sub_nodes.push_back(_nodes + boost::lexical_cast<int>(items[i]));
        // node.sub_nodes[i] = _nodes + boost::lexical_cast<int>(items[i]);
      }
      // node.sub_node_num = items.size();
    } else {
      //没有孩子节点，当前节点是叶节点
      _leaf_node_map[node.id] = &node;
      // node.sub_node_num = 0;
    }
    if (strs.size() > 4) {
      node.height = boost::lexical_cast<int16_t>(strs[4]);
    }
    ++idx;
    ++linenum;
  }
  _head = _nodes + _total_node_num - 1;
  LOG(INFO) << "all lines:" << linenum << ", all tree nodes:" << idx;
  return 0;
}
void Tree::print_tree() {
  /*
  std::queue<Node*> q;
  if (_head) {
      q.push(_head);
  }
  while (!q.empty()) {
      const Node* node = q.front();
      q.pop();
      std::cout << "node_id: " << node->id << std::endl;
      std::cout << "node_embedding: ";
      for (int i = 0; i != node->embedding.size(); ++i) {
          std::cout << node->embedding[i] << " ";
      }
      std::cout << std::endl;
      if (node->parent_node) {
          std::cout << "parent_idx: " << node->parent_node - _nodes <<
  std::endl;
      }
      if (node->sub_node_num > 0) {
          for (int i = 0; i != node->sub_node_num; ++i) {
          std::cout << "child_idx" << i << ": " << node->sub_nodes[i] - _nodes
  << std::endl;
          }
      }
      std::cout << "-------------------------------------" << std::endl;
      for (int i = 0; i != node->sub_node_num; ++i) {
          Node* tmp_node = node->sub_nodes[i];
          q.push(tmp_node);
      }
  }
  */
}
int Tree::dump_tree(const uint64_t table_id, int fea_value_dim,
                    const std::string tree_path) {

  // pull sparse
  std::vector<uint64_t> fea_keys;
  std::vector<float*> pull_result_ptr;
  fea_keys.reserve(_total_node_num);
  pull_result_ptr.reserve(_total_node_num);

  for (size_t i = 0; i != _total_node_num; ++i) {
    _nodes[i].embedding.resize(fea_value_dim);
    fea_keys.push_back(_nodes[i].id);
    pull_result_ptr.push_back(_nodes[i].embedding.data());
  }
  std::vector<::std::future<int32_t>> pull_sparse_status;
  pull_sparse_status.resize(0);
  auto fleet_ptr = FleetWrapper::GetInstance();
  auto status = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr.data(), table_id, fea_keys.data(), fea_keys.size());
  pull_sparse_status.push_back(std::move(status));
  for (auto& t : pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    }
  }

  std::cout<< "\ndone pull_sparse";
  int ret;
  std::shared_ptr<FILE> fp =
      paddle::framework::fs_open(tree_path, "w", &ret, "");
  std::string first_line = boost::lexical_cast<std::string>(_total_node_num) +
                           "\t" +
                           boost::lexical_cast<std::string>(_tree_height);
  fwrite(first_line.c_str(), first_line.length(), 1, &*fp);
  std::string line_break_str("\n");
  std::string line("");
  for (size_t i = 0; i != _total_node_num; ++i) {
    line = line_break_str;
    const Node& node = _nodes[i];
    line += boost::lexical_cast<std::string>(node.id) + "\t";
    if (!node.embedding.empty()) {
      for (size_t j = 0; j != node.embedding.size() - 1; ++j) {
        line += boost::lexical_cast<std::string>(node.embedding[j]) + " ";
      }
      line += boost::lexical_cast<std::string>(
          node.embedding[node.embedding.size() - 1]);
    } else {
      LOG(WARNING) << "node_idx[" << i << "], id[" << node.id << "] "
                   << "has no embeddings";
    }
    line += "\t";
    if (node.parent_node) {
      line += boost::lexical_cast<std::string>(node.parent_node - _nodes);
    }
    line += "\t";
    if (node.sub_nodes.size() > 0) {
      for (uint32_t j = 0; j < node.sub_nodes.size() - 1; ++j) {
        line +=
            boost::lexical_cast<std::string>(node.sub_nodes[j] - _nodes) + " ";
      }
      line += boost::lexical_cast<std::string>(
          node.sub_nodes[node.sub_nodes.size() - 1] - _nodes);
    }
    line += "\t" + boost::lexical_cast<std::string>(node.height);
    fwrite(line.c_str(), line.length(), 1, &*fp);
  }
  return 0;
}

bool Tree::trace_back(uint64_t id,
                      std::vector<std::pair<uint64_t, uint32_t>>* ids) {
  ids->clear();
  std::unordered_map<uint64_t, Node*>::iterator find_it =
      _leaf_node_map.find(id);
  if (find_it == _leaf_node_map.end()) {
    return false;
  } else {
    uint32_t height = 0;
    Node* node = find_it->second;
    while (node != NULL) {
      height++;
      ids->emplace_back(node->id, 0);
      node = node->parent_node;
    }
    for (auto& pair_id : *ids) {
      pair_id.second = height--;
    }
  }
  return true;
}

Node* Tree::get_node() { return _nodes; }
size_t Tree::get_total_node_num() { return _total_node_num; }

}  // end namespace framework
}  // end namespace paddle
