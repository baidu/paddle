/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/graph_helper.h"
#include <queue>
#include <stack>

DEFINE_string(print_sub_graph_dir, "",
              "FLAGS_print_sub_graph_dir is used "
              "to print the nodes of sub_graphs.");

namespace paddle {
namespace framework {
namespace ir {
namespace {
void SortHelper(const std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>,
                               ir::NodeComp> &adj_list,
                ir::Node *node, std::unordered_set<ir::Node *> *visited,
                std::vector<ir::Node *> *ret) {
  visited->insert(node);

  for (auto adj : adj_list.at(node)) {
    if (visited->find(adj) == visited->end()) {
      SortHelper(adj_list, adj, visited, ret);
    }
  }

  VLOG(5) << "topology sort insert: " << node->Name() << " "
          << reinterpret_cast<void *>(node) << " input " << node->inputs.size();
  ret->push_back(node);
}

bool HasCircleHelper(
    ir::Node *node,
    const std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
        &adj_list,
    std::unordered_set<ir::Node *> *visited,
    std::unordered_set<ir::Node *> *in_trace,
    std::vector<std::vector<ir::Node *>> *circles) {
  if (visited->find(node) == visited->end()) {
    visited->insert(node);
    in_trace->insert(node);

    for (ir::Node *in : adj_list.at(node)) {
      if (visited->find(in) == visited->end() &&
          HasCircleHelper(in, adj_list, visited, in_trace, circles)) {
        return true;
      } else if (in_trace->find(in) != in_trace->end()) {
        if (circles != nullptr) {
          std::vector<ir::Node *> circle;
          circle.emplace_back(in);
          ir::Node *p = in;
          for (auto &adj : adj_list.at(p)) {
            if (in_trace->count(adj)) {
              circle.emplace_back(adj);
              p = adj;
            }
          }
          circles->emplace_back(circle);
        }
        return true;
      }
    }
  }
  in_trace->erase(node);
  return false;
}

bool HasCircleInternal(
    const std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
        &adj_list,
    std::vector<std::vector<ir::Node *>> *circles) {
  std::unordered_set<ir::Node *> visited;
  std::unordered_set<ir::Node *> in_trace;
  for (auto &adj : adj_list) {
    if (HasCircleHelper(adj.first, adj_list, &visited, &in_trace, circles)) {
      return true;
    }
  }
  return false;
}
}  // namespace

bool HasCircle(const Graph &graph) {
  return HasCircleInternal(BuildOperationAdjList(graph), nullptr);
}

bool VarDescIsConsistency(const Graph &graph) {
  std::unordered_map<std::string, std::unordered_set<ir::Node *>>
      var_name2node_set;
  for (ir::Node *node : graph.Nodes()) {
    if (node->IsVar() && node->Var()) {
      var_name2node_set[node->Var()->Name()].emplace(node);
    }
  }
  for (auto &iter : var_name2node_set) {
    auto &first_node = *iter.second.begin();
    bool is_persistable = std::any_of(iter.second.begin(), iter.second.end(),
                                      [&first_node](const ir::Node *node) {
                                        return node->Var()->Persistable();
                                      });
    if (is_persistable) {
      bool is_consistency =
          std::all_of(iter.second.begin(), iter.second.end(),
                      [&first_node](const ir::Node *node) {
                        return *node->Var() == *first_node->Var();
                      });
      if (!is_consistency) return false;
    }
  }
  return true;
}
bool FindCircleSubGraph(const Graph &graph,
                        std::vector<std::vector<ir::Node *>> *circles) {
  return HasCircleInternal(BuildOperationAdjList(graph), circles);
}

std::vector<ir::Node *> TopologySortOperations(const Graph &graph) {
  std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
      adj_list = BuildOperationAdjList(graph);
  PADDLE_ENFORCE_EQ(HasCircleInternal(adj_list, nullptr), false,
                    platform::errors::InvalidArgument(
                        "Generated graph shouldn't contain cycle."));
  std::unordered_set<ir::Node *> visited;
  std::vector<ir::Node *> ret;
  for (auto adj : adj_list) {
    if (visited.find(adj.first) == visited.end()) {
      SortHelper(adj_list, adj.first, &visited, &ret);
    }
  }

  return ret;
}

// Build operator inlink edge table.
std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
BuildOperationAdjList(const Graph &graph) {
  std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
      adj_list;

  for (auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::set<ir::Node *, ir::NodeComp>();
    }
    for (auto &var : n->inputs) {
      for (auto &adj_n : var->inputs) {
        PADDLE_ENFORCE_EQ(
            adj_n->NodeType(), ir::Node::Type::kOperation,
            platform::errors::InvalidArgument(
                "Node(%s)'s type(%d) must be kOperation type.", adj_n->Name(),
                static_cast<int>(adj_n->NodeType())));
        VLOG(4) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                << " -> " << n->Name() << reinterpret_cast<void *>(n)
                << "  via " << var->Name() << reinterpret_cast<void *>(var);
        adj_list[n].insert(adj_n);
      }
    }
  }
  return adj_list;
}

// Build operator outlink edge table.
std::map<ir::Node *, std::unordered_set<ir::Node *>> BuildOperationOutAdjList(
    const Graph &graph) {
  std::map<ir::Node *, std::unordered_set<ir::Node *>> adj_list;

  for (auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;
    if (adj_list.find(n) == adj_list.end()) {
      adj_list[n] = std::unordered_set<ir::Node *>();
    }
    for (auto &var : n->outputs) {
      for (auto &adj_n : var->outputs) {
        PADDLE_ENFORCE_EQ(
            adj_n->NodeType(), ir::Node::Type::kOperation,
            platform::errors::InvalidArgument(
                "Node(%s)'s type(%d) must be kOperation type.", adj_n->Name(),
                static_cast<int>(adj_n->NodeType())));
        VLOG(40) << "adj " << adj_n->Name() << reinterpret_cast<void *>(adj_n)
                 << " -> " << n->Name() << reinterpret_cast<void *>(n)
                 << "  via " << var->Name() << reinterpret_cast<void *>(var);
        adj_list[n].insert(adj_n);
      }
    }
  }
  return adj_list;
}

std::vector<ir::Node *> OpDFSSort(const Graph &graph) {
  auto edge_table = BuildOperationOutAdjList(graph);
  std::stack<Node *> stack;
  for (auto &ele : edge_table) {
    if (ele.first->inputs.empty()) {
      // find the input ops (those without input vars)
      stack.push(ele.first);
    } else {
      // find the ops with only persistable vars as inputs.
      bool all_persistable = true;
      for (auto *input : ele.first->inputs) {
        if (!(input->IsVar() && input->Var() && input->Var()->Persistable())) {
          all_persistable = false;
        }
      }
      if (all_persistable) {
        stack.push(ele.first);
      }
    }
  }

  std::vector<Node *> res;
  // start from the feed op and DFS
  std::unordered_set<Node *> unique_set;
  while (!stack.empty()) {
    // will start from the last feed by default.
    auto cur = stack.top();
    stack.pop();
    unique_set.insert(cur);
    res.push_back(cur);

    for (auto *op : edge_table[cur]) {
      if (!unique_set.count(op)) {
        stack.push(op);
      }
    }
  }
  return res;
}

std::vector<ir::Node *> TopologyDfsSortOperations(const Graph &graph) {
  std::vector<ir::Node *> nodes;
  std::unordered_map<Node *, int> in_degree;

  auto set_out_ops_ready = [&](Node *var) {
    for (auto *op : var->outputs) {
      --in_degree[op];
    }
  };
  // build in_degree
  for (auto *node : graph.Nodes()) {
    if (node->IsOp()) {
      in_degree[node] += node->inputs.size();
    } else if (node->IsVar() && node->inputs.empty()) {
      // put all the inputs of the whole graph ready.
      set_out_ops_ready(node);
    }
  }

  std::deque<Node *> op_queue;
  // first visit
  for (auto &node : OpDFSSort(graph)) {
    if (node->IsOp()) {
      op_queue.push_back(node);
    }
  }

  // traverse the graph
  int num_ops = op_queue.size();
  while (num_ops) {
    for (auto it = op_queue.begin(); it != op_queue.end(); it++) {
      auto *&cur_op = *it;
      if (!cur_op || in_degree[cur_op] > 0) continue;
      // visit this node
      // put all the output var of this op valid.
      for (auto *out_var : cur_op->outputs) {
        if (!out_var) continue;
        set_out_ops_ready(out_var);
      }
      VLOG(8) << "visit " << cur_op->Name();
      nodes.push_back(cur_op);

      cur_op = nullptr;
      num_ops--;
    }
  }

  return nodes;
}

size_t GraphNum(const Graph &graph) {
  std::unordered_set<ir::Node *> nodes(graph.Nodes());
  std::unordered_set<ir::Node *> visited_nodes;
  visited_nodes.reserve(nodes.size());
  std::deque<ir::Node *> q_nodes;
  std::vector<std::unordered_set<ir::Node *>> graph_nodes;
  std::unordered_set<ir::Node *> g_nodes;
  // q_set used to record records in the queue.
  std::unordered_set<ir::Node *> q_set;
  size_t graph_count = 0;

  auto traverse_nodes = [&visited_nodes, &q_nodes,
                         &q_set](const std::vector<ir::Node *> &nodes) {
    for (auto n : nodes) {
      if (visited_nodes.count(n) == 0 && q_set.count(n) == 0) {
        q_nodes.push_back(n);
        q_set.insert(n);
      }
    }
  };

  while (visited_nodes.size() != nodes.size()) {
    if (!q_nodes.empty()) {
      auto cur_node = q_nodes.front();
      q_nodes.pop_front();
      q_set.erase(cur_node);
      visited_nodes.insert(cur_node);
      g_nodes.insert(cur_node);
      traverse_nodes(cur_node->inputs);
      traverse_nodes(cur_node->outputs);
    } else {
      ++graph_count;
      if (g_nodes.size()) {
        graph_nodes.emplace_back(g_nodes);
      }
      g_nodes.clear();
      for (auto &n : nodes) {
        if (visited_nodes.count(n) == 0) {
          q_nodes.push_back(n);
          q_set.insert(n);
          break;
        }
      }
    }
  }

  if (g_nodes.size()) {
    graph_nodes.emplace_back(g_nodes);
  }

  if (FLAGS_print_sub_graph_dir.size()) {
    if (graph_nodes.size() > 1) {
      std::stringstream out;
      for (auto &g_n : graph_nodes) {
        out << "graph_nodes: " << g_n.size() << "\n";
      }
      out << "\n\n";
      for (auto &g_n : graph_nodes) {
        out << "graph_nodes: " << g_n.size();
        for (auto &node : g_n) {
          out << "\nNode: " << node->Name() << " in [";
          for (auto &n : node->inputs) {
            out << n->Name() << ", ";
          }
          out << "], out[";
          for (auto &n : node->outputs) {
            out << n->Name() << ", ";
          }
          out << "]";
        }
        out << "\n\n\n";
      }
      std::unique_ptr<std::ostream> fout(
          new std::ofstream(FLAGS_print_sub_graph_dir));
      PADDLE_ENFORCE_EQ(fout->good(), true,
                        platform::errors::Unavailable(
                            "Can not open file %s for printing the graph.",
                            FLAGS_print_sub_graph_dir));
      *fout << out.str();
    }
  }

  return graph_count;
}

void CleanIndividualNodes(Graph *graph) {
  std::unordered_set<Node *> nodes2rm;
  for (auto *node : graph->Nodes()) {
    if (node->inputs.empty() && node->outputs.empty()) {
      nodes2rm.insert(node);
    }
  }

  for (auto *node : nodes2rm) {
    graph->RemoveNode(node);
  }
}

std::vector<Node *> TopologyVarientSort(const Graph &graph,
                                        SortKind sort_kind) {
  switch (sort_kind) {
    case SortKind::TS:
      return framework::ir::TopologySortOperations(graph);
    default:
      return framework::ir::TopologyDfsSortOperations(graph);
  }
}

class DescOrderComparator {
 public:
  bool operator()(const Node *n1, const Node *n2) {
    return (n1->DescOrder() > n2->DescOrder()) ||
           ((n1->DescOrder() == n2->DescOrder()) &&
            (n1->ToString() > n2->ToString()));
  }
};

std::vector<ir::Node *> TopologySortGraphByDescOrder(const Graph &graph) {
  std::vector<ir::Node *> sorted_ops;
  std::priority_queue<Node *, std::vector<Node *>, DescOrderComparator> q;
  std::unordered_map<Node *, std::unordered_set<Node *>> in_ops;
  std::unordered_map<Node *, std::unordered_set<Node *>> out_ops;

  // ensure all op node in 'in_ops' and 'out_ops'
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    in_ops.emplace(n, std::unordered_set<Node *>());
    out_ops.emplace(n, std::unordered_set<Node *>());
  }

  // record all op's input op and output op
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    // traverse all input op
    for (const auto &var : n->inputs) {
      for (const auto &in : var->inputs) {
        // use at instead of [] to prevent no unrecorded op node
        in_ops.at(n).insert(in);
        out_ops.at(in).insert(n);
      }
    }
  }

  // find topology entrance
  for (const auto &n : graph.Nodes()) {
    if (!n->IsOp()) continue;

    if (in_ops.at(n).empty()) {
      q.push(n);
    }
  }

  // topological sorting
  while (!q.empty()) {
    // Do not get by reference!!! The element will pop later.
    const auto cur_op = q.top();
    q.pop();

    sorted_ops.push_back(cur_op);
    for (const auto &out : out_ops.at(cur_op)) {
      PADDLE_ENFORCE_GT(in_ops.at(out).count(cur_op), 0,
                        platform::errors::InvalidArgument(
                            "We find %s in %s's output list, "
                            "but cannot find %s in %s's input list. "
                            "Please ensure graph completely.",
                            out->Name().c_str(), cur_op->Name().c_str(),
                            cur_op->Name().c_str(), out->Name().c_str()));
      in_ops.at(out).erase(cur_op);

      // push if in-degree is 0
      if (in_ops.at(out).empty()) {
        q.push(out);
      }
    }
  }

  PADDLE_ENFORCE_EQ(
      sorted_ops.size(), in_ops.size(),
      platform::errors::InvalidArgument("Topological sorting incompletely, "
                                        "only sorted %zd op but total %zd.",
                                        sorted_ops.size(), in_ops.size()));

  return sorted_ops;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
