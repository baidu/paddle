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

#pragma once

#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace framework {
namespace ir {

// Compare nodes via node id.
class Graph;

struct NodeComp {
  bool operator()(ir::Node *const &node1, ir::Node *const &node2) const {
    return node1->id() < node2->id();
  }
};

// Test if the graph contains circle.
bool HasCircle(const Graph &graph);

// Check if the var desc of node is consistency.
// The graph may have the same name node, for example, parameter
// is the input of operator and it also is the output of optimizer.
// For the persistable variable, the var_desc of the nodes with
// the same node name should be equal.
bool VarDescIsConsistency(const Graph &graph);

// Find All Circles for debugging,
// store all subgraph in circles.
bool FindCircleSubGraph(const Graph &graph,
                        std::vector<std::vector<ir::Node *>> *circles);

size_t GraphNum(const Graph &graph);

// Topology Sort the operations in the graph from inputs to outputs.
// `graph` cannot contain circle.
std::vector<ir::Node *> TopologySortOperations(const Graph &graph);

// Topological sort, but try to DFS.
std::vector<ir::Node *> TopologyDfsSortOperations(const Graph &graph);

// Different kinds to sort the operators in a graph to a sequence.
enum class SortKind {
  // Topological Search
  TS = 0,
  // Topological and Depth First Search
  TDFS
};

// Several kinds of topological sort.
std::vector<Node *> TopologyVarientSort(const Graph &graph, SortKind sort_kind);

// Clean the nodes that doesn't connect to others.
void CleanIndividualNodes(Graph *graph);

// Build an adjacency list of operations for the `graph`.
std::map<ir::Node *, std::set<ir::Node *, ir::NodeComp>, ir::NodeComp>
BuildOperationAdjList(const Graph &graph);

template <typename T>
std::vector<T *> FilterByNodeWrapper(const Graph &graph) {
  std::vector<T *> ret;
  for (ir::Node *n : graph.Nodes()) {
    if (n->IsWrappedBy<T>()) ret.push_back(&n->Wrapper<T>());
  }
  return ret;
}

std::vector<ir::Node *> TopologySortGraphByDescOrder(const Graph &graph);

}  // namespace ir
}  // namespace framework
}  // namespace paddle
