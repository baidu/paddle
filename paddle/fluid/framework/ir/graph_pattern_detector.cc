// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/graph_traits.h"
#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLog;
using string::Style;

size_t PDPattern::id_ = 0UL;

PDNode *PDPattern::NewNode(const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(
        node_map_.count(name), 0UL,
        platform::errors::PreconditionNotMet(
            "PDNode's name should be unique, get duplicate [%s]", name));
  }

  nodes_.emplace_back(new PDNode(this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::NewNode(PDNode::teller_t &&teller, const std::string &name) {
  if (!name.empty()) {
    PADDLE_ENFORCE_EQ(
        node_map_.count(name), 0UL,
        platform::errors::PreconditionNotMet(
            "PDNode's name should be unique, get duplicate [%s]", name));
  }

  nodes_.emplace_back(new PDNode(std::move(teller), this, name));
  auto *cur = nodes_.back().get();
  node_map_[name] = cur;
  return cur;
}

PDNode *PDPattern::RetrieveNode(const std::string &id) const {
  auto it = node_map_.find(id);
  if (it == node_map_.end()) {
    return nullptr;
  }

  return it->second;
}

void PDPattern::AddEdge(PDNode *a, PDNode *b) {
  PADDLE_ENFORCE_NOT_NULL(
      a, platform::errors::NotFound("PDNode %s is not found.", a->name()));
  PADDLE_ENFORCE_NOT_NULL(
      b, platform::errors::NotFound("PDNode %s is not found.", b->name()));
  PADDLE_ENFORCE_NE(a, b, platform::errors::PermissionDenied(
                              "Cannot connect the same node in the graph."));
  edges_.emplace_back(a, b);
}

void GraphPatternDetector::operator()(Graph *graph,
                                      GraphPatternDetector::handle_t handler) {
  if (!MarkPDNodesInGraph(*graph)) {
    return;
  }

  auto subgraphs = DetectPatterns();
  UniquePatterns(&subgraphs);
  SortSubgraphs(&subgraphs);
  RemoveOverlappedMatch(&subgraphs);
  ValidateByNodeRole(&subgraphs);

  if (subgraphs.empty()) return;

  int id = 0;
  for (auto &g : subgraphs) {
    VLOG(3) << "optimizing #" << id++ << " subgraph";
    handler(g, graph);
  }
}

bool GraphPatternDetector::MarkPDNodesInGraph(const ir::Graph &graph) {
  VLOG(3) << "mark pdnodes in graph";
  if (graph.Nodes().empty()) return false;

  for (auto &node : GraphTraits::DFS(graph)) {
    for (const auto &pdnode : pattern_.nodes()) {
      if (pdnode->Tell(&node)) {
        VLOG(4) << "Node " << node.Name() << " marked as " << pdnode->name();
        pdnodes2nodes_[pdnode.get()].insert(&node);
      }
    }
  }
  // Check to early stop if some PDNode can't find matched Node.
  for (auto &pdnode : pattern_.nodes()) {
    if (!pdnodes2nodes_.count(pdnode.get())) {
      VLOG(4) << pdnode->name() << " can't find matched Node, early stop";
      // return false;
    }
  }
  VLOG(3) << pdnodes2nodes_.size() << " nodes marked";

  return !pdnodes2nodes_.empty();
}

// The intermediate Nodes can only link to the nodes inside the pattern, or this
// subgraph will be dropped.
void GraphPatternDetector::ValidateByNodeRole(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  std::vector<GraphPatternDetector::subgraph_t> result;

  subgraphs->erase(
      std::remove_if(
          subgraphs->begin(), subgraphs->end(),
          [](const GraphPatternDetector::subgraph_t &subgraph) -> bool {
            // Collect the inputs and outputs.
            std::set<Node *> ios;
            for (auto &item : subgraph) {
              if (!item.first->IsIntermediate()) {
                ios.insert(item.second);
              }
            }
            for (auto &item : subgraph) {
              if (item.first->IsIntermediate()) {
                for (auto *x : item.second->inputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
                for (auto *x : item.second->outputs) {
                  if (!ios.count(x)) {
                    return true;
                  }
                }
              }
            }
            return false;
          }),
      subgraphs->end());
}

struct HitGroup {
  std::map<PDNode *, Node *> roles;

  bool Match(Node *node, PDNode *pat) {
    if (nodes_.count(node)) {
      if (roles.count(pat) && roles[pat] == node) return true;
      return false;
    } else {
      if (roles.count(pat) && roles[pat] != node) return false;
      return true;
    }
  }

  void Register(Node *node, PDNode *pat) {
    roles[pat] = node;
    nodes_.insert(node);
  }

 private:
  std::set<Node *> nodes_;
};

// Tell whether Node a links to b.
bool IsNodesLink(Node *a, Node *b) {
  for (auto *node : a->outputs) {
    if (b == node) {
      return true;
    }
  }
  return false;
}

std::vector<GraphPatternDetector::subgraph_t>
GraphPatternDetector::DetectPatterns() {
  // Init empty subgraphs.
  std::vector<GraphPatternDetector::subgraph_t> result;
  std::vector<HitGroup> init_groups;
  std::array<std::vector<HitGroup>, 2> bi_records;
  auto *first_pnode = pattern_.edges().empty() ? pattern().nodes().front().get()
                                               : pattern_.edges().front().first;
  if (!pdnodes2nodes_.count(first_pnode)) return result;
  for (auto *node : pdnodes2nodes_[first_pnode]) {
    HitGroup group;
    group.roles[first_pnode] = node;
    init_groups.emplace_back(group);
  }

  int step = 0;
  bi_records[0] = std::move(init_groups);

  // Extend a PDNode to subgraphs by deducing the connection relations defined
  // in edges of PDNodes.
  for (const auto &edge : pattern_.edges()) {
    VLOG(4) << "check " << edge.first->name() << " -> " << edge.second->name();
    // TODO(Superjomn) Fix bug here, the groups might be duplicate here.
    // Each role has two PDNodes, which indicates two roles.
    // Detect two Nodes that can match these two roles and they are connected.
    auto &pre_groups = bi_records[step % 2];
    auto &cur_groups = bi_records[1 - (step++ % 2)];
    cur_groups.clear();
    if (pre_groups.empty()) break;
    // source -> target
    for (Node *source : pdnodes2nodes_[edge.first]) {
      for (Node *target : pdnodes2nodes_[edge.second]) {
        VLOG(8) << "check " << source->id() << " -- " << target->id();
        // TODO(Superjomn) add some prune strategies.
        for (const auto &group : pre_groups) {
          if (IsNodesLink(source, target)) {
            HitGroup new_group = group;
            bool flag = new_group.Match(source, edge.first) &&
                        new_group.Match(target, edge.second);
            if (flag) {
              new_group.Register(source, edge.first);
              new_group.Register(target, edge.second);
              cur_groups.push_back(new_group);
              // TODO(Superjomn) need to unique
            }
          }
        }
      }
    }
    VLOG(3) << "step " << step << " get records: " << cur_groups.size();
    for (auto &group : cur_groups) {
      for (auto &item : group.roles) {
        VLOG(4) << "node " << item.second->id() << " as " << item.first->name();
      }
      VLOG(4) << "=========================================================";
    }
  }

  for (auto &group : bi_records[step % 2]) {
    GraphPatternDetector::subgraph_t subgraph;
    for (auto &role : group.roles) {
      subgraph.emplace(role.first, role.second);
    }
    result.emplace_back(subgraph);
  }
  return result;
}

struct GraphItemLessThan {
  bool operator()(const std::pair<PDNode *, Node *> &a,
                  const std::pair<PDNode *, Node *> &b) {
    if (a.first != b.first) {
      return a.first < b.first;
    } else {
      return a.second < b.second;
    }
  }
};

// TODO(Superjomn) enhance the function as it marks unique unique as duplicates
// see https://github.com/PaddlePaddle/Paddle/issues/13550
void GraphPatternDetector::UniquePatterns(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  std::vector<GraphPatternDetector::subgraph_t> result;

  std::set<size_t> set;
  std::hash<std::string> hasher;
  for (auto &g : *subgraphs) {
    // Sort the items in the sub-graph, and transform to a string key.
    std::vector<std::pair<PDNode *, Node *>> sorted_keys(g.begin(), g.end());
    std::sort(sorted_keys.begin(), sorted_keys.end(), GraphItemLessThan());
    std::stringstream ss;
    for (auto &item : sorted_keys) {
      ss << item.first << ":" << item.second;
    }
    auto key = hasher(ss.str());
    if (!set.count(key)) {
      result.emplace_back(g);
      set.insert(key);
    }
  }
  *subgraphs = result;
}

void GraphPatternDetector::SortSubgraphs(
    std::vector<GraphPatternDetector::subgraph_t> *subgraphs) {
  if (subgraphs->empty()) return;
  bool has_bn_add_act = false;
  for (auto &subgraph : *subgraphs) {
    for (auto &item : subgraph) {
      if (item.first->name().find("bn_add_act") != std::string::npos) {
        has_bn_add_act = true;
        break;
      }
    }
  }
  if (!has_bn_add_act) {
    return;
  }

  std::sort(
      subgraphs->begin(), subgraphs->end(),
      [](const GraphPatternDetector::subgraph_t &a,
         const GraphPatternDetector::subgraph_t &b) {
        for (auto &item : a) {
          if (item.first->name().find("bn_add_act") != std::string::npos &&
              item.first->name().find("bn_reserve_space") !=
                  std::string::npos) {
            auto it_b = b.find(item.first);
            if (it_b != b.end()) {
              if (item.second->Name() != it_b->second->Name()) {
                return item.second->Name() < it_b->second->Name();
              } else {
                return false;
              }
            } else {
              return false;
            }
          }
        }
        return false;
      });
}

void GraphPatternDetector::RemoveOverlappedMatch(
    std::vector<subgraph_t> *subgraphs) {
  std::vector<subgraph_t> result;
  std::set<Node *> node_set;

  for (const auto &subgraph : *subgraphs) {
    bool valid = true;
    for (auto &item : subgraph) {
      if (item.first->IsIntermediate() && node_set.count(item.second)) {
        valid = false;
        break;
      }
    }
    if (valid) {
      for (auto &item : subgraph) {
        node_set.insert(item.second);
      }
      result.push_back(subgraph);
    }
  }
  *subgraphs = result;
}

std::string PDPattern::DotString() const {
  using inference::analysis::Dot;
  Dot dot;
  int id = 0;
  // Create Nodes
  std::unordered_map<PDNode *, std::string> node2dot;
  for (const auto &node : nodes()) {
    std::string node_id = "Node" + std::to_string(id++);
    dot.AddNode(node_id, {}, node->name());
    node2dot[node.get()] = node_id;
  }
  // Create Edges
  for (const auto &edge : edges()) {
    if (!node2dot.count(edge.first) || !node2dot.count(edge.second)) {
      LOG(ERROR) << "no node " << edge.first << " " << edge.second;
      continue;
    }
    auto &src = node2dot.at(edge.first);
    auto &trg = node2dot.at(edge.second);
    dot.AddEdge(src, trg, {});
  }
  return dot.Build();
}

PDNode &PDNode::LinksTo(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(this, x);
  }
  return *this;
}

PDNode &PDNode::LinksFrom(const std::vector<PDNode *> &others) {
  // extend outlinks.
  for (PDNode *x : others) {
    pattern_->AddEdge(x, this);
  }
  return *this;
}

PDNode *PDNode::assert_is_op() {
  asserts_.emplace_back([](Node *x) { return x && x->IsOp(); });
  return this;
}

PDNode *PDNode::assert_is_op(const std::string &op_type) {
  asserts_.emplace_back([op_type](Node *x) {
    return x && x->IsOp() && x->Op()->Type() == op_type;
  });
  return this;
}

PDNode *PDNode::assert_is_var() {
  asserts_.emplace_back([](Node *x) { return x && x->IsVar(); });
  return this;
}

PDNode *PDNode::assert_var_dtype(proto::VarType::Type dtype) {
  assert_is_var();
  asserts_.emplace_back(
      [dtype](Node *x) { return x->Var()->GetDataType() == dtype; });
  return this;
}

PDNode *PDNode::assert_is_not_ctrl_var() {
  asserts_.emplace_back([](Node *x) { return x && !x->IsCtrlVar(); });
  return this;
}

PDNode *PDNode::assert_var_not_persistable() {
  assert_is_var();
  asserts_.emplace_back([](Node *x) { return !x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_persistable_var() {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) { return x->Var()->Persistable(); });
  return this;
}

PDNode *PDNode::assert_is_op_nth_input(const std::string &op_type,
                                       const std::string &argument, int nth) {
  assert_is_var();
  assert_is_op_input(op_type);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_nth_output(const std::string &op_type,
                                        const std::string &argument, int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op->Op()->Type() == op_type &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_input_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->inputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_output_of_op(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type &&
          op->outputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_op_output(const std::string &op_type,
                                    const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_output(op_type, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_op_input(const std::string &op_type) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op->Op()->Type() == op_type) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_not_op_input(const std::string &argument) {
  assert_is_op();
  asserts_.emplace_back([=](Node *x) {
    auto &ins = x->Op()->Inputs();
    auto iter = ins.find(argument);
    return iter == ins.end() || iter->second.empty();
  });
  return this;
}

PDNode *PDNode::assert_is_op_input(const std::string &op_type,
                                   const std::string &argument) {
  assert_is_var();
  assert_is_op_nth_input(op_type, argument, 0);
  return this;
}

PDNode *PDNode::assert_op_has_n_inputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->inputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_op_has_n_outputs(const std::string &op_type, size_t n) {
  assert_is_op(op_type);
  asserts_.emplace_back([=](Node *x) { return x->outputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_has_n_inputs(size_t n) {
  asserts_.emplace_back([=](Node *x) { return x->inputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_has_n_outputs(size_t n) {
  asserts_.emplace_back([=](Node *x) { return x->outputs.size() == n; });
  return this;
}

PDNode *PDNode::assert_more(PDNode::teller_t &&teller) {
  asserts_.emplace_back(std::move(teller));
  return this;
}

PDNode *PDNode::assert_is_ops(const std::unordered_set<std::string> &op_types) {
  asserts_.emplace_back([op_types](Node *x) {
    return x && x->IsOp() && op_types.count(x->Op()->Type());
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument, int nth) {
  assert_is_var();
  assert_is_ops_input(op_types);
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthInput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_nth_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument, int nth) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op->IsOp() && op_types.count(op->Op()->Type()) &&
          IsNthOutput(x, op, argument, nth))
        return true;
    }
    return false;
  });
  return this;
}
PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_output(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_output(op_types, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type())) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_ops_input(
    const std::unordered_set<std::string> &op_types,
    const std::string &argument) {
  assert_is_var();
  assert_is_ops_nth_input(op_types, argument, 0);
  return this;
}

PDNode *PDNode::assert_is_only_input_of_ops(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->outputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type()) &&
          op->inputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

PDNode *PDNode::assert_is_only_output_of_ops(
    const std::unordered_set<std::string> &op_types) {
  assert_is_var();
  asserts_.emplace_back([=](Node *x) {
    for (auto *op : x->inputs) {
      if (op && op->IsOp() && op->Op() && op_types.count(op->Op()->Type()) &&
          op->outputs.size() == 1) {
        return true;
      }
    }
    return false;
  });
  return this;
}

bool VarLinksToOp(Node *node, const std::string &op_type) {
  for (auto *out : node->outputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

bool IsNthInput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE_EQ(
      var->IsVar(), true,
      platform::errors::InvalidArgument(
          "First parameter of function IsNthInput must be Node::Var"));
  PADDLE_ENFORCE_EQ(
      op->IsOp(), true,
      platform::errors::InvalidArgument(
          "Second parameter of function IsNthInput must be Node::Op"));
  if (!HasInput(op, argument) || op->Op()->Input(argument).size() <= nth)
    return false;
  return var->Name() == op->Op()->Input(argument)[nth];
}

bool HasInput(Node *op, const std::string &argument) {
  PADDLE_ENFORCE_EQ(
      op->IsOp(), true,
      platform::errors::InvalidArgument(
          "First parameter of function HasInput must be Node::Op"));
  auto const &names = op->Op()->InputNames();
  if (std::find(names.begin(), names.end(), argument) == names.end())
    return false;
  return true;
}

bool HasOutput(Node *op, const std::string &argument) {
  PADDLE_ENFORCE_EQ(
      op->IsOp(), true,
      platform::errors::InvalidArgument(
          "First parameter of function HasOuput must be Node::Op"));
  auto const &names = op->Op()->OutputNames();
  if (std::find(names.begin(), names.end(), argument) == names.end())
    return false;
  return true;
}

bool IsNthOutput(Node *var, Node *op, const std::string &argument, size_t nth) {
  PADDLE_ENFORCE_EQ(
      var->IsVar(), true,
      platform::errors::InvalidArgument(
          "First parameter of function IsNthOutput must be Node::Var"));
  PADDLE_ENFORCE_EQ(
      op->IsOp(), true,
      platform::errors::InvalidArgument(
          "Second parameter of function IsNthOutput must be Node::Op"));
  if (!HasOutput(op, argument) || op->Op()->Output(argument).size() <= nth)
    return false;
  return var->Name() == op->Op()->Output(argument)[nth];
}

void GraphSafeRemoveNodes(Graph *graph,
                          const std::unordered_set<const Node *> &nodes) {
  for (auto *node : nodes) {
    graph->RemoveNode(const_cast<Node *>(node));
  }

  for (auto *node : graph->Nodes()) {
    for (auto it = node->inputs.begin(); it != node->inputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->inputs.erase(it);
      } else {
        it++;
      }
    }
    for (auto it = node->outputs.begin(); it != node->outputs.end();) {
      if (nodes.count(*it)) {
        it = const_cast<Node *>(node)->outputs.erase(it);
      } else {
        it++;
      }
    }
  }
}

bool VarLinksFromOp(Node *node, const std::string &op_type) {
  for (auto *out : node->inputs) {
    if (out->IsOp() && out->Op()->Type() == op_type) {
      return true;
    }
  }
  return false;
}

PDNode *patterns::ConvBN::operator()(paddle::framework::ir::PDNode *conv_input,
                                     const std::string &conv_type,
                                     bool with_eltwise_add) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);

  PDNode *eltwise_op = nullptr;
  if (with_eltwise_add) {
    eltwise_op =
        pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  }
  auto *batch_norm_op =
      pattern->NewNode(batch_norm_repr())->assert_is_op("batch_norm");
  // Create variables
  // Conv Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input(conv_type, "Filter");

  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type);

  PDNode *eltwise_y_in_var = nullptr;
  PDNode *eltwise_out_var = nullptr;
  if (with_eltwise_add) {
    // Conv output as Bias input
    conv_out_var->assert_is_op_input("elementwise_add", "X");
    // Bias
    eltwise_y_in_var = pattern->NewNode(eltwise_y_in_repr())
                           ->assert_is_op_input("elementwise_add", "Y")
                           ->AsInput();
    eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("elementwise_add");
  } else {
    // Conv output as BN input
    conv_out_var->assert_is_op_input("batch_norm", "X");
  }

  // BN Scale
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->AsInput()
                           ->assert_is_persistable_var()
                           ->assert_is_op_input("batch_norm", "Scale")
                           ->assert_has_n_outputs(1);
  // BN Bias
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_is_op_input("batch_norm", "Bias")
                          ->assert_has_n_outputs(1);
  // BN Mean
  auto *bn_mean_var = pattern->NewNode(bn_mean_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_is_op_input("batch_norm", "Mean")
                          ->assert_has_n_outputs(1);
  // BN Variance
  auto *bn_variance_var = pattern->NewNode(bn_variance_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input("batch_norm", "Variance")
                              ->assert_has_n_outputs(1);

  // BN output
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("batch_norm", "Y");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("batch_norm", "MeanOut")
                              ->assert_has_n_outputs(0);

  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->AsOutput()
          ->assert_is_op_output("batch_norm", "VarianceOut")
          ->assert_has_n_outputs(0);

  auto *bn_saved_mean_var = pattern->NewNode(bn_saved_mean_repr())
                                ->AsOutput()
                                ->assert_is_op_output("batch_norm", "SavedMean")
                                ->assert_has_n_outputs(0);

  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("batch_norm", "SavedVariance")
          ->assert_has_n_outputs(0);

  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});

  if (with_eltwise_add) {
    eltwise_op->LinksFrom({conv_out_var, eltwise_y_in_var})
        .LinksTo({eltwise_out_var});
    batch_norm_op
        ->LinksFrom({eltwise_out_var, bn_scale_var, bn_bias_var, bn_mean_var,
                     bn_variance_var})
        .LinksTo({bn_out_var, bn_mean_out_var, bn_variance_out_var,
                  bn_saved_mean_var, bn_saved_variance_var});
  } else {
    batch_norm_op
        ->LinksFrom({conv_out_var, bn_scale_var, bn_bias_var, bn_mean_var,
                     bn_variance_var})
        .LinksTo({bn_out_var, bn_mean_out_var, bn_variance_out_var,
                  bn_saved_mean_var, bn_saved_variance_var});
  }
  return bn_out_var;
}

PDNode *patterns::ConvActivation::operator()(
    paddle::framework::ir::PDNode *conv_input, std::string conv_type,
    std::string activation_type) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);
  auto *activation_op =
      pattern->NewNode(activation_repr())->assert_is_op(activation_type);
  // Create variables
  // Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input(conv_type, "Filter");
  // intermediate variable, will be removed in the IR after fuse.
  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type)
                           ->assert_is_op_input(activation_type);
  // output
  auto *activation_out_var = pattern->NewNode(activation_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output(activation_type);

  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});
  activation_op->LinksFrom({conv_out_var}).LinksTo({activation_out_var});
  return activation_out_var;
}

PDNode *patterns::SeqConvEltAddRelu::operator()(
    paddle::framework::ir::PDNode *seqconv_input) {
  // Create Operators
  seqconv_input->assert_is_op_input("sequence_conv", "X");
  auto *seqconv_op = pattern->NewNode(seqconv_repr())
                         ->assert_is_op("sequence_conv")
                         ->assert_op_attr<bool>("paddingTrainable", false)
                         ->assert_op_attr<int>("contextStride", 1);

  auto *eltadd_op =
      pattern->NewNode(eltadd_repr())->assert_is_op("elementwise_add");
  auto *relu_op = pattern->NewNode(relu_repr())->assert_is_op("relu");
  // Create variables
  // Filter
  auto *seqconv_weight_var =
      pattern->NewNode(seqconv_weight_repr())
          ->AsInput()
          ->assert_is_persistable_var()
          ->assert_is_op_input("sequence_conv", "Filter");
  // Bias
  auto *eltadd_bias_var = pattern->NewNode(eltadd_bias_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add");
  // intermediate variable, will be removed in the IR after fuse.
  auto *seqconv_out_var = pattern->NewNode(seqconv_out_repr())
                              ->AsIntermediate()
                              ->assert_is_only_output_of_op("sequence_conv")
                              ->assert_is_op_input("elementwise_add");
  auto *eltadd_out_var = pattern->NewNode(eltadd_out_repr())
                             ->AsIntermediate()
                             ->assert_is_only_output_of_op("elementwise_add")
                             ->assert_is_only_input_of_op("relu");
  // output
  auto *relu_out_var = pattern->NewNode(relu_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("relu");

  seqconv_op->LinksFrom({seqconv_input, seqconv_weight_var})
      .LinksTo({seqconv_out_var});
  eltadd_op->LinksFrom({seqconv_out_var, eltadd_bias_var})
      .LinksTo({eltadd_out_var});
  relu_op->LinksFrom({eltadd_out_var}).LinksTo({relu_out_var});
  return relu_out_var;
}

PDNode *patterns::FC::operator()(paddle::framework::ir::PDNode *x,
                                 bool with_bias, bool with_relu) {
  // Create shared nodes.
  x->assert_is_op_input("mul", "X");
  auto *mul = pattern->NewNode(mul_repr())->assert_is_op("mul");

  auto *mul_w_var = pattern->NewNode(w_repr())
                        ->AsInput()
                        ->assert_is_persistable_var()
                        ->assert_is_op_input("mul", "Y");

  auto *mul_out_var =
      pattern->NewNode(mul_out_repr())->assert_is_op_output("mul");

  // Add links.
  mul->LinksFrom({x, mul_w_var}).LinksTo({mul_out_var});
  if (!with_bias) {  // not with bias
    return mul_out_var;
  } else {  // with bias
    mul_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
    // Create operators.
    auto *elementwise_add = pattern->NewNode(elementwise_add_repr())
                                ->assert_is_op("elementwise_add");
    // Create variables.
    auto *bias = pattern->NewNode(bias_repr())
                     ->assert_is_op_input("elementwise_add")
                     ->assert_is_persistable_var()
                     ->AsInput();

    auto *elementwise_add_out_var =
        pattern->NewNode(elementwise_add_out_repr())
            ->AsOutput()
            ->assert_is_op_output("elementwise_add");

    elementwise_add->LinksFrom({mul_out_var, bias})
        .LinksTo({elementwise_add_out_var});
    if (!with_relu) {
      return elementwise_add_out_var;
    } else {
      elementwise_add_out_var->AsIntermediate()->assert_is_op_input("relu");
      // Create operators.
      auto *relu = pattern->NewNode(relu_repr())->assert_is_op("relu");
      auto *relu_out_var = pattern->NewNode(relu_out_repr())
                               ->AsOutput()
                               ->assert_is_op_output("relu");

      relu->LinksFrom({elementwise_add_out_var}).LinksTo({relu_out_var});
      return relu_out_var;
    }
  }
}

PDNode *patterns::FCMKLDNN::operator()(paddle::framework::ir::PDNode *x,
                                       bool with_bias) {
  // Create shared nodes.
  x->assert_is_op_input("fc", "Input");

  auto *fc_op = pattern->NewNode(fc_repr())->assert_is_op("fc");
  // Create variables
  // Input
  auto *input_var = pattern->NewNode(input_repr())
                        ->AsInput()
                        ->assert_is_op_input("fc", "Input");
  // Filter
  auto *fc_weight_var = pattern->NewNode(weights_repr())
                            ->AsInput()
                            ->assert_is_op_input("fc", "W");
  // Bias
  auto *fc_bias_var = pattern->NewNode(bias_repr())
                          ->AsInput()
                          ->assert_is_op_input("fc", "Bias");
  // Output
  auto *fc_out_var = pattern->NewNode(output_repr())
                         ->AsOutput()
                         ->assert_is_op_output("fc", "Out")
                         ->assert_is_only_output_of_op("fc");

  fc_op->LinksFrom({input_var, fc_weight_var, fc_bias_var})
      .LinksTo({fc_out_var});
  return fc_out_var;
}

PDNode *patterns::FCActOneDNN::operator()(const std::string &act_type) {
  auto *fc = pattern->NewNode(fc_repr())->assert_is_op("fc");
  auto *fc_out = pattern->NewNode(fc_out_repr())
                     ->assert_is_op_output("fc", "Out")
                     ->assert_is_op_input(act_type);
  auto *act =
      pattern->NewNode(act_repr())->assert_is_op(act_type)->AsIntermediate();
  auto *act_out = pattern->NewNode(act_out_repr())
                      ->assert_is_op_output(act_type, "Out")
                      ->AsOutput();

  fc->LinksTo({fc_out});
  act->LinksFrom({fc_out}).LinksTo({act_out});

  return act_out;
}

PDNode *patterns::Embedding::operator()(PDNode *x) {
  x->assert_is_op_input("lookup_table", "Ids");
  auto *lookup_table_op =
      pattern->NewNode(lookup_table_repr())->assert_is_op("lookup_table");
#define NEW_NODE(arg__, io__)                    \
  auto *arg__ = pattern->NewNode(arg__##_repr()) \
                    ->assert_is_op_##io__("lookup_table", #arg__);

  NEW_NODE(W, input);

  NEW_NODE(Out, output);
#undef NEW_NODE

  lookup_table_op->LinksFrom({x, W});
  lookup_table_op->LinksTo({Out});
  return Out;
}

PDNode *patterns::LSTM::operator()(PDNode *x) {
  x->assert_is_op_input("lstm", "Input");
  auto *lstm_op = pattern->NewNode(lstm_repr())->assert_is_op("lstm");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("lstm", #arg__);

  // Currently, the H0 and C0 are optional
  // TODO(Superjomn) upgrade the fuse framework to support optional.
  // NEW_NODE(H0, input);
  // NEW_NODE(C0, input);
  NEW_NODE(Weight, input);
  NEW_NODE(Bias, input);

  NEW_NODE(Hidden, output);
  NEW_NODE(Cell, output);
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchCellPreAct, output);
#undef NEW_NODE

  lstm_op->LinksFrom({x, Weight, Bias});
  lstm_op->LinksTo({Hidden, Cell, BatchGate, BatchCellPreAct});
  return Hidden;
}

PDNode *patterns::GRU::operator()(PDNode *x) {
  x->assert_is_op_input("gru", "Input");
  auto *gru_op = pattern->NewNode(gru_repr())->assert_is_op("gru");
#define NEW_NODE(arg__, io__) \
  auto *arg__ =               \
      pattern->NewNode(arg__##_repr())->assert_is_op_##io__("gru", #arg__);

  NEW_NODE(Weight, input);
  // TODO(Superjomn): upgrade the fuse framework to support optional.
  // H0 and bias are optional
  NEW_NODE(Bias, input);  // also optional
  // NEW_NODE(H0, input);

  NEW_NODE(Hidden, output);
  // below are intermediate
  NEW_NODE(BatchGate, output);
  NEW_NODE(BatchResetHiddenPrev, output);
  NEW_NODE(BatchHidden, output);
#undef NEW_NODE

  BatchGate->AsIntermediate();
  BatchResetHiddenPrev->AsIntermediate();
  BatchHidden->AsIntermediate();

  gru_op->LinksFrom({x, Weight, Bias});
  gru_op->LinksTo({Hidden, BatchGate, BatchResetHiddenPrev, BatchHidden});
  return Hidden;
}

PDNode *patterns::ActElewiseAdd::operator()(
    paddle::framework::ir::PDNode *in_var,
    std::unordered_set<std::string> act_types) {
  in_var->assert_is_ops_input(act_types, "X");

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);
  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_not_ctrl_var()
                          ->assert_is_ops_output(act_types);
  act_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto *ele_x_var = pattern->NewNode(ele_x_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add")
                        ->AsInput();
  auto *elementwise_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *elewise_add_out = pattern->NewNode(elewise_add_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("elementwise_add", "Out");

  act->LinksFrom({in_var}).LinksTo({act_out_var});
  elementwise_add->LinksFrom({act_out_var, ele_x_var})
      .LinksTo({elewise_add_out});

  return elewise_add_out;
}

PDNode *patterns::BatchNormAct::operator()(
    paddle::framework::ir::PDNode *bn_x_var,
    std::unordered_set<std::string> act_types) {
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm", "Bias");
  auto *bn_variance_var = pattern->NewNode(bn_variance_repr())
                              ->assert_is_op_input("batch_norm", "Variance");
  auto *bn_mean_var = pattern->NewNode(bn_mean_repr())
                          ->assert_is_op_input("batch_norm", "Mean");

  auto *bn = pattern->NewNode(batch_norm_repr())
                 ->assert_is_op("batch_norm")
                 ->assert_is_not_op_input("MomentumTensor")
                 ->assert_op_attr<bool>("is_test", false)
                 ->assert_op_attr<bool>("use_global_stats", false)
                 ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_output("batch_norm", "ReserveSpace");
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->assert_is_op_output("batch_norm", "Y")
                         ->assert_has_n_outputs(1);

  bn_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  bn->LinksFrom(
        {bn_x_var, bn_scale_var, bn_bias_var, bn_variance_var, bn_mean_var})
      .LinksTo({bn_mean_out_var, bn_variance_out_var, bn_saved_variance_var,
                bn_saved_mean_var, bn_reserve_space, bn_out_var});
  act->LinksFrom({bn_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::BatchNormActGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_grad_types) {
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *bn_grad = pattern->NewNode(batch_norm_grad_repr())
                      ->assert_is_op("batch_norm_grad")
                      ->assert_op_attr<bool>("use_global_stats", false)
                      ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_ops_input(act_grad_types, "Out");
  auto *d_intermediate_var =
      pattern->NewNode(d_itermediate_out_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1);
  auto *bn_x_var = pattern->NewNode(bn_x_repr())
                       ->assert_is_op_input("batch_norm_grad", "X")
                       ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");
  // ReserveSpace as the output is equal to:
  // data_layout == 'NHWC' && FLAGS_cudnn_batchnorm_spatial_persistent == true
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_input("batch_norm_grad", "ReserveSpace");
  auto *d_bn_x_var =
      pattern->NewNode(d_bn_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"));
  auto *d_bn_scale_var =
      pattern->NewNode(d_bn_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn_bias_var =
      pattern->NewNode(d_bn_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));

  act_grad->LinksFrom({d_act_out_var, act_out_var})
      .LinksTo({d_intermediate_var});

  bn_grad
      ->LinksFrom({bn_x_var, d_intermediate_var, bn_scale_var, bn_bias_var,
                   bn_saved_mean_var, bn_saved_variance_var, bn_reserve_space})
      .LinksTo({d_bn_x_var, d_bn_scale_var, d_bn_bias_var});

  return bn_grad;
}

PDNode *patterns::BatchNormActOneDNN::operator()(const std::string &act_type) {
  auto *bn_x = pattern->NewNode(bn_in_repr())
                   ->AsInput()
                   ->assert_is_op_input("batch_norm", "X");
  auto *bn = pattern->NewNode(batch_norm_repr())->assert_is_op("batch_norm");
  auto *bn_out = pattern->NewNode(bn_out_repr())
                     ->assert_is_op_output("batch_norm", "Y")
                     ->assert_is_op_input(act_type);
  auto *act =
      pattern->NewNode(act_repr())->assert_is_op(act_type)->AsIntermediate();
  auto *act_out = pattern->NewNode(act_out_repr())
                      ->assert_is_op_output(act_type, "Out")
                      ->AsOutput();

  bn->LinksFrom({bn_x}).LinksTo({bn_out});
  act->LinksFrom({bn_out}).LinksTo({act_out});

  return act_out;
}

PDNode *patterns::BatchNormAddAct::operator()(
    paddle::framework::ir::PDNode *bn_x_var,
    std::unordered_set<std::string> act_types) {
  bn_x_var->assert_is_op_input("batch_norm", "X")
      ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm", "Bias");

  auto *bn = pattern->NewNode(batch_norm_repr())
                 ->assert_is_op("batch_norm")
                 ->assert_is_not_op_input("MomentumTensor")
                 ->assert_op_attr<bool>("is_test", false)
                 ->assert_op_attr<bool>("use_global_stats", false)
                 ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *bn_mean_out_var = pattern->NewNode(bn_mean_out_repr())
                              ->assert_is_op_output("batch_norm", "MeanOut");
  auto *bn_variance_out_var =
      pattern->NewNode(bn_variance_out_repr())
          ->assert_is_op_output("batch_norm", "VarianceOut");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_output("batch_norm", "SavedVariance");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_output("batch_norm", "SavedMean");
  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_output("batch_norm", "ReserveSpace");
  auto *bn_out_var = pattern->NewNode(bn_out_repr())
                         ->assert_is_op_output("batch_norm", "Y")
                         ->assert_var_dtype(proto::VarType::FP16);

  bn_out_var->assert_is_op_input("elementwise_add");

  auto *elewise_add =
      pattern->NewNode(elewise_add_repr())->assert_is_op("elementwise_add");

  auto *elewise_add_in_var = pattern->NewNode(elewise_add_in_repr())
                                 ->assert_is_not_ctrl_var()
                                 ->assert_is_op_input("elementwise_add")
                                 ->assert_var_dtype(proto::VarType::FP16);

  auto *elewise_add_out_var =
      pattern->NewNode(elewise_add_out_repr())
          ->assert_is_op_output("elementwise_add", "Out")
          ->assert_has_n_outputs(1);

  elewise_add_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  bn->LinksFrom({bn_x_var, bn_scale_var, bn_bias_var})
      .LinksTo({bn_mean_out_var, bn_variance_out_var, bn_saved_variance_var,
                bn_saved_mean_var, bn_reserve_space, bn_out_var});
  elewise_add->LinksFrom({elewise_add_in_var, bn_out_var})
      .LinksTo({elewise_add_out_var});
  act->LinksFrom({elewise_add_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::BatchNormAddActGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_grad_types) {
  auto *act_grad =
      pattern->NewNode(act_grad_repr())->assert_is_ops(act_grad_types);
  auto *elewise_add_grad = pattern->NewNode(elewise_add_grad_repr())
                               ->assert_is_op("elementwise_add_grad");
  auto *bn_grad = pattern->NewNode(batch_norm_grad_repr())
                      ->assert_is_op("batch_norm_grad")
                      ->assert_op_attr<bool>("use_global_stats", false)
                      ->assert_op_attr<std::string>("data_layout", "NHWC");

  auto *act_out_var = pattern->NewNode(act_out_repr())
                          ->assert_is_ops_input(act_grad_types, "Out");
  auto *d_act_x_var =
      pattern->NewNode(d_act_x_repr())
          ->assert_is_ops_output(act_grad_types, GradVarName("X"))
          ->assert_has_n_outputs(1);  // d_act_x

  d_act_x_var->AsIntermediate()->assert_is_op_input("elementwise_add_grad");

  auto *d_elewise_add_in_var =
      pattern->NewNode(d_elewise_add_in_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad")
          ->assert_var_dtype(proto::VarType::FP16);  // d_add_in_1
  auto *d_bn_out_var =
      pattern->NewNode(d_bn_out_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad")
          ->assert_var_dtype(proto::VarType::FP16);  // d_add_in_2

  d_bn_out_var->assert_is_op_input("batch_norm_grad", GradVarName("Y"));

  auto *bn_x_var = pattern->NewNode(bn_x_repr())
                       ->assert_is_op_input("batch_norm_grad", "X")
                       ->assert_var_dtype(proto::VarType::FP16);
  auto *bn_scale_var = pattern->NewNode(bn_scale_repr())
                           ->assert_is_op_input("batch_norm_grad", "Scale");
  auto *bn_bias_var = pattern->NewNode(bn_bias_repr())
                          ->assert_is_op_input("batch_norm_grad", "Bias");
  auto *bn_saved_mean_var =
      pattern->NewNode(bn_saved_mean_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedMean");
  auto *bn_saved_variance_var =
      pattern->NewNode(bn_saved_variance_repr())
          ->assert_is_op_input("batch_norm_grad", "SavedVariance");

  auto *bn_reserve_space =
      pattern->NewNode(bn_reserve_space_repr())
          ->assert_is_op_input("batch_norm_grad", "ReserveSpace");
  auto *d_bn_x_var =
      pattern->NewNode(d_bn_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("X"))
          ->assert_var_dtype(proto::VarType::FP16);
  auto *d_bn_scale_var =
      pattern->NewNode(d_bn_scale_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Scale"));
  auto *d_bn_bias_var =
      pattern->NewNode(d_bn_bias_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("batch_norm_grad", GradVarName("Bias"));

  act_grad->LinksFrom({d_act_out_var, act_out_var}).LinksTo({d_act_x_var});

  elewise_add_grad->LinksFrom({d_act_x_var})
      .LinksTo({d_elewise_add_in_var, d_bn_out_var});

  bn_grad
      ->LinksFrom({bn_x_var, d_bn_out_var, bn_scale_var, bn_bias_var,
                   bn_saved_mean_var, bn_saved_variance_var, bn_reserve_space})
      .LinksTo({d_bn_x_var, d_bn_scale_var, d_bn_bias_var});

  return bn_grad;
}

PDNode *patterns::ElewiseAddAct::operator()(
    paddle::framework::ir::PDNode *ele_x_var,
    std::unordered_set<std::string> act_types) {
  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_op_input("elementwise_add", "Y");

  auto *ele_add =
      pattern->NewNode(ele_add_repr())->assert_is_op("elementwise_add");

  auto *ele_out_var = pattern->NewNode(elewise_add_out_repr())
                          ->assert_is_op_output("elementwise_add", "Out");

  ele_out_var->AsIntermediate()->assert_is_ops_input(act_types);

  auto *act = pattern->NewNode(act_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_output(act_types, "Out");

  ele_add->LinksFrom({ele_x_var, ele_y_var}).LinksTo({ele_out_var});
  act->LinksFrom({ele_out_var}).LinksTo({act_out_var});

  return act_out_var;
}

PDNode *patterns::ElewiseAddActInplaceGrad::operator()(
    paddle::framework::ir::PDNode *d_act_out_var,
    std::unordered_set<std::string> act_types) {
  // act_grad: in["Out", "Out@GRAD"], out["X@GRAD"]
  // ele_add_grad: in["Y", "Out@GRAD"], out["X@GRAD", "Y@GRAD"]
  auto *act_grad = pattern->NewNode(act_grad_repr())->assert_is_ops(act_types);

  auto *act_out_var =
      pattern->NewNode(act_out_repr())->assert_is_ops_input(act_types, "Out");

  auto *d_intermediate_var =
      pattern->NewNode(d_itermediate_out_repr())
          ->assert_is_ops_output(act_types, GradVarName("X"));

  act_grad->LinksFrom({d_act_out_var, act_out_var})
      .LinksTo({d_intermediate_var});

  auto *ele_y_var = pattern->NewNode(ele_y_repr())
                        ->assert_is_not_ctrl_var()
                        ->assert_is_op_input("elementwise_add_grad", "Y");

  auto *ele_add_grad = pattern->NewNode(ele_add_grad_repr())
                           ->assert_is_op("elementwise_add_grad");

  auto *d_ele_x_var =
      pattern->NewNode(d_ele_x_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("X"));

  auto *d_ele_y_var =
      pattern->NewNode(d_ele_y_repr())
          ->assert_is_not_ctrl_var()
          ->assert_is_op_output("elementwise_add_grad", GradVarName("Y"));

  ele_add_grad->LinksFrom({d_intermediate_var, ele_y_var})
      .LinksTo({d_ele_x_var, d_ele_y_var});

  return ele_add_grad;
}

// conv_type: conv2d, conv3d, conv2d_transpose
PDNode *patterns::ConvBias::operator()(
    paddle::framework::ir::PDNode *conv_input, std::string conv_type) {
  // Create Operators
  conv_input->assert_is_op_input(conv_type, "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op(conv_type);
  auto *eltiwse_op =
      pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  // Create variables
  // Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input(conv_type, "Filter");
  // intermediate variable, will be removed in the IR after fuse.
  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op(conv_type)
                           ->assert_is_op_input("elementwise_add");
  // Bias stored in elementwise_add
  auto *eltwise_bias_var = pattern->NewNode(eltwise_bias_repr())
                               ->AsInput()
                               ->assert_is_persistable_var()
                               ->assert_is_op_input("elementwise_add", "Y");
  // output
  auto *eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                              ->AsOutput()
                              ->assert_is_op_output("elementwise_add");
  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});
  eltiwse_op->LinksFrom({conv_out_var, eltwise_bias_var})
      .LinksTo({eltwise_out_var});
  return eltwise_out_var;
}

PDNode *patterns::Conv::operator()() {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");

  auto input_var = pattern->NewNode(conv_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("conv2d", "Input");

  auto filter_var = pattern->NewNode(conv_filter_repr())
                        ->AsInput()
                        ->assert_is_op_input("conv2d", "Filter");

  auto output_var = pattern->NewNode(conv_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output("conv2d", "Output");

  conv_op->LinksFrom({input_var, filter_var}).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::Transpose::operator()() {
  auto prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();

  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");

  auto transpose_in = pattern->NewNode(transpose_in_repr())
                          ->AsInput()
                          ->assert_is_op_input("transpose2");
  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsOutput()
                           ->assert_is_op_output("transpose2", "Out");

  auto next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  prev_op->LinksTo({transpose_in});
  transpose_op->LinksFrom({transpose_in}).LinksTo({transpose_out});
  next_op->LinksFrom({transpose_out});
  return transpose_out;
}

PDNode *patterns::Reshape::operator()() {
  auto prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();

  auto reshape_op =
      pattern->NewNode(reshape_op_repr())->assert_is_op("reshape2");

  auto reshape_in = pattern->NewNode(reshape_in_repr())
                        ->AsInput()
                        ->assert_is_op_input("reshape2", "X");
  auto reshape_out = pattern->NewNode(reshape_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("reshape2", "Out");

  auto next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  prev_op->LinksTo({reshape_in});
  reshape_op->LinksFrom({reshape_in}).LinksTo({reshape_out});
  next_op->LinksFrom({reshape_out});
  return reshape_out;
}

PDNode *patterns::Matmul::operator()() {
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y = pattern->NewNode(matmul_in_y_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "Y");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::Squeeze2Matmul::operator()() {
  auto squeeze2_in_x = pattern->NewNode(squeeze2_in_x_repr())
                           ->assert_is_op_input("squeeze2", "X")
                           ->AsInput();
  auto squeeze2_op =
      pattern->NewNode(squeeze2_op_repr())->assert_is_op("squeeze2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("squeeze2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  squeeze2_op->LinksFrom({squeeze2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::Reshape2Matmul::operator()() {
  auto reshape2_in_x = pattern->NewNode(reshape2_in_x_repr())
                           ->assert_is_op_input("reshape2", "X")
                           ->AsInput();
  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("reshape2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  reshape2_op->LinksFrom({reshape2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::MatmulWithInputOps::operator()() {
  auto prev_op_x = pattern->NewNode(prev_op_x_repr())->assert_is_op();
  auto prev_op_y = pattern->NewNode(prev_op_y_repr())->assert_is_op();

  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y = pattern->NewNode(matmul_in_y_repr())
                         ->AsInput()
                         ->assert_is_op_input("matmul", "Y");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  prev_op_x->LinksTo({matmul_in_x});
  prev_op_y->LinksTo({matmul_in_y});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::Flatten2Matmul::operator()() {
  auto flatten2_in_x = pattern->NewNode(flatten2_in_x_repr())
                           ->assert_is_op_input("flatten2", "X")
                           ->AsInput();
  auto flatten2_op =
      pattern->NewNode(flatten2_op_repr())->assert_is_op("flatten2");
  auto matmul_in_x = pattern->NewNode(matmul_in_x_repr())
                         ->assert_is_op_output("flatten2", "Out")
                         ->assert_is_op_input("matmul", "X");
  auto matmul_in_y =
      pattern->NewNode(matmul_in_y_repr())->assert_is_op_input("matmul", "Y");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");
  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  flatten2_op->LinksFrom({flatten2_in_x}).LinksTo({matmul_in_x});
  matmul_op->LinksFrom({matmul_in_x, matmul_in_y}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::ConvResidual::operator()(bool with_residual_data) {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");

  if (!with_residual_data) {
    conv_op->assert_more([&](Node *x) {
      auto node_names = x->Op()->InputNames();
      if (!HasInput(x, "ResidualData") ||
          x->Op()->Input("ResidualData").size() == 0)
        return true;
      return false;
    });
  }

  auto input_var = pattern->NewNode(conv_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("conv2d", "Input");

  auto filter_var = pattern->NewNode(conv_filter_repr())
                        ->AsInput()
                        ->assert_is_op_input("conv2d", "Filter");

  auto output_var = pattern->NewNode(conv_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output("conv2d", "Output");

  std::vector<PDNode *> links_from{input_var, filter_var};

  if (with_residual_data) {
    auto res_conn_var = pattern->NewNode(conv_residual_data_repr())
                            ->AsInput()
                            ->assert_is_op_input("conv2d", "ResidualData");
    links_from.push_back(res_conn_var);
  }

  conv_op->LinksFrom(links_from).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::Pool::operator()() {
  auto pool_op = pattern->NewNode(pool_op_repr())->assert_is_op("pool2d");

  auto input_var = pattern->NewNode(pool_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("pool2d", "X");

  auto output_var = pattern->NewNode(pool_output_repr())
                        ->AsOutput()
                        ->assert_is_op_output("pool2d", "Out");

  pool_op->LinksFrom({input_var}).LinksTo({output_var});
  return output_var;
}

PDNode *patterns::ElementwiseAdd::operator()(PDNode *x_var, PDNode *y_var) {
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");

  x_var->AsInput()->assert_is_op_input("elementwise_add", "X");
  y_var->AsInput()->assert_is_op_input("elementwise_add", "Y");
  auto out_var = pattern->NewNode(elementwise_add_out_repr())
                     ->AsOutput()
                     ->assert_is_op_output("elementwise_add", "Out");

  elementwise_add_op->LinksFrom({x_var, y_var});
  elementwise_add_op->LinksTo({out_var});

  return out_var;
}

PDNode *patterns::Concat::operator()() {
  auto concat_op = pattern->NewNode(concat_op_repr())->assert_is_op("concat");

  auto output_var = pattern->NewNode(concat_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("concat", "Out");

  concat_op->LinksTo({output_var});
  return output_var;
}

PDNode *patterns::ConcatReLU::operator()() {
  auto concat_op = pattern->NewNode(concat_op_repr())->assert_is_op("concat");
  auto relu_op = pattern->NewNode(relu_op_repr())->assert_is_op("relu");

  auto concat_out =
      pattern->NewNode(concat_out_repr())->assert_is_op_output("concat", "Out");

  auto relu_out = pattern->NewNode(relu_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("relu", "Out");

  concat_op->LinksTo({concat_out});
  relu_op->LinksFrom({concat_out}).LinksTo({relu_out});

  return relu_out;
}

PDNode *patterns::ConvConcatReLU::operator()() {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto concat_op = pattern->NewNode(concat_op_repr())->assert_is_op("concat");
  auto relu_op = pattern->NewNode(relu_op_repr())->assert_is_op("relu");

  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d", "Output");

  auto concat_out = pattern->NewNode(concat_out_repr())
                        ->assert_is_op_output("concat", "Out")
                        ->assert_is_op_input("relu", "X");

  auto relu_out = pattern->NewNode(relu_out_repr())
                      ->AsOutput()
                      ->assert_is_op_output("relu", "Out");

  conv_op->LinksTo({conv_out});
  concat_op->LinksFrom({conv_out}).LinksTo({concat_out});
  relu_op->LinksFrom({concat_out}).LinksTo({relu_out});

  return relu_out;
}

PDNode *patterns::OpRequant::operator()() {
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return node->Op()->HasAttr("Scale_out") ? true : false;
                    });
  auto requant_in = pattern->NewNode(requant_in_repr())
                        ->assert_is_op_input("requantize", "Input");
  auto requant_op =
      pattern->NewNode(requant_op_repr())->assert_is_op("requantize");
  auto requant_out = pattern->NewNode(requant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("requantize", "Output");

  any_op->LinksTo({requant_in});
  requant_op->LinksFrom({requant_in}).LinksTo({requant_out});
  return requant_out;
}

PDNode *patterns::RequantOp::operator()() {
  auto requant_in = pattern->NewNode(requant_in_repr())
                        ->assert_is_op_input("requantize", "Input");
  auto requant_op =
      pattern->NewNode(requant_op_repr())->assert_is_op("requantize");
  auto requant_out = pattern->NewNode(requant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("requantize", "Output");
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return (node->Op()->HasAttr("Scale_in") ||
                              node->Op()->HasAttr("Scale_x") ||
                              node->Op()->HasAttr("Scale_y"));
                    });

  requant_op->LinksFrom({requant_in}).LinksTo({requant_out});
  any_op->LinksFrom({requant_out});
  return any_op;
}

PDNode *patterns::OpDequant::operator()() {
  auto any_op = pattern->NewNode(any_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      return (node->Op()->HasAttr("force_fp32_output") ||
                              node->Op()->HasProtoAttr("force_fp32_output"));
                    });
  auto dequant_in = pattern->NewNode(dequant_in_repr())
                        ->assert_is_op_input("dequantize", "Input");
  auto dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");
  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize", "Output");

  any_op->LinksTo({dequant_in});
  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  return dequant_out;
}

PDNode *patterns::DequantScale::operator()() {
  // Create Operators
  auto dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");

  auto dequant_out = pattern->NewNode(dequant_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("dequantize", "Output");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");

  dequant_op->LinksTo({dequant_out});
  scale_op->LinksFrom({dequant_out}).LinksTo({scale_out});

  return scale_out;
}

PDNode *patterns::ScaleQuant::operator()() {
  auto scale_in = pattern->NewNode(scale_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("scale", "X");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");

  auto quant_in = pattern->NewNode(quant_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("quantize", "Input");
  auto quant_op = pattern->NewNode(quant_op_repr())->assert_is_op("quantize");

  scale_op->LinksFrom({scale_in}).LinksTo({quant_in});
  quant_op->LinksFrom({quant_in});

  return quant_op;
}

PDNode *patterns::QuantConv::operator()() {
  auto quant_in = pattern->NewNode(quant_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("quantize", "Input");
  auto quant_op = pattern->NewNode(quant_op_repr())->assert_is_op("quantize");

  auto conv_in = pattern->NewNode(conv_in_repr())
                     ->AsInput()
                     ->assert_is_op_input("conv2d", "Input");
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  conv_op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });

  quant_op->LinksFrom({quant_in}).LinksTo({conv_in});
  conv_op->LinksFrom({conv_in});

  return quant_op;
}

PDNode *patterns::ScaleMatmul::operator()() {
  auto scale_in = pattern->NewNode(scale_in_repr())
                      ->AsInput()
                      ->assert_is_op_input("scale", "X");
  auto scale_op = pattern->NewNode(scale_op_repr())->assert_is_op("scale");
  auto scale_out = pattern->NewNode(scale_out_repr())
                       ->AsOutput()
                       ->assert_is_op_output("scale", "Out");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  scale_op->LinksFrom({scale_in}).LinksTo({scale_out});
  matmul_op->LinksFrom({scale_out});
  return matmul_op;
}

PDNode *patterns::PriorBox::operator()() {
  auto prior_box_op =
      pattern->NewNode(prior_box_op_repr())->assert_is_op("prior_box");

  auto input_var = pattern->NewNode(prior_box_input_repr())
                       ->AsInput()
                       ->assert_is_op_input("prior_box", "Input");

  auto image_var = pattern->NewNode(prior_box_image_repr())
                       ->AsInput()
                       ->assert_is_op_input("prior_box", "Image");

  auto boxes_var = pattern->NewNode(prior_box_boxes_repr())
                       ->AsOutput()
                       ->assert_is_op_output("prior_box", "Boxes");

  auto variances_var = pattern->NewNode(prior_box_variances_repr())
                           ->AsOutput()
                           ->assert_is_op_output("prior_box", "Variances");

  prior_box_op->LinksFrom({input_var, image_var})
      .LinksTo({boxes_var, variances_var});
  return boxes_var;
}

std::unordered_set<std::string> conv_act_set({"identity", "relu"});

PDNode *patterns::ConvElementwiseaddAct::operator()(PDNode *conv_in) {
  conv_in->AsInput();
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->AsIntermediate();

  auto act_op = pattern->NewNode(act_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      auto op_type = node->Name();
                      return conv_act_set.count(op_type);
                    });

  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_var()
                     // is activation op's output.
                     ->assert_more([&](Node *node) {
                       for (auto *in_op : node->inputs) {
                         if (conv_act_set.count(in_op->Name())) {
                           return true;
                         }
                       }
                       return false;
                     })
                     ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter});
  conv_out->LinksFrom({conv_op});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});
  act_op->LinksFrom({elementwise_add_out}).LinksTo({act_out});

  return act_out;
}

PDNode *patterns::ConvElementwiseadd2Act::operator()(PDNode *conv_in) {
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->assert_is_op_input("elementwise_add", "Y")
                                 ->AsIntermediate();

  auto elementwise_add_op_1 = pattern->NewNode(elementwise_add_op_1_repr())
                                  ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y_1 = pattern->NewNode(elementwise_add_in_y_1_repr())
                                    ->assert_is_op_input("elementwise_add", "X")
                                    ->AsInput();
  auto elementwise_add_out_1 = pattern->NewNode(elementwise_add_out_1_repr())
                                   ->assert_is_op_output("elementwise_add")
                                   ->AsIntermediate();

  auto act_op = pattern->NewNode(act_op_repr())
                    ->assert_is_op()
                    ->assert_more([&](Node *node) {
                      auto op_type = node->Name();
                      return conv_act_set.count(op_type);
                    });
  auto act_out = pattern->NewNode(act_out_repr())
                     ->assert_is_var()
                     // is activation op's output.
                     ->assert_more([&](Node *node) {
                       for (auto *in_op : node->inputs) {
                         if (conv_act_set.count(in_op->Name())) {
                           return true;
                         }
                       }
                       return false;
                     })
                     ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter}).LinksTo({conv_out});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});
  elementwise_add_op_1->LinksFrom({elementwise_add_out, elementwise_add_in_y_1})
      .LinksTo({elementwise_add_out_1});
  act_op->LinksFrom({elementwise_add_out_1}).LinksTo({act_out});
  return act_out;
}

PDNode *patterns::ConvElementwiseadd::operator()(PDNode *conv_in) {
  conv_in->AsInput();
  auto conv_op = pattern->NewNode(conv_op_repr())->assert_is_op("conv2d");
  auto conv_out = pattern->NewNode(conv_out_repr())
                      ->assert_is_op_output("conv2d")
                      ->assert_is_op_input("elementwise_add", "X")
                      ->AsIntermediate();
  auto conv_filter = pattern->NewNode(conv_filter_repr())
                         ->assert_is_op_input("conv2d", "Filter")
                         ->AsInput();
  auto elementwise_add_op = pattern->NewNode(elementwise_add_op_repr())
                                ->assert_is_op("elementwise_add");
  auto elementwise_add_in_y = pattern->NewNode(elementwise_add_in_y_repr())
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("elementwise_add", "Y")
                                  ->AsInput();
  auto elementwise_add_out = pattern->NewNode(elementwise_add_out_repr())
                                 ->assert_is_op_output("elementwise_add")
                                 ->AsOutput();

  conv_op->LinksFrom({conv_in, conv_filter});
  conv_out->LinksFrom({conv_op});
  elementwise_add_op->LinksFrom({conv_out, elementwise_add_in_y})
      .LinksTo({elementwise_add_out});

  return elementwise_add_out;
}

PDNode *patterns::ConvAffineChannel::operator()(
    paddle::framework::ir::PDNode *conv_input, bool with_eltwise_add) {
  // Create Operators
  conv_input->assert_is_op_input("conv2d", "Input");
  auto *conv_op = pattern->NewNode(conv_repr())->assert_is_op("conv2d");

  PDNode *eltwise_op = nullptr;
  if (with_eltwise_add) {
    eltwise_op =
        pattern->NewNode(eltwise_repr())->assert_is_op("elementwise_add");
  }

  auto *affine_channel_op =
      pattern->NewNode(affine_channel_repr())->assert_is_op("affine_channel");
  // Create variables
  // Conv Filter
  auto *conv_weight_var = pattern->NewNode(conv_weight_repr())
                              ->AsInput()
                              ->assert_is_persistable_var()
                              ->assert_is_op_input("conv2d", "Filter");

  auto *conv_out_var = pattern->NewNode(conv_out_repr())
                           ->AsIntermediate()
                           ->assert_is_only_output_of_op("conv2d");

  PDNode *eltwise_y_in_var = nullptr;
  PDNode *eltwise_out_var = nullptr;
  if (with_eltwise_add) {
    // Conv output as Bias input
    conv_out_var->assert_is_op_input("elementwise_add", "X");
    // Bias
    eltwise_y_in_var = pattern->NewNode(eltwise_y_in_repr())
                           ->assert_is_op_input("elementwise_add", "Y")
                           ->AsInput();
    eltwise_out_var = pattern->NewNode(eltwise_out_repr())
                          ->AsIntermediate()
                          ->assert_is_only_output_of_op("elementwise_add");
  } else {
    // Conv output as AffineChannel input
    conv_out_var->assert_is_op_input("affine_channel", "X");
  }

  // AC Scale
  auto *ac_scale_var = pattern->NewNode(ac_scale_repr())
                           ->AsInput()
                           ->assert_is_persistable_var()
                           ->assert_has_n_outputs(1)
                           ->assert_is_op_input("affine_channel", "Scale");
  // AC Bias
  auto *ac_bias_var = pattern->NewNode(ac_bias_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_has_n_outputs(1)
                          ->assert_is_op_input("affine_channel", "Bias");

  // AC output
  auto *ac_out_var = pattern->NewNode(ac_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("affine_channel");

  conv_op->LinksFrom({conv_input, conv_weight_var}).LinksTo({conv_out_var});

  if (with_eltwise_add) {
    eltwise_op->LinksFrom({conv_out_var, eltwise_y_in_var})
        .LinksTo({eltwise_out_var});
    affine_channel_op->LinksFrom({eltwise_out_var, ac_scale_var, ac_bias_var})
        .LinksTo({ac_out_var});
  } else {
    affine_channel_op->LinksFrom({conv_out_var, ac_scale_var, ac_bias_var})
        .LinksTo({ac_out_var});
  }
  return ac_out_var;
}

PDNode *patterns::DequantQuantAny::operator()() {
  auto *dequant_in = pattern->NewNode(dequant_in_repr())
                         ->AsInput()
                         ->assert_is_op_input("dequantize", "Input");

  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize", "Output");

  auto *quant_op = pattern->NewNode(quant_op_repr())
                       ->assert_is_op("quantize")
                       ->AsIntermediate();

  auto *quant_out = pattern->NewNode(quant_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("quantize");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksFrom({dequant_in}).LinksTo({dequant_out});
  quant_op->LinksFrom({dequant_out}).LinksTo({quant_out});
  next_op->LinksFrom({quant_out});

  return quant_out;
}

PDNode *patterns::DequantAny::operator()() {
  auto *dequant_op =
      pattern->NewNode(dequant_op_repr())->assert_is_op("dequantize");

  auto *dequant_out = pattern->NewNode(dequant_out_repr())
                          ->AsOutput()
                          ->assert_is_op_output("dequantize", "Output");

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();

  dequant_op->LinksTo({dequant_out});
  next_op->LinksFrom({dequant_out});

  return dequant_out;
}

PDNode *patterns::MultipleQuantize::operator()() {
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  // find nodes that are inputs to quantize operators
  prev_out->assert_more([&](Node *node) {
    int counter = std::count_if(
        node->outputs.begin(), node->outputs.end(), [&](Node const *iter) {
          return iter && iter->IsOp() && iter->Op()->Type() == "quantize";
        });
    return (counter > 1);
  });

  return prev_out;
}

PDNode *patterns::QuantizePlacement::operator()(
    const std::unordered_set<std::string> &quantize_enabled_op_types) {
  std::unordered_set<std::string> supported_op_types =
      std::unordered_set<std::string>(
          {"concat", "conv2d", "elementwise_add", "fc", "matmul", "pool2d",
           "prior_box", "relu", "reshape2", "transpose2", "fusion_gru"});
  if (!quantize_enabled_op_types.empty()) {
    supported_op_types = quantize_enabled_op_types;
  }
  auto *op = pattern->NewNode(op_repr())->assert_is_ops(supported_op_types);
  return op;
}

PDNode *patterns::Bfloat16Placement::operator()(
    const std::unordered_set<std::string> &bfloat16_enabled_op_types) {
  std::unordered_set<std::string> supported_op_types =
      std::unordered_set<std::string>(
          {"concat",          "conv2d",          "conv2d_transpose",
           "elementwise_add", "elementwise_mul", "fc",
           "fusion_gru",      "fusion_lstm",     "gelu",
           "layer_norm",      "matmul",          "matmul_v2",
           "pool2d",          "prelu",           "relu",
           "reshape2",        "softmax",         "split",
           "squeeze",         "squeeze2",        "sum",
           "transpose2"});
  if (!bfloat16_enabled_op_types.empty()) {
    supported_op_types = bfloat16_enabled_op_types;
  }
  auto *op = pattern->NewNode(op_repr())->assert_is_ops(supported_op_types);
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<bool>("use_mkldnn") ||
           node->Op()->Type() == "reshape2";
  });
  return op;
}

PDNode *patterns::OrphanedBfloat16::operator()() {
  auto *prev_op = pattern->NewNode(prev_op_repr())->assert_is_op();
  prev_op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "float32";
  });
  auto *prev_out = pattern->NewNode(prev_out_repr())->AsOutput();

  auto *op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  auto *op_out = pattern->NewNode(op_out_repr())->AsOutput();

  auto *next_op = pattern->NewNode(next_op_repr())->assert_is_op();
  next_op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "float32";
  });

  prev_op->LinksTo({prev_out});
  op->LinksFrom({prev_out}).LinksTo({op_out});
  next_op->LinksFrom({op_out});
  return next_op;
}

PDNode *patterns::LastBfloat16Ops::operator()() {
  auto *op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  auto *op_out = pattern->NewNode(op_out_repr())->AsOutput();
  op->LinksTo({op_out});
  return op_out;
}

PDNode *patterns::FirstBfloat16Ops::operator()() {
  auto *op_in = pattern->NewNode(op_in_repr())->AsInput();

  auto *op = pattern->NewNode(op_repr())->assert_is_op();
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });

  op->LinksFrom({op_in});
  return op;
}

PDNode *patterns::DuplicatedInputs::operator()() {
  auto op = pattern->NewNode(op_repr())->assert_is_ops({"concat", "sum"});
  op->assert_more([&](Node *node) {
    return node->Op()->GetAttrIfExists<std::string>("mkldnn_data_type") ==
           "bfloat16";
  });
  return op;
}

PDNode *patterns::MKLDNNInPlace::operator()() {
  const std::unordered_set<std::string> &supported_op_types = {
      "abs", "gelu", "leaky_relu", "relu", "softmax", "sqrt", "swish", "tanh"};

  auto possible_inplace_op = pattern->NewNode(inplace_to_be_op_repr())
                                 ->assert_is_ops(supported_op_types);

  auto input = pattern->NewNode(inplace_to_be_op_in_repr())
                   ->assert_is_ops_input(supported_op_types)
                   ->AsInput();
  auto output = pattern->NewNode(inplace_to_be_op_out_repr())
                    ->assert_is_ops_output(supported_op_types)
                    ->AsOutput();

  auto next_op = pattern->NewNode(next_op_repr())->assert_is_op();
  auto next_output = pattern->NewNode(next_op_out_repr())->AsOutput();

  // Check if op is MKL-DNN enabled
  possible_inplace_op->assert_op_attr("use_mkldnn", true);

  // linked structure
  possible_inplace_op->LinksTo({output});
  possible_inplace_op->LinksFrom({input});
  next_op->LinksFrom({output});
  next_op->LinksTo({next_output});

  return possible_inplace_op;
}

// a -> transpose_op(1) -> transpose_out_a -> flatten_op(1) -> flatten_out_a
// b -> transpose_op(2) -> transpose_out_b -> flatten_op(2) -> flatten_out_b
// ...
// z -> transpose_op(n) -> transpose_out_z -> flatten_op(n) -> flatten_out_z
// flatten_out_a -> concat_op  flatten_out_b -> concat_op ... flatten_out_z ->
// concat_op
PDNode *patterns::TransposeFlattenConcat::operator()(
    std::vector<PDNode *> conv_in, int times) {
  // The times represents the repeat times of the
  // {trans, trans_out, flatten, flatten_out}
  const int kNumFields = 4;
  const int kTransOutOffset = 1;
  const int kFlattenOffset = 2;
  const int kFlattenOutOffset = 3;

  std::vector<PDNode *> nodes;

  for (int i = 0; i < times; i++) {
    nodes.push_back(
        pattern->NewNode(GetNodeName("transpose" + std::to_string(i)))
            ->assert_is_op("transpose2"));
    nodes.push_back(
        pattern->NewNode(GetNodeName("transpose_out" + std::to_string(i)))
            ->assert_is_op_output("transpose2")
            ->assert_is_op_input("flatten2", "X")
            ->AsIntermediate());
    nodes.push_back(pattern->NewNode(GetNodeName("flatten" + std::to_string(i)))
                        ->assert_is_op("flatten2"));

    nodes.push_back(
        pattern->NewNode(GetNodeName("flatten_out" + std::to_string(i)))
            ->assert_is_op_output("flatten2")
            ->assert_is_op_nth_input("concat", "X", i)
            ->AsIntermediate());
  }

  auto concat_op = pattern->NewNode(GetNodeName("concat"))
                       ->assert_is_op("concat")
                       ->assert_op_has_n_inputs("concat", times);
  auto concat_out = pattern->NewNode(GetNodeName("concat_out"))
                        ->assert_is_op_output("concat")
                        ->AsOutput();

  std::vector<PDNode *> flatten_outs;
  for (int i = 0; i < times; i++) {
    conv_in[i]->AsInput();
    // trans
    nodes[i * kNumFields]->LinksFrom({conv_in[i]});
    // trans_out
    nodes[i * kNumFields + kTransOutOffset]->LinksFrom({nodes[i * kNumFields]});
    // flatten
    nodes[i * kNumFields + kFlattenOffset]->LinksFrom(
        {nodes[i * kNumFields + kTransOutOffset]});
    // flatten_out
    nodes[i * kNumFields + kFlattenOutOffset]->LinksFrom(
        {nodes[i * kNumFields + kFlattenOffset]});
    flatten_outs.push_back(nodes[i * kNumFields + kFlattenOutOffset]);
  }

  concat_op->LinksFrom(flatten_outs).LinksTo({concat_out});
  return concat_out;
}

void patterns::DeleteDropoutOpPattern::operator()() {
  auto any_op_out = pattern->NewNode(any_op_out_repr())
                        ->assert_is_op_input("dropout", "X")
                        ->AsInput();

  auto dropout_op =
      pattern->NewNode(dropout_op_repr())->assert_is_op("dropout");

  auto dropout_op_out = pattern->NewNode(dropout_op_out_repr())
                            ->assert_is_op_output("dropout", "Out")
                            ->AsIntermediate();

  auto dropout_op_outmask = pattern->NewNode(dropout_op_outmask_repr())
                                ->assert_is_op_output("dropout", "Mask")
                                ->AsOutput();
  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  dropout_op->LinksFrom({any_op_out});
  dropout_op_out->LinksFrom({dropout_op});
  dropout_op_outmask->LinksFrom({dropout_op});
  any_op2->LinksFrom({dropout_op_out});
}

void patterns::DeleteQuantOpFuse::operator()(PDNode *input_act_node,
                                             const std::string &quant_type) {
  auto *input_scale_node = pattern->NewNode(GetNodeName("input_scale_node"))
                               ->assert_is_op_input(quant_type, "InScale")
                               ->AsInput();
  auto *quant_node =
      pattern->NewNode(GetNodeName("quant_node"))->assert_is_op(quant_type);
  auto *output_scale_node = pattern->NewNode(GetNodeName("output_scale_node"))
                                ->assert_is_op_output(quant_type, "OutScale")
                                ->AsOutput();
  auto *output_act_node = pattern->NewNode(GetNodeName("output_act_node"))
                              ->assert_is_op_output(quant_type, "Out")
                              ->AsOutput();
  quant_node->LinksFrom({input_scale_node, input_act_node});
  output_scale_node->LinksFrom({quant_node});
  output_act_node->LinksFrom({quant_node});
}

void patterns::DequantOpFuse::operator()(PDNode *quantized_op_input,
                                         const std::string &quantized_op_type,
                                         const std::string &dequant_type,
                                         const std::string &weight_name) {
  auto *quantized_op_weight =
      pattern->NewNode(GetNodeName("quantized_op_weight"))
          ->assert_is_op_input(quantized_op_type, weight_name)
          ->AsInput();
  auto *quantized_op = pattern->NewNode(GetNodeName("quantized_op"))
                           ->assert_is_op(quantized_op_type);
  auto *quantized_op_out = pattern->NewNode(GetNodeName("quantized_op_out"))
                               ->assert_is_op_output(quantized_op_type)
                               ->assert_is_op_input(dequant_type, "X");
  auto *dequant_op =
      pattern->NewNode(GetNodeName("dequant_op"))->assert_is_op(dequant_type);
  auto *dequant_op_out = pattern->NewNode(GetNodeName("dequant_op_out"))
                             ->assert_is_op_output(dequant_type, "Out")
                             ->AsOutput();
  PDNode *dequant_channel_scale = nullptr;
  if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
    dequant_channel_scale =
        pattern->NewNode(GetNodeName("dequant_channel_scale"))
            ->assert_is_op_nth_input(dequant_type, "Scales", 0)
            ->AsInput();
  }
  quantized_op->LinksFrom({quantized_op_input, quantized_op_weight});
  quantized_op_out->LinksFrom({quantized_op});

  if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
    dequant_op->LinksFrom({quantized_op_out, dequant_channel_scale});
  } else {
    dequant_op->LinksFrom({quantized_op_out});
  }
  dequant_op_out->LinksFrom({dequant_op});
}

void patterns::ShuffleChannelPattern::operator()(PDNode *reshape1_in) {
  auto reshape1_op =
      pattern->NewNode(reshape1_op_repr())->assert_is_op("reshape2");
  reshape1_op->assert_more([&](Node *x) {
    return BOOST_GET_CONST(std::vector<int>, x->Op()->GetAttr("shape"))
               .size() == 5;
  });

  auto reshape1_out = pattern->NewNode(reshape1_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->assert_is_op_input("transpose2")
                          ->AsIntermediate();

  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->assert_is_op_output("transpose2", "Out")
                           ->assert_is_op_input("reshape2")
                           ->AsIntermediate();

  auto reshape2_op =
      pattern->NewNode(reshape2_op_repr())->assert_is_op("reshape2");
  auto reshape2_out = pattern->NewNode(reshape2_out_repr())
                          ->assert_is_op_output("reshape2", "Out")
                          ->AsOutput();

  reshape1_op->LinksFrom({reshape1_in});
  reshape1_out->LinksFrom({reshape1_op});
  transpose_op->LinksFrom({reshape1_out});
  transpose_out->LinksFrom({transpose_op});
  reshape2_op->LinksFrom({transpose_out});
  reshape2_out->LinksFrom({reshape2_op});
}

void patterns::DeleteQuantDequantOpPattern::operator()() {
  auto any_op_out =
      pattern->NewNode(any_op_out_repr())
          ->assert_is_op_input(
              "fake_quantize_dequantize_moving_average_abs_max", "X")
          ->AsInput();

  auto quant_dequant_op_inscale =
      pattern->NewNode(quant_dequant_op_inscale_repr())
          ->assert_is_op_input(
              "fake_quantize_dequantize_moving_average_abs_max", "InScale")
          ->AsInput();
  auto quant_dequant_op =
      pattern->NewNode(quant_dequant_op_repr())
          ->assert_is_op("fake_quantize_dequantize_moving_average_abs_max");

  auto quant_dequant_out =
      pattern->NewNode(quant_dequant_op_out_repr())
          ->assert_is_op_output(
              "fake_quantize_dequantize_moving_average_abs_max", "Out")
          ->AsIntermediate();

  auto quant_dequant_op_outscale =
      pattern->NewNode(quant_dequant_op_outscale_repr())
          ->assert_is_op_output(
              "fake_quantize_dequantize_moving_average_abs_max", "OutScale")
          ->AsOutput();
  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  quant_dequant_op->LinksFrom({any_op_out, quant_dequant_op_inscale});
  quant_dequant_op_outscale->LinksFrom({quant_dequant_op});
  quant_dequant_out->LinksFrom({quant_dequant_op});
  any_op2->LinksFrom({quant_dequant_out});
}

void patterns::DeleteQuantDequantFilterOpPattern::operator()() {
  auto quant_dequant_op_x =
      pattern->NewNode(quant_dequant_op_x_repr())
          ->assert_is_ops_input(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "X")
          ->AsInput();

  auto quant_dequant_op =
      pattern->NewNode(quant_dequant_op_repr())
          ->assert_is_ops({"fake_channel_wise_quantize_dequantize_abs_max",
                           "fake_quantize_dequantize_abs_max"});

  auto quant_dequant_out =
      pattern->NewNode(quant_dequant_op_out_repr())
          ->assert_is_ops_output(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "Out")
          ->AsIntermediate();

  auto quant_dequant_op_outscale =
      pattern->NewNode(quant_dequant_op_outscale_repr())
          ->assert_is_ops_output(
              {"fake_channel_wise_quantize_dequantize_abs_max",
               "fake_quantize_dequantize_abs_max"},
              "OutScale")
          ->AsOutput();
  auto any_op2 = pattern->NewNode(any_op2_repr())->assert_is_op()->AsOutput();

  quant_dequant_op->LinksFrom({quant_dequant_op_x});
  quant_dequant_op_outscale->LinksFrom({quant_dequant_op});
  quant_dequant_out->LinksFrom({quant_dequant_op});
  any_op2->LinksFrom({quant_dequant_out});
}

PDNode *patterns::ReshapeTransposeMatmulPattern::operator()(
    bool with_reshape_xshape, bool with_transpose_xshape) {
  auto reshape_op =
      pattern->NewNode(reshape_op_repr())->assert_is_op("reshape2");
  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  auto reshape_in = pattern->NewNode(reshape_in_repr())
                        ->AsInput()
                        ->assert_is_op_input("reshape2", "X");

  auto reshape_out = pattern->NewNode(reshape_out_repr())
                         ->AsIntermediate()
                         ->assert_is_op_input("transpose2", "X")
                         ->assert_is_op_output("reshape2", "Out");
  if (!with_reshape_xshape)
    reshape_out->assert_is_only_output_of_op("reshape2");

  auto reshape_xshape = with_reshape_xshape
                            ? pattern->NewNode(reshape_xshape_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("reshape2", "XShape")
                            : nullptr;

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsIntermediate()
                           ->assert_is_op_input("matmul")
                           ->assert_is_op_output("transpose2", "Out");
  if (!with_transpose_xshape)
    transpose_out->assert_is_only_output_of_op("transpose2");

  auto transpose_xshape =
      with_transpose_xshape
          ? pattern->NewNode(transpose_xshape_repr())
                ->AsIntermediate()
                ->assert_is_op_output("transpose2", "XShape")
          : nullptr;

  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsOutput()
                        ->assert_is_op_output("matmul", "Out");

  reshape_op->LinksFrom({reshape_in}).LinksTo({reshape_out});
  if (with_reshape_xshape) reshape_op->LinksTo({reshape_xshape});
  transpose_op->LinksFrom({reshape_out}).LinksTo({transpose_out});
  if (with_transpose_xshape) transpose_op->LinksTo({transpose_xshape});
  matmul_op->LinksFrom({transpose_out}).LinksTo({matmul_out});
  return matmul_out;
}

PDNode *patterns::MatmulTransposeReshapePattern::operator()() {
  auto reshape_op =
      pattern->NewNode(reshape_op_repr())->assert_is_op("reshape2");
  auto transpose_op =
      pattern->NewNode(transpose_op_repr())->assert_is_op("transpose2");
  auto matmul_op = pattern->NewNode(matmul_op_repr())->assert_is_op("matmul");

  auto matmul_out = pattern->NewNode(matmul_out_repr())
                        ->AsInput()
                        ->assert_is_op_output("matmul", "Out")
                        ->assert_is_op_input("transpose2", "X");

  auto transpose_out = pattern->NewNode(transpose_out_repr())
                           ->AsIntermediate()
                           ->assert_is_op_output("transpose2", "Out")
                           ->assert_is_op_input("reshape2", "X");

  auto transpose_out_xshape = pattern->NewNode(transpose_out_xshape_repr())
                                  ->AsIntermediate()
                                  ->assert_is_op_output("transpose2", "XShape");

  auto reshape_out = pattern->NewNode(reshape_out_repr())
                         ->AsOutput()
                         ->assert_is_op_output("reshape2");

  auto reshape_out_xshape = pattern->NewNode(reshape_out_xshape_repr())
                                ->AsIntermediate()
                                ->assert_is_op_output("reshape2", "XShape");

  matmul_op->LinksTo({matmul_out});
  transpose_op->LinksTo({transpose_out_xshape});
  reshape_op->LinksTo({reshape_out_xshape});
  transpose_op->LinksFrom({matmul_out}).LinksTo({transpose_out});
  reshape_op->LinksFrom({transpose_out}).LinksTo({reshape_out});
  return reshape_out;
}

PDNode *patterns::FusionGru::operator()() {
  auto op = pattern->NewNode(op_repr())->assert_is_op("fusion_gru");
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "X");
  auto weight_h = pattern->NewNode(weight_h_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_gru", "WeightH");
  auto weight_x = pattern->NewNode(weight_x_repr())
                      ->AsInput()
                      ->assert_is_op_input("fusion_gru", "WeightX");
  auto out = pattern->NewNode(out_repr())
                 ->AsOutput()
                 ->assert_is_op_output("fusion_gru", "Hidden");
  op->LinksFrom({x, weight_h, weight_x}).LinksTo({out});
  return out;
}

PDNode *patterns::TwoFusionGruConcat::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "X");
  auto gru1 =
      pattern->NewNode(gru1_repr())
          ->assert_is_op("fusion_gru")
          ->assert_more([&](Node *node) {
            return node->Op()->GetAttrIfExists<bool>("is_reverse") == false;
          });
  auto gru2 =
      pattern->NewNode(gru2_repr())
          ->assert_is_op("fusion_gru")
          ->assert_more([&](Node *node) {
            return node->Op()->GetAttrIfExists<bool>("is_reverse") == true;
          });
  auto wh1 = pattern->NewNode(wh1_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightH");
  auto wh2 = pattern->NewNode(wh2_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightH");
  auto wx1 = pattern->NewNode(wx1_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightX");
  auto wx2 = pattern->NewNode(wx2_repr())
                 ->AsInput()
                 ->assert_is_op_input("fusion_gru", "WeightX");
  auto b1 = pattern->NewNode(b1_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "Bias");
  auto b2 = pattern->NewNode(b2_repr())->AsInput()->assert_is_op_input(
      "fusion_gru", "Bias");
  auto h1 = pattern->NewNode(h1_repr())
                ->AsOutput()
                ->assert_is_op_output("fusion_gru", "Hidden")
                ->assert_is_op_input("concat")
                ->AsIntermediate();
  auto h2 = pattern->NewNode(h2_repr())
                ->AsOutput()
                ->assert_is_op_output("fusion_gru", "Hidden")
                ->assert_is_op_input("concat")
                ->AsIntermediate();
  auto concat = pattern->NewNode(concat_repr())->assert_is_op("concat");
  auto out = pattern->NewNode(out_repr())
                 ->AsOutput()
                 ->assert_is_op_output("concat", "Out");
  gru1->LinksFrom({x, wh1, wx1, b1}).LinksTo({h1});
  gru2->LinksFrom({x, wh2, wx2, b2}).LinksTo({h2});
  concat->LinksFrom({h1, h2}).LinksTo({out});
  return out;
}

PDNode *patterns::MultiGruSeq::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "multi_gru", "X");
  auto gru1 = pattern->NewNode(gru1_repr())->assert_is_op("multi_gru");
  auto wx11 = pattern->NewNode(wx11_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 0);
  auto wx12 = pattern->NewNode(wx12_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 1);
  auto wh11 = pattern->NewNode(wh11_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 0);
  auto wh12 = pattern->NewNode(wh12_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 1);
  auto b11 = pattern->NewNode(b11_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 0);
  auto b12 = pattern->NewNode(b12_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 1);
  auto h1 = pattern->NewNode(h1_repr())
                ->AsOutput()
                ->assert_is_op_output("multi_gru", "Hidden")
                ->assert_is_op_input("multi_gru", "X")
                ->AsIntermediate();
  auto gru2 = pattern->NewNode(gru2_repr())->assert_is_op("multi_gru");
  auto wx21 = pattern->NewNode(wx21_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 0);
  auto wx22 = pattern->NewNode(wx22_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightX", 1);
  auto wh21 = pattern->NewNode(wh21_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 0);
  auto wh22 = pattern->NewNode(wh22_repr())
                  ->AsInput()
                  ->assert_is_op_nth_input("multi_gru", "WeightH", 1);
  auto b21 = pattern->NewNode(b21_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 0);
  auto b22 = pattern->NewNode(b22_repr())
                 ->AsInput()
                 ->assert_is_op_nth_input("multi_gru", "Bias", 1);
  auto h2 = pattern->NewNode(h2_repr())->AsOutput()->assert_is_op_output(
      "multi_gru", "Hidden");
  gru1->LinksFrom({x, wx11, wx12, wh11, wh12, b11, b12}).LinksTo({h1});
  gru2->LinksFrom({h1, wx21, wx22, wh21, wh22, b21, b22}).LinksTo({h2});
  return h2;
}

PDNode *patterns::MultiGru::operator()() {
  auto x = pattern->NewNode(x_repr())->AsInput()->assert_is_op_input(
      "multi_gru", "X");
  auto gru = pattern->NewNode(gru_repr())->assert_is_op("multi_gru");
  auto wx = pattern->NewNode(wx_repr())->AsInput()->assert_is_op_nth_input(
      "multi_gru", "WeightX", 0);
  auto wh = pattern->NewNode(wh_repr())->AsInput()->assert_is_op_nth_input(
      "multi_gru", "WeightH", 0);
  auto h = pattern->NewNode(h_repr())->AsOutput()->assert_is_op_output(
      "multi_gru", "Hidden");
  gru->LinksFrom({x, wx, wh}).LinksTo({h});
  return h;
}

PDNode *patterns::LayerNorm::operator()() {
  auto *x = pattern->NewNode(x_repr())->AsInput()->assert_is_ops_input(
      {"reduce_mean", "elementwise_sub"});
  auto *x_mean = pattern->NewNode(x_mean_repr())->assert_is_op("reduce_mean");
  auto *x_mean_out = pattern->NewNode(x_mean_out_repr())
                         ->assert_is_op_output("reduce_mean", "Out")
                         ->assert_is_op_input("elementwise_sub", "Y")
                         ->AsIntermediate();
  auto *x_sub_mean =
      pattern->NewNode(x_sub_mean_repr())->assert_is_op("elementwise_sub");
  auto *x_sub_mean_out =
      pattern->NewNode(x_sub_mean_out_repr())
          ->assert_is_op_output("elementwise_sub")
          ->assert_is_ops_input({"elementwise_pow", "elementwise_div"}, "X")
          ->AsIntermediate();
  auto *sqr_pow = pattern->NewNode(sqr_pow_repr())
                      ->assert_is_op_input("elementwise_pow", "Y")
                      ->assert_is_persistable_var()
                      ->AsInput();
  auto *x_sub_mean_sqr =
      pattern->NewNode(x_sub_mean_sqr_repr())->assert_is_op("elementwise_pow");
  auto *x_sub_mean_sqr_out = pattern->NewNode(x_sub_mean_sqr_out_repr())
                                 ->assert_is_op_output("elementwise_pow")
                                 ->assert_is_op_input("reduce_mean")
                                 ->AsIntermediate();
  auto *std_dev = pattern->NewNode(std_dev_repr())->assert_is_op("reduce_mean");
  auto *std_dev_out = pattern->NewNode(std_dev_out_repr())
                          ->assert_is_op_output("reduce_mean")
                          ->assert_is_op_input("elementwise_add")
                          ->AsIntermediate();
  auto *eps = pattern->NewNode(eps_repr())
                  ->assert_is_op_input("elementwise_add", "Y")
                  ->assert_is_persistable_var()
                  ->AsInput();
  auto *std_dev_eps =
      pattern->NewNode(std_dev_eps_repr())->assert_is_op("elementwise_add");
  auto *std_dev_eps_out = pattern->NewNode(std_dev_eps_out_repr())
                              ->assert_is_op_output("elementwise_add")
                              ->assert_is_op_input("sqrt")
                              ->AsIntermediate();
  auto *std_dev_eps_sqrt =
      pattern->NewNode(std_dev_eps_sqrt_repr())->assert_is_op("sqrt");
  auto *std_dev_eps_sqrt_out = pattern->NewNode(std_dev_eps_sqrt_out_repr())
                                   ->assert_is_op_output("sqrt")
                                   ->assert_is_op_input("elementwise_div", "Y")
                                   ->AsIntermediate();
  auto *division =
      pattern->NewNode(division_repr())->assert_is_op("elementwise_div");
  auto *division_out = pattern->NewNode(division_out_repr())
                           ->assert_is_op_output("elementwise_div")
                           ->assert_is_op_input("elementwise_mul")
                           ->AsIntermediate();
  auto *gamma = pattern->NewNode(gamma_repr())
                    ->assert_is_op_input("elementwise_mul", "Y")
                    ->assert_is_persistable_var()
                    ->AsInput();
  auto *scale = pattern->NewNode(scale_repr())->assert_is_op("elementwise_mul");
  auto *scale_out = pattern->NewNode(scale_out_repr())
                        ->assert_is_op_output("elementwise_mul")
                        ->assert_is_op_input("elementwise_add")
                        ->AsIntermediate();
  auto *beta = pattern->NewNode(beta_repr())
                   ->assert_is_op_input("elementwise_add", "Y")
                   ->assert_is_persistable_var()
                   ->AsInput();
  auto *shift = pattern->NewNode(shift_repr())->assert_is_op("elementwise_add");
  auto *shift_out = pattern->NewNode(shift_out_repr())
                        ->assert_is_op_output("elementwise_add")
                        ->AsOutput();

  /*
   *            X
   *           / \
   *          /   reduce_mean "u(x)"
   *          \   /
   *      elementwise_sub     "x - u(x)"
   *      /           \    2
   *      |            \  /
   *      |      elementwise_pow  "(x - u(x))^2"
   *      |             |
   *      |       reduce_mean     "sigma^2 = 1/C*Sum{(x - u(x))^2}"
   *      |             |     eps
   *      |             |     /
   *      |       elementwise_add "sigma^2 + epsilon"
   *      \             |
   *       \           sqrt       "sqrt(sigma^2 + epsilon)"
   *        \          /
   *         \        /
   *       elementwise_div        "lnorm = {x-u(x)}/{sqrt(sigma^2 + epsilon)}"
   *              |
   *       gamma  |
   *          \   |
   *       elementwise_mul        "scale: gamma(C) * lnorm"
   *              |
   *        beta  |
   *          \   |
   *       elementwise_add        "shift: gamma(C) * lnorm + beta(C)"
   */

  x_mean->LinksFrom({x}).LinksTo({x_mean_out});
  x_sub_mean->LinksFrom({x, x_mean_out}).LinksTo({x_sub_mean_out});
  x_sub_mean_sqr->LinksFrom({x_sub_mean_out, sqr_pow})
      .LinksTo({x_sub_mean_sqr_out});
  std_dev->LinksFrom({x_sub_mean_sqr_out}).LinksTo({std_dev_out});
  std_dev_eps->LinksFrom({std_dev_out, eps}).LinksTo({std_dev_eps_out});

  std_dev_eps_sqrt->LinksFrom({std_dev_eps_out})
      .LinksTo({std_dev_eps_sqrt_out});
  division->LinksFrom({x_sub_mean_out, std_dev_eps_sqrt_out})
      .LinksTo({division_out});
  scale->LinksFrom({division_out, gamma}).LinksTo({scale_out});
  shift->LinksFrom({scale_out, beta}).LinksTo({shift_out});

  return shift_out;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
