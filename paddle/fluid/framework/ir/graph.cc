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

#include <memory>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/operator.h"

DEFINE_bool(convert_all_blocks, false,
            "Convert all blocks in program into SSAgraphs");

namespace paddle {
namespace framework {
namespace ir {

Graph::Graph(const ProgramDesc &program)
    : Graph(program, 0, program.Block(0).AllOps().size()) {}

Graph::Graph(const ProgramDesc &program, const int64_t start_op_index,
             const int64_t end_op_index)
    : program_(program), main_graph_(nullptr) {
  PADDLE_ENFORCE_GE(start_op_index, 0,
                    platform::errors::InvalidArgument(
                        "Required start_op_index >= 0, but received "
                        "start_op_index = %d",
                        start_op_index));
  PADDLE_ENFORCE_GE(end_op_index, start_op_index,
                    platform::errors::InvalidArgument(
                        "Required end_op_index >= start_op_index, but received "
                        "end_op_index: %d < start_op_index: %d",
                        end_op_index, start_op_index));
  PADDLE_ENFORCE_GE(
      program_.Size(), 1,
      platform::errors::InvalidArgument("Can't construct a graph from this "
                                        "program, it doesn't have a block"));

  const int64_t block_op_size = program_.Block(0).AllOps().size();
  PADDLE_ENFORCE_LE(end_op_index, block_op_size,
                    platform::errors::InvalidArgument(
                        "Required end_op_index <= block_op_size, but received "
                        "end_op_index: %d > block_op_size: %d",
                        end_op_index, block_op_size));
  if (FLAGS_convert_all_blocks) {
    // NOTE(levi): start_op_index and end_op_index only work on the first
    // sub_graph.
    std::unique_ptr<Graph> first_sub_graph = std::make_unique<Graph>(
        program_.Block(0), this, start_op_index, end_op_index);
    sub_graphs_.push_back(std::move(first_sub_graph));
    for (size_t idx = 1; idx < program_.Size(); ++idx) {
      std::unique_ptr<Graph> sub_graph =
          std::make_unique<Graph>(program_.Block(idx), this);
      sub_graphs_.push_back(std::move(sub_graph));
    }
  } else {
    auto var_nodes = InitFromProgram(program_, start_op_index, end_op_index);
    ResolveHazard(var_nodes);
  }
}

Graph::Graph(const BlockDesc &block, const Graph *main_graph)
    : Graph(block, main_graph, 0, block.AllOps().size()) {}

Graph::Graph(const BlockDesc &block, const Graph *main_graph,
             const int64_t start_op_index, const int64_t end_op_index)
    : main_graph_(main_graph) {
  auto var_nodes = InitFromBlock(block, start_op_index, end_op_index);
  ResolveHazard(var_nodes);
}

// TODO(levi): delete this interface after when we can convert all
// blocks into sub_graphs.
std::map<std::string, std::vector<ir::Node *>> Graph::InitFromProgram(
    const ProgramDesc &program, const int64_t start_op_index,
    const int64_t end_op_index) {
  VLOG(3) << "block in program:" << program_.Size();
  return InitFromBlock(program.Block(0), start_op_index, end_op_index);
}

std::map<std::string, std::vector<ir::Node *>> Graph::InitFromBlock(
    const BlockDesc &block, const int64_t start_op_index,
    const int64_t end_op_index) {
  std::unordered_map<std::string, VarDesc *> all_vars;
  // var nodes for each var name, will have multiple versions in SSA
  std::map<std::string, std::vector<ir::Node *>> var_nodes;
  for (auto *var : block.AllVars()) {
    all_vars.emplace(var->Name(), var);
  }

  int desc_order = 0;
  auto not_visited_vars = all_vars;
  auto all_ops = block.AllOps();
  PADDLE_ENFORCE_LE(
      end_op_index, all_ops.size(),
      platform::errors::InvalidArgument(
          "Required end_op_index <= %d, but received end_op_index = %d",
          all_ops.size(), end_op_index));

  for (auto i = start_op_index; i < end_op_index; ++i) {
    auto *op = all_ops[i];
    VLOG(3) << "create OpNode by " << op->Type();
    ir::Node *node = CreateOpNode(op);
    node->SetDescOrder(desc_order);
    ++desc_order;
    // For input args, reuse the same var name if it was created before.
    // Otherwise, create a new one.
    for (auto &each_var_name : op->InputArgumentNames()) {
      not_visited_vars.erase(each_var_name);
      ir::Node *var = nullptr;
      if (var_nodes.find(each_var_name) != var_nodes.end()) {
        var = var_nodes.at(each_var_name).back();
      } else if (all_vars.count(each_var_name) != 0) {
        var = CreateVarNode(all_vars.at(each_var_name));
        var_nodes[each_var_name].push_back(var);
      } else {
        // Operation input var can be optional (dispensable). Which means
        // the operation doesn't really need the var at runtime. In this
        // case, the no-existed var is ready at the beginning.
        var = CreateEmptyNode(each_var_name, ir::Node::Type::kVariable);
        var_nodes[each_var_name].push_back(var);
      }
      node->inputs.push_back(var);
      var->outputs.push_back(node);
    }
    // For output args, always create a new var.
    std::unordered_set<std::string> out_arg_set;
    for (auto &each_var_name : op->OutputArgumentNames()) {
      not_visited_vars.erase(each_var_name);
      if (each_var_name != kEmptyVarName) {
        PADDLE_ENFORCE_EQ(out_arg_set.count(each_var_name), 0,
                          platform::errors::InvalidArgument(
                              "The input Program is invalid. Variable %s occurs"
                              " in output of %s multiple times.",
                              each_var_name, op->Type()));
        out_arg_set.insert(each_var_name);
      }

      ir::Node *var = nullptr;
      if (all_vars.count(each_var_name) != 0) {
        var = CreateVarNode(all_vars.at(each_var_name));
      } else {
        // Operation output vars can be @EMPTY@. For example, while_grad
        // can have multi @EMPTY@ outputs with no VarDesc.
        // TODO(panyx0718): Add a test.
        var = CreateEmptyNode(each_var_name, ir::Node::Type::kVariable);
      }
      var_nodes[each_var_name].push_back(var);
      node->outputs.push_back(var);
      var->inputs.push_back(node);
    }
  }

  if (end_op_index < static_cast<int64_t>(all_ops.size()) ||
      start_op_index > 0) {
    is_partial_ = true;
  }
  if (!is_partial_) {
    for (auto &pair : not_visited_vars) {
      const auto &var_name = pair.first;
      auto *var_desc = pair.second;
      if (var_name != kEmptyVarName) {
        VLOG(10) << "Create isolated var node " << var_name;
        var_nodes[var_name].push_back(CreateVarNode(var_desc));
      }
    }
  }

  Set<const std::vector<OpDesc *>>(
      details::kStaleProgramOpDescs,
      new std::vector<OpDesc *>(all_ops.begin() + start_op_index,
                                all_ops.begin() + end_op_index));
  VLOG(3)
      << "kStaleProgramOpDescs.size: "
      << Get<const std::vector<OpDesc *>>(details::kStaleProgramOpDescs).size();
  return var_nodes;
}

void Graph::ResolveHazard(
    const std::map<std::string, std::vector<ir::Node *>> &var_nodes) {
  /**
   * We should handle write after read(WAR) and write after write(WAW) here.
   * Because some of the operators of the program can be executed parallelly.
   * So, to make the program running in the right order, we should add the
   * dependence of WAR and WAW.
   *
   *
   * https://en.wikipedia.org/wiki/Hazard_(computer_architecture)#Write_after_read_(WAR)
   */

  for (auto &var : var_nodes) {
    auto &versions = var.second;
    if (versions.size() <= 1) continue;

    auto it_new = versions.rbegin();
    auto it_old = versions.rbegin();
    ++it_old;
    for (; it_old != versions.rend(); it_new = it_old, ++it_old) {
      VLOG(3) << "deal with var: " << (*it_new)->Name();
      ir::Node *write_op =
          (*it_new)->inputs.empty() ? nullptr : (*it_new)->inputs[0];
      const auto &read_ops = (*it_old)->outputs;

      PADDLE_ENFORCE_NOT_NULL(
          write_op, platform::errors::NotFound(
                        "The generate operator of variable %s is null.",
                        (*it_new)->Name()));

      // Add write after write dependence
      ir::Node *upstream_op =
          (*it_old)->inputs.empty() ? nullptr : (*it_old)->inputs[0];
      // TODO(zcd): Add a test.
      if (upstream_op && upstream_op != write_op) {
        ir::Node *dep_var = CreateControlDepVar();
        write_op->inputs.push_back(dep_var);
        upstream_op->outputs.push_back(dep_var);
        VLOG(10) << "add dep_var:" << dep_var->Name();
        dep_var->outputs.push_back(write_op);
        dep_var->inputs.push_back(upstream_op);
      }

      for (auto *read_op : read_ops) {
        // Manually add a dependency var from read_op to write_op;
        if (read_op == write_op) {
          // Read Write is the same op.
          continue;
        }
        // 2 ops might have been connected via other vars.
        bool has_dep = false;
        for (ir::Node *r_out : read_op->outputs) {
          for (ir::Node *w_in : write_op->inputs) {
            if (r_out == w_in) {
              has_dep = true;
              break;
            }
          }
        }
        if (has_dep) continue;

        ir::Node *dep_var = CreateControlDepVar();
        VLOG(10) << "add dep_var:" << dep_var->Name();
        read_op->outputs.push_back(dep_var);
        dep_var->inputs.push_back(read_op);
        write_op->inputs.push_back(dep_var);
        dep_var->outputs.push_back(write_op);
      }
    }
  }
}

std::shared_ptr<Graph> Graph::Clone() {
  PADDLE_ENFORCE_EQ(
      this->IsMainGraph(), true,
      platform::errors::InvalidArgument(
          "This graph is a sub_graph, and can't be cloned individually"));
  if (FLAGS_convert_all_blocks) {
    auto cloned_graph = std::make_shared<Graph>(this->program_);
    cloned_graph->ReleaseSubGraphs();
    for (size_t idx = 0; idx < this->program_.Size(); ++idx) {
      cloned_graph->AddSubGraph(this->CloneSubGraph(idx));
    }
    return cloned_graph;
  } else {
    auto cloned_graph = std::make_shared<Graph>(this->program_);
    cloned_graph->ReleaseNodes();
    cloned_graph->num_node_created_ = 0;
    std::unordered_map<ir::Node *, ir::Node *> origin_to_cloned;
    for (auto *n : this->node_set_) {
      PADDLE_ENFORCE_NOT_NULL(n, platform::errors::InvalidArgument(
                                     "The node to be cloned is nullptr."));
      ir::Node *cloned_node = nullptr;
      if (n->IsCtrlVar()) {
        cloned_node = cloned_graph->CreateControlDepVar();
      } else if (!n->var_desc_ && !n->op_desc_) {  // empty node
        cloned_node = cloned_graph->CreateEmptyNode(n->Name(), n->NodeType());
      } else if (n->IsVar()) {
        cloned_node = cloned_graph->CreateVarNode(n->Var());
      } else if (n->IsOp()) {
        cloned_node = cloned_graph->CreateOpNode(n->Op());
      }
      PADDLE_ENFORCE_NOT_NULL(
          cloned_node,
          platform::errors::InvalidArgument(
              "Failed to clone new node from original node in graph."));
      origin_to_cloned[n] = cloned_node;
    }
    for (auto *n : this->node_set_) {
      for (auto it = n->inputs.begin(); it != n->inputs.end(); it++) {
        origin_to_cloned[n]->inputs.push_back(origin_to_cloned[*it]);
      }
      for (auto it = n->outputs.begin(); it != n->outputs.end(); it++) {
        origin_to_cloned[n]->outputs.push_back(origin_to_cloned[*it]);
      }
    }
    return cloned_graph;
  }
}

std::unique_ptr<Graph> Graph::CloneSubGraph(const size_t idx) {
  PADDLE_ENFORCE_EQ(
      this->IsMainGraph(), true,
      platform::errors::InvalidArgument("This graph is not main_graph"));
  PADDLE_ENFORCE_LT(
      idx, this->sub_graphs_.size(),
      platform::errors::InvalidArgument("Invalid sub_graph index"));
  std::unique_ptr<Graph> cloned_sub_graph =
      std::make_unique<Graph>(this->program_.Block(idx), this);
  cloned_sub_graph->ReleaseNodes();
  cloned_sub_graph->num_node_created_ = 0;
  std::unordered_map<ir::Node *, ir::Node *> origin_to_cloned;
  for (auto *n : this->sub_graphs_.at(idx)->Nodes()) {
    PADDLE_ENFORCE_NOT_NULL(n, platform::errors::InvalidArgument(
                                   "The node to be cloned is nullptr."));
    ir::Node *cloned_node = nullptr;
    if (n->IsCtrlVar()) {
      cloned_node = cloned_sub_graph->CreateControlDepVar();
    } else if (!n->var_desc_ && !n->op_desc_) {  // empty node
      cloned_node = cloned_sub_graph->CreateEmptyNode(n->Name(), n->NodeType());
    } else if (n->IsVar()) {
      cloned_node = cloned_sub_graph->CreateVarNode(n->Var());
    } else if (n->IsOp()) {
      cloned_node = cloned_sub_graph->CreateOpNode(n->Op());
    }
    PADDLE_ENFORCE_NOT_NULL(
        cloned_node,
        platform::errors::InvalidArgument(
            "Failed to clone new node from original node in graph."));
    origin_to_cloned[n] = cloned_node;
  }
  for (auto *n : this->sub_graphs_.at(idx)->Nodes()) {
    for (auto it = n->inputs.begin(); it != n->inputs.end(); it++) {
      origin_to_cloned[n]->inputs.push_back(origin_to_cloned[*it]);
    }
    for (auto it = n->outputs.begin(); it != n->outputs.end(); it++) {
      origin_to_cloned[n]->outputs.push_back(origin_to_cloned[*it]);
    }
  }
  return cloned_sub_graph;
}

bool IsControlDepVar(const ir::Node &var) {
  return var.Name().find(ir::Node::kControlDepVarName) != std::string::npos;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
