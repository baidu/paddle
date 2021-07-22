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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;
class Graph;

/*
* Remove the sum op of all gradients of the backward op.
* And remove the dependecies of the optimizer related to the
* same backward op.
*
* Before this pass:
*
* forward_op1 forward_op2
*     |            |
*  grad_op1    grad_op2
*        \      /
*          \  /
*         sum_op
*           |
*         sgd_op
*
* After this pass:
* forward_op1 forward_op2
*     |            |
*  grad_op1    grad_op2
*     |            |
*  sgd_op1      sgd_op2
*
* sgd_op1 and sgd_op2 will update the same weight which holds the same
* memory, so we could benefits from the acceleration
*/
class LockFreeOptimizePass : public Pass {
 public:
  virtual ~LockFreeOptimizePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

 private:
  // Create a new sgd node via current optimizer node
  ir::Node* CreateNewSGDNode(ir::Graph* graph, ir::Node* forward_node,
                             ir::Node* backward_node, ir::Node* grad_sum_node,
                             ir::Node* optimize_node) const;

  // Replace the input weight's optimizers
  void ReplaceUpstreamNode(ir::Node* upstream_node,
                           ir::Node* old_optimizer_node,
                           ir::Node* new_optimizer_node) const;

  // Replace the output weight's optimizers
  void ReplaceAllDownstreamNode(ir::Node* old_optimizer_node,
                                ir::Node* new_optimizer_node) const;

  // Find all weight variables in graph
  bool FindAllWeightVars(ir::Graph* graph) const;

  // Find the forward_op node via the backward_op node
  ir::Node* FindForwardOpViaBackwardOp(ir::Graph* graph,
                                       ir::Node* backward_node) const;

  std::vector<ir::Node*> FindConnectedNode(ir::Node* upstream_node,
                                           ir::Node* downstream_node) const;

  inline bool IsOpNamed(ir::Node* node, const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(node,
                            platform::errors::InvalidArgument(
                                "Input argument node cannot be nullptr."));

    return node->NodeType() == Node::Type::kOperation && node->Name() == name;
  }

  inline bool IsVarNamed(ir::Node* node, const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(node,
                            platform::errors::InvalidArgument(
                                "Input argument node cannot be nullptr."));

    return node->NodeType() == Node::Type::kVariable && node->Name() == name;
  }

  inline bool IsVarNameEndsWith(ir::Node* node, const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(node,
                            platform::errors::InvalidArgument(
                                "Input argument node cannot be nullptr."));

    return node->NodeType() == Node::Type::kVariable &&
           paddle::string::ends_with(node->Name(), name);
  }

  inline bool IsVarNameContains(ir::Node* node, const std::string& name) const {
    PADDLE_ENFORCE_NOT_NULL(node,
                            platform::errors::InvalidArgument(
                                "Input argument node cannot be nullptr."));

    return node->NodeType() == Node::Type::kVariable &&
           node->Name().find(name) != std::string::npos;
  }

  inline bool IsControlDepFrom(ir::Node* ctrl_dep_node, ir::Node* node) const {
    PADDLE_ENFORCE_NOT_NULL(
        ctrl_dep_node, platform::errors::InvalidArgument(
                           "Input argument ctrl_dep_node cannot be nullptr."));
    PADDLE_ENFORCE_NOT_NULL(node,
                            platform::errors::InvalidArgument(
                                "Input argument node cannot be nullptr."));

    return IsControlDepVar(*ctrl_dep_node) &&
           ctrl_dep_node->inputs.size() >= 1u &&
           ctrl_dep_node->inputs[0] == node;
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
