/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/simplify_with_basic_ops_pass.h"

#include "glog/logging.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * This pass is to simplify the Grpah, it may contains:
 * - replace comlicated op with basic op
 * - remove some unnecessary op
 *
 * In the current implementation, it supports:
 * - remove dropout_op (upscale_in_train) or
 *   replace dropout_op with scale_op (downgrade_in_infer) when is_test is true
 */
class Graph;

SimplifyWithBasicOpsPass::SimplifyWithBasicOpsPass() {
  AddOpCompat(OpCompat("scale"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("scale")
      .IsNumGE(0.f)
      .IsNumLE(1.f)
      .End()
      .AddAttr("bias")
      .IsNumEQ(0.f)
      .End()
      .AddAttr("bias_after_scale")
      .IsNumEQ(true)
      .End();
}

void SimplifyWithBasicOpsPass::ApplyImpl(Graph* graph) const {
  VLOG(3) << "Simplify the Graph with basic ops.";
  std::unordered_set<const Node*> del_node_set;
  for (Node* n : graph->Nodes()) {
    if (n->IsOp() && n->Op()) {
      if (n->Op()->Type() == "dropout") {
        SimplifyDropout(graph, n, &del_node_set);
      }
    }
  }

  GraphSafeRemoveNodes(graph, del_node_set);
}

bool SimplifyWithBasicOpsPass::SimplifyDropout(
    Graph* graph, Node* n,
    std::unordered_set<const Node*>* del_node_set) const {
  OpDesc* dropout_op_desc = n->Op();
  bool is_test = false;
  // In the model used in test_analyzer_bert, the is_test's AttrType of
  // dropout_op is INT.
  if (dropout_op_desc->HasAttr("is_test")) {
    if (dropout_op_desc->GetAttrType("is_test") == proto::AttrType::BOOLEAN) {
      is_test = BOOST_GET_CONST(bool, dropout_op_desc->GetAttr("is_test"));
    } else if (dropout_op_desc->GetAttrType("is_test") ==
               proto::AttrType::INT) {
      is_test = BOOST_GET_CONST(int, dropout_op_desc->GetAttr("is_test")) == 0
                    ? false
                    : true;
    }
  }

  if (!is_test) {
    return false;
  }

  Node* dropout_x = GetInputVar(n, dropout_op_desc->Input("X")[0]);
  Node* dropout_out = GetOutputVar(n, dropout_op_desc->Output("Out")[0]);

  bool upscale_in_train = false;
  // Once the dropout_implementation's AttrType is BOOLEAN, but now is STRING.
  if (dropout_op_desc->HasAttr("dropout_implementation")) {
    if (dropout_op_desc->GetAttrType("dropout_implementation") ==
        proto::AttrType::BOOLEAN) {
      upscale_in_train = BOOST_GET_CONST(
          bool, dropout_op_desc->GetAttr("dropout_implementation"));
    } else if (dropout_op_desc->GetAttrType("dropout_implementation") ==
               proto::AttrType::STRING) {
      upscale_in_train =
          BOOST_GET_CONST(std::string,
                          dropout_op_desc->GetAttr("dropout_implementation")) ==
          "upscale_in_train";
    }
  }

  if (upscale_in_train) {
    // dropout_op can be deleted.
    // dropout_x -> dropout_op -> dropout_out -> next_op -> next_out
    //   |
    //  \|/
    // dropout_x -> next_op -> next_out
    // Check whether dropout_x is some next_op's output
    bool dropout_x_is_reused_as_output = false;
    for (auto* next_op : dropout_out->outputs) {
      for (auto* next_out : next_op->outputs) {
        if (next_out == dropout_x ||
            next_out->Var()->Name() == dropout_x->Var()->Name()) {
          dropout_x_is_reused_as_output = true;
          break;
        }
      }
      if (dropout_x_is_reused_as_output) {
        break;
      }
    }
    if (dropout_x_is_reused_as_output) {
      VarDesc new_var_desc(*dropout_x->Var());
      new_var_desc.SetName("simplify_with_basic_ops_" + dropout_x->Name());
      auto* new_var_node = graph->CreateVarNode(&new_var_desc);
      for (auto* out_op : dropout_x->outputs) {
        if (out_op != n) {
          ReplaceInputVar(out_op, dropout_x, new_var_node);
        }
      }
      for (auto* in_op : dropout_x->inputs) {
        ReplaceOutputVar(in_op, dropout_x, new_var_node);
      }
      dropout_x = new_var_node;
    }
    for (auto* next_op : dropout_out->outputs) {
      ReplaceInputVar(next_op, dropout_out, dropout_x);
    }

    del_node_set->insert(dropout_out);
  } else {
    // Use a scale_op replaces the dropout_op
    // dropout_x -> dropout_op -> dropout_out -> next_op -> next_out
    //   |
    //  \|/
    // dropout_x -> scale_op -> dropout_out -> next_op -> next_out
    float scale =
        1.0f - BOOST_GET_CONST(float, dropout_op_desc->GetAttr("dropout_prob"));

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("scale");
    new_op_desc.SetInput("X", {dropout_x->Name()});
    new_op_desc.SetOutput("Out", {dropout_out->Name()});
    new_op_desc.SetAttr("scale", scale);
    new_op_desc.SetAttr("bias", static_cast<float>(0));
    new_op_desc.SetAttr("bias_after_scale", true);

    if (!IsCompat(new_op_desc)) {
      LOG(WARNING) << "Basic ops pass in scale op compat failed.";
      return false;
    }

    auto* scale_op_node = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(dropout_x, scale_op_node);
    IR_NODE_LINK_TO(scale_op_node, dropout_out);
  }

  del_node_set->insert(n);
  return true;
}

Node* SimplifyWithBasicOpsPass::GetInputVar(Node* n,
                                            const std::string& name) const {
  for (auto* in : n->inputs) {
    if (in->Name() == name) {
      return in;
    }
  }
  return nullptr;
}

Node* SimplifyWithBasicOpsPass::GetOutputVar(Node* n,
                                             const std::string& name) const {
  for (auto* out : n->outputs) {
    if (out->Name() == name) {
      return out;
    }
  }
  return nullptr;
}

void SimplifyWithBasicOpsPass::ReplaceInputVar(Node* op, Node* old_var,
                                               Node* new_var) const {
  if (op->IsOp() && op->Op()) {
    new_var->outputs.push_back(op);
    for (size_t i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == old_var) {
        op->inputs[i] = new_var;
        op->Op()->RenameInput(old_var->Name(), new_var->Name());
      }
    }
  }
}

void SimplifyWithBasicOpsPass::ReplaceOutputVar(Node* op, Node* old_var,
                                                Node* new_var) const {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(simplify_with_basic_ops_pass,
              paddle::framework::ir::SimplifyWithBasicOpsPass);
