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

#include "paddle/fluid/framework/ir/repeated_fc_relu_fuse_pass.h"
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

#define MAX_NUM_FC 10

namespace paddle {
namespace framework {
namespace ir {

static bool IsInputOfFC(Node* n) {
  if (n && n->IsVar() && VarLinksToOp(n, "fc")) {
    return true;
  }
  return false;
}

static bool IsOutputOfFC(Node* n) {
  if (n && n->IsVar() && VarLinksFromOp(n, "fc") && n->inputs.size() == 1U) {
    return true;
  }
  return false;
}

static bool IsFCWithAct(Node* n, const std::string& act_type = "relu") {
  if (n && n->IsOp() && n->Op() && n->Op()->Type() == "fc" &&
      n->inputs.size() == 3U && n->outputs.size() == 1U) {
    return BOOST_GET_CONST(std::string, n->Op()->GetAttr("activation_type")) ==
           act_type;
  }
  return false;
}

static bool IsFCWithPaddingWeights(Node* n) {
  bool res = false;
  if (n && n->IsOp() && n->Op() && n->Op()->Type() == "fc" &&
      n->inputs.size() == 3U && n->outputs.size() == 1U) {
    if (n->Op()->HasAttr("padding_weights")) {
      res = BOOST_GET_CONST(bool, n->Op()->GetAttr("padding_weights"));
    }
  }
  return res;
}

static bool IsParamOfFC(Node* n, const std::string& param_name) {
  if (IsInputOfFC(n) && n->inputs.empty()) {
    for (auto* out : n->outputs) {
      if (out->Op()->Type() == "fc" &&
          n->Name() == out->Op()->Input(param_name)[0]) {
        return true;
      }
    }
  }
  return false;
}

static int FindFCIdx(Node* x, const std::string& act_type = "relu") {
  if (!IsInputOfFC(x)) {
    return -1;
  }
  for (size_t k = 0; k < x->outputs.size(); ++k) {
    auto* out_op = x->outputs[k];
    if (IsFCWithAct(out_op, act_type) && out_op->outputs.size() == 1U) {
      return k;
    }
  }
  return -1;
}

static int FindInputIdx(Node* n, const std::string& name,
                        const std::string& act_type = "relu") {
  if (!IsFCWithAct(n, act_type)) {
    return -1;
  }
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    if (n->inputs[i]->Name() == n->Op()->Input(name)[0]) {
      return i;
    }
  }
  return -1;
}

void BuildRepeatedFCReluPattern(PDPattern* pattern,
                                const std::string& name_scope, int num_fc) {
  auto var_next_is_fc_act = [=](Node* x, const std::string& act_type = "relu",
                                bool check_in_has_only_one_out = true,
                                int fc_idx = 0) -> bool {
    if (!IsInputOfFC(x)) {
      return false;
    }
    if (check_in_has_only_one_out && x->outputs.size() != 1U) {
      return false;
    }
    auto* fc_op = x->outputs[fc_idx];
    return IsFCWithAct(fc_op, act_type) && fc_op->outputs.size() == 1U;
  };

  // in -> fc -> out
  // Current x is in, return fc's out which is next fc's input.
  auto next_var_of_part = [=](Node* x, int fc_idx = 0) -> Node* {
    return x->outputs[fc_idx]->outputs[0];
  };

  auto var_next_is_fc_act_repeated_n_times = [=](
      Node* x, int repeated_times, const std::string& act_type = "relu",
      bool check_in_has_only_one_out = true) -> bool {
    for (int i = 0; i < repeated_times; ++i) {
      if (!var_next_is_fc_act(x, act_type,
                              i == 0 && check_in_has_only_one_out)) {
        return false;
      }
      x = next_var_of_part(x);
    }
    return true;
  };

  // x is output of fc
  auto var_before_is_fc_act = [=](Node* x, const std::string& act_type = "relu",
                                  bool at_top = false) -> bool {
    if (!IsOutputOfFC(x)) {
      return false;
    }
    auto* fc_op = x->inputs[0];
    if (!IsFCWithAct(fc_op, act_type) || fc_op->inputs.size() != 3U) {
      return false;
    }
    for (auto* fc_i : fc_op->inputs) {
      if (!fc_i->inputs.empty()) {
        if (at_top) {
          return true;
        } else {
          return VarLinksFromOp(fc_i, "fc");
        }
      }
    }
    return false;
  };

  auto before_var_of_part = [=](Node* x) -> Node* {
    auto* fc_op = x->inputs[0];
    for (auto* in : fc_op->inputs) {
      if (!in->inputs.empty()) {
        // w and bias has no input.
        return in;
      }
    }
    return nullptr;
  };

  auto var_before_is_fc_act_repeated_n_times = [=](
      Node* x, int repeated_times,
      const std::string& act_type = "relu") -> bool {
    for (int i = 0; i < repeated_times; ++i) {
      if (!var_before_is_fc_act(x, act_type, i == repeated_times - 1)) {
        return false;
      }
      x = before_var_of_part(x);
    }
    return true;
  };

  PDNode* fc_input_var_0 = nullptr;
  std::vector<PDNode*> fc_output_var(num_fc);
  std::vector<PDNode*> fc_weight_var(num_fc);
  std::vector<PDNode*> fc_bias_var(num_fc);
  std::vector<PDNode*> fc_ops(num_fc);

  for (int i = 0; i < num_fc; ++i) {
    if (i == 0) {
      fc_input_var_0 = pattern->NewNode(
          [=](Node* x) {
            if (x->outputs.size() <= 0 || x->inputs.size() <= 0U) {
              return false;
            }
            if (x->IsVar() && x->Var() && x->Var()->GetShape().size() > 2) {
              VLOG(3) << "repeated fc relu only supports input dims = 2, so it "
                         "is not applied.";
              return false;
            }
            int fc_idx = FindFCIdx(x);
            if (fc_idx < 0) {
              return false;
            } else if (fc_idx == 0) {
              return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu");
            } else {
              x = next_var_of_part(x, fc_idx);
              return var_next_is_fc_act_repeated_n_times(
                  x, std::max(1, num_fc - i - 1), "relu");
            }
          },
          name_scope + "/fc_in_0");
    }

    fc_weight_var[i] = pattern->NewNode(
        [=](Node* x) {
          if (!IsParamOfFC(x, "W")) {
            return false;
          }
          auto* fc_op = x->outputs[0];
          int input_idx = FindInputIdx(fc_op, "Input", "relu");
          return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                 var_before_is_fc_act_repeated_n_times(fc_op->inputs[input_idx],
                                                       i, "relu");
        },
        name_scope + "/fc_weight_" + std::to_string(i));

    fc_bias_var[i] = pattern->NewNode(
        [=](Node* x) {
          if (!IsParamOfFC(x, "Bias")) {
            return false;
          }
          auto* fc_op = x->outputs[0];
          int input_idx = FindInputIdx(fc_op, "Input", "relu");
          return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                 var_before_is_fc_act_repeated_n_times(fc_op->inputs[input_idx],
                                                       i, "relu");
        },
        name_scope + "/fc_bias_" + std::to_string(i));

    fc_output_var[i] = pattern->NewNode(
        [=](Node* x) {
          if (!IsOutputOfFC(x)) {
            return false;
          }
          x = before_var_of_part(x);
          if (i == 0 && x->outputs.size() > 0U) {
            if (x->inputs.size() <= 0U) {
              return false;
            }
            int fc_idx = FindFCIdx(x);
            if (fc_idx < 0) {
              return false;
            } else if (fc_idx == 0) {
              return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu");
            } else {
              x = next_var_of_part(x, fc_idx);
              return var_next_is_fc_act_repeated_n_times(
                  x, std::max(1, num_fc - i - 1), "relu");
            }
          } else {
            return var_next_is_fc_act_repeated_n_times(x, num_fc - i, "relu") &&
                   x->inputs.size() > 0 &&
                   var_before_is_fc_act_repeated_n_times(x, i, "relu");
          }
        },
        name_scope + "/fc_out_" + std::to_string(i));

    fc_ops[i] = pattern->NewNode(
        [=](Node* x) {
          if (!IsFCWithAct(x, "relu") || IsFCWithPaddingWeights(x)) {
            return false;
          }
          auto* fc_out_var = x->outputs[0];
          return fc_out_var && fc_out_var->IsVar() &&
                 fc_out_var->outputs.size() == 1 &&
                 var_next_is_fc_act_repeated_n_times(fc_out_var, num_fc - i - 1,
                                                     "relu") &&
                 var_before_is_fc_act_repeated_n_times(fc_out_var, i + 1,
                                                       "relu");
        },
        name_scope + "/fc_op_" + std::to_string(i));

    if (i == 0) {
      fc_ops[i]
          ->LinksFrom({fc_input_var_0, fc_weight_var[i], fc_bias_var[i]})
          .LinksTo({fc_output_var[i]});
    } else {
      fc_ops[i]
          ->LinksFrom({fc_output_var[i - 1], fc_weight_var[i], fc_bias_var[i]})
          .LinksTo({fc_output_var[i]});
    }
  }
}

static int BuildFusion(Graph* graph, const std::string& name_scope,
                       int num_fc) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  BuildRepeatedFCReluPattern(pattern, name_scope, num_fc);

  auto retrieve_node = [](const std::string& name,
                          const GraphPatternDetector::subgraph_t& subgraph,
                          const PDPattern& pat) -> Node* {
    PADDLE_ENFORCE_GT(subgraph.count(pat.RetrieveNode(name)), 0,
                      platform::errors::NotFound(
                          "Pattern has no node called %s.", name.c_str()));
    Node* p = subgraph.at(pat.RetrieveNode(name));
    PADDLE_ENFORCE_NOT_NULL(p, platform::errors::NotFound(
                                   "Subgraph has no node %s.", name.c_str()));
    return p;
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    LOG(INFO) << "handle Repeated FC Act fuse";
    std::vector<Node*> weights_vars(num_fc);
    std::vector<Node*> bias_vars(num_fc);
    std::vector<Node*> relu_vars(num_fc - 1);

    std::vector<std::string> weight_names(num_fc);
    std::vector<std::string> bias_names(num_fc);
    std::vector<std::string> relu_names(num_fc - 1);

    auto& fused_pattern = gpd.pattern();
    for (int i = 0; i < num_fc; ++i) {
      if (i < num_fc - 1) {
        relu_vars[i] =
            retrieve_node(name_scope + "/fc_out_" + std::to_string(i), subgraph,
                          fused_pattern);
        relu_names[i] = relu_vars[i]->Name();
      }

      weights_vars[i] =
          retrieve_node(name_scope + "/fc_weight_" + std::to_string(i),
                        subgraph, fused_pattern);
      weight_names[i] = weights_vars[i]->Name();

      bias_vars[i] = retrieve_node(name_scope + "/fc_bias_" + std::to_string(i),
                                   subgraph, fused_pattern);
      bias_names[i] = bias_vars[i]->Name();
    }

    auto* input_var =
        retrieve_node(name_scope + "/fc_in_0", subgraph, fused_pattern);
    auto* last_out_var =
        retrieve_node(name_scope + "/fc_out_" + std::to_string(num_fc - 1),
                      subgraph, fused_pattern);

    // Create New OpDesc
    OpDesc op_desc;
    op_desc.SetType("fusion_repeated_fc_relu");
    op_desc.SetInput("X", {input_var->Name()});
    op_desc.SetInput("W", weight_names);
    op_desc.SetInput("Bias", bias_names);
    op_desc.SetOutput("ReluOut", relu_names);
    op_desc.SetOutput("Out", {last_out_var->Name()});

    auto* op = graph->CreateOpNode(&op_desc);
    IR_NODE_LINK_TO(input_var, op);
    for (size_t i = 0; i < weights_vars.size(); ++i) {
      IR_NODE_LINK_TO(weights_vars[i], op);
      IR_NODE_LINK_TO(bias_vars[i], op);
    }
    for (size_t i = 0; i < relu_vars.size(); ++i) {
      IR_NODE_LINK_TO(op, relu_vars[i]);
    }
    IR_NODE_LINK_TO(op, last_out_var);

    std::unordered_set<const Node*> marked_nodes;
    for (auto& item : subgraph) {
      marked_nodes.insert(item.second);
    }
    for (size_t i = 0; i < weights_vars.size(); ++i) {
      marked_nodes.erase(weights_vars[i]);
      marked_nodes.erase(bias_vars[i]);
    }
    for (size_t i = 0; i < relu_vars.size(); ++i) {
      marked_nodes.erase(relu_vars[i]);
    }
    marked_nodes.erase(input_var);
    marked_nodes.erase(last_out_var);
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };

  gpd(graph, handler);
  return fusion_count;
}

void RepeatedFCReluFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = 0;
  for (int i = MAX_NUM_FC; i > 1; --i) {
    fusion_count +=
        BuildFusion(graph, name_scope_ + "/" + std::to_string(i), i);
  }
  AddStatis(fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(repeated_fc_relu_fuse_pass,
              paddle::framework::ir::RepeatedFCReluFusePass);
REGISTER_PASS_CAPABILITY(repeated_fc_relu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("fc", 0)
            .EQ("relu", 0));
