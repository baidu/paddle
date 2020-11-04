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

#include "paddle/fluid/framework/ir/fc_reshape_elementwise_layernorm_fuse_pass.h"
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FCReshapeElementwiseLayerNorm : public PatternBase {
  FCReshapeElementwiseLayerNorm(PDPattern *pattern,
                                const std::string &name_scope)
      : PatternBase(pattern, name_scope, "fc_reshape_elementwise_layernorm") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_fc_reshape_elementwise_layernorm);
  PATTERN_DECL_NODE(reshape_1);
  PATTERN_DECL_NODE(fc_1);
  PATTERN_DECL_NODE(gelu);
  PATTERN_DECL_NODE(fc_2);
  PATTERN_DECL_NODE(reshape_2);
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(layer_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(reshape_1_out);
  PATTERN_DECL_NODE(fc_w_1);
  PATTERN_DECL_NODE(fc_bias_1);
  PATTERN_DECL_NODE(fc_out_1);
  PATTERN_DECL_NODE(gelu_out);
  PATTERN_DECL_NODE(fc_w_2);
  PATTERN_DECL_NODE(fc_bias_2);
  PATTERN_DECL_NODE(fc_out_2);  // (x,fc_w,fc_bias) -> fc_out

  PATTERN_DECL_NODE(reshape_2_out);

  // PATTERN_DECL_NODE(elementwise_input);
  PATTERN_DECL_NODE(
      elementwise_out);  // (fc_out,elementwise_input) -> elementwise_out
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};

PDNode *FCReshapeElementwiseLayerNorm::operator()(PDNode *x) {
  // Create nodes for reshape_1 op
  x->assert_is_op_input("reshape2");
  auto *reshape_1 =
      pattern->NewNode(reshape_1_repr())->assert_is_op("reshape2");
  auto *reshape_1_out_var =
      pattern->NewNode(reshape_1_out_repr())->assert_is_op_output("reshape2");

  // Add links for reshape_1 op.
  reshape_1->LinksFrom({x}).LinksTo({reshape_1_out_var});

  // Create nodes for fc_1 op.
  reshape_1_out_var->AsIntermediate()->assert_is_op_input("fc", "Input");
  auto *fc_1 = pattern->NewNode(fc_1_repr())->assert_is_op("fc");
  auto *fc_w_var_1 = pattern->NewNode(fc_w_1_repr())
                         ->AsInput()
                         ->assert_is_persistable_var()
                         ->assert_is_op_input("fc", "W");
  auto *fc_bias_var_1 = pattern->NewNode(fc_bias_1_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()
                            ->assert_is_op_input("fc", "Bias");
  auto *fc_out_var_1 =
      pattern->NewNode(fc_out_1_repr())->assert_is_op_output("fc");

  // Add links for fc op.
  fc_1->LinksFrom({reshape_1_out_var, fc_w_var_1, fc_bias_var_1})
      .LinksTo({fc_out_var_1});

  // Create nodes for gelu op.
  fc_out_var_1->AsIntermediate()->assert_is_op_input("gelu");
  auto *gelu = pattern->NewNode(gelu_repr())->assert_is_op("gelu");
  auto gelu_out_var =
      pattern->NewNode(gelu_out_repr())->assert_is_op_output("gelu");

  // Add links for gelu op.
  gelu->LinksFrom({fc_out_var_1}).LinksTo({gelu_out_var});

  // Create nodes for fc op.
  gelu_out_var->AsIntermediate()->assert_is_op_input("fc", "Input");
  auto *fc_2 = pattern->NewNode(fc_2_repr())->assert_is_op("fc");
  auto *fc_w_var_2 = pattern->NewNode(fc_w_2_repr())
                         ->AsInput()
                         ->assert_is_persistable_var()
                         ->assert_is_op_input("fc", "W");
  auto *fc_bias_var_2 = pattern->NewNode(fc_bias_2_repr())
                            ->AsInput()
                            ->assert_is_persistable_var()
                            ->assert_is_op_input("fc", "Bias");
  auto *fc_out_var_2 =
      pattern->NewNode(fc_out_2_repr())->assert_is_op_output("fc");

  // Add links for fc op.
  fc_2->LinksFrom({gelu_out_var, fc_w_var_2, fc_bias_var_2})
      .LinksTo({fc_out_var_2});

  // Create nodes for reshape_2 op
  fc_out_var_2->AsIntermediate()->assert_is_op_input("reshape2");
  auto *reshape_2 =
      pattern->NewNode(reshape_2_repr())->assert_is_op("reshape2");
  auto *reshape_2_out_var =
      pattern->NewNode(reshape_2_out_repr())->assert_is_op_output("reshape2");
  // Add links for reshape_2 op.
  reshape_2->LinksFrom({fc_out_var_2}).LinksTo({reshape_2_out_var});

  // Create nodes for elementwise_add op.
  reshape_2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");

  auto *elementwise_out_var = pattern->NewNode(elementwise_out_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("elementwise_add");

  // Add links for elementwise_add op.
  elementwise->LinksFrom({reshape_2_out_var, x}).LinksTo({elementwise_out_var});

  // Create nodes for layer_norm op.
  elementwise_out_var->AsIntermediate()->assert_is_op_input("layer_norm");
  auto *layer_norm =
      pattern->NewNode(layer_norm_repr())->assert_is_op("layer_norm");
  auto *layer_norm_bias_var = pattern->NewNode(layer_norm_bias_repr())
                                  ->AsInput()
                                  ->assert_is_persistable_var()
                                  ->assert_is_op_input("layer_norm", "Bias");
  auto *layer_norm_scale_var = pattern->NewNode(layer_norm_scale_repr())
                                   ->AsInput()
                                   ->assert_is_persistable_var()
                                   ->assert_is_op_input("layer_norm", "Scale");

  auto *layer_norm_out_var = pattern->NewNode(layer_norm_out_repr())
                                 ->AsOutput()
                                 ->assert_is_op_output("layer_norm", "Y");
  auto *layer_norm_mean_var = pattern->NewNode(layer_norm_mean_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("layer_norm", "Mean");
  auto *layer_norm_variance_var =
      pattern->NewNode(layer_norm_variance_repr())
          ->AsOutput()
          ->assert_is_op_output("layer_norm", "Variance");

  // Add links for layer_norm op.
  layer_norm
      ->LinksFrom(
          {elementwise_out_var, layer_norm_bias_var, layer_norm_scale_var})
      .LinksTo(
          {layer_norm_out_var, layer_norm_mean_var, layer_norm_variance_var});
  return layer_norm_out_var;
}

}  // namespace patterns

template <typename T>
static bool IsEqual(const std::vector<T> &x, const std::vector<T> &y) {
  if (!(x.size() > 0U && y.size() > 0U) || x.size() != y.size()) {
    return false;
  }
  for (size_t i = 0; i < x.size(); ++i) {
    if (x[i] != y[i]) {
      return false;
    }
  }
  return true;
}

void FCReshapeElementwiseLayerNormFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("fc_reshape_elementwise_layernorm_fuse", graph);
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("fc_reshape_elementwise_layernorm_fuse/x")
                ->AsInput()
                ->assert_is_op_input("reshape2");
  patterns::FCReshapeElementwiseLayerNorm fused_pattern(
      gpd.mutable_pattern(), "fc_reshape_elementwise_layernorm_fuse");
  fused_pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    VLOG(4) << "handle FCReshapeElementwiseLayerNorm fuse";
    GET_IR_NODE_FROM_SUBGRAPH(reshape_1, reshape_1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_1_out, reshape_1_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_1, fc_1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_w_1, fc_w_1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias_1, fc_bias_1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out_1, fc_out_1, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gelu, gelu, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(gelu_out, gelu_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_2, fc_2, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_w_2, fc_w_2, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias_2, fc_bias_2, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out_2, fc_out_2, fused_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape_2, reshape_2, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_2_out, reshape_2_out, fused_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale, layer_norm_scale,
                              fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance, layer_norm_variance,
                              fused_pattern);

    if (reshape_2_out->outputs.size() > 1U ||
        elementwise_out->outputs.size() > 1U) {
      VLOG(1) << "output check failed!!!!!";
      VLOG(1) << "reshape_out->outputs.size(): "
              << reshape_2_out->outputs.size();
      VLOG(1) << "elementwise_out->outputs.size(): "
              << elementwise_out->outputs.size();
      // When reshape_out or elementwise_out are used as input of other
      // operators, we
      // cannon fuse.
      return;
    }

    std::unordered_set<const Node *> del_node_set;

    // Create an FusedFCReshapeElementwiseLayerNorm op node
    OpDesc new_desc;
    new_desc.SetType("fused_fc_reshape_elementwise_layernorm");

    // inputs
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("W_1", {fc_w_1->Name()});
    new_desc.SetInput("Bias0_1", {fc_bias_1->Name()});
    new_desc.SetInput("W_2", {fc_w_2->Name()});
    new_desc.SetInput("Bias0_2", {fc_bias_2->Name()});
    new_desc.SetInput("Y", {subgraph.at(x)->Name()});
    new_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_desc.SetInput("Bias1", {layer_norm_bias->Name()});

    // outputs
    new_desc.SetOutput("Out", {layer_norm_out->Name()});
    bool lnm_has_output = layer_norm_mean->outputs.size() > 0U;
    if (lnm_has_output) {
      new_desc.SetOutput("Mean", {layer_norm_mean->Name()});
    } else {
      del_node_set.insert(layer_norm_mean);
    }
    bool lnv_has_output = layer_norm_variance->outputs.size() > 0U;
    if (lnv_has_output) {
      new_desc.SetOutput("Variance", {layer_norm_variance->Name()});
    } else {
      del_node_set.insert(layer_norm_variance);
    }

    // attrs
    new_desc.SetAttr("x_num_col_dims_1",
                     fc_1->Op()->GetAttr("in_num_col_dims"));
    new_desc.SetAttr("x_num_col_dims_2",
                     fc_2->Op()->GetAttr("in_num_col_dims"));
    new_desc.SetAttr("shape", reshape_2->Op()->GetAttr("shape"));
    new_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));
    new_desc.SetAttr("begin_norm_axis",
                     layer_norm->Op()->GetAttr("begin_norm_axis"));
    new_desc.SetAttr("activation_type_1",
                     fc_1->Op()->GetAttr("activation_type"));
    new_desc.SetAttr("activation_type_2",
                     fc_2->Op()->GetAttr("activation_type"));

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(reshape_1);
    del_node_set.insert(fc_1);
    del_node_set.insert(gelu);
    del_node_set.insert(fc_2);
    del_node_set.insert(reshape_2);
    del_node_set.insert(elementwise);
    del_node_set.insert(layer_norm);
    del_node_set.insert(reshape_1_out);
    del_node_set.insert(fc_out_1);
    del_node_set.insert(gelu_out);
    del_node_set.insert(fc_out_2);
    del_node_set.insert(reshape_2_out);
    del_node_set.insert(elementwise_out);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(fc_w_1, fused_node);
    IR_NODE_LINK_TO(fc_bias_1, fused_node);
    IR_NODE_LINK_TO(fc_w_2, fused_node);
    IR_NODE_LINK_TO(fc_bias_2, fused_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);
    if (lnm_has_output) {
      IR_NODE_LINK_TO(fused_node, layer_norm_mean);
    }
    if (lnv_has_output) {
      IR_NODE_LINK_TO(fused_node, layer_norm_variance);
    }

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(fc_reshape_elementwise_layernorm_fuse_pass,
              paddle::framework::ir::FCReshapeElementwiseLayerNormFusePass);
