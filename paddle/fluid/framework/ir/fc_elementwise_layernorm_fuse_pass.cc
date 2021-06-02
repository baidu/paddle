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

#include "paddle/fluid/framework/ir/fc_elementwise_layernorm_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {
class Node;
}  // namespace ir
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

struct FCElementwiseLayerNorm : public PatternBase {
  FCElementwiseLayerNorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "fc_elementwise_layernorm") {}

  PDNode *operator()(PDNode *x);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_fc_elementwise_layernorm);
  PATTERN_DECL_NODE(fc);
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(layer_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(fc_w);
  PATTERN_DECL_NODE(fc_bias);
  PATTERN_DECL_NODE(fc_out);  // (x,fc_w,fc_bias) -> fc_out
  PATTERN_DECL_NODE(elementwise_input);
  PATTERN_DECL_NODE(
      elementwise_out);  // (fc_out,elementwise_input) -> elementwise_out
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};

PDNode *FCElementwiseLayerNorm::operator()(PDNode *x) {
  // Create nodes for fc op.
  x->assert_is_op_input("fc", "Input");
  auto *fc = pattern->NewNode(fc_repr())->assert_is_op("fc");
  auto *fc_w_var = pattern->NewNode(fc_w_repr())
                       ->AsInput()
                       ->assert_is_persistable_var()
                       ->assert_is_op_input("fc", "W");
  auto *fc_bias_var = pattern->NewNode(fc_bias_repr())
                          ->AsInput()
                          ->assert_is_persistable_var()
                          ->assert_is_op_input("fc", "Bias");
  auto *fc_out_var = pattern->NewNode(fc_out_repr())->assert_is_op_output("fc");

  // Add links for fc op.
  fc->LinksFrom({x, fc_w_var, fc_bias_var}).LinksTo({fc_out_var});

  // Create nodes for elementwise_add op.
  fc_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");
  auto *elementwise_input_var = pattern->NewNode(elementwise_input_repr())
                                    ->assert_is_op_input("elementwise_add");

  auto *elementwise_out_var = pattern->NewNode(elementwise_out_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("elementwise_add");

  // Add links for elementwise_add op.
  elementwise->LinksFrom({fc_out_var, elementwise_input_var})
      .LinksTo({elementwise_out_var});

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

FCElementwiseLayerNormFusePass::FCElementwiseLayerNormFusePass(){
  AddOpCompat(OpCompat("fc"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("in_num_col_dims")
      .IsNumGE(1)
      .End()
      .AddAttr("activation_type")
      .End();

  AddOpCompat(OpCompat("layer_norm"))
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsOptional()
      .End()

      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End()
      .AddAttr("begin_norm_axis")
      .IsNumGT(0)
      .End()


  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddInput("Out")
      .IsTensor()
      .End()


  AddOpCompat(OpCompat("fc_elementwise_layernorm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias0")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias1")
      .IsTensor()
      .End()

      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("Mean")
      .IsOptional()
      .End()
      .AddOutput("Variance")
      .IsOptional()
      .End()

      .AddAttr("x_num_col_dims")
      .End()
      .AddAttr("epsilon")
      .End()
      .AddAttr("begin_norm_axis")
      .End()
      .AddAttr("activation_type")
      .End()
}

void FCElementwiseLayerNormFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(graph,
                          platform::errors::InvalidArgument(
                              "Pointer to graph argument should not be NULL."));
  FusePassBase::Init("fc_elementwise_layernorm_fuse", graph);
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("fc_elementwise_layernorm_fuse/x")
                ->AsInput()
                ->assert_is_op_input("fc", "Input");
  patterns::FCElementwiseLayerNorm fused_pattern(
      gpd.mutable_pattern(), "fc_elementwise_layernorm_fuse");
  fused_pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle FCElementwiseLayerNorm fuse";
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_w, fc_w, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_bias, fc_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(fc_out, fc_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise, elementwise, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_input, elementwise_input,
                              fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_out, elementwise_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_bias, layer_norm_bias, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_scale, layer_norm_scale,
                              fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_mean, layer_norm_mean, fused_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_variance, layer_norm_variance,
                              fused_pattern);

    if (!IsEqual(fc_out->Var()->GetShape(),
                 elementwise_input->Var()->GetShape())) {
      return;
    }

    int begin_norm_axis =
        BOOST_GET_CONST(int, layer_norm->Op()->GetAttr("begin_norm_axis"));
    auto layer_norm_x_dims = fc_out->Var()->GetShape();
    auto layer_norm_x_mat_dims = framework::flatten_to_2d(
        framework::make_ddim(layer_norm_x_dims), begin_norm_axis);
    if (fc_w->Var()->GetShape()[1] != layer_norm_x_mat_dims[1]) {
      return;
    }

    if (fc_out->outputs.size() > 1U || elementwise_out->outputs.size() > 1U) {
      // When fc_out or elementwise_out are used as input of other operators, we
      // cannon fuse.
      return;
    }

    std::unordered_set<const Node *> del_node_set;

    // Create an FusedFCElementwiseLayerNorm op node
    OpDesc new_desc;
    new_desc.SetType("fused_fc_elementwise_layernorm");

    // inputs
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("W", {fc_w->Name()});
    new_desc.SetInput("Bias0", {fc_bias->Name()});
    new_desc.SetInput("Y", {elementwise_input->Name()});
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
    new_desc.SetAttr("x_num_col_dims", fc->Op()->GetAttr("in_num_col_dims"));
    new_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));
    new_desc.SetAttr("begin_norm_axis",
                     layer_norm->Op()->GetAttr("begin_norm_axis"));
    new_desc.SetAttr("activation_type", fc->Op()->GetAttr("activation_type"));

    if (!IsCompat(new_desc)) {
      LOG(WARNING) << "Fc fuse pass in out fc op compat failed.";
      return;
    }

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(fc);
    del_node_set.insert(elementwise);
    del_node_set.insert(layer_norm);
    del_node_set.insert(fc_out);
    del_node_set.insert(elementwise_out);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(fc_w, fused_node);
    IR_NODE_LINK_TO(fc_bias, fused_node);
    IR_NODE_LINK_TO(elementwise_input, fused_node);
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

REGISTER_PASS(fc_elementwise_layernorm_fuse_pass,
              paddle::framework::ir::FCElementwiseLayerNormFusePass);
