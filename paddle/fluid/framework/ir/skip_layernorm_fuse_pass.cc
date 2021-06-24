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

#include "paddle/fluid/framework/ir/skip_layernorm_fuse_pass.h"

#include <string>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

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

struct SkipLayerNorm : public PatternBase {
  SkipLayerNorm(PDPattern *pattern, const std::string &name_scope)
      : PatternBase(pattern, name_scope, "skip_layernorm") {}

  PDNode *operator()(PDNode *x, PDNode *y);

  // declare operator node's name
  PATTERN_DECL_NODE(fused_skipe_layernorm);
  PATTERN_DECL_NODE(elementwise);
  PATTERN_DECL_NODE(layer_norm);
  // declare variable node's name
  PATTERN_DECL_NODE(
      elementwise_out);  // (elementwise_input_x,elementwise_input_y) ->
                         // elementwise_out
  PATTERN_DECL_NODE(layer_norm_bias);
  PATTERN_DECL_NODE(layer_norm_scale);
  PATTERN_DECL_NODE(layer_norm_out);
  PATTERN_DECL_NODE(layer_norm_mean);
  PATTERN_DECL_NODE(layer_norm_variance);
};

PDNode *SkipLayerNorm::operator()(PDNode *x, PDNode *y) {
  // Create nodes for elementwise add op.
  x->assert_is_op_input("elementwise_add", "X");
  y->assert_is_op_input("elementwise_add", "Y");
  auto *elementwise =
      pattern->NewNode(elementwise_repr())->assert_is_op("elementwise_add");
  auto *elementwise_out_var = pattern->NewNode(elementwise_out_repr())
                                  ->AsOutput()
                                  ->assert_is_op_output("elementwise_add");

  // Add links for elementwise_add op.
  elementwise->LinksFrom({x, y}).LinksTo({elementwise_out_var});

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

void SkipLayerNormFusePass::ApplyImpl(ir::Graph *graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::PreconditionNotMet("graph should not be null."));
  FusePassBase::Init("skip_layernorm_fuse", graph);
  int found_subgraph_count = 0;

  GraphPatternDetector gpd;
  auto *x = gpd.mutable_pattern()
                ->NewNode("skip_layernorm_fuse/x")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "X")
                ->assert_var_not_persistable();
  auto *y = gpd.mutable_pattern()
                ->NewNode("skip_layernorm_fuse/y")
                ->AsInput()
                ->assert_is_op_input("elementwise_add", "Y")
                ->assert_var_not_persistable();
  patterns::SkipLayerNorm fused_pattern(gpd.mutable_pattern(),
                                        "skip_layernorm_fuse");
  fused_pattern(x, y);

  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *graph) {
    if (subgraph.count(x) <= 0 || subgraph.count(y) <= 0) {
      LOG(WARNING) << "The subgraph is empty.";
      return;
    }

    if (!IsCompat(subgraph, graph)) {
      LOG(WARNING) << "skip_layernorm pass in op compat failed.";
      return;
    }

    VLOG(4) << "handle SkipLayerNorm fuse";
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

    std::unordered_set<const Node *> del_node_set;

    // Create an SkipLayerNorm op node
    OpDesc new_desc;
    new_desc.SetType("skip_layernorm");

    // inputs
    new_desc.SetInput("X", {subgraph.at(x)->Name()});
    new_desc.SetInput("Y", {subgraph.at(y)->Name()});
    new_desc.SetInput("Scale", {layer_norm_scale->Name()});
    new_desc.SetInput("Bias", {layer_norm_bias->Name()});

    if (elementwise->Op()->HasAttr("out_threshold")) {
      new_desc.SetAttr("enable_int8", true);
    }

    // outputs
    new_desc.SetOutput("Out", {layer_norm_out->Name()});

    // attrs
    new_desc.SetAttr("epsilon", layer_norm->Op()->GetAttr("epsilon"));
    new_desc.SetAttr("begin_norm_axis",
                     layer_norm->Op()->GetAttr("begin_norm_axis"));

    auto fused_node = graph->CreateOpNode(&new_desc);  // OpDesc will be copied.

    del_node_set.insert(elementwise);
    del_node_set.insert(layer_norm);
    del_node_set.insert(elementwise_out);
    del_node_set.insert(layer_norm_mean);
    del_node_set.insert(layer_norm_variance);
    GraphSafeRemoveNodes(graph, del_node_set);

    IR_NODE_LINK_TO(subgraph.at(x), fused_node);
    IR_NODE_LINK_TO(subgraph.at(y), fused_node);
    IR_NODE_LINK_TO(layer_norm_scale, fused_node);
    IR_NODE_LINK_TO(layer_norm_bias, fused_node);
    IR_NODE_LINK_TO(fused_node, layer_norm_out);

    found_subgraph_count++;
  };

  gpd(graph, handler);
  AddStatis(found_subgraph_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(skip_layernorm_fuse_pass,
              paddle::framework::ir::SkipLayerNormFusePass);
REGISTER_PASS_CAPABILITY(skip_layernorm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("elementwise_add", 1)
            .EQ("layer_norm", 0));
