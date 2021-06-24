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

#include "paddle/fluid/framework/ir/conv_elementwise_add_fuse_pass.h"

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                    \
  GET_IR_NODE(conv_op);              \
  GET_IR_NODE(conv_out);             \
  GET_IR_NODE(conv_filter);          \
  GET_IR_NODE(elementwise_add_op);   \
  GET_IR_NODE(elementwise_add_in_y); \
  GET_IR_NODE(elementwise_add_out);

ConvElementwiseAddFusePass::ConvElementwiseAddFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("ResidualData")
      .IsOptional()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC", "AnyLayout"})
      .End();

  AddOpCompat(OpCompat("elementwise_add"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumEQ(1)
      .End();
}

void ConvElementwiseAddFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "conv_elementwise_add_fuse";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input("conv2d", "Input")
                ->AsInput();

  patterns::ConvElementwiseadd pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_NODES;

    auto base_op_desc = *conv_op->Op()->Proto();
    std::string bias_name = elementwise_add_in_y->Name();
    std::string output_name = elementwise_add_out->Name();

    std::string act_type = "identity";
    framework::OpDesc new_op_desc(base_op_desc, nullptr);
    new_op_desc.SetType("conv2d_fusion");
    new_op_desc.SetInput("Bias", {bias_name});
    new_op_desc.SetInput("ResidualData", {});
    new_op_desc.SetAttr("activation", act_type);
    new_op_desc.SetOutput("Output", {output_name});
    new_op_desc.SetAttr("is_test", true);
    new_op_desc.SetAttr("use_cudnn", false);
    auto* elementwise_add_op_desc = elementwise_add_op->Op();
    auto out_threshold_attr =
        elementwise_add_op_desc->GetNullableAttr("out_threshold");
    // set the out_threshold of the elementwise add op to be the out_threshold
    // of the conv2d_fusion
    if (out_threshold_attr.which()) {
      new_op_desc.SetAttr("out_threshold", out_threshold_attr);
    }
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto* new_conv_op = graph->CreateOpNode(&new_op_desc);

    // Link inputs and outputs.
    PADDLE_ENFORCE_NE(
        subgraph.count(x), 0,
        platform::errors::NotFound("Detector did not find input x of conv2d."));
    auto* conv_in_node = subgraph.at(x);

    IR_NODE_LINK_TO(conv_in_node, new_conv_op);          // Input
    IR_NODE_LINK_TO(conv_filter, new_conv_op);           // Filter
    IR_NODE_LINK_TO(elementwise_add_in_y, new_conv_op);  // Bias
    IR_NODE_LINK_TO(new_conv_op, elementwise_add_out);   // Output

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, {conv_op, conv_out, elementwise_add_op});
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_elementwise_add_fuse_pass,
              paddle::framework::ir::ConvElementwiseAddFusePass);
REGISTER_PASS_CAPABILITY(conv_elementwise_add_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1));
