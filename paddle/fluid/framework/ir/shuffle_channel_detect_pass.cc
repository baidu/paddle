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

#include <string>

#include "paddle/fluid/framework/ir/shuffle_channel_detect_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES             \
  GET_IR_NODE(reshape1_op);   \
  GET_IR_NODE(reshape1_out);  \
  GET_IR_NODE(transpose_op);  \
  GET_IR_NODE(transpose_out); \
  GET_IR_NODE(reshape2_op);   \
  GET_IR_NODE(reshape2_out);

void ShuffleChannelDetectPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "shufflechannel_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input("reshape2", "X")
                ->AsInput();

  patterns::ShuffleChannelPattern pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    PADDLE_ENFORCE_GT(
        subgraph.count(x), 0,
        platform::errors::NotFound("Detector did not find input X."));
    auto* input_node = subgraph.at(x);
    auto reshape1_desc = reshape1_op->Op();
    auto reshape2_desc = reshape2_op->Op();
    std::string input_name = input_node->Name();
    std::string output_name = reshape2_out->Name();

    auto reshape1_shape =
        BOOST_GET_CONST(std::vector<int>, reshape1_desc->GetAttr("shape"));
    auto reshape2_shape =
        BOOST_GET_CONST(std::vector<int>, reshape2_desc->GetAttr("shape"));

    int i_c = reshape1_shape[2];
    int o_c = reshape2_shape[1];
    int group = o_c / i_c;

    framework::OpDesc new_op_desc;
    new_op_desc.SetType("shuffle_channel");
    new_op_desc.SetInput("X", {input_name});
    new_op_desc.SetOutput("Out", {output_name});

    new_op_desc.SetAttr("group", group);
    new_op_desc.Flush();

    // Create a new node for the fused op.
    auto* new_op = graph->CreateOpNode(&new_op_desc);

    IR_NODE_LINK_TO(input_node, new_op);
    IR_NODE_LINK_TO(new_op, reshape2_out);

    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, {reshape1_op, reshape1_out, transpose_op,
                                 transpose_out, reshape2_op});
    LOG_FIRST_N(WARNING, 1)
        << "There is fluid.layers.shuffle_channel API already, maybe you can "
           "use it instead of (reshape + transpose + reshape)";
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(shuffle_channel_detect_pass,
              paddle::framework::ir::ShuffleChannelDetectPass);
REGISTER_PASS_CAPABILITY(shuffle_channel_detect_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("reshape2", 0)
            .EQ("transpose2", 0));
