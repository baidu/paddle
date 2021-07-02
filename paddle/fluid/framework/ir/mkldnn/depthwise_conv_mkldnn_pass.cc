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

#include "paddle/fluid/framework/ir/mkldnn/depthwise_conv_mkldnn_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

#define GET_NODE(id, pattern)                                     \
  PADDLE_ENFORCE_NE(subgraph.count(pattern.RetrieveNode(#id)), 0, \
                    platform::errors::InvalidArgument(            \
                        "Pattern has no Node called %s.", #id));  \
  auto* id = subgraph.at(pattern.RetrieveNode(#id));              \
  PADDLE_ENFORCE_NOT_NULL(                                        \
      id, platform::errors::InvalidArgument("Subgraph has no node %s.", #id));

DepthwiseConvMKLDNNPass::DepthwiseConvMKLDNNPass() {
  AddOpCompat(OpCompat("depthwise_conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddInput("ResidualData")
      .IsOptional()
      .IsTensor()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("paddings")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("padding_algorithm")
      // mobilenet-ssd has no "padding_algorithm"
      .IsOptional()
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NHWC", "NCHW", "AnyLayout"})
      .End();
}

void DepthwiseConvMKLDNNPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init("depthwise_conv_mkldnn_pass", graph);
  GraphPatternDetector gpd;

  auto* pattern = gpd.mutable_pattern();
  pattern->NewNode("depthwise_conv")
      ->assert_is_op("depthwise_conv2d")
      ->assert_op_attr("use_mkldnn", true);

  int found_depthwise_conv_mkldnn_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass op compat failed.";
      return;
    }
    VLOG(3) << "handle DepthwiseConvMKLDNN fuse";
    GET_NODE(depthwise_conv, (*pattern));
    depthwise_conv->Op()->SetType("conv2d");
    found_depthwise_conv_mkldnn_count++;
  };

  gpd(graph, handler);
  AddStatis(found_depthwise_conv_mkldnn_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(depthwise_conv_mkldnn_pass,
              paddle::framework::ir::DepthwiseConvMKLDNNPass);
REGISTER_PASS_CAPABILITY(depthwise_conv_mkldnn_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination().LE(
            "depthwise_conv2d", 1));
