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

#include "paddle/fluid/framework/ir/conv_affine_channel_fuse_pass.h"

#include <cmath>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Node;

#define GET_CONV_BN_NODES(pattern_name)                                    \
  /* OPERATORS */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern_name);                     \
  GET_IR_NODE_FROM_SUBGRAPH(affine_channel, affine_channel, pattern_name); \
  /* CONV inputs */                                                        \
  GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, pattern_name);       \
  /* CONV outputs */                                                       \
  GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, pattern_name);             \
  /* Affine Channel inputs */                                              \
  GET_IR_NODE_FROM_SUBGRAPH(ac_scale, ac_scale, pattern_name);             \
  GET_IR_NODE_FROM_SUBGRAPH(ac_bias, ac_bias, pattern_name);               \
  /* Affine channel outputs */                                             \
  GET_IR_NODE_FROM_SUBGRAPH(ac_out, ac_out, pattern_name); /* Out */

void recompute_bias_and_weights(const Scope* scope, ir::Node* conv_weight,
                                const ir::Node& ac_scale,
                                const LoDTensor& ac_bias_tensor,
                                LoDTensor* eltwise_y_in_tensor) {
  using EigenVectorArrayMap =
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
  using ConstEigenVectorArrayMap =
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
  using EigenMatrixArrayMap = Eigen::Map<
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Re-compute bias of conv2d from AffineChannel
  PADDLE_ENFORCE_EQ(
      eltwise_y_in_tensor->dims(), ac_bias_tensor.dims(),
      platform::errors::InvalidArgument(
          "Tensor elementwise y(%d) and activation bias(%d) must have same "
          "dimension.",
          eltwise_y_in_tensor->dims().size(), ac_bias_tensor.dims().size()));

  auto* scale_tensor = scope->FindVar(ac_scale.Name())->GetMutable<LoDTensor>();

  ConstEigenVectorArrayMap scale_array(scale_tensor->data<float>(),
                                       scale_tensor->numel(), 1);
  ConstEigenVectorArrayMap ac_bias_array(ac_bias_tensor.data<float>(),
                                         ac_bias_tensor.numel(), 1);

  EigenVectorArrayMap eltwise_y_in_array(
      eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
      eltwise_y_in_tensor->numel(), 1);

  eltwise_y_in_array = (eltwise_y_in_array * scale_array) + ac_bias_array;

  // Re-compute weight of conv2d from AffineChannel
  auto* weights = scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
  auto weights_shape = weights->dims();
  auto weights_shape_2d = flatten_to_2d(weights_shape, 1);
  auto* weights_data = weights->mutable_data<float>(platform::CPUPlace());

  EigenMatrixArrayMap weights_array_2d(weights_data, weights_shape_2d[0],
                                       weights_shape_2d[1]);

  weights_array_2d.colwise() *= scale_array;

  // Check for subnormal values that slows down convolution execution
  for (int i = 0; i < weights->numel(); ++i) {
    if (std::fpclassify(weights_data[i]) == FP_SUBNORMAL) weights_data[i] = 0;
  }
}

ConvAffineChannelFusePass::ConvAffineChannelFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
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
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC"})
      .End();

  AddOpCompat(OpCompat("affine_channel"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("data_layout")
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

void ConvAffineChannelFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvAffineChannel conv_ac_pattern(gpd.mutable_pattern(),
                                              name_scope_);
  conv_ac_pattern(conv_input, false /*with_eltwise_add*/);

  int found_conv_ac_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "ConvAffineChannelFusePass in op compat failed.";
      return;
    }

    VLOG(4) << "handle ConvAffineChannel fuse";

    GET_CONV_BN_NODES(conv_ac_pattern);

    // Create eltwise_y (conv bias) variable
    VarDesc eltwise_y_in_desc(
        patterns::PDNodeName(name_scope_, "eltwise_y_in"));
    eltwise_y_in_desc.SetPersistable(true);
    auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);
    auto* eltwise_y_in_tensor =
        scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();

    // Get affine_channel bias
    auto* ac_bias_tensor =
        scope->FindVar(ac_bias->Name())->GetMutable<LoDTensor>();

    // Initialize eltwise_y
    eltwise_y_in_tensor->Resize(ac_bias_tensor->dims());
    std::fill_n(eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
                eltwise_y_in_tensor->numel(), 0.0f);

    // update weights and biases
    recompute_bias_and_weights(scope, conv_weight, *ac_scale, *ac_bias_tensor,
                               eltwise_y_in_tensor);

    // create an elementwise add node.
    OpDesc desc;
    desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
    desc.SetInput("Y", std::vector<std::string>({eltwise_y_in_node->Name()}));
    desc.SetOutput("Out", std::vector<std::string>({ac_out->Name()}));
    desc.SetType("elementwise_add");
    desc.SetAttr("axis", 1);
    desc.SetAttr("use_mkldnn", conv->Op()->GetAttrIfExists<bool>("use_mkldnn"));

    if (!IsCompat(desc)) {
      LOG(WARNING) << "ConvAffineChannelFusePass in out fc op compat failed.";
      return;
    }

    auto eltwise_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

    GraphSafeRemoveNodes(graph, {ac_scale, ac_bias, affine_channel});

    IR_NODE_LINK_TO(conv_out, eltwise_op);
    IR_NODE_LINK_TO(eltwise_y_in_node, eltwise_op);
    IR_NODE_LINK_TO(eltwise_op, ac_out);
    found_conv_ac_count++;
  };

  gpd(graph, handler);

  AddStatis(found_conv_ac_count);
}

ConvEltwiseAddAffineChannelFusePass::ConvEltwiseAddAffineChannelFusePass() {
  AddOpCompat(OpCompat("conv2d"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("ResidualData")
      .IsTensor()
      .IsOptional()
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
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .IsType<std::vector<int>>()
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC"})
      .End();

  AddOpCompat(OpCompat("affine_channel"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("data_layout")
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
      .IsIntIn({1, -1})
      .End();
}

void ConvEltwiseAddAffineChannelFusePass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));

  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvAffineChannel conv_ac_pattern(gpd.mutable_pattern(),
                                              name_scope_);
  conv_ac_pattern(conv_input, true /*with_eltwise_add*/);

  int found_conv_ac_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING)
          << "ConvEltwiseAddAffineChannelFusePass in op compat failed.";
      return;
    }

    VLOG(4) << "handle ConvBN fuse";

    GET_CONV_BN_NODES(conv_ac_pattern);
    // OPERATORS
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_ac_pattern);
    // BIAS inputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_y_in, eltwise_y_in, conv_ac_pattern);
    // BIAS outputs
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_ac_pattern);

    // Get eltwise_y (conv bias) variable
    auto* eltwise_y_in_tensor =
        scope->FindVar(eltwise_y_in->Name())->GetMutable<LoDTensor>();

    // Get batch norm bias
    auto* ac_bias_tensor =
        scope->FindVar(ac_bias->Name())->GetMutable<LoDTensor>();

    recompute_bias_and_weights(scope, conv_weight, *ac_scale, *ac_bias_tensor,
                               eltwise_y_in_tensor);

    // Update the elementwise_add node
    eltwise->Op()->SetAttr("axis", 1);
    eltwise->Op()->SetOutput("Out", std::vector<std::string>({ac_out->Name()}));

    GraphSafeRemoveNodes(graph,
                         {ac_scale, ac_bias, affine_channel, eltwise_out});

    IR_NODE_LINK_TO(eltwise, ac_out);

    found_conv_ac_count++;
  };

  gpd(graph, handler);
  AddStatis(found_conv_ac_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_affine_channel_fuse_pass,
              paddle::framework::ir::ConvAffineChannelFusePass);
REGISTER_PASS(conv_eltwiseadd_affine_channel_fuse_pass,
              paddle::framework::ir::ConvEltwiseAddAffineChannelFusePass);
REGISTER_PASS_CAPABILITY(conv_affine_channel_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .EQ("affine_channel", 0));
REGISTER_PASS_CAPABILITY(conv_eltwiseadd_affine_channel_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1)
            .EQ("affine_channel", 0));
