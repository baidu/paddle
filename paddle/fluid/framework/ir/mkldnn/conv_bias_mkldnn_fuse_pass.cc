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

#include "paddle/fluid/framework/ir/mkldnn/conv_bias_mkldnn_fuse_pass.h"

#include <functional>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

ConvBiasFusePass::ConvBiasFusePass() {
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
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC"})
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
      .IsNumEQ(-1)
      .End();
}

Conv2DTransposeBiasFusePass::Conv2DTransposeBiasFusePass() {
  AddOpCompat(OpCompat("conv2d_transpose"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("Filter")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Output")
      .IsTensor()
      .End()
      .AddAttr("output_padding")
      .End()
      .AddAttr("output_size")
      .IsNumGE(1)
      .End()
      .AddAttr("groups")
      .IsNumGE(1)
      .End()
      .AddAttr("dilations")
      .End()
      .AddAttr("strides")
      .End()
      .AddAttr("paddings")
      .End()
      .AddAttr("padding_algorithm")
      .IsStringIn({"EXPLICIT", "SAME", "VALID"})
      .End()
      .AddAttr("data_format")
      .IsStringIn({"NCHW", "NHWC"})
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
      .IsNumEQ(-1)
      .End();
}

template <typename BinaryOperation>
LoDTensor tensor_apply_eltwise(const LoDTensor& vec_a, const LoDTensor& vec_b,
                               BinaryOperation f) {
  PADDLE_ENFORCE_EQ(vec_a.dims(), vec_b.dims(),
                    platform::errors::InvalidArgument(
                        "Input two tensors must have same shape, but they are "
                        "different: %s, %s.",
                        vec_a.dims(), vec_b.dims()));
  LoDTensor vec_y;
  vec_y.Resize(vec_a.dims());
  const float* a = vec_a.data<float>();
  const float* b = vec_b.data<float>();
  float* y = vec_y.mutable_data<float>(platform::CPUPlace());
  for (int i = 0; i < vec_a.numel(); i++) {
    y[i] = f(a[i], b[i]);
  }
  return vec_y;
}

void ConvBiasFusePass::ApplyImpl(ir::Graph* graph) const {
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
          ->assert_is_op_input(type(), "Input");
  patterns::ConvBias conv_bias_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bias_pattern(conv_input, type());
  int found_conv_bias_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBias fuse";
    GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight,
                              conv_bias_pattern);                      // Filter
    GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, conv_bias_pattern);  // tmp
    GET_IR_NODE_FROM_SUBGRAPH(conv, conv, conv_bias_pattern);  // CONV op
    // bias
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_bias, eltwise_bias, conv_bias_pattern);
    // output
    GET_IR_NODE_FROM_SUBGRAPH(eltwise_out, eltwise_out, conv_bias_pattern);
    // elementwise_add op
    GET_IR_NODE_FROM_SUBGRAPH(eltwise, eltwise, conv_bias_pattern);

    PADDLE_ENFORCE_NE(
        subgraph.count(conv_input), 0,
        platform::errors::NotFound("Detector did not find conv input."));

    // check compat
    if (!IsCompat(subgraph, g)) {
      VLOG(3) << "Pass in op compat failed.";
      return;
    }

    // check if fuse can be done and if MKL-DNN should be used
    FuseOptions fuse_option = FindFuseOption(*conv, *eltwise);
    if (fuse_option == DO_NOT_FUSE || fuse_option == FUSE_NATIVE) {
      VLOG(3) << "do not perform " + type() + "+bias fuse";
      return;
    }

    auto* eltwise_bias_tensor =
        scope->FindVar(eltwise_bias->Name())->GetMutable<LoDTensor>();

    auto input_names = conv->Op()->InputNames();
    bool has_bias = std::find(input_names.begin(), input_names.end(), "Bias") !=
                    input_names.end();
    if (has_bias && conv->Op()->Input("Bias").size() > 0) {
      auto conv_bias_names = conv->Op()->Input("Bias");
      // add eltwise bias to existing conv bias
      PADDLE_ENFORCE_EQ(conv_bias_names.size(), 1,
                        platform::errors::NotFound("Can not find var Bias."));
      auto* conv_bias_var = scope->FindVar(conv_bias_names[0]);
      auto* conv_bias_tensor = conv_bias_var->GetMutable<LoDTensor>();
      PADDLE_ENFORCE_EQ(
          conv_bias_tensor->dims(), eltwise_bias_tensor->dims(),
          platform::errors::InvalidArgument(
              "Conv bias tensor and eltwise bias tensor "
              "must have same shape, but they are different: %s, %s.",
              conv_bias_tensor->dims(), eltwise_bias_tensor->dims()));
      *conv_bias_tensor = tensor_apply_eltwise(
          *conv_bias_tensor, *eltwise_bias_tensor, std::plus<float>());

      conv->Op()->SetOutput("Output",
                            std::vector<std::string>({eltwise_out->Name()}));

      GraphSafeRemoveNodes(graph, {eltwise, conv_out});

      IR_NODE_LINK_TO(conv, eltwise_out);
    } else {
      // take eltwise bias as conv bias
      OpDesc desc;

      desc.SetInput(
          "Input", std::vector<std::string>({subgraph.at(conv_input)->Name()}));
      desc.SetInput("Filter", std::vector<std::string>({conv_weight->Name()}));
      desc.SetInput("Bias", std::vector<std::string>({eltwise_bias->Name()}));
      desc.SetOutput("Output", std::vector<std::string>({eltwise_out->Name()}));
      desc.SetType(type());

      for (auto& attr : conv->Op()->GetAttrMap()) {
        desc.SetAttr(attr.first, attr.second);
      }
      auto conv_bias_node = g->CreateOpNode(&desc);

      IR_NODE_LINK_TO(subgraph.at(conv_input), conv_bias_node);
      IR_NODE_LINK_TO(conv_weight, conv_bias_node);
      IR_NODE_LINK_TO(eltwise_bias, conv_bias_node);
      IR_NODE_LINK_TO(conv_bias_node, eltwise_out);

      GraphSafeRemoveNodes(graph, {conv, eltwise, conv_out});
    }

    found_conv_bias_count++;
  };
  gpd(graph, handler);
  AddStatis(found_conv_bias_count);
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
REGISTER_PASS(conv_bias_mkldnn_fuse_pass,
              paddle::framework::ir::ConvBiasFusePass);
REGISTER_PASS_CAPABILITY(conv_bias_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d", 1)
            .LE("elementwise_add", 1));

REGISTER_PASS(conv_transpose_bias_mkldnn_fuse_pass,
              paddle::framework::ir::Conv2DTransposeBiasFusePass);
REGISTER_PASS_CAPABILITY(conv_transpose_bias_mkldnn_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .LE("conv2d_transpose", 2)
            .LE("elementwise_add", 1));

REGISTER_PASS(conv3d_bias_mkldnn_fuse_pass,
              paddle::framework::ir::Conv3DBiasFusePass);
