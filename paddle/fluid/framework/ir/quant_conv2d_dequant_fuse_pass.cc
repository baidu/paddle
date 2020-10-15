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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/ir/graph_viz_pass.h"
#include "paddle/fluid/framework/ir/quant_conv2d_dequant_fuse_pass.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

// Delete quant op before quantized ops, and set input scale in the attr of
// quantized ops
void DeleteQuant(ir::Graph* graph, Scope* scope,
                 const std::string& quant_type) {
  const std::string pattern_name = "delete_quant_fuse";
  GraphPatternDetector gpd;
  auto* input_act_node = gpd.mutable_pattern()
                             ->NewNode("input_act_node")
                             ->assert_is_op_input(quant_type, "X")
                             ->AsInput();

  // Create pattern
  patterns::DeleteQuantOpFuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(input_act_node, quant_type);

  // extract input scale from quant op input to set it in attr of all quantized
  // ops linked from it
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    PADDLE_ENFORCE_EQ(
        subgraph.count(input_act_node), true,
        platform::errors::NotFound(
            "Input act node(%s) not found in QuantDequantFuse pass.",
            input_act_node->name()));
    Node* input_act = subgraph.at(input_act_node);
    Node* input_scale = subgraph.at(pattern.GetPDNode("input_scale_node"));
    Node* quant = subgraph.at(pattern.GetPDNode("quant_node"));
    Node* output_scale = subgraph.at(pattern.GetPDNode("output_scale_node"));
    Node* output_act = subgraph.at(pattern.GetPDNode("output_act_node"));
    int bit_length = BOOST_GET_CONST(int, quant->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);

    // Get input scale from tensor
    std::string input_scale_var_name = quant->Op()->Input("InScale").front();
    PADDLE_ENFORCE_NOT_NULL(
        scope, platform::errors::InvalidArgument(
                   "Scope in QuantDequantFuse pass should not be null."));
    const LoDTensor& input_scale_tensor =
        scope->FindVar(input_scale_var_name)->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        paddle::platform::is_cpu_place(input_scale_tensor.place()), true,
        platform::errors::InvalidArgument(
            "Input scale tensor's place should be CPU."));
    const float* input_scale_data = input_scale_tensor.data<float>();
    float in_scale = input_scale_data[0];
    float scale_value = in_scale / range;

    // Set input scale in attr, and relink nodes
    std::string input_act_name = input_act->Var()->Name();
    std::string output_act_name = output_act->Var()->Name();
    auto outlinks = output_act->outputs;
    for (auto* quantized_node : outlinks) {
      auto op_desc = quantized_node->Op();
      std::string quantized_op_type = op_desc->Type();
      if (quantized_op_type == "conv2d" ||
          quantized_op_type == "conv2d_fusion" ||
          quantized_op_type == "depthwise_conv2d" ||
          quantized_op_type == "fc" ||
          quantized_op_type == "conv2d_transpose") {
        op_desc->SetAttr("Input_scale", scale_value);
      } else if (quantized_op_type == "mul") {
        op_desc->SetAttr("X_scale", scale_value);
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "Unsupported quantized op type %s.", quantized_op_type));
      }
      op_desc->SetAttr("bit_length", bit_length);
      op_desc->RenameInput(output_act_name, input_act_name);
      op_desc->Flush();
      IR_NODE_LINK_TO(input_act, quantized_node);
    }
    // Delete nodes and edges
    std::unordered_set<const Node*> nodes2rm = {input_scale, quant,
                                                output_scale, output_act};
    GraphSafeRemoveNodes(graph, nodes2rm);
  };
  gpd(graph, handler);
}

// Delete dequant op after quantized ops, and convert weight from fp32 range to
// int8 range
void FuseDequant(ir::Graph* graph, Scope* scope,
                 const std::string& quantized_op_type,
                 const std::string& dequant_type) {
  std::string weight_name = "";
  std::string input_name = "";
  if (quantized_op_type == "conv2d" ||
      quantized_op_type == "depthwise_conv2d" ||
      quantized_op_type == "conv2d_fusion" ||
      quantized_op_type == "conv2d_transpose") {
    weight_name = "Filter";
    input_name = "Input";
  } else if (quantized_op_type == "mul") {
    weight_name = "Y";
    input_name = "X";
  } else if (quantized_op_type == "fc") {
    weight_name = "W";
    input_name = "Input";
  } else {
    PADDLE_THROW(platform::errors::Unimplemented(
        "QuantDequantFuse: We only support conv2d, conv2d_fusion, "
        "conv2d_transpose, fc, mul for "
        "now."));
  }
  const std::string pattern_name = "dequant_fuse";
  GraphPatternDetector gpd;

  auto* quantized_op_input =
      gpd.mutable_pattern()
          ->NewNode("quantized_op_input")
          ->assert_is_op_input(quantized_op_type, input_name)
          ->AsInput();

  // Create pattern
  patterns::DequantOpFuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(quantized_op_input, quantized_op_type, dequant_type, weight_name);

  // Create new op desc
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    PADDLE_ENFORCE_EQ(
        subgraph.count(quantized_op_input), true,
        platform::errors::NotFound("Quantized op input node(%s) did not find "
                                   "in QuantDequantFuse pass.",
                                   quantized_op_input->name()));
    Node* quantized_op_input_node = subgraph.at(quantized_op_input);
    Node* quantized_op_weight_node =
        subgraph.at(pattern.GetPDNode("quantized_op_weight"));
    Node* quantized_op_node = subgraph.at(pattern.GetPDNode("quantized_op"));
    Node* dequant_op_node = subgraph.at(pattern.GetPDNode("dequant_op"));
    Node* dequant_op_out_node =
        subgraph.at(pattern.GetPDNode("dequant_op_out"));

    std::unordered_set<const Node*> nodes2rm = {};
    int bit_length =
        BOOST_GET_CONST(int, quantized_op_node->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);
    std::vector<float> weight_scale;

    // Get weight scale
    if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
      Node* dequant_channel_scale_node =
          subgraph.at(pattern.GetPDNode("dequant_channel_scale"));
      auto scales_name = dequant_op_node->Op()->Input("Scales");
      PADDLE_ENFORCE_EQ(
          scales_name.size(), 2,
          platform::errors::InvalidArgument(
              "Scales size in channel-wise dequantize op should be 2, got %d.",
              scales_name.size()));
      const LoDTensor& channel_scale_tensor =
          scope->FindVar(scales_name[0])->Get<LoDTensor>();
      PADDLE_ENFORCE_EQ(
          paddle::platform::is_cpu_place(channel_scale_tensor.place()), true,
          platform::errors::InvalidArgument(
              "Channel scale tensor's place should be CPU."));
      const float* channel_scale_data = channel_scale_tensor.data<float>();
      for (int i = 0; i < channel_scale_tensor.numel(); i++) {
        weight_scale.push_back(channel_scale_data[i] / range);
      }
      nodes2rm.insert(dequant_channel_scale_node);
    } else {
      float max_range =
          BOOST_GET_CONST(float, dequant_op_node->Op()->GetAttr("max_range"));
      weight_scale.push_back((range * range) / max_range / range);
    }

    // Convert weight to fp32 range
    auto* weight_tensor =
        scope->Var(quantized_op_weight_node->Name())->GetMutable<LoDTensor>();
    auto w_dims = weight_tensor->dims();
    // If quantized op is fc, weight scale size = 1;
    // If quantized op is conv2d, weight scale size = weight dims[0]
    // If quantized op is conv2d_transpose, weight scale size = weight dims[1]
    bool valid_scale_size =
        (weight_scale.size() == 1 ||
         weight_scale.size() == static_cast<size_t>(w_dims[0]) ||
         weight_scale.size() == static_cast<size_t>(w_dims[1]));
    PADDLE_ENFORCE_EQ(
        valid_scale_size, true,
        platform::errors::InvalidArgument(
            "TRT int8 quant: invalid scale size(%d).", weight_scale.size()));
    float* quantized_weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());
    for (int j = 0; j < weight_tensor->numel(); j++) {
      if (weight_scale.size() == 1) {
        quantized_weight_data[j] *= weight_scale[0];
      } else {
        if (quantized_op_type == "conv2d_transpose") {
          int inner_size = w_dims[2] * w_dims[3];
          quantized_weight_data[j] *=
              weight_scale[(j / inner_size) % w_dims[1]];
        } else {
          int inner_size = w_dims[1] * w_dims[2] * w_dims[3];
          quantized_weight_data[j] *= weight_scale[j / inner_size];
        }
      }
    }

    // create new op_desc
    auto base_op_desc = *quantized_op_node->Op()->Proto();
    std::string new_input = quantized_op_input_node->Name();
    std::string new_output = dequant_op_out_node->Name();

    framework::OpDesc new_op_desc(base_op_desc, nullptr);
    new_op_desc.SetType(quantized_op_type);
    new_op_desc.SetAttr("enable_int8", true);
    if (quantized_op_type == "conv2d" || quantized_op_type == "conv2d_fusion" ||
        quantized_op_type == "depthwise_conv2d" ||
        quantized_op_type == "conv2d_transpose") {
      new_op_desc.SetInput("Input", {new_input});
      new_op_desc.SetOutput("Output", {new_output});
    } else if (quantized_op_type == "fc") {
      new_op_desc.SetInput("Input", {new_input});
      new_op_desc.SetOutput("Out", {new_output});
    } else if (quantized_op_type == "mul") {
      new_op_desc.SetInput("X", {new_input});
      new_op_desc.SetOutput("Out", {new_output});
    }
    new_op_desc.SetAttr("weight_scale", weight_scale);
    new_op_desc.Flush();
    auto* new_op = graph->CreateOpNode(&new_op_desc);
    IR_NODE_LINK_TO(quantized_op_input_node, new_op);
    IR_NODE_LINK_TO(quantized_op_weight_node, new_op);
    IR_NODE_LINK_TO(new_op, dequant_op_out_node);
    // Delete nodes and edges
    nodes2rm.insert(quantized_op_node);
    nodes2rm.insert(dequant_op_node);
    GraphSafeRemoveNodes(graph, nodes2rm);
  };
  gpd(graph, handler);
}

void QuantDequantFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "quant_dequant_fuse";
  FusePassBase::Init(pattern_name, graph);

  std::unordered_set<std::string> dequant_types = {
      "fake_channel_wise_dequantize_max_abs", "fake_dequantize_max_abs"};
  std::unordered_set<std::string> quant_types = {
      "fake_quantize_range_abs_max", "fake_quantize_moving_average_abs_max"};
  std::unordered_set<std::string> quantized_op_types = {
      "conv2d", "mul", "depthwise_conv2d", "fc", "conv2d_transpose"};
  auto* scope = param_scope();

  for (auto& quant_type : quant_types) {
    DeleteQuant(graph, scope, quant_type);
  }
  for (auto& dequant_type : dequant_types) {
    for (auto& quantized_op_type : quantized_op_types) {
      FuseDequant(graph, scope, quantized_op_type, dequant_type);
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_conv2d_dequant_fuse_pass,
              paddle::framework::ir::QuantDequantFusePass);

REGISTER_PASS_CAPABILITY(tensorrt_subgraph_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("conv2d", 0)
            .EQ("fc", 0)
            .LE("conv2d_transpose", 1)
            .EQ("fake_quantize_abs_max", 0)
            .EQ("fake_quantize_range_abs_max", 0)
            .EQ("fake_quantize_moving_average_abs_max", 0)
            .EQ("fake_channel_wise_quantize_abs_max", 0)
            .EQ("fake_dequantize_max_abs", 0));
