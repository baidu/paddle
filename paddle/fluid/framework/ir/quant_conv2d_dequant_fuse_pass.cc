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

namespace paddle {
namespace framework {
namespace ir {

void RunQuantDequant(ir::Graph* graph, Scope* scope, int times,
                     const std::string& op_type, const std::string& quant_type,
                     const std::string& dequant_type,
                     bool use_trt_or_anakin = false) {
  const std::string pattern_name = "quant_dequant_fuse";
  int kNumFields = 5;
  const int kQuantizedWeightOffset = 0;
  const int kQuantizedOpOffset = 1;
  const int kQuantizedOpOutOffset = 2;
  const int kDequantOpOffset = 3;
  const int kDequantOpOutOffset = 4;
  const int kDequantOpWeightScaleOffset = 5;

  if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
    kNumFields += 1;
  }

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("x")
                ->assert_is_op_input(quant_type, "X")
                ->AsInput();

  std::string quantized_op_type = op_type;
  std::string weight_name = "";
  if (op_type == "conv2d" || op_type == "depthwise_conv2d" ||
      op_type == "conv2d_fusion") {
    weight_name = "Filter";
  } else if (op_type == "mul") {
    weight_name = "Y";
  } else if (op_type == "fc") {
    weight_name = "W";
  } else {
    PADDLE_ENFORCE(
        "QuantDequantFuse: We only support conv2d, conv2d_fusion, fc, mul for "
        "now.");
  }

  patterns::QuantDequantOpFuse pattern(gpd.mutable_pattern(), pattern_name);
  pattern(x, quantized_op_type, weight_name, times, quant_type, dequant_type);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    PADDLE_ENFORCE(subgraph.count(x));
    auto* input_node = subgraph.at(x);
    Node* quant_op_in_scale =
        subgraph.at(pattern.GetPDNode("quant_op_in_scale"));
    Node* quant_op = subgraph.at(pattern.GetPDNode("quant_op"));
    Node* quant_op_out_scale =
        subgraph.at(pattern.GetPDNode("quant_op_out_scale"));
    Node* quant_op_out = subgraph.at(pattern.GetPDNode("quant_op_out"));

    std::vector<Node*> nodes;
    for (int i = 0; i < times; i++) {
      nodes.push_back(subgraph.at(
          pattern.GetPDNode("quantized_op_weight" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("quantized_op" + std::to_string(i))));
      nodes.push_back(subgraph.at(
          pattern.GetPDNode("quantized_op_out" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("dequant_op" + std::to_string(i))));
      nodes.push_back(
          subgraph.at(pattern.GetPDNode("dequant_op_out" + std::to_string(i))));
      if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
        nodes.push_back(subgraph.at(
            pattern.GetPDNode("dequant_channel_scale" + std::to_string(i))));
      }
    }

    int bit_length = boost::get<int>(quant_op->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);
    // Prepare input scale
    std::string input_scale_var_name = quant_op->Op()->Input("InScale").front();
    PADDLE_ENFORCE(scope);
    const LoDTensor& input_scale_tensor =
        scope->FindVar(input_scale_var_name)->Get<LoDTensor>();

    PADDLE_ENFORCE(paddle::platform::is_cpu_place(input_scale_tensor.place()));
    const float* input_scale_data = input_scale_tensor.data<float>();
    float input_scale = input_scale_data[0];
    std::unordered_set<const Node*> delete_nodes;

    for (int i = 0; i < times; i++) {
      std::vector<float> weight_scale;

      // Get weight scale from dequant op.
      if (dequant_type == "fake_channel_wise_dequantize_max_abs") {
        auto scales_name =
            nodes[i * kNumFields + kDequantOpOffset]->Op()->Input("Scales");
        PADDLE_ENFORCE(scales_name.size() == 2);
        const LoDTensor& channel_scale_tensor =
            scope->FindVar(scales_name[0])->Get<LoDTensor>();
        PADDLE_ENFORCE(
            paddle::platform::is_cpu_place(channel_scale_tensor.place()));
        const float* channel_scale_data = channel_scale_tensor.data<float>();
        for (int i = 0; i < channel_scale_tensor.numel(); i++) {
          weight_scale.push_back(channel_scale_data[i]);
        }
        delete_nodes.insert(
            nodes[i * kNumFields + kDequantOpWeightScaleOffset]);
      } else {
        float max_range = boost::get<float>(
            nodes[i * kNumFields + kDequantOpOffset]->Op()->GetAttr(
                "max_range"));
        weight_scale.push_back((range * range) / max_range);
      }

      // If not runing under trt or anakin, modify the weight type from float to
      // int8
      if (!use_trt_or_anakin) {
        auto quantized_op_weight =
            nodes[i * kNumFields + kQuantizedWeightOffset];
        auto* weight_tensor =
            scope->Var(quantized_op_weight->Name())->GetMutable<LoDTensor>();
        LoDTensor temp_tensor;
        framework::TensorCopy(*weight_tensor, platform::CPUPlace(),
                              &temp_tensor);
        const float* temp_weight_data = temp_tensor.data<float>();
        int8_t* quantized_weight_data =
            weight_tensor->mutable_data<int8_t>(platform::CPUPlace());
        for (int i = 0; i < temp_tensor.numel(); i++) {
          quantized_weight_data[i] = static_cast<int8_t>(temp_weight_data[i]);
        }
      }

      // create new op_desc
      auto base_op_desc =
          *nodes[i * kNumFields + kQuantizedOpOffset]->Op()->Proto();
      std::string new_input = input_node->Name();
      std::string new_output =
          nodes[i * kNumFields + kDequantOpOutOffset]->Name();

      framework::OpDesc new_op_desc(base_op_desc, nullptr);
      new_op_desc.SetType(quantized_op_type);
      new_op_desc.SetAttr("enable_int8", true);

      if (quantized_op_type == "conv2d" ||
          quantized_op_type == "conv2d_fusion" ||
          quantized_op_type == "depthwise_conv2d") {
        new_op_desc.SetInput("Input", {new_input});
        new_op_desc.SetAttr("Input_scale", input_scale);
        new_op_desc.SetOutput("Output", {new_output});
      } else if (quantized_op_type == "fc") {
        new_op_desc.SetInput("Input", {new_input});
        new_op_desc.SetAttr("Input_scale", input_scale);
        new_op_desc.SetOutput("Out", {new_output});
      } else if (quantized_op_type == "mul") {
        new_op_desc.SetInput("X", {new_input});
        new_op_desc.SetAttr("X_scale", input_scale);
        new_op_desc.SetOutput("Out", {new_output});
      }

      new_op_desc.SetAttr("weight_scale", weight_scale);
      new_op_desc.Flush();
      auto* new_op = graph->CreateOpNode(&new_op_desc);
      IR_NODE_LINK_TO(input_node, new_op);
      IR_NODE_LINK_TO(nodes[i * kNumFields + kQuantizedWeightOffset], new_op);
      IR_NODE_LINK_TO(new_op, nodes[i * kNumFields + kDequantOpOutOffset]);

      delete_nodes.insert(nodes[i * kNumFields + kQuantizedOpOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kQuantizedOpOutOffset]);
      delete_nodes.insert(nodes[i * kNumFields + kDequantOpOffset]);
    }

    delete_nodes.insert(quant_op_in_scale);
    delete_nodes.insert(quant_op);
    delete_nodes.insert(quant_op_out);
    delete_nodes.insert(quant_op_out_scale);
    // Delete the unneeded nodes.
    GraphSafeRemoveNodes(graph, delete_nodes);
  };
  gpd(graph, handler);
}

void QuantDequantFusePass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "quant_dequant_fuse";
  FusePassBase::Init(pattern_name, graph);

  const bool use_trt_or_anakin = Get<bool>("use_trt_or_anakin");

  std::unordered_set<std::string> dequant_types = {
      "fake_dequantize_max_abs", "fake_channel_wise_dequantize_max_abs"};
  std::unordered_set<std::string> quant_types = {
      "fake_quantize_range_abs_max", "fake_quantize_moving_average_abs_max"};

  std::unordered_set<std::string> quantized_op_types;
  if (use_trt_or_anakin) {
    quantized_op_types = {"conv2d", "mul", "depthwise_conv2d"};
  } else {
    quantized_op_types = {"mul"};
  }

  auto* scope = param_scope();

  for (auto& dequant_type : dequant_types) {
    for (auto& quant_type : quant_types) {
      for (auto& op_type : quantized_op_types) {
        for (int i = 6; i >= 1; i--) {
          RunQuantDequant(graph, scope, i, op_type, quant_type, dequant_type,
                          use_trt_or_anakin);
        }
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(quant_conv2d_dequant_fuse_pass,
              paddle::framework::ir::QuantDequantFusePass);
