// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_pass.h"

#include <sstream>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using EigenVectorArrayMap = Eigen::Map<Eigen::Array<double, Eigen::Dynamic, 1>>;
using EigenVectorArrayMapFloat =
    Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
using string::PrettyLogDetail;

namespace {

void UnlinkNodes(ir::Node* a, ir::Node* b) {
  a->outputs.erase(std::remove(a->outputs.begin(), a->outputs.end(), b),
                   a->outputs.end());
  b->inputs.erase(std::remove(b->inputs.begin(), b->inputs.end(), a),
                  b->inputs.end());
}

void LogCannotQuantizeOp(Node* op, const char* details = nullptr) {
  std::stringstream msg_ss;
  msg_ss << "Cannot quantize operator " << op->Name()
         << " (type: " << op->Op()->Type() << ", id: " << op->id() << ").";
  if (details) msg_ss << " " << details;
  PrettyLogDetail(msg_ss.str().c_str());
}

void LogScaleIsMissingForVarName(const std::string& name) {
  VLOG(4) << "Quantization scale for the variable " << name << " is missing.";
}

void LogScaleIsMissingForVarNode(Node* node) {
  LogScaleIsMissingForVarName(node->Name());
}

void LogQuantizationDisabled(Node* op) {
  std::stringstream msg_ss;
  VLOG(4) << "Qantization skipped for operator " << op->Name()
          << " (type: " << op->Op()->Type() << ", id: " << op->id()
          << "). Attribute mkldnn_data_type != \"int8\".";
}

}  // namespace

enum { U8_MAX = 255, S8_MAX = 127 };

void CPUQuantizePass::QuantizeInput(Graph* g, Node* op, Node* input,
                                    std::string input_name, double scale_to_one,
                                    bool is_input_unsigned,
                                    std::string scale_attr_name, float shift,
                                    std::string shift_attr_name) const {
  auto inputs = op->Op()->InputNames();
  bool name_found =
      std::find(inputs.begin(), inputs.end(), input_name) != inputs.end();
  PADDLE_ENFORCE_EQ(name_found, true,
                    platform::errors::InvalidArgument(
                        "Var(%s) isn't the input of the %s operator.",
                        input_name, op->Op()->Type()));
  unsigned max = is_input_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_to_one * max;

  // Create quantize output variable
  VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
  auto* quantize_out_node = g->CreateVarNode(&quantize_out_desc);

  // create a quantize op node
  OpDesc q_desc;
  q_desc.SetType("quantize");
  q_desc.SetInput("Input", std::vector<std::string>({input->Name()}));
  q_desc.SetOutput("Output",
                   std::vector<std::string>({quantize_out_node->Name()}));
  q_desc.SetAttr("Scale", scale);
  q_desc.SetAttr("Shift", shift);
  q_desc.SetAttr("is_negative_input", !is_input_unsigned);

  q_desc.SetAttr("output_format",
                 Has("data_layout") ? Get<std::string>("data_layout") : "NHWC");
  auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

  // update op's input
  op->Op()->SetInput(input_name,
                     std::vector<std::string>({quantize_out_node->Name()}));

  // link quantize op
  UnlinkNodes(input, op);
  IR_NODE_LINK_TO(input, quantize_op);
  IR_NODE_LINK_TO(quantize_op, quantize_out_node);
  IR_NODE_LINK_TO(quantize_out_node, op);

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
  if (!shift_attr_name.empty()) op->Op()->SetAttr(shift_attr_name, shift);
}

void CPUQuantizePass::QuantizeInputs(Graph* g, Node* op, std::string input_name,
                                     bool are_inputs_unsigned,
                                     std::string scale_attr_name, float shift,
                                     std::string shift_attr_name) const {
  auto inputs = op->inputs;
  auto output = op->outputs[0];
  PADDLE_ENFORCE_GE(inputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s inputs(%d) must be equal or greater than 1.",
                        op->Name(), inputs.size()));
  PADDLE_ENFORCE_EQ(op->outputs.size(), 1,
                    platform::errors::InvalidArgument(
                        "OP(%s)'s outputs(%d) must be equal to 1.", op->Name(),
                        op->outputs.size()));

  // create a quantize op desc prototype
  OpDesc q_desc;
  q_desc.SetType("quantize");

  std::vector<Node*> quantize_out_nodes(inputs.size());
  std::vector<std::string> quantize_out_node_names(inputs.size());

  double scale_out = GetScaleValueForNode(output);
  unsigned max = are_inputs_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_out * max;

  for (size_t i = 0; i < inputs.size(); i++) {
    // Create quantize output variable
    VarDesc quantize_out_desc(patterns::PDNodeName("quantize", "out"));
    quantize_out_nodes[i] = g->CreateVarNode(&quantize_out_desc);
    quantize_out_node_names[i] = quantize_out_nodes[i]->Name();

    q_desc.SetAttr("Scale", scale);
    q_desc.SetAttr("Shift", shift);
    q_desc.SetInput("Input", std::vector<std::string>({inputs[i]->Name()}));
    q_desc.SetOutput("Output",
                     std::vector<std::string>({quantize_out_node_names[i]}));
    q_desc.SetAttr("is_negative_input", !are_inputs_unsigned);
    auto quantize_op = g->CreateOpNode(&q_desc);  // OpDesc will be copied.

    // link quantize op
    UnlinkNodes(inputs[i], op);
    IR_NODE_LINK_TO(inputs[i], quantize_op);
    IR_NODE_LINK_TO(quantize_op, quantize_out_nodes[i]);
    IR_NODE_LINK_TO(quantize_out_nodes[i], op);
  }

  // update op's input
  op->Op()->SetInput(input_name, quantize_out_node_names);

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
  if (!shift_attr_name.empty()) op->Op()->SetAttr(shift_attr_name, shift);
}

void CPUQuantizePass::DequantizeOutput(Graph* g, Node* op, Node* output,
                                       std::string output_name,
                                       double scale_to_one, bool is_unsigned,
                                       std::string scale_attr_name) const {
  auto outputs = op->Op()->OutputNames();
  bool name_found =
      std::find(outputs.begin(), outputs.end(), output_name) != outputs.end();
  PADDLE_ENFORCE_EQ(name_found, true,
                    platform::errors::InvalidArgument(
                        "Var(%s) isn't the output of the %s operator.",
                        output_name, op->Op()->Type()));
  unsigned max = is_unsigned ? U8_MAX : S8_MAX;
  float scale = scale_to_one * max;

  // Create dequantize input variable
  VarDesc dequantize_in_desc(patterns::PDNodeName("dequantize", "in"));
  auto* dequantize_in_node = g->CreateVarNode(&dequantize_in_desc);

  // create a dequantize op node for output.
  OpDesc deq_desc;
  deq_desc.SetType("dequantize");
  deq_desc.SetInput("Input",
                    std::vector<std::string>({dequantize_in_node->Name()}));
  deq_desc.SetOutput("Output", std::vector<std::string>({output->Name()}));
  deq_desc.SetAttr("Scale", scale);
  auto dequantize_op = g->CreateOpNode(&deq_desc);  // OpDesc will be copied.

  // update op's output
  op->Op()->SetOutput(output_name,
                      std::vector<std::string>({dequantize_in_node->Name()}));

  // link dequantize op
  UnlinkNodes(op, output);
  IR_NODE_LINK_TO(op, dequantize_in_node);
  IR_NODE_LINK_TO(dequantize_in_node, dequantize_op);
  IR_NODE_LINK_TO(dequantize_op, output);

  if (!scale_attr_name.empty()) op->Op()->SetAttr(scale_attr_name, scale);
}

bool CPUQuantizePass::AreScalesPresentForVarNames(
    std::vector<std::string> names) const {
  auto& scales = Get<VarQuantScale>("quant_var_scales");
  bool present = true;
  for (auto name : names) {
    if (scales.find(name) == scales.end()) {
      present = false;
      LogScaleIsMissingForVarName(name);
    }
  }
  return present;
}

bool CPUQuantizePass::AreScalesPresentForNodes(
    std::initializer_list<Node*> nodes) const {
  auto& scales = Get<VarQuantScale>("quant_var_scales");
  bool present = true;
  for (auto node : nodes) {
    if (scales.count(node->Name()) == 0) {
      present = false;
      LogScaleIsMissingForVarNode(node);
    }
  }
  return present;
}

std::pair<bool, LoDTensor> CPUQuantizePass::GetScaleDataByName(
    const std::string& name) const {
  auto& scales = Get<VarQuantScale>("quant_var_scales");
  return scales.at(name);
}

std::pair<bool, LoDTensor> CPUQuantizePass::GetScaleDataForNode(
    const Node* node) const {
  return GetScaleDataByName(node->Name());
}

LoDTensor CPUQuantizePass::GetScaleTensorByName(const std::string& name) const {
  return GetScaleDataByName(name).second;
}

LoDTensor CPUQuantizePass::GetScaleTensorForNode(const Node* node) const {
  return GetScaleDataForNode(node).second;
}

double CPUQuantizePass::GetScaleValueForNode(const Node* node,
                                             bool* is_unsigned) const {
  auto scale_data = GetScaleDataForNode(node);
  if (is_unsigned != nullptr) *is_unsigned = scale_data.first;
  return scale_data.second.data<double>()[0];
}

bool CPUQuantizePass::IsOpDequantized(const Node* node) const {
  return node->Op()->Type() == "dequantize" ||
         platform::HasOpINT8DataType(node->Op());
}

bool CPUQuantizePass::IsOpQuantized(const Node* node) const {
  return node->Op()->Type() == "quantize" ||
         platform::HasOpINT8DataType(node->Op());
}

void CPUQuantizePass::QuantizeConv(Graph* graph,
                                   bool with_residual_data) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::ConvResidual conv_pattern{pattern, name_scope_};
  conv_pattern(with_residual_data);

  int quantize_conv_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize conv2d op";
    GET_IR_NODE_FROM_SUBGRAPH(conv_op, conv_op, conv_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(conv_op->Op())) {
      LogQuantizationDisabled(conv_op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(conv_filter, conv_filter, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_input, conv_input, conv_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(conv_output, conv_output, conv_pattern);

    auto has_output_scale = AreScalesPresentForNodes({conv_output});
    if (with_residual_data && !has_output_scale) {
      LogCannotQuantizeOp(conv_op,
                          "Conv op with ResidualData input cannot be quantized "
                          "without output scale.");
      return;
    }

    if (with_residual_data) {
      GET_IR_NODE_FROM_SUBGRAPH(conv_residual_data, conv_residual_data,
                                conv_pattern);
      if (!AreScalesPresentForNodes(
              {conv_input, conv_filter, conv_residual_data})) {
        LogCannotQuantizeOp(conv_op);
        return;
      }

      bool is_residual_unsigned{false};
      auto residual_scale =
          GetScaleValueForNode(conv_residual_data, &is_residual_unsigned);

      QuantizeInput(g, conv_op, conv_residual_data, "ResidualData",
                    residual_scale, is_residual_unsigned, "Scale_in_eltwise");
    } else {
      if (!AreScalesPresentForNodes({conv_input, conv_filter})) {
        LogCannotQuantizeOp(conv_op);
        return;
      }
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(conv_input, &is_input_unsigned);
    QuantizeInput(g, conv_op, conv_input, "Input", input_scale,
                  is_input_unsigned, "Scale_in");

    auto filter_scale_tensor = GetScaleTensorForNode(conv_filter);
    EigenVectorArrayMap eigen_tensor{filter_scale_tensor.data<double>(),
                                     filter_scale_tensor.numel()};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        filter_scale_tensor.data<double>(),
        filter_scale_tensor.data<double>() + filter_scale_tensor.numel()};

    conv_op->Op()->SetAttr("Scale_weights", filter_scale);

    // if quantization scale is missing for output tensor, return fp32 data
    if (has_output_scale) {
      bool is_output_unsigned{false};
      auto output_scale =
          GetScaleValueForNode(conv_output, &is_output_unsigned);
      DequantizeOutput(g, conv_op, conv_output, "Output", output_scale,
                       is_output_unsigned, "Scale_out");
    } else {
      conv_op->Op()->SetAttr("force_fp32_output", true);
    }

    // change threshold in bounded ReLu
    if (conv_op->Op()->GetAttrIfExists<std::string>("fuse_activation") ==
        "relu6") {
      float scale_out =
          BOOST_GET_CONST(float, conv_op->Op()->GetAttr("Scale_out"));
      float threshold =
          BOOST_GET_CONST(float, conv_op->Op()->GetAttr("fuse_alpha"));
      conv_op->Op()->SetAttr("fuse_alpha", scale_out * threshold);
    }

    ++quantize_conv_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_conv_count);

  std::stringstream msg_ss;
  msg_ss << "---    quantized " << quantize_conv_count << " conv2d ops";
  if (with_residual_data) msg_ss << " with residual connection";
  PrettyLogDetail(msg_ss.str().c_str());
}

void CPUQuantizePass::QuantizeFc(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::FCMKLDNN fc_pattern{pattern, name_scope_};
  auto* fc_input = gpd.mutable_pattern()
                       ->NewNode("fc_quantizer/input")
                       ->AsInput()
                       ->assert_is_op_input("fc", "Input");
  fc_pattern(fc_input, false);

  int quantize_fc_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize fc op";
    GET_IR_NODE_FROM_SUBGRAPH(fc, fc, fc_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(fc->Op())) {
      LogQuantizationDisabled(fc);
      return;
    }
    if (!fc->Op()->GetAttrIfExists<bool>("use_mkldnn")) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(weights, weights, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(input, input, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(output, output, fc_pattern);

    if (!AreScalesPresentForNodes({input, weights})) {
      LogCannotQuantizeOp(fc);
      return;
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(input, &is_input_unsigned);
    QuantizeInput(g, fc, input, "Input", input_scale, is_input_unsigned,
                  "Scale_in");

    auto weight_scale_tensor = GetScaleTensorForNode(weights);
    EigenVectorArrayMap eigen_tensor{weight_scale_tensor.data<double>(),
                                     weight_scale_tensor.numel()};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> filter_scale{
        weight_scale_tensor.data<double>(),
        weight_scale_tensor.data<double>() + weight_scale_tensor.numel()};

    fc->Op()->SetAttr("Scale_weights", filter_scale);

    // if quantization scale is missing for output tensor, return fp32 data
    if (AreScalesPresentForNodes({output})) {
      bool is_output_unsigned{false};
      auto output_scale = GetScaleValueForNode(output, &is_output_unsigned);
      DequantizeOutput(g, fc, output, "Out", output_scale, is_output_unsigned,
                       "Scale_out");
    } else {
      fc->Op()->SetAttr("force_fp32_output", true);
    }

    ++quantize_fc_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_fc_count);

  std::stringstream msg_ss;
  msg_ss << "---    quantized " << quantize_fc_count << " fc ops";
  PrettyLogDetail(msg_ss.str().c_str());
}

void CPUQuantizePass::QuantizePool(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Pool pool_pattern{pattern, name_scope_};
  pool_pattern();

  int quantize_pool_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize pool2d op";
    GET_IR_NODE_FROM_SUBGRAPH(pool_op, pool_op, pool_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(pool_op->Op())) {
      LogQuantizationDisabled(pool_op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(pool_input, pool_input, pool_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(pool_output, pool_output, pool_pattern);

    if (!AreScalesPresentForNodes({pool_input, pool_output})) {
      LogCannotQuantizeOp(pool_op);
      return;
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(pool_input, &is_input_unsigned);
    QuantizeInput(g, pool_op, pool_input, "X", input_scale, is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(pool_output, &is_output_unsigned);
    DequantizeOutput(g, pool_op, pool_output, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_pool_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_pool_count);

  PrettyLogDetail("---    quantized %d pool2d ops", quantize_pool_count);
}

void CPUQuantizePass::QuantizeConcat(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Concat concat_pattern{pattern, name_scope_};
  concat_pattern();

  int quantize_concat_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize concat op";
    GET_IR_NODE_FROM_SUBGRAPH(concat_op, concat_op, concat_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(concat_op->Op())) {
      LogQuantizationDisabled(concat_op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(concat_out, concat_out, concat_pattern);

    if (!AreScalesPresentForNodes({concat_out})) {
      LogCannotQuantizeOp(concat_op);
      return;
    }

    // if all inputs were unsigned, then the output was set to unsigned
    // during the scale calculation step
    bool are_all_inputs_unsigned{false};
    auto output_scale =
        GetScaleValueForNode(concat_out, &are_all_inputs_unsigned);

    QuantizeInputs(g, concat_op, "X", are_all_inputs_unsigned);

    DequantizeOutput(g, concat_op, concat_out, "Out", output_scale,
                     are_all_inputs_unsigned);

    ++quantize_concat_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_concat_count);

  PrettyLogDetail("---    quantized %d concat ops", quantize_concat_count);
}

void CPUQuantizePass::QuantizePriorBox(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::PriorBox prior_box_pattern{pattern, name_scope_};
  prior_box_pattern();

  int quantize_prior_box_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize prior_box op";
    GET_IR_NODE_FROM_SUBGRAPH(prior_box_op, prior_box_op, prior_box_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(prior_box_op->Op())) {
      LogQuantizationDisabled(prior_box_op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(prior_box_input, prior_box_input,
                              prior_box_pattern);

    if (!AreScalesPresentForNodes({prior_box_input})) {
      LogCannotQuantizeOp(prior_box_op);
      return;
    }

    bool is_input_unsigned{false};
    auto input_scale =
        GetScaleValueForNode(prior_box_input, &is_input_unsigned);
    QuantizeInput(g, prior_box_op, prior_box_input, "Input", input_scale,
                  is_input_unsigned);

    ++quantize_prior_box_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_prior_box_count);

  PrettyLogDetail("---    quantized %d prior_box ops",
                  quantize_prior_box_count);
}

void CPUQuantizePass::QuantizeTranspose(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Transpose transpose_pattern{pattern, name_scope_};
  transpose_pattern();

  int quantize_transpose_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize transpose op";
    GET_IR_NODE_FROM_SUBGRAPH(transpose_op, transpose_op, transpose_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(transpose_op->Op())) {
      LogQuantizationDisabled(transpose_op);
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, transpose_pattern);

    // skip if prev op and next op is not quantized
    if (!(IsOpDequantized(prev_op)) && !(IsOpQuantized(next_op))) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(transpose_in, transpose_in, transpose_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose_out, transpose_out, transpose_pattern);

    if (!AreScalesPresentForNodes({transpose_in, transpose_out})) {
      LogCannotQuantizeOp(transpose_op);
      return;
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(transpose_in, &is_input_unsigned);
    QuantizeInput(g, transpose_op, transpose_in, "X", input_scale,
                  is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale =
        GetScaleValueForNode(transpose_out, &is_output_unsigned);
    DequantizeOutput(g, transpose_op, transpose_out, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_transpose_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_transpose_count);

  PrettyLogDetail("---    quantized %d transpose ops",
                  quantize_transpose_count);
}

void CPUQuantizePass::QuantizeReshape(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::Reshape reshape_pattern{pattern, name_scope_};
  reshape_pattern();

  int quantize_reshape_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize reshape op";
    GET_IR_NODE_FROM_SUBGRAPH(reshape_op, reshape_op, reshape_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(reshape_op->Op())) {
      LogQuantizationDisabled(reshape_op);
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op, prev_op, reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, reshape_pattern);

    // skip if prev op and next op is not quantized
    if (!(IsOpDequantized(prev_op)) && !(IsOpQuantized(next_op))) {
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(reshape_in, reshape_in, reshape_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape_out, reshape_out, reshape_pattern);

    if (!AreScalesPresentForNodes({reshape_in, reshape_out})) {
      LogCannotQuantizeOp(reshape_op);
      return;
    }

    bool is_input_unsigned{false};
    auto input_scale = GetScaleValueForNode(reshape_in, &is_input_unsigned);
    QuantizeInput(g, reshape_op, reshape_in, "X", input_scale,
                  is_input_unsigned);

    bool is_output_unsigned{false};
    auto output_scale = GetScaleValueForNode(reshape_out, &is_output_unsigned);
    DequantizeOutput(g, reshape_op, reshape_out, "Out", output_scale,
                     is_output_unsigned);

    ++quantize_reshape_count;
  };

  gpd(graph, handler);
  AddStatis(quantize_reshape_count);

  PrettyLogDetail("---    quantized %d reshape ops", quantize_reshape_count);
}

void CPUQuantizePass::QuantizeMatmul(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::MatmulWithInputOps matmul_pattern{pattern, name_scope_};
  matmul_pattern();

  int quantize_matmul_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize matmul op";
    GET_IR_NODE_FROM_SUBGRAPH(matmul_op, matmul_op, matmul_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(matmul_op->Op())) {
      LogQuantizationDisabled(matmul_op);
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(prev_op_x, prev_op_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(prev_op_y, prev_op_y, matmul_pattern);

    // skip if prev ops are not quantized
    if (!IsOpDequantized(prev_op_x) || !IsOpDequantized(prev_op_y)) {
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_x, matmul_in_x, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_in_y, matmul_in_y, matmul_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_out, matmul_out, matmul_pattern);

    if (!AreScalesPresentForNodes({matmul_in_x, matmul_in_y})) {
      LogCannotQuantizeOp(matmul_op);
      return;
    }

    bool is_x_unsigned{false}, is_y_unsigned{false};
    auto input_x_scale = GetScaleValueForNode(matmul_in_x, &is_x_unsigned);
    auto input_y_scale = GetScaleValueForNode(matmul_in_y, &is_y_unsigned);
    PADDLE_ENFORCE_EQ(is_x_unsigned, is_y_unsigned,
                      platform::errors::InvalidArgument(
                          "Matmul inputs should have the same "
                          "attribute of signed/unsigned, but they "
                          "are different: x(%d), y(%d).",
                          is_x_unsigned, is_y_unsigned));
    QuantizeInput(g, matmul_op, matmul_in_x, "X", input_x_scale, is_x_unsigned,
                  "Scale_x");
    QuantizeInput(g, matmul_op, matmul_in_y, "Y", input_y_scale, is_y_unsigned,
                  "Scale_y");

    // if quantization scale is missing for output tensor, return fp32 data
    if (AreScalesPresentForNodes({matmul_out})) {
      bool is_output_unsigned{false};
      auto output_scale = GetScaleValueForNode(matmul_out, &is_output_unsigned);
      DequantizeOutput(g, matmul_op, matmul_out, "Out", output_scale,
                       is_output_unsigned, "Scale_out");
    } else {
      matmul_op->Op()->SetAttr("force_fp32_output", true);
    }

    ++quantize_matmul_count;
  };
  gpd(graph, handler);
  AddStatis(quantize_matmul_count);

  PrettyLogDetail("---    quantized %d matmul ops", quantize_matmul_count);
}

void CPUQuantizePass::QuantizeElementwiseAdd(Graph* graph) const {
  GraphPatternDetector gpd;
  auto pattern = gpd.mutable_pattern();
  patterns::ElementwiseAdd elementwise_add_pattern{pattern, name_scope_};

  elementwise_add_pattern(
      pattern->NewNode(elementwise_add_pattern.elementwise_add_x_repr()),
      pattern->NewNode(elementwise_add_pattern.elementwise_add_y_repr()));

  int quantize_elementwise_add_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize elementwise_add op";
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_op, elementwise_add_op,
                              elementwise_add_pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(elementwise_add_op->Op())) {
      LogQuantizationDisabled(elementwise_add_op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_x, elementwise_add_x,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_y, elementwise_add_y,
                              elementwise_add_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              elementwise_add_pattern);

    if (!AreScalesPresentForNodes({elementwise_add_x, elementwise_add_y})) {
      LogCannotQuantizeOp(elementwise_add_op);
      return;
    }

    bool is_x_unsigned{false}, is_y_unsigned{false};
    auto input_x_scale =
        GetScaleValueForNode(elementwise_add_x, &is_x_unsigned);
    auto input_y_scale =
        GetScaleValueForNode(elementwise_add_y, &is_y_unsigned);

    // TODO(sfraczek): add support for different signness
    if (is_x_unsigned != is_y_unsigned) {
      LogCannotQuantizeOp(elementwise_add_op,
                          "ElementwiseAdd inputs must be of the same type.");
      return;
    }

    QuantizeInput(g, elementwise_add_op, elementwise_add_x, "X", input_x_scale,
                  is_x_unsigned, "Scale_x");
    QuantizeInput(g, elementwise_add_op, elementwise_add_y, "Y", input_y_scale,
                  is_y_unsigned, "Scale_y");

    // if quantization scale is missing for output tensor, return fp32 data
    if (AreScalesPresentForNodes({elementwise_add_out})) {
      bool is_output_unsigned{false};
      auto output_scale =
          GetScaleValueForNode(elementwise_add_out, &is_output_unsigned);
      DequantizeOutput(g, elementwise_add_op, elementwise_add_out, "Out",
                       output_scale, is_output_unsigned, "Scale_out");
    } else {
      elementwise_add_op->Op()->SetAttr("force_fp32_output", true);
    }

    ++quantize_elementwise_add_count;
  };
  gpd(graph, handler);
  AddStatis(quantize_elementwise_add_count);

  PrettyLogDetail("---    quantized %d elementwise_add ops",
                  quantize_elementwise_add_count);
}

void CPUQuantizePass::QuantizeFusionGru(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::FusionGru pattern{gpd.mutable_pattern(), name_scope_};
  pattern();

  int quantize_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize fusion_gru op";
    GET_IR_NODE_FROM_SUBGRAPH(op, op, pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(op->Op())) {
      LogQuantizationDisabled(op);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(x, x, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(weight_h, weight_h, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(weight_x, weight_x, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(out, out, pattern);

    if (!AreScalesPresentForNodes({x, weight_x})) {
      LogCannotQuantizeOp(op);
      return;
    }

    bool is_x_unsigned{false};
    auto input_x_scale = GetScaleValueForNode(x, &is_x_unsigned);

    double input_x_shift{128.};
    if (is_x_unsigned) input_x_shift = 0.;

    QuantizeInput(g, op, x, "X", input_x_scale, is_x_unsigned, "Scale_data",
                  input_x_shift, "Shift_data");

    auto weight_scale_tensor = GetScaleTensorForNode(weight_x);
    EigenVectorArrayMap eigen_tensor{weight_scale_tensor.data<double>(),
                                     weight_scale_tensor.numel()};
    eigen_tensor *= static_cast<double>(S8_MAX);
    std::vector<float> scale_weights{
        weight_scale_tensor.data<double>(),
        weight_scale_tensor.data<double>() + weight_scale_tensor.numel()};

    op->Op()->SetAttr("Scale_weights", scale_weights);
    // return fp32 data
    op->Op()->SetAttr("force_fp32_output", true);

    ++quantize_count;
  };
  gpd(graph, handler);
  AddStatis(quantize_count);

  PrettyLogDetail("---    quantized %d fusion_gru ops", quantize_count);
}

void CPUQuantizePass::QuantizeMultiGru(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::MultiGru pattern{gpd.mutable_pattern(), name_scope_};
  pattern();

  int quantize_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "Quantize multi_gru op";
    GET_IR_NODE_FROM_SUBGRAPH(gru, gru, pattern);

    // skip if should not be quantized
    if (!platform::HasOpINT8DataType(gru->Op())) {
      LogQuantizationDisabled(gru);
      return;
    }

    GET_IR_NODE_FROM_SUBGRAPH(x, x, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(wx, wx, pattern);
    GET_IR_NODE_FROM_SUBGRAPH(h, h, pattern);

    auto wx_names = gru->Op()->Input("WeightX");
    if (!AreScalesPresentForNodes({x}) ||
        !AreScalesPresentForVarNames(wx_names)) {
      LogCannotQuantizeOp(gru);
      return;
    }

    bool is_x_unsigned{false};
    auto input_x_scale = GetScaleValueForNode(x, &is_x_unsigned);

    double input_x_shift{128.};
    if (is_x_unsigned) input_x_shift = 0.;

    QuantizeInput(g, gru, x, "X", input_x_scale, is_x_unsigned, "Scale_data",
                  input_x_shift, "Shift_data");

    auto* scope = param_scope();
    int wx_size = wx_names.size();
    std::vector<std::string> w_scale_var_names;
    for (int i = 0; i < wx_size; ++i) {
      auto scale_tensor_src = GetScaleTensorByName(wx_names[i]);
      EigenVectorArrayMap eigen_tensor_src{scale_tensor_src.data<double>(),
                                           scale_tensor_src.numel()};

      VarDesc scale_var_desc(patterns::PDNodeName("multi_gru", "w_scale"));

      scale_var_desc.SetShape(framework::vectorize(scale_tensor_src.dims()));
      scale_var_desc.SetDataType(proto::VarType::FP32);
      scale_var_desc.SetLoDLevel(scale_tensor_src.lod().size());
      scale_var_desc.SetPersistable(true);
      auto* w_scale_node = g->CreateVarNode(&scale_var_desc);

      auto* w_scale_tensor_dst =
          scope->Var(w_scale_node->Name())->GetMutable<LoDTensor>();
      w_scale_tensor_dst->Resize(scale_tensor_src.dims());
      auto* dst_data =
          w_scale_tensor_dst->mutable_data<float>(platform::CPUPlace());
      EigenVectorArrayMapFloat eigen_tensor_dst{dst_data,
                                                w_scale_tensor_dst->numel()};
      eigen_tensor_dst =
          eigen_tensor_src.cast<float>() * static_cast<float>(S8_MAX);
      w_scale_var_names.push_back(w_scale_node->Name());
      IR_NODE_LINK_TO(w_scale_node, gru);
    }

    gru->Op()->SetInput("Scale_weights", w_scale_var_names);
    // return fp32 data
    gru->Op()->SetAttr("force_fp32_output", true);

    ++quantize_count;
  };
  gpd(graph, handler);
  AddStatis(quantize_count);

  PrettyLogDetail("---    quantized %d multi_gru ops", quantize_count);
}

void CPUQuantizePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Quantizing the graph.";
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  FusePassBase::Init(name_scope_, graph);

  PADDLE_ENFORCE_NOT_NULL(param_scope(), platform::errors::InvalidArgument(
                                             "Scope cannot be nullptr."));

  QuantizeConv(graph, false /* with_residual_data */);
  QuantizeConv(graph, true /* with_residual_data */);
  QuantizePool(graph);
  QuantizeConcat(graph);
  QuantizePriorBox(graph);
  QuantizeTranspose(graph);
  QuantizeFc(graph);
  QuantizeReshape(graph);
  QuantizeMatmul(graph);
  QuantizeElementwiseAdd(graph);
  QuantizeFusionGru(graph);
  QuantizeMultiGru(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_pass, paddle::framework::ir::CPUQuantizePass)
    .RequirePassAttr("quant_var_scales");
