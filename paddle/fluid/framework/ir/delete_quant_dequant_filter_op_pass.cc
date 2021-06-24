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

#include "paddle/fluid/framework/ir/delete_quant_dequant_filter_op_pass.h"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {
namespace ir {

#define GET_IR_NODE(node__) GET_IR_NODE_FROM_SUBGRAPH(node__, node__, pattern);
#define GET_NODES                         \
  GET_IR_NODE(quant_dequant_op_x);        \
  GET_IR_NODE(quant_dequant_op);          \
  GET_IR_NODE(quant_dequant_op_out);      \
  GET_IR_NODE(quant_dequant_op_outscale); \
  GET_IR_NODE(any_op2);

DeleteQuantDequantFilterOpPass::DeleteQuantDequantFilterOpPass() {
  AddOpCompat(OpCompat("fake_quantize_dequantize_abs_max"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("OutScale")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsIntIn({8, 16})
      .End();
  AddOpCompat(OpCompat("fake_channel_wise_quantize_dequantize_abs_max"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("OutScale")
      .IsTensor()
      .End()
      .AddAttr("bit_length")
      .IsIntIn({8, 16})
      .End()
      .AddAttr("quant_axis")
      .IsIntIn({0, 1})
      .End();
}
// Delete quant_dequant_op, then quantize and dequantize weight
void DeleteQuantDequantFilterOpPass::ApplyImpl(ir::Graph* graph) const {
  const std::string pattern_name = "delete_quantdequant_filter_op_pattern";
  FusePassBase::Init(pattern_name, graph);

  GraphPatternDetector gpd;

  // Create pattern
  patterns::DeleteQuantDequantFilterOpPattern pattern(gpd.mutable_pattern(),
                                                      pattern_name);
  pattern();
  auto* scope = param_scope();
  int found_count = 0;

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_NODES;

    if (!IsCompat(*quant_dequant_op->Op())) {
      LOG(WARNING) << "quant_dequant_op in delete_quant_dequant_filter_op_pass "
                      "compat check failed.";
      return;
    }
    std::unordered_set<const Node*> nodes2rm = {};
    int bit_length =
        BOOST_GET_CONST(int, quant_dequant_op->Op()->GetAttr("bit_length"));
    int range = ((1 << (bit_length - 1)) - 1);
    std::vector<float> weight_scale;
    std::string quant_dequant_op_out_name = quant_dequant_op_out->Var()->Name();

    auto* any_op2_desc = any_op2->Op();
    auto var_map = any_op2_desc->Inputs();
    std::string arg_name = "";
    for (auto& name_m : var_map) {
      if (std::find(name_m.second.begin(), name_m.second.end(),
                    quant_dequant_op_out_name) != name_m.second.end()) {
        arg_name = name_m.first;
        break;
      }
    }
    PADDLE_ENFORCE_GT(arg_name.size(), 0, platform::errors::InvalidArgument(
                                              "can not find the input %s.",
                                              quant_dequant_op_out_name));
    any_op2_desc->SetAttr("enable_int8", true);
    any_op2_desc->SetAttr("bit_length", bit_length);
    // modify the any_op2's inputs
    any_op2_desc->Flush();
    auto dequant_type = quant_dequant_op->Op()->Type();
    auto quantized_op_type = any_op2_desc->Type();
    // get weight tensor
    auto* weight_tensor =
        scope->GetVar(quant_dequant_op_x->Name())->GetMutable<LoDTensor>();
    auto w_dims = weight_tensor->dims();
    float* quantized_weight_data =
        weight_tensor->mutable_data<float>(platform::CPUPlace());

    // Get weight scale
    if (dequant_type == "fake_channel_wise_quantize_dequantize_abs_max") {
      auto scales_name = quant_dequant_op->Op()->Output("OutScale");
      PADDLE_ENFORCE_EQ(scales_name.size(), 1,
                        platform::errors::InvalidArgument(
                            "Scales size in channel-wise quant dequantize op "
                            "should be 1, got %d.",
                            scales_name.size()));
      const LoDTensor& channel_scale_tensor =
          scope->GetVar(scales_name[0])->Get<LoDTensor>();
      PADDLE_ENFORCE(
          paddle::platform::is_cpu_place(channel_scale_tensor.place()),
          platform::errors::InvalidArgument(
              "Channel scale tensor's place should be CPU."));
      // compute the channel wise abs max of the weight tensor
      int quant_axis =
          BOOST_GET_CONST(int, quant_dequant_op->Op()->GetAttr("quant_axis"));

      PADDLE_ENFORCE_EQ(quant_axis == 0 || quant_axis == 1, true,
                        platform::errors::InvalidArgument(
                            "'quant_axis' should be 0 or 1, but "
                            "the received is %d",
                            quant_axis));

      const int64_t channel = w_dims[quant_axis];
      weight_scale.resize(channel, 0);
      if (quant_axis == 0) {
        const int64_t channel_size = weight_tensor->numel() / channel;
        for (int64_t i = 0; i < channel; i++) {
          auto* start = quantized_weight_data + i * channel_size;
          for (int64_t j = 0; j < channel_size; j++) {
            weight_scale[i] = std::max(std::abs(start[j]), weight_scale[i]);
          }
        }
      } else if (quant_axis == 1) {
        const int64_t step_i = weight_tensor->numel() / w_dims[0];
        const int64_t step_j = weight_tensor->numel() / (w_dims[0] * w_dims[1]);
        for (int64_t i = 0; i < w_dims[0]; i++) {
          for (int64_t j = 0; j < w_dims[1]; j++) {
            auto* start = quantized_weight_data + i * step_i + j * step_j;
            float abs_max = 0;
            for (int64_t k = 0; k < step_j; k++) {
              abs_max = std::max(std::abs(start[k]), abs_max);
            }
            weight_scale[j] = std::max(weight_scale[j], abs_max);
          }
        }
      }
      for (int i = 0; i < channel; i++) {
        PADDLE_ENFORCE_NE(weight_scale[i], 0,
                          platform::errors::InvalidArgument(
                              "Weight scale should be nonzero, but get zero."));
        weight_scale[i] = range / weight_scale[i];
      }
    } else {
      auto scale_name = quant_dequant_op_outscale->Name();
      // compute the abs max of the weight tensor
      float abs_max_weight = 0.;
      for (int j = 0; j < weight_tensor->numel(); j++) {
        abs_max_weight =
            std::max(abs_max_weight, std::abs(quantized_weight_data[j]));
      }
      PADDLE_ENFORCE_NE(abs_max_weight, 0,
                        platform::errors::InvalidArgument(
                            "Weight scale should be nonzero, but get zero"));
      weight_scale.push_back((range * range) / abs_max_weight / range);
    }

    nodes2rm.insert(quant_dequant_op_outscale);

    // perform quantize dequantize operations
    // If quantized op is not channel wise, weight scale size = 1;
    // If quantized op is conv2d, weight scale size = weight dims[0]
    // If quantized op is conv2d_transpose, weight scale size = weight dims[1]
    if (dequant_type == "fake_quantize_dequantize_abs_max") {
      PADDLE_ENFORCE_EQ(
          weight_scale.size(), 1,
          platform::errors::InvalidArgument(
              "%s op weight dequantized by [fake_quantize_dequantize_max_abs] "
              "requires weight scale size = 1, but got %d.",
              quantized_op_type, weight_scale.size()));
      for (int j = 0; j < weight_tensor->numel(); j++) {
        // quantized
        quantized_weight_data[j] = quantized_weight_data[j] * weight_scale[0];
        quantized_weight_data[j] = std::round(quantized_weight_data[j]);
        // dequantized
        quantized_weight_data[j] /= weight_scale[0];
      }
    } else if (quantized_op_type == "mul" || quantized_op_type == "matmul" ||
               quantized_op_type == "fc") {
      if (dequant_type == "fake_channel_wise_quantize_dequantize_abs_max") {
        PADDLE_ENFORCE_EQ(
            weight_scale.size(), static_cast<size_t>(w_dims[1]),
            platform::errors::InvalidArgument(
                "mul op weight dequantized by "
                "[fake_channel_wise_quantize_dequantize_abs_max] requires "
                "weight scale "
                "size = 2nd dim of mul's weight, which is %zu, but got %zu.",
                static_cast<size_t>(w_dims[1]), weight_scale.size()));
        for (int j = 0; j < weight_tensor->numel(); j++) {
          // quantized
          PADDLE_ENFORCE_NE(
              weight_scale[j % w_dims[1]], 0,
              platform::errors::InvalidArgument(
                  "fc op weight scale should be nonzero, but get zero"));
          quantized_weight_data[j] =
              quantized_weight_data[j] * weight_scale[j % w_dims[1]];
          quantized_weight_data[j] = std::round(quantized_weight_data[j]);
          // dequantized
          quantized_weight_data[j] /= weight_scale[j % w_dims[1]];
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported quantized op type: %s", quantized_op_type));
      }
    } else if (quantized_op_type == "conv2d" ||
               quantized_op_type == "depthwise_conv2d") {
      if (dequant_type == "fake_channel_wise_quantize_dequantize_abs_max") {
        PADDLE_ENFORCE_EQ(
            weight_scale.size(), static_cast<size_t>(w_dims[0]),
            platform::errors::InvalidArgument(
                "conv2d op requires weight scale size = channel size of the "
                "weight, which is %zu, but got %zu.",
                static_cast<size_t>(w_dims[0]), weight_scale.size()));
        int inner_size = w_dims[1] * w_dims[2] * w_dims[3];
        for (int j = 0; j < weight_tensor->numel(); j++) {
          // quantized
          PADDLE_ENFORCE_NE(
              weight_scale[j / inner_size], 0,
              platform::errors::InvalidArgument(
                  "conv2d op weight scale should be nonzero, but get zero"));
          quantized_weight_data[j] =
              quantized_weight_data[j] * weight_scale[j / inner_size];
          quantized_weight_data[j] = std::round(quantized_weight_data[j]);
          // dequantized
          quantized_weight_data[j] /= weight_scale[j / inner_size];
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported quantized op type: %s", quantized_op_type));
      }
    } else if (quantized_op_type == "conv2d_transpose") {
      if (dequant_type == "fake_channel_wise_quantize_dequantize_abs_max") {
        PADDLE_ENFORCE_EQ(
            weight_scale.size(), static_cast<size_t>(w_dims[0]),
            platform::errors::InvalidArgument(
                "conv2d_transpose op requires weight scale size = channel size "
                "of the "
                "weight, which is %zu, but got %zu.",
                static_cast<size_t>(w_dims[1]), weight_scale.size()));
        int inner_size = w_dims[2] * w_dims[3];
        for (int j = 0; j < weight_tensor->numel(); j++) {
          // quantized
          PADDLE_ENFORCE_NE(weight_scale[(j / inner_size) % w_dims[1]], 0,
                            platform::errors::InvalidArgument(
                                "conv2d_transpose op weight scale should be "
                                "nonzero, but get zero"));
          quantized_weight_data[j] = quantized_weight_data[j] *
                                     weight_scale[(j / inner_size) % w_dims[1]];
          quantized_weight_data[j] = std::round(quantized_weight_data[j]);
          // dequantized
          quantized_weight_data[j] /=
              weight_scale[(j / inner_size) % w_dims[1]];
        }
      } else {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unsupported quantized op type: %s", quantized_op_type));
      }
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unsupported quantized op type: %s", quantized_op_type));
    }
    nodes2rm.insert(quant_dequant_op_out);

    // link weight in quant_dequant_op_x to any_op2
    any_op2_desc->RenameInput(quant_dequant_op_out->Var()->Name(),
                              quant_dequant_op_x->Var()->Name());
    any_op2_desc->SetAttr("weight_scale", weight_scale);
    any_op2_desc->Flush();
    IR_NODE_LINK_TO(quant_dequant_op_x, any_op2);
    nodes2rm.insert(quant_dequant_op);
    GraphSafeRemoveNodes(graph, nodes2rm);
    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(delete_quant_dequant_filter_op_pass,
              paddle::framework::ir::DeleteQuantDequantFilterOpPass);
