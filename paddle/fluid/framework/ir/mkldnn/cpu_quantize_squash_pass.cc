// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file eint8_outcept in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either eint8_outpress or
// implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/framework/ir/mkldnn/cpu_quantize_squash_pass.h"

#include <string>
#include <vector>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;

using string::PrettyLogDetail;

void CPUQuantizeSquashPass::FindNodesToKeep(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantAny deq_any_pattern{gpd.mutable_pattern(), "deqant_any"};
  deq_any_pattern();

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, deq_any_pattern);

    if (nodes_keep_counter->find(dequant_out) == nodes_keep_counter->end())
      (*nodes_keep_counter)[dequant_out] = 1;
    else
      (*nodes_keep_counter)[dequant_out] += 1;

    found_count++;
  };
  gpd(graph, handler);
  AddStatis(found_count);
}

void CPUQuantizeSquashPass::DequantQuantSquash(
    Graph* graph,
    std::unordered_map<const Node*, int>* nodes_keep_counter) const {
  GraphPatternDetector gpd;
  patterns::DequantQuantAny squash_pattern{gpd.mutable_pattern(), "squash"};
  squash_pattern();

  int found_dequant_quant_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash requantize-quantize ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(dequant_in, dequant_in, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_op, quant_op, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(quant_out, quant_out, squash_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(next_op, next_op, squash_pattern);

    auto* next_op_desc = next_op->Op();
    float dequant_scale =
        BOOST_GET_CONST(float, dequant_op->Op()->GetAttr("Scale"));
    float quant_scale =
        BOOST_GET_CONST(float, quant_op->Op()->GetAttr("Scale"));
    float dequant_shift = dequant_op->Op()->GetAttrIfExists<float>("Shift");
    float quant_shift = quant_op->Op()->GetAttrIfExists<float>("Shift");
    PADDLE_ENFORCE_NE(
        nodes_keep_counter->find(dequant_out), nodes_keep_counter->end(),
        platform::errors::NotFound("The dequant output node is not found."));

    // check if dequantize op should be kept or removed, decrease the counter
    bool keep_dequant = (*nodes_keep_counter)[dequant_out]-- > 1;

    if (dequant_scale == quant_scale && dequant_shift == quant_shift) {
      // squash dequantize-quantize to nothing
      auto quant_out_var_name = quant_out->Name();
      auto next_op_inputs = next_op_desc->InputNames();
      for (const auto& name : next_op_inputs) {
        auto input_names = next_op_desc->Input(name);
        std::replace(input_names.begin(), input_names.end(), quant_out_var_name,
                     dequant_in->Name());
        next_op_desc->SetInput(name, input_names);
      }

      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant_op, quant_out});
      else
        GraphSafeRemoveNodes(graph,
                             {dequant_op, quant_op, dequant_out, quant_out});

      IR_NODE_LINK_TO(dequant_in, next_op);

      found_dequant_quant_count++;
    } else {
      // squash dequantize-quantize to requantize op
      OpDesc desc;
      desc.SetType("requantize");
      desc.SetInput("Input", std::vector<std::string>({dequant_in->Name()}));
      desc.SetOutput("Output", std::vector<std::string>({quant_out->Name()}));
      desc.SetAttr("Scale_in", dequant_scale);
      desc.SetAttr("Shift_in", dequant_shift);
      desc.SetAttr("Scale_out", quant_scale);
      desc.SetAttr("Shift_out", quant_shift);

      auto requant_op = g->CreateOpNode(&desc);

      if (keep_dequant)
        GraphSafeRemoveNodes(graph, {quant_op});
      else
        GraphSafeRemoveNodes(graph, {dequant_op, quant_op, dequant_out});

      IR_NODE_LINK_TO(dequant_in, requant_op);
      IR_NODE_LINK_TO(requant_op, quant_out);

      found_dequant_quant_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_dequant_quant_count);
  PrettyLogDetail("---    squashed %d dequantize-quantize pairs",
                  found_dequant_quant_count);
}

// op+requant squash if op has Scale_out attr
// conv2d and fc
void CPUQuantizeSquashPass::OpRequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::OpRequant op_requant_pattern{gpd.mutable_pattern(), "op_requant"};
  op_requant_pattern();

  int found_requant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash op-requantize ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(any_op, any_op, op_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_in, requant_in, op_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_op, requant_op, op_requant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_out, requant_out, op_requant_pattern);

    if (requant_in->outputs.size() == 1) {
      std::string any_op_output_name;
      for (auto name : any_op->Op()->OutputNames())
        for (auto output_name : any_op->Op()->Output(name))
          if (output_name == requant_in->Name()) any_op_output_name = name;

      PADDLE_ENFORCE_NE(
          any_op_output_name.empty(), true,
          platform::errors::NotFound("Operator before requantize operator(%s) "
                                     "should have requantize input as output.",
                                     requant_in->Name()));

      float requant_scale_out =
          BOOST_GET_CONST(float, requant_op->Op()->GetAttr("Scale_out"));
      any_op->Op()->SetAttr("Scale_out", requant_scale_out);
      any_op->Op()->SetOutput(any_op_output_name,
                              std::vector<std::string>({requant_out->Name()}));
      IR_NODE_LINK_TO(any_op, requant_out);
      GraphSafeRemoveNodes(graph, {requant_in, requant_op});
      found_requant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_requant_squash_count);
  PrettyLogDetail("---    squashed %d requantize ops",
                  found_requant_squash_count);
}

// requant-op squash if op has Scale_in, Scale_x, Scale_y attr
// conv2d, fc, matmul
void CPUQuantizeSquashPass::RequantOpSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::RequantOp requant_op_pattern{gpd.mutable_pattern(), "requant_op"};
  requant_op_pattern();

  int found_requant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash requantize-op ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(requant_in, requant_in, requant_op_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_op, requant_op, requant_op_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(requant_out, requant_out, requant_op_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(any_op, any_op, requant_op_pattern);

    if (requant_out->outputs.size() == 1) {
      std::string any_op_input_name;
      for (auto name : any_op->Op()->InputNames())
        for (auto input_name : any_op->Op()->Input(name))
          if (input_name == requant_out->Name()) any_op_input_name = name;

      PADDLE_ENFORCE_NE(any_op_input_name.empty(), true,
                        platform::errors::NotFound(
                            "The operator after requantize operator(%s) "
                            "should have requantize output as input.",
                            requant_out->Name()));
      float requant_scale_in =
          boost::get<float>(requant_op->Op()->GetAttr("Scale_in"));

      auto scale_name = "Scale_in";
      if (any_op->Op()->Type() == "matmul")
        scale_name = any_op_input_name == "X" ? "Scale_x" : "Scale_y";

      PADDLE_ENFORCE_EQ(
          requant_op->Op()->GetAttrIfExists<float>("Scale_out"),
          any_op->Op()->GetAttrIfExists<float>(scale_name),
          platform::errors::InvalidArgument(
              "The operator after requantize should have input "
              "scale(%f) equal to requantize output scale(%f).",
              any_op->Op()->GetAttrIfExists<float>(scale_name),
              requant_op->Op()->GetAttrIfExists<float>("Scale_out")));
      any_op->Op()->SetAttr(scale_name, requant_scale_in);
      any_op->Op()->SetInput(any_op_input_name,
                             std::vector<std::string>({requant_in->Name()}));
      IR_NODE_LINK_TO(requant_in, any_op);
      GraphSafeRemoveNodes(graph, {requant_op, requant_out});
      found_requant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_requant_squash_count);
  PrettyLogDetail("---    squashed %d requantize ops",
                  found_requant_squash_count);
}

// squash dequant with previous op if that op has force_fp32_output attr
// conv2d, fc, matmul
void CPUQuantizeSquashPass::OpDequantSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::OpDequant op_dequant_pattern{gpd.mutable_pattern(), "op_dequant"};
  op_dequant_pattern();

  int found_op_dequant_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash op-dequant ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(any_op, any_op, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_in, dequant_in, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, op_dequant_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, op_dequant_pattern);

    if (dequant_in->outputs.size() == 1) {
      auto output_name = "Out";
      if (any_op->Op()->Type() == "conv2d") {
        // do not squash if fuse residual connection is true
        // because residual fusion does not support force output with fp32
        if (any_op->Op()->GetAttrIfExists<bool>("fuse_residual_connection"))
          return;
        output_name = "Output";
      }
      any_op->Op()->SetAttr("force_fp32_output", true);
      any_op->Op()->SetOutput(output_name,
                              std::vector<std::string>({dequant_out->Name()}));
      IR_NODE_LINK_TO(any_op, dequant_out);
      GraphSafeRemoveNodes(graph, {dequant_in, dequant_op});
      found_op_dequant_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_op_dequant_squash_count);
  PrettyLogDetail("---    squashed %d dequant with ops",
                  found_op_dequant_squash_count);
}

void CPUQuantizeSquashPass::MultipleQuantizeSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::MultipleQuantize multiple_quantize_pattern{gpd.mutable_pattern(),
                                                       "multiple_quantize"};
  multiple_quantize_pattern();

  int found_multiple_quantize_squash_count = 0;
  int removed_quantize = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "fuse multiple quantize ops";

    GET_IR_NODE_FROM_SUBGRAPH(prev_out, prev_out, multiple_quantize_pattern);

    auto* first_quant_op = *(std::find_if(
        prev_out->outputs.begin(), prev_out->outputs.end(), [&](Node* node) {
          return (node->IsOp() && node->Op()->Type() == "quantize");
        }));
    auto* first_quant_out = first_quant_op->outputs[0];
    float scale = first_quant_op->Op()->GetAttrIfExists<float>("Scale");
    float shift = first_quant_op->Op()->GetAttrIfExists<float>("Shift");

    PADDLE_ENFORCE_NE(scale, 0,
                      platform::errors::InvalidArgument(
                          "Quantize scale(%f) should not be equal 0.", scale));

    for (int iter = prev_out->outputs.size() - 1; iter >= 0; iter--) {
      auto quant_op = prev_out->outputs[iter];
      if (quant_op->IsOp() && quant_op->Op()->Type() == "quantize" &&
          quant_op->id() != first_quant_op->id() &&
          quant_op->Op()->GetAttrIfExists<float>("Scale") == scale &&
          quant_op->Op()->GetAttrIfExists<float>("Shift") == shift) {
        auto quant_out = quant_op->outputs[0];
        auto last_op = quant_out->outputs[0];

        std::string last_op_input_name;
        for (auto name : last_op->Op()->InputNames())
          for (auto input_name : last_op->Op()->Input(name))
            if (input_name == quant_out->Name()) last_op_input_name = name;

        PADDLE_ENFORCE_NE(
            last_op_input_name.empty(), true,
            platform::errors::NotFound("Operator after quantize operator(%s) "
                                       "should has quantize output as input.",
                                       quant_out->Name()));
        last_op->Op()->SetInput(
            last_op_input_name,
            std::vector<std::string>({first_quant_out->Name()}));
        IR_NODE_LINK_TO(first_quant_out, last_op);
        GraphSafeRemoveNodes(graph, {quant_op, quant_out});
        removed_quantize++;
      }
    }
    found_multiple_quantize_squash_count++;
  };
  gpd(graph, handler);
  AddStatis(found_multiple_quantize_squash_count);
  PrettyLogDetail("---    squashed %d quantize op", removed_quantize);
}

// squash scale with dequant
void CPUQuantizeSquashPass::DequantScaleSquash(Graph* graph) const {
  GraphPatternDetector gpd;
  patterns::DequantScale dequant_scale_pattern{gpd.mutable_pattern(),
                                               "dequant_scale"};
  dequant_scale_pattern();

  int found_dequant_scale_squash_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "squash dequant-scale ops pair";

    GET_IR_NODE_FROM_SUBGRAPH(dequant_op, dequant_op, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(dequant_out, dequant_out, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_op, scale_op, dequant_scale_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, dequant_scale_pattern);

    if (dequant_out->outputs.size() == 1 &&
        scale_op->Op()->GetAttrIfExists<float>("bias") == 0.0) {
      auto dequant_scale = dequant_op->Op()->GetAttrIfExists<float>("Scale");
      auto scale_scale = scale_op->Op()->GetAttrIfExists<float>("scale");

      PADDLE_ENFORCE_GT(dequant_scale, 0.0f,
                        platform::errors::InvalidArgument(
                            "Dequantize scale(%f) should have positive value.",
                            dequant_scale));
      PADDLE_ENFORCE_GT(scale_scale, 0.0f,
                        platform::errors::InvalidArgument(
                            "Scale(%f) of scale op should have positive value.",
                            scale_scale));

      dequant_op->Op()->SetAttr("Scale", dequant_scale / scale_scale);
      dequant_op->Op()->SetOutput(
          "Output", std::vector<std::string>({scale_out->Name()}));
      IR_NODE_LINK_TO(dequant_op, scale_out);
      GraphSafeRemoveNodes(graph, {dequant_out, scale_op});
      found_dequant_scale_squash_count++;
    }
  };
  gpd(graph, handler);
  AddStatis(found_dequant_scale_squash_count);
  PrettyLogDetail("---    squashed %d scale with dequant",
                  found_dequant_scale_squash_count);
}

void CPUQuantizeSquashPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph,
      platform::errors::InvalidArgument(
          "The graph in function CPUQuantizeSquashPass::ApplyImpl is null."));
  FusePassBase::Init("cpu_quantize_squash_pass", graph);

  std::unordered_map<const Node*, int> nodes_keep_counter;
  FindNodesToKeep(graph, &nodes_keep_counter);
  DequantQuantSquash(graph, &nodes_keep_counter);
  OpRequantSquash(graph);
  RequantOpSquash(graph);
  OpDequantSquash(graph);
  MultipleQuantizeSquash(graph);
  DequantScaleSquash(graph);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(cpu_quantize_squash_pass,
              paddle::framework::ir::CPUQuantizeSquashPass);
