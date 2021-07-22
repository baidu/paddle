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

#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace framework {
namespace ir {

class Node;

MulLstmFusePass::MulLstmFusePass() {
  AddOpCompat(OpCompat("lstm"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("H0")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("C0")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("Weight")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Hidden")
      .IsTensor()
      .End()
      .AddOutput("Cell")
      .IsTensor()
      .End()
      .AddOutput("BatchGate")
      .IsTensor()
      .End()
      .AddOutput("BatchCellPreAct")
      .IsTensor()
      .End()
      .AddAttr("use_peepholes")
      .IsType<bool>()
      .End()
      .AddAttr("is_reverse")
      .IsType<bool>()
      .End()
      .AddAttr("gate_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End()
      .AddAttr("cell_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End()
      .AddAttr("candidate_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End();
  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
      .End();
}

FCLstmFusePass::FCLstmFusePass() {
  AddOpCompat(OpCompat("lstm"))
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("H0")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("C0")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("Weight")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddOutput("Hidden")
      .IsTensor()
      .End()
      .AddOutput("Cell")
      .IsTensor()
      .End()
      .AddOutput("BatchGate")
      .IsTensor()
      .End()
      .AddOutput("BatchCellPreAct")
      .IsTensor()
      .End()
      .AddAttr("use_peepholes")
      .IsType<bool>()
      .End()
      .AddAttr("is_reverse")
      .IsType<bool>()
      .End()
      .AddAttr("gate_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End()
      .AddAttr("cell_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End()
      .AddAttr("candidate_activation")
      .IsStringIn({"sigmoid", "tanh", "relu", "identity"})
      .End();
  AddOpCompat(OpCompat("mul"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Y")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("x_num_col_dims")
      .IsNumEQ(1)
      .End()
      .AddAttr("y_num_col_dims")
      .IsNumEQ(1)
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
      .IsNumGE(-1)
      .End();
}

int FCLstmFusePass::BuildFusion(Graph* graph, const std::string& name_scope,
                                Scope* scope, bool with_fc_bias) const {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Build pattern
  PDNode* x = pattern->NewNode(patterns::PDNodeName(name_scope, "x"))
                  ->assert_is_op_input("mul")
                  ->assert_var_not_persistable();
  patterns::FC fc_pattern(pattern, name_scope);

  auto* fc_out = fc_pattern(x, with_fc_bias, /* with_relu */ false);
  patterns::LSTM lstm_pattern(pattern, name_scope);
  lstm_pattern(fc_out);

  // Create New OpDesc
  auto lstm_creator = [&](Node* lstm, Node* input, Node* weight_x,
                          Node* weight_h, Node* bias, Node* hidden, Node* cell,
                          Node* xx, Node* fc_bias, const bool use_mkldnn) {
    OpDesc op_desc;
    op_desc.SetType("fusion_lstm");
#define SET_IN(Key, node__) op_desc.SetInput(#Key, {node__->Name()});
    SET_IN(X, input);
    SET_IN(WeightX, weight_x);
    SET_IN(WeightH, weight_h);
    SET_IN(Bias, bias);
#undef SET_IN
    if (with_fc_bias) {
      // Add FC-bias with LSTM-bias and create a new weight
      PADDLE_ENFORCE_NOT_NULL(
          scope, platform::errors::InvalidArgument("Scope cannot be nullptr."));
      auto* lstm_bias_var = scope->FindVar(bias->Name());
      auto* fc_bias_var = scope->FindVar(fc_bias->Name());
      PADDLE_ENFORCE_NOT_NULL(lstm_bias_var,
                              platform::errors::InvalidArgument(
                                  "Lstm bias var ptr cannot be nullptr."));
      PADDLE_ENFORCE_NOT_NULL(fc_bias_var,
                              platform::errors::InvalidArgument(
                                  "FC bias var ptr cannot be nullptr."));
      auto* lstm_bias_tensor =
          lstm_bias_var->GetMutable<framework::LoDTensor>();
      const auto& fc_bias_tensor = fc_bias_var->Get<framework::LoDTensor>();

      auto lstm_bias_data =
          lstm_bias_tensor->mutable_data<float>(platform::CPUPlace());
      auto* fc_bias_data = fc_bias_tensor.data<float>();

      for (int i = 0; i < lstm_bias_tensor->numel(); i++) {
        lstm_bias_data[i] += fc_bias_data[i];
      }
    }

    op_desc.SetInput("H0", {});
    op_desc.SetInput("C0", {});
    op_desc.SetOutput("Hidden", {hidden->Name()});
    op_desc.SetOutput("Cell", {cell->Name()});
    op_desc.SetOutput("XX", {xx->Name()});
    op_desc.SetAttr("is_reverse", lstm->Op()->GetAttr("is_reverse"));
    op_desc.SetAttr("use_peepholes", lstm->Op()->GetAttr("use_peepholes"));
    op_desc.SetAttr("use_mkldnn", use_mkldnn);
    // TODO(TJ): get from attr
    op_desc.SetAttr("use_seq", true);

// Create temp variables.
#define OP_SET_OUT(x)                            \
  const std::string x = patterns::UniqueKey(#x); \
  op_desc.SetOutput(#x, {x});

    OP_SET_OUT(BatchedGate);
    OP_SET_OUT(BatchedCellPreAct);
    OP_SET_OUT(BatchedInput);
    OP_SET_OUT(CheckedCell);
    OP_SET_OUT(BatchedCell);
    OP_SET_OUT(BatchedHidden);
    OP_SET_OUT(ReorderedH0);
    OP_SET_OUT(ReorderedC0);
#undef OP_SET_OUT

    auto* op = graph->CreateOpNode(&op_desc);

    IR_NODE_LINK_TO(input, op);
    IR_NODE_LINK_TO(weight_x, op);
    IR_NODE_LINK_TO(weight_h, op);
    IR_NODE_LINK_TO(bias, op);
    IR_NODE_LINK_TO(op, hidden);
    IR_NODE_LINK_TO(op, cell);
    IR_NODE_LINK_TO(op, xx);

#define IR_NODE(x)                                 \
  VarDesc key_##x(x);                              \
  key_##x.SetPersistable(false);                   \
  auto* node_##x = graph->CreateVarNode(&key_##x); \
  IR_NODE_LINK_TO(op, node_##x);

    IR_NODE(BatchedGate);
    IR_NODE(BatchedCellPreAct);
    IR_NODE(BatchedInput);
    IR_NODE(CheckedCell);
    IR_NODE(BatchedCell);
    IR_NODE(BatchedHidden);
    IR_NODE(ReorderedH0);
    IR_NODE(ReorderedC0);
#undef IR_NODE

    return op;
  };

  int fusion_count{0};

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    GET_IR_NODE_FROM_SUBGRAPH(lstm, lstm, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Weight, Weight, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Bias, Bias, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Hidden, Hidden, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchCellPreAct, BatchCellPreAct, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(BatchGate, BatchGate, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(Cell, Cell, lstm_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, fc_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, fc_pattern);
    const bool use_mkldnn =
        (mul->Op()->GetAttrIfExists<bool>("use_mkldnn") &&
         lstm->Op()->GetAttrIfExists<std::string>("gate_activation") ==
             "sigmoid" &&
         lstm->Op()->GetAttrIfExists<std::string>("cell_activation") ==
             "tanh" &&
         lstm->Op()->GetAttrIfExists<std::string>("candidate_activation") ==
             "tanh");

    if (with_fc_bias) {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, elementwise_add_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(fc_bias, bias, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out, fc_pattern);
      GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add, fc_pattern);
      lstm_creator(lstm, subgraph.at(x), w, Weight, Bias, Hidden, Cell, fc_out,
                   fc_bias, use_mkldnn);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, lstm, elementwise_add, mul_out, BatchGate, BatchCellPreAct});
      GraphSafeRemoveNodes(graph, marked_nodes);
    } else {
      GET_IR_NODE_FROM_SUBGRAPH(fc_out, mul_out, fc_pattern);
      lstm_creator(lstm, subgraph.at(x), w, Weight, Bias, Hidden, Cell, fc_out,
                   nullptr, use_mkldnn);
      // Remove unneeded nodes.
      std::unordered_set<const Node*> marked_nodes(
          {mul, lstm, BatchGate, BatchCellPreAct});
      GraphSafeRemoveNodes(graph, marked_nodes);
    }

    ++fusion_count;
  };

  gpd(graph, handler);

  return fusion_count;
}

void MulLstmFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), false /*with_fc_bias*/);

  AddStatis(fusion_count);
}

void FCLstmFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count =
      BuildFusion(graph, name_scope_, param_scope(), true /*with_fc_bias*/);

  AddStatis(fusion_count);

  string::PrettyLogDetail("---    fused %d pairs of fc lstm patterns",
                          fusion_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mul_lstm_fuse_pass, paddle::framework::ir::MulLstmFusePass);
REGISTER_PASS(fc_lstm_fuse_pass, paddle::framework::ir::FCLstmFusePass);

REGISTER_PASS_CAPABILITY(fc_lstm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("lstm", 0)
            .EQ("fusion_lstm", 0));
REGISTER_PASS_CAPABILITY(mul_lstm_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .EQ("lstm", 0)
            .EQ("fusion_lstm", 0));
