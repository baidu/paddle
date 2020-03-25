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

#include "paddle/fluid/framework/ir/multihead_matmul_fuse_pass.h"
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace ir {
namespace patterns {

static void ReplaceOutputVar(Node* op, Node* old_var, Node* new_var) {
  if (op->IsOp() && op->Op()) {
    new_var->inputs.push_back(op);
    for (size_t i = 0; i < op->outputs.size(); ++i) {
      if (op->outputs[i] == old_var) {
        op->outputs[i] = new_var;
        op->Op()->RenameOutput(old_var->Name(), new_var->Name());
      }
    }
  }
}

static int BuildFusion(Graph* graph, const std::string& name_scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  MultiHeadMatmulPattern multihead_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("X"))->assert_var_not_persistable();

  multihead_pattern(x);
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* x, Node* mul0, Node* mul1, Node* mul2, Node* mul0_out,
      Node* mul1_out, Node* mul2_out, Node* eltadd0_b, Node* eltadd1_b,
      Node* eltadd2_b, Node* eltadd_qk_b, Node* reshape2,
      Node* reshape2_qkv_out, Node* scale, Node* scale_out) {
    auto scale_attr = boost::get<float>(scale->Op()->GetAttr("scale"));
    // auto scale_bias = boost::get<float>(scale->Op()->GetAttr("bias"));
    // bool after_scale =
    //    boost::get<bool>(scale->Op()->GetAttr("bias_after_scale"));

    // create multihead
    OpDesc multihead_op_desc;

    // create tmp tensor
    VarDesc k_var_desc(*mul1_out->Var());
    k_var_desc.SetName("K" + mul1_out->Name());
    auto* k_var_node = graph->CreateVarNode(&k_var_desc);

    VarDesc q_var_desc(*mul0_out->Var());
    q_var_desc.SetName("Q" + mul0_out->Name());
    auto* q_var_node = graph->CreateVarNode(&q_var_desc);

    VarDesc v_var_desc(*mul2_out->Var());
    v_var_desc.SetName("V" + mul2_out->Name());
    auto* v_var_node = graph->CreateVarNode(&v_var_desc);

    auto reshape_desc = reshape2->Op();
    int head_number =
        boost::get<std::vector<int>>(reshape_desc->GetAttr("shape")).at(2);

    ReplaceOutputVar(mul0, mul0_out, q_var_node);
    ReplaceOutputVar(mul1, mul1_out, k_var_node);
    ReplaceOutputVar(mul2, mul2_out, v_var_node);

    multihead_op_desc.SetType("multihead_matmul");
    multihead_op_desc.SetInput("Q", {q_var_node->Name()});
    multihead_op_desc.SetInput("K", {k_var_node->Name()});
    multihead_op_desc.SetInput("V", {v_var_node->Name()});

    multihead_op_desc.SetInput("BiasQ", {eltadd0_b->Name()});
    multihead_op_desc.SetInput("BiasK", {eltadd1_b->Name()});
    multihead_op_desc.SetInput("BiasV", {eltadd2_b->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);
    IR_NODE_LINK_TO(q_var_node, multihead);
    IR_NODE_LINK_TO(k_var_node, multihead);
    IR_NODE_LINK_TO(v_var_node, multihead);

    IR_NODE_LINK_TO(eltadd0_b, multihead);
    IR_NODE_LINK_TO(eltadd1_b, multihead);
    IR_NODE_LINK_TO(eltadd2_b, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out, reshape2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out, transpose2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1_out, reshape2_1_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1_out, transpose2_1_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2_out, reshape2_2_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2_out, transpose2_2_out,
                              multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out, softmax_qk_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out, matmul_qkv_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out, reshape2_qkv_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv, transpose2_qkv,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out, transpose2_qkv_out,
                              multihead_pattern);

    fuse_creater(layer_norm, mul0, mul1, mul2, mul0_out, mul1_out, mul2_out,
                 eltadd0_b, eltadd1_b, eltadd2_b, eltadd_qk_b, reshape2_0,
                 reshape2_qkv_out, scale, scale_out);

    std::unordered_set<const Node*> marked_nodes(
        {eltadd0,
         eltadd1,
         eltadd2,
         eltadd0_out,
         eltadd1_out,
         eltadd2_out,
         reshape2_0,
         reshape2_1,
         reshape2_2,
         reshape2_0_out,
         reshape2_1_out,
         reshape2_2_out,
         transpose2_0,
         transpose2_1,
         transpose2_2,
         transpose2_0_out,
         transpose2_1_out,
         transpose2_2_out,
         matmul_qk,
         matmul_qk_out,
         eltadd_qk,
         eltadd_qk_out,
         softmax_qk,
         softmax_qk_out,  // dropout_qk, dropout_qk_out,
         transpose2_qkv,
         transpose2_qkv_out,
         matmul_qkv,
         matmul_qkv_out,
         mul0_out,
         mul1_out,
         mul2_out,
         reshape2_qkv,
         scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

PDNode* MultiHeadMatmulPattern::operator()(paddle::framework::ir::PDNode* x) {
  // Create shared nodes.
  auto* layer_norm = pattern->NewNode(layer_norm_repr());

  auto* layer_norm_out_var = pattern->NewNode(layer_norm_out_repr());
  layer_norm_out_var->assert_is_op_input("mul");

  // First path with scale
  auto* mul0 = pattern->NewNode(mul0_repr())->assert_is_op("mul");
  auto* mul0_w_var = pattern->NewNode(mul0_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul0_out_var =
      pattern->NewNode(mul0_out_repr())->assert_is_op_output("mul");

  decltype(mul0) eltadd0;
  decltype(mul0) eltadd0_b_var;
  decltype(mul0) eltadd0_out_var;

  mul0_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  eltadd0 = pattern->NewNode(eltadd0_repr())->assert_is_op("elementwise_add");
  eltadd0_b_var = pattern->NewNode(eltadd0_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd0_out_var = pattern->NewNode(eltadd0_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd0_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_0 =
      pattern->NewNode(reshape2_0_repr())->assert_is_op("reshape2");

  auto* reshape2_0_out_var =
      pattern->NewNode(reshape2_0_out_repr())->assert_is_op_output("reshape2");
  reshape2_0_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_0 =
      pattern->NewNode(transpose2_0_repr())->assert_is_op("transpose2");
  auto* transpose2_0_out_var = pattern->NewNode(transpose2_0_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_0_out_var->AsIntermediate()->assert_is_op_input("scale");

  auto* scale = pattern->NewNode(scale_repr())->assert_is_op("scale");
  auto* scale_out_var =
      pattern->NewNode(scale_out_repr())->assert_is_op_output("scale");
  scale_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul_qk = pattern->NewNode(matmul_qk_repr())->assert_is_op("matmul");
  auto* matmul_qk_out_var =
      pattern->NewNode(matmul_qk_out_repr())->assert_is_op_output("matmul");
  matmul_qk_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");

  auto* eltadd_qk =
      pattern->NewNode(eltadd_qk_repr())->assert_is_op("elementwise_add");
  auto* eltadd_qk_b_var = pattern->NewNode(eltadd_qk_b_repr())
                              ->AsInput()
                              ->assert_is_op_input("elementwise_add", "Y");
  auto* eltadd_qk_out_var = pattern->NewNode(eltadd_qk_out_repr())
                                ->assert_is_op_output("elementwise_add");
  eltadd_qk_out_var->AsIntermediate()->assert_is_op_input("softmax");

  auto* softmax_qk =
      pattern->NewNode(softmax_qk_repr())->assert_is_op("softmax");
  auto* softmax_qk_out_var =
      pattern->NewNode(softmax_qk_out_repr())->assert_is_op_output("softmax");
  softmax_qk_out_var->AsIntermediate()->assert_is_op_input("matmul");

  auto* matmul_qkv =
      pattern->NewNode(matmul_qkv_repr())->assert_is_op("matmul");
  auto* matmul_qkv_out_var =
      pattern->NewNode(matmul_qkv_out_repr())->assert_is_op_output("matmul");
  matmul_qkv_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_qkv =
      pattern->NewNode(transpose2_qkv_repr())->assert_is_op("transpose2");
  auto* transpose2_qkv_out_var = pattern->NewNode(transpose2_qkv_out_repr())
                                     ->assert_is_op_output("transpose2");
  transpose2_qkv_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_qkv =
      pattern->NewNode(reshape2_qkv_repr())->assert_is_op("reshape2");
  auto* reshape2_qkv_out_var = pattern->NewNode(reshape2_qkv_out_repr())
                                   ->assert_is_op_output("reshape2");
  reshape2_qkv_out_var->assert_is_op_input("mul");

  // Second path to matmul
  auto* mul1 = pattern->NewNode(mul1_repr())->assert_is_op("mul");
  auto* mul1_w_var = pattern->NewNode(mul1_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul1_out_var =
      pattern->NewNode(mul1_out_repr())->assert_is_op_output("mul");

  decltype(mul1) eltadd1;
  decltype(mul1) eltadd1_b_var;
  decltype(mul1) eltadd1_out_var;

  mul1_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd1 = pattern->NewNode(eltadd1_repr())->assert_is_op("elementwise_add");
  eltadd1_b_var = pattern->NewNode(eltadd1_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd1_out_var = pattern->NewNode(eltadd1_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd1_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_1 =
      pattern->NewNode(reshape2_1_repr())->assert_is_op("reshape2");

  auto* reshape2_1_out_var =
      pattern->NewNode(reshape2_1_out_repr())->assert_is_op_output("reshape2");
  reshape2_1_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_1 =
      pattern->NewNode(transpose2_1_repr())->assert_is_op("transpose2");
  auto* transpose2_1_out_var = pattern->NewNode(transpose2_1_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_1_out_var->AsIntermediate()->assert_is_op_input(
      "matmul");  // link to matmul qk

  // Third path to matmul
  auto* mul2 = pattern->NewNode(mul2_repr())->assert_is_op("mul");
  auto* mul2_w_var = pattern->NewNode(mul2_w_repr())
                         ->AsInput()
                         ->assert_is_op_input("mul", "Y");
  auto* mul2_out_var =
      pattern->NewNode(mul2_out_repr())->assert_is_op_output("mul");

  decltype(mul2) eltadd2;
  decltype(mul2) eltadd2_b_var;
  decltype(mul2) eltadd2_out_var;

  mul2_out_var->AsIntermediate()->assert_is_op_input("elementwise_add");
  eltadd2 = pattern->NewNode(eltadd2_repr())->assert_is_op("elementwise_add");
  eltadd2_b_var = pattern->NewNode(eltadd2_b_repr())
                      ->AsInput()
                      ->assert_is_op_input("elementwise_add", "Y");

  eltadd2_out_var = pattern->NewNode(eltadd2_out_repr())
                        ->assert_is_op_output("elementwise_add");
  eltadd2_out_var->AsIntermediate()->assert_is_op_input("reshape2");

  auto* reshape2_2 =
      pattern->NewNode(reshape2_2_repr())->assert_is_op("reshape2");

  auto* reshape2_2_out_var =
      pattern->NewNode(reshape2_2_out_repr())->assert_is_op_output("reshape2");
  reshape2_2_out_var->AsIntermediate()->assert_is_op_input("transpose2");

  auto* transpose2_2 =
      pattern->NewNode(transpose2_2_repr())->assert_is_op("transpose2");
  auto* transpose2_2_out_var = pattern->NewNode(transpose2_2_out_repr())
                                   ->assert_is_op_output("transpose2");
  transpose2_2_out_var->AsIntermediate()->assert_is_op_input(
      "matmul");  // link to matmul qkv

  // Link all nodes together
  layer_norm->LinksFrom({x}).LinksTo({layer_norm_out_var});
  // Q path
  mul0->LinksFrom({layer_norm_out_var, mul0_w_var}).LinksTo({mul0_out_var});
  eltadd0->LinksFrom({mul0_out_var, eltadd0_b_var}).LinksTo({eltadd0_out_var});

  reshape2_0->LinksFrom({eltadd0_out_var}).LinksTo({reshape2_0_out_var});
  transpose2_0->LinksFrom({reshape2_0_out_var}).LinksTo({transpose2_0_out_var});
  scale->LinksFrom({transpose2_0_out_var}).LinksTo({scale_out_var});
  // K path
  mul1->LinksFrom({layer_norm_out_var, mul1_w_var}).LinksTo({mul1_out_var});
  eltadd1->LinksFrom({mul1_out_var, eltadd1_b_var}).LinksTo({eltadd1_out_var});
  reshape2_1->LinksFrom({eltadd1_out_var}).LinksTo({reshape2_1_out_var});
  transpose2_1->LinksFrom({reshape2_1_out_var}).LinksTo({transpose2_1_out_var});
  // compute q*k
  matmul_qk->LinksFrom({scale_out_var, transpose2_1_out_var})
      .LinksTo({matmul_qk_out_var});
  eltadd_qk->LinksFrom({matmul_qk_out_var, eltadd_qk_b_var})
      .LinksTo({eltadd_qk_out_var});
  softmax_qk->LinksFrom({eltadd_qk_out_var}).LinksTo({softmax_qk_out_var});
  // V  path
  mul2->LinksFrom({layer_norm_out_var, mul2_w_var}).LinksTo({mul2_out_var});
  eltadd2->LinksFrom({mul2_out_var, eltadd2_b_var}).LinksTo({eltadd2_out_var});
  reshape2_2->LinksFrom({eltadd2_out_var}).LinksTo({reshape2_2_out_var});
  transpose2_2->LinksFrom({reshape2_2_out_var}).LinksTo({transpose2_2_out_var});
  // compute q*k*v
  matmul_qkv->LinksFrom({softmax_qk_out_var, transpose2_2_out_var})
      .LinksTo({matmul_qkv_out_var});
  transpose2_qkv->LinksFrom({matmul_qkv_out_var})
      .LinksTo({transpose2_qkv_out_var});
  reshape2_qkv->LinksFrom({transpose2_qkv_out_var})
      .LinksTo({reshape2_qkv_out_var});

  return transpose2_2_out_var;
}

static int BuildFusionV2(Graph* graph, const std::string& name_scope,
                         Scope* scope) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();

  // Create pattern.
  MultiHeadMatmulPattern multihead_pattern(pattern, name_scope);

  PDNode* x =
      pattern->NewNode(patterns::UniqueKey("X"))->assert_var_not_persistable();

  multihead_pattern(x);
  // Create New OpDesc
  auto fuse_creater = [&](
      Node* layer_norm_out, Node* mul0, Node* mul1, Node* mul2, Node* mul0_out,
      Node* mul1_out, Node* mul2_out, Node* mul0_w, Node* mul1_w, Node* mul2_w,
      Node* eltadd0_b, Node* eltadd1_b, Node* eltadd2_b, Node* eltadd_qk_b,
      Node* reshape2, Node* reshape2_qkv_out, Node* scale, Node* scale_out) {
    auto scale_attr = boost::get<float>(scale->Op()->GetAttr("scale"));

    auto* mul0_op_desc = mul0->Op();
    auto* mul1_op_desc = mul1->Op();
    auto* mul2_op_desc = mul2->Op();

    // mul (B * S * Hidden) x (Hidden * 3 * N * H) = (B * S * 3 * N * H)
    // bias (B * S * 3 * N * H) + bias (3 * N * H)
    // Transpose (B * S * 3 * N * H) -> (3 * B * N * S * H)
    auto* wq_tensor = scope->FindVar(mul0_w->Name())->GetMutable<LoDTensor>();
    auto* wk_tensor = scope->FindVar(mul1_w->Name())->GetMutable<LoDTensor>();
    auto* wv_tensor = scope->FindVar(mul2_w->Name())->GetMutable<LoDTensor>();

    auto* bq_tensor =
        scope->FindVar(eltadd0_b->Name())->GetMutable<LoDTensor>();
    auto* bk_tensor =
        scope->FindVar(eltadd1_b->Name())->GetMutable<LoDTensor>();
    auto* bv_tensor =
        scope->FindVar(eltadd2_b->Name())->GetMutable<LoDTensor>();

    auto combined_w_dims =
        framework::make_ddim({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    auto combined_bias_dims = framework::make_ddim({3, bq_tensor->dims()[0]});

    // create a new var in scope
    VarDesc combined_w_desc(
        patterns::PDNodeName(name_scope, "multi_head_combined_weight"));
    combined_w_desc.SetShape({wq_tensor->dims()[0], 3, wq_tensor->dims()[1]});
    combined_w_desc.SetDataType(wq_tensor->type());
    combined_w_desc.SetLoDLevel(mul0_w->Var()->GetLoDLevel());
    combined_w_desc.SetPersistable(true);

    // create a new var in scope
    VarDesc combined_bias_desc(
        patterns::PDNodeName(name_scope, "multi_head_combined_bias"));
    combined_bias_desc.SetShape({3, bq_tensor->dims()[0]});
    combined_bias_desc.SetDataType(bq_tensor->type());
    combined_bias_desc.SetLoDLevel(eltadd0_b->Var()->GetLoDLevel());
    combined_bias_desc.SetPersistable(true);

    auto* combined_w_node = graph->CreateVarNode(&combined_w_desc);
    auto* combined_w_tensor =
        scope->Var(combined_w_node->Name())->GetMutable<LoDTensor>();
    combined_w_tensor->Resize(combined_w_dims);

    // Combine the three fc weights together.
    if (mul0_op_desc->HasAttr("enable_int8") &&
        boost::get<bool>(mul0_op_desc->GetAttr("enable_int8"))) {
      auto* combined_w_data =
          combined_w_tensor->mutable_data<int8_t>(platform::CPUPlace());

      auto* wq_data = wq_tensor->data<int8_t>();
      auto* wk_data = wk_tensor->data<int8_t>();
      auto* wv_data = wv_tensor->data<int8_t>();

      std::vector<int8_t*> w_vec = {wq_data, wk_data, wv_data};
      int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];

      for (int i = 0; i < dims_h; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < dims_w; k++) {
            int out_index = i * (3 * dims_w) + j * dims_w + k;
            int in_index = i * dims_w + k;
            combined_w_data[out_index] = w_vec[j][in_index];
          }
        }
      }
    } else {
      auto* combined_w_data =
          combined_w_tensor->mutable_data<float>(platform::CPUPlace());

      auto* wq_data = wq_tensor->data<float>();
      auto* wk_data = wk_tensor->data<float>();
      auto* wv_data = wv_tensor->data<float>();

      std::vector<float*> w_vec = {wq_data, wk_data, wv_data};
      int dims_h = combined_w_dims[0], dims_w = combined_w_dims[2];

      for (int i = 0; i < dims_h; i++) {
        for (int j = 0; j < 3; j++) {
          for (int k = 0; k < dims_w; k++) {
            int out_index = i * (3 * dims_w) + j * dims_w + k;
            int in_index = i * dims_w + k;
            combined_w_data[out_index] = w_vec[j][in_index];
          }
        }
      }
    }

    auto* bq_data = bq_tensor->data<float>();
    auto* bk_data = bk_tensor->data<float>();
    auto* bv_data = bv_tensor->data<float>();

    scope->EraseVars({mul0_w->Name(), mul1_w->Name(), mul2_w->Name()});
    auto* combined_bias_node = graph->CreateVarNode(&combined_bias_desc);
    auto* combined_bias_tensor =
        scope->Var(combined_bias_node->Name())->GetMutable<LoDTensor>();

    combined_bias_tensor->Resize(combined_bias_dims);
    auto* combined_bias_data =
        combined_bias_tensor->mutable_data<float>(platform::CPUPlace());
    size_t bias_size = bq_tensor->numel();
    memcpy(combined_bias_data, bq_data, sizeof(float) * bias_size);
    memcpy(combined_bias_data + bias_size, bk_data, sizeof(float) * bias_size);
    memcpy(combined_bias_data + 2 * bias_size, bv_data,
           sizeof(float) * bias_size);

    scope->EraseVars({eltadd0_b->Name(), eltadd1_b->Name(), eltadd2_b->Name()});

    auto reshape_desc = reshape2->Op();
    int head_number =
        boost::get<std::vector<int>>(reshape_desc->GetAttr("shape")).at(2);

    OpDesc multihead_op_desc;
    multihead_op_desc.SetType("multihead_matmul");

    multihead_op_desc.SetInput("Input", {layer_norm_out->Name()});
    multihead_op_desc.SetInput("W", {combined_w_node->Name()});
    multihead_op_desc.SetInput("Bias", {combined_bias_node->Name()});
    multihead_op_desc.SetInput("BiasQK", {eltadd_qk_b->Name()});

    multihead_op_desc.SetOutput("Out", {reshape2_qkv_out->Name()});
    multihead_op_desc.SetAttr("alpha", scale_attr);
    multihead_op_desc.SetAttr("head_number", head_number);

    if (mul0_op_desc->HasAttr("enable_int8") &&
        boost::get<bool>(mul0_op_desc->GetAttr("enable_int8"))) {
      PADDLE_ENFORCE_EQ(mul0_op_desc->HasAttr("X_scale") &&
                            mul0_op_desc->HasAttr("weight_scale") &&
                            mul1_op_desc->HasAttr("weight_scale") &&
                            mul2_op_desc->HasAttr("weight_scale"),
                        true,
                        platform::errors::InvalidArgument(
                            "mul in multihead_matmul is set to use int8, but "
                            "no input scale or weight scales found."));
      multihead_op_desc.SetAttr("enable_int8",
                                mul0_op_desc->GetAttr("enable_int8"));
      multihead_op_desc.SetAttr("X_scale", mul0_op_desc->GetAttr("X_scale"));
      multihead_op_desc.SetAttr("weight0_scale",
                                mul0_op_desc->GetAttr("weight_scale"));
      multihead_op_desc.SetAttr("weight1_scale",
                                mul1_op_desc->GetAttr("weight_scale"));
      multihead_op_desc.SetAttr("weight2_scale",
                                mul2_op_desc->GetAttr("weight_scale"));
    }

    auto* multihead = graph->CreateOpNode(&multihead_op_desc);

    IR_NODE_LINK_TO(layer_norm_out, multihead);
    IR_NODE_LINK_TO(combined_w_node, multihead);
    IR_NODE_LINK_TO(combined_bias_node, multihead);
    IR_NODE_LINK_TO(eltadd_qk_b, multihead);

    IR_NODE_LINK_TO(multihead, reshape2_qkv_out);
  };

  int fusion_count{0};
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    // GET_IR_NODE_FROM_SUBGRAPH(dropout_out, dropout_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm, layer_norm, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(layer_norm_out, layer_norm_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul0, mul0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_out, mul0_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul0_w, mul0_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0, reshape2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_0_out, reshape2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0, transpose2_0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_0_out, transpose2_0_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale, scale, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(scale_out, scale_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul1, mul1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_out, mul1_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul1_w, mul1_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1, reshape2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_1_out, reshape2_1_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1, transpose2_1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_1_out, transpose2_1_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(mul2, mul2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_out, mul2_out, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul2_w, mul2_w, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2, reshape2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_2_out, reshape2_2_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2, transpose2_2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_2_out, transpose2_2_out,
                              multihead_pattern);

    // nodes need be removed
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0, eltadd0, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_b, eltadd0_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd0_out, eltadd0_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd1, eltadd1, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_b, eltadd1_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd1_out, eltadd1_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd2, eltadd2, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_b, eltadd2_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd2_out, eltadd2_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk, matmul_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qk_out, matmul_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk, eltadd_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_b, eltadd_qk_b, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(eltadd_qk_out, eltadd_qk_out, multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk, softmax_qk, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(softmax_qk_out, softmax_qk_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv, matmul_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(matmul_qkv_out, matmul_qkv_out,
                              multihead_pattern);

    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv, reshape2_qkv, multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(reshape2_qkv_out, reshape2_qkv_out,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv, transpose2_qkv,
                              multihead_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(transpose2_qkv_out, transpose2_qkv_out,
                              multihead_pattern);

    fuse_creater(layer_norm_out, mul0, mul1, mul2, mul0_out, mul1_out, mul2_out,
                 mul0_w, mul1_w, mul2_w, eltadd0_b, eltadd1_b, eltadd2_b,
                 eltadd_qk_b, reshape2_0, reshape2_qkv_out, scale, scale_out);

    std::unordered_set<const Node*> marked_nodes({eltadd0,
                                                  eltadd1,
                                                  eltadd2,
                                                  eltadd0_b,
                                                  eltadd1_b,
                                                  eltadd2_b,
                                                  eltadd0_out,
                                                  eltadd1_out,
                                                  eltadd2_out,
                                                  reshape2_0,
                                                  reshape2_1,
                                                  reshape2_2,
                                                  reshape2_0_out,
                                                  reshape2_1_out,
                                                  reshape2_2_out,
                                                  transpose2_0,
                                                  transpose2_1,
                                                  transpose2_2,
                                                  transpose2_0_out,
                                                  transpose2_1_out,
                                                  transpose2_2_out,
                                                  matmul_qk,
                                                  matmul_qk_out,
                                                  eltadd_qk,
                                                  eltadd_qk_out,
                                                  softmax_qk,
                                                  softmax_qk_out,
                                                  transpose2_qkv,
                                                  transpose2_qkv_out,
                                                  matmul_qkv,
                                                  matmul_qkv_out,
                                                  mul0,
                                                  mul1,
                                                  mul2,
                                                  mul0_out,
                                                  mul1_out,
                                                  mul2_out,
                                                  mul0_w,
                                                  mul1_w,
                                                  mul2_w,
                                                  reshape2_qkv,
                                                  scale});
    // Remove unneeded nodes.
    GraphSafeRemoveNodes(graph, marked_nodes);
    ++fusion_count;
  };
  gpd(graph, handler);

  return fusion_count;
}

}  // namespace patterns

void MultiHeadMatmulFusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);

  int fusion_count = patterns::BuildFusion(graph, name_scope_);
  AddStatis(fusion_count);
}

void MultiHeadMatmulV2FusePass::ApplyImpl(Graph* graph) const {
  FusePassBase::Init(name_scope_, graph);
  auto* scope = param_scope();
  PADDLE_ENFORCE_NOT_NULL(
      scope,
      platform::errors::Fatal(
          "During the multiheadMatmul pass, The scope should not be null."));

  patterns::BuildFusionV2(graph, name_scope_, scope);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(multihead_matmul_fuse_pass,
              paddle::framework::ir::MultiHeadMatmulFusePass);

REGISTER_PASS(multihead_matmul_fuse_pass_v2,
              paddle::framework::ir::MultiHeadMatmulV2FusePass);
