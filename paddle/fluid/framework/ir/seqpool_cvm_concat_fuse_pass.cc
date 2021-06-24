/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/framework/ir/seqpool_cvm_concat_fuse_pass.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {

class Graph;
class Node;

namespace {
static PDNode* BuildCVMConcatPattern(PDPattern* pattern) {
  auto cvm_behind_x = [](Node* x) -> bool {
    Node* adj = x->inputs[0];
    Node* alt = x->inputs[0]->inputs[0];
    return x && adj && adj->IsVar() && alt->IsOp() &&
           alt->Op()->Type() == "cvm";
  };
  auto* concat_op_node = pattern->NewNode("concat_op")
                             ->assert_is_op("concat")
                             ->assert_op_attr<int>("axis", 1)
                             ->assert_more(cvm_behind_x);
  return concat_op_node;
}

static void GetConcatNodes(ir::Graph* graph, std::vector<Node*>* concat_nodes) {
  GraphPatternDetector gpd;
  auto* pattern = gpd.mutable_pattern();
  auto concat_op_node = BuildCVMConcatPattern(pattern);
  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    Node* concat_op = subgraph.at(concat_op_node);
    concat_nodes->push_back(concat_op);
  };
  gpd(graph, handler);
}
}  // anonymous namespace

SeqPoolCVMConcatFusePass::SeqPoolCVMConcatFusePass() {
  AddOpCompat(OpCompat("sequence_pool"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddOutput("MaxIndex")
      .IsTensor()
      .IsOptional()
      .End()
      .AddAttr("pooltype")
      .IsStringIn({"AVERAGE", "SUM", "SQRT", "LAST", "FIRST", "MAX"})
      .End()
      .AddAttr("pad_value")
      .End();
  AddOpCompat(OpCompat("cvm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("CVM")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddAttr("use_cvm")
      .IsBoolEQ(true)
      .End();
  AddOpCompat(OpCompat("concat"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("AxisTensor")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End()
      .AddAttr("axis")
      .IsNumGE(1)
      .End();
}

void SeqPoolCVMConcatFusePass::ApplyImpl(ir::Graph* graph) const {
  FusePassBase::Init("seqpool_cvm_concat_fuse", graph);
  std::vector<Node*> concat_nodes;
  GetConcatNodes(graph, &concat_nodes);

  int count = 0;
  for (auto* concat_node : concat_nodes) {
    GraphPatternDetector gpd;
    auto* pattern = gpd.mutable_pattern();
    auto concat_before_x = [=](Node* x) -> bool {
      return x && x->outputs[0] == concat_node;
    };
    PDNode* seqpool_in_var_node =
        pattern->NewNode("seqpool_in_var")
            ->assert_is_only_input_of_op("sequence_pool");
    PDNode* seqpool_op_node =
        pattern->NewNode("seqpool_op")
            ->assert_is_op("sequence_pool")
            ->assert_op_attr<std::string>("pooltype", "SUM");
    PDNode* seqpool_out_var_node =
        pattern->NewNode("seqpool_out_var")
            ->assert_is_op_nth_output("sequence_pool", "Out", 0)
            ->assert_is_op_nth_input("cvm", "X", 0);
    PDNode* seqpool_idx_out_var_node =
        pattern->NewNode("seqpool_idx_out_var")
            ->assert_is_op_nth_output("sequence_pool", "MaxIndex", 0);
    PDNode* cvm_op_node =
        pattern->NewNode("cvm_op")->assert_is_op("cvm")->assert_op_attr<bool>(
            "use_cvm", true);
    PDNode* cvm_out_var_node = pattern->NewNode("cvm_op_out_var")
                                   ->assert_is_op_nth_output("cvm", "Y", 0)
                                   ->assert_more(concat_before_x);
    PDNode* cvm_cvm_in_var_node = pattern->NewNode("cvm_cvm_in_var")
                                      ->assert_is_op_nth_input("cvm", "CVM", 0);

    seqpool_op_node->LinksFrom({seqpool_in_var_node})
        .LinksTo({seqpool_out_var_node, seqpool_idx_out_var_node});
    seqpool_out_var_node->LinksFrom({seqpool_op_node}).LinksTo({cvm_op_node});
    cvm_op_node->LinksTo({cvm_out_var_node})
        .LinksFrom({cvm_cvm_in_var_node, seqpool_out_var_node});

    std::unordered_map<std::string, Node*> ins_to_concat;
    std::vector<Node*> subgraph_ins;
    std::vector<std::string> subgraph_ins_name;
    std::unordered_set<const Node*> marked_nodes;

    Node* cvm_input_of_cvm;
    Node* concat_out_var = concat_node->outputs[0];

    GraphPatternDetector::handle_t handler = [&](
        const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
      Node* seqpool_in_var = subgraph.at(seqpool_in_var_node);
      Node* seqpool_op = subgraph.at(seqpool_op_node);
      Node* seqpool_out_var = subgraph.at(seqpool_out_var_node);
      Node* seqpool_idx_out_var = subgraph.at(seqpool_idx_out_var_node);
      Node* cvm_op = subgraph.at(cvm_op_node);
      Node* cvm_out_var = subgraph.at(cvm_out_var_node);
      cvm_input_of_cvm = subgraph.at(cvm_cvm_in_var_node);
      marked_nodes.insert({seqpool_op, seqpool_out_var, seqpool_idx_out_var,
                           cvm_op, cvm_out_var, concat_node});
      ins_to_concat[cvm_out_var->Name()] = seqpool_in_var;
    };
    gpd(graph, handler);

    if (!ins_to_concat.empty()) {
      for (const auto* in : concat_node->inputs) {
        subgraph_ins.push_back(ins_to_concat.at(in->Name()));
        subgraph_ins_name.push_back(ins_to_concat.at(in->Name())->Name());
      }

      // Create New OpDesc
      OpDesc op_desc;
      op_desc.SetType("fusion_seqpool_cvm_concat");
      op_desc.SetInput("X", subgraph_ins_name);
      op_desc.SetInput("CVM", {cvm_input_of_cvm->Name()});
      op_desc.SetAttr("pooltype", std::string("SUM"));
      op_desc.SetAttr("use_cvm", true);
      op_desc.SetAttr("axis", concat_node->Op()->GetAttr("axis"));
      op_desc.SetOutput("Out", {concat_out_var->Name()});
      auto* op = graph->CreateOpNode(&op_desc);

      for (size_t i = 0; i < subgraph_ins.size(); ++i) {
        IR_NODE_LINK_TO(subgraph_ins[i], op);
      }
      IR_NODE_LINK_TO(cvm_input_of_cvm, op);
      IR_NODE_LINK_TO(op, concat_out_var);

      GraphSafeRemoveNodes(graph, marked_nodes);
      count++;
    }
  }
  AddStatis(count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(seqpool_cvm_concat_fuse_pass,
              paddle::framework::ir::SeqPoolCVMConcatFusePass);
