// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/generate_pass.h"

namespace paddle {
namespace framework {
namespace ir {

GeneratePass::GeneratePass(const std::string& binary_str) {
  multi_pass_desc_.ParseFromString(binary_str);
}

GeneratePass::GeneratePass(const proto::PassDesc& pass_desc) {
  multi_pass_desc_.add_pass_descs()->CopyFrom(pass_desc);
}

void GeneratePass::ApplyImpl(Graph* graph) const {
  VerifyDesc();
  for (const proto::PassDesc& pass_desc : multi_pass_desc_.pass_descs()) {
    GraphPatternDetector detector;
    InitPattern(detector.mutable_pattern(), pass_desc);
    detector(graph, Substitute(detector.pattern(), pass_desc));
  }
  VerifyGraph();
}

void GeneratePass::InitPattern(PDPattern* pattern,
                               const proto::PassDesc& pass_desc) const {
  const proto::BlockDesc& block = pass_desc.pattern().blocks(0);
  // 1. process Op and out Var
  for (int index = 0; index < block.ops_size(); ++index) {
    const proto::OpDesc& op = block.ops(index);
    PDNode* op_pdnode =
        pattern->NewNode(string::Sprintf("%s.%d", op.type(), index));
    op_pdnode->assert_is_op(op.type());
    for (const proto::OpDesc::Var& out : op.outputs()) {
      PDNode* out_pdnode = pattern->NewNode(out.arguments(0));
      out_pdnode->AsOutput()->assert_is_op_output(op.type());
      pattern->AddEdge(op_pdnode, out_pdnode);
    }
  }
  // 2. process in Var and out Var
  for (int index = 0; index < block.ops_size(); ++index) {
    const proto::OpDesc& op = block.ops(index);
    PDNode* op_pdnode =
        pattern->RetrieveNode(string::Sprintf("%s.%d", op.type(), index));
    for (const proto::OpDesc::Var& in : op.inputs()) {
      PDNode* in_pdnode = pattern->RetrieveNode(in.arguments(0));
      if (nullptr != in_pdnode) {
        // out Var used by Op in pattern is intermediate role
        in_pdnode->AsIntermediate();
      } else {
        in_pdnode = pattern->NewNode(in.arguments(0));
        in_pdnode->AsInput()->assert_is_op_input(op.type());
      }
      // in_pdnode->assert_is_persistable_var();
      pattern->AddEdge(in_pdnode, op_pdnode);
    }
  }
}

GraphPatternDetector::handle_t GeneratePass::Substitute(
    const PDPattern& pattern, const proto::PassDesc& pass_desc) const {
  GraphPatternDetector::handle_t handler = [&](
      const GraphPatternDetector::subgraph_t& subgraph, Graph* graph) {
    const proto::BlockDesc& block = pass_desc.replace().blocks(0);
    std::unordered_set<const Node*> remove_nodes;
    for (const auto& pdnode : pattern.nodes()) {
      remove_nodes.emplace(subgraph.at(pdnode.get()));
    }
    std::map<std::string, Node*> var_node_map;
    // var_node_map from VarMap
    for (const proto::PassDesc::VarMap& var_map : pass_desc.var_maps()) {
      PDNode* pd_node = pattern.RetrieveNode(var_map.pattern_var());
      Node* node = subgraph.at(pd_node);
      var_node_map.insert({var_map.replace_var(), node});
      remove_nodes.erase(node);
    }
    for (const proto::OpDesc& op : block.ops()) {
      std::vector<Node *> in_nodes, out_nodes;
      OpDesc op_desc;
      op_desc.SetType(op.type());
      for (const proto::OpDesc::Var& in : op.inputs()) {
        Node* in_node = nullptr;
        auto iter = var_node_map.find(in.arguments(0));
        if (iter != var_node_map.end()) {
          in_node = iter->second;
          op_desc.SetInput(in.parameter(), {in_node->Name()});
          in_nodes.push_back(in_node);
        } else {
          // create node
        }
      }
      for (const proto::OpDesc::Var& out : op.outputs()) {
        Node* out_node = nullptr;
        auto iter = var_node_map.find(out.arguments(0));
        if (iter != var_node_map.end()) {
          out_node = iter->second;
          op_desc.SetOutput(out.parameter(), {out_node->Name()});
          out_nodes.push_back(out_node);
        } else {
          // create node
        }
      }
      Node* op_node = graph->CreateOpNode(&op_desc);
      for (Node* node : in_nodes) {
        IR_NODE_LINK_TO(node, op_node);
      }
      for (Node* node : out_nodes) {
        IR_NODE_LINK_TO(op_node, node);
      }
    }
    GraphSafeRemoveNodes(graph, remove_nodes);
  };
  return handler;
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
