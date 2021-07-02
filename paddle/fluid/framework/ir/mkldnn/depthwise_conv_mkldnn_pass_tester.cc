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

#include "paddle/fluid/framework/ir/mkldnn/depthwise_conv_mkldnn_pass.h"

#include <gtest/gtest.h>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs, bool use_mkldnn = false) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  op->SetAttr("use_mkldnn", use_mkldnn);
  op->SetAttr("name", name);
  op->SetAttr("groups", 1);
  op->SetAttr("padding_algorithm", std::string("EXPLICIT"));
  op->SetAttr("data_format", std::string("NCHW"));
  op->SetAttr("strides", std::vector<int>({1, 1}));
  op->SetAttr("dilations", std::vector<int>({1, 1}));
  op->SetAttr("paddings", std::vector<int>({0, 0}));
  op->SetInput("Input", {inputs[0]});
  op->SetInput("Filter", {inputs[1]});
  op->SetInput("Bias", {inputs[2]});
  op->SetOutput("Output", outputs);
}

// (a, weights, bias)->depthwise conv mkldnn->b
// (b, weights2, bias2)->depthwise conv no mkldnn->c
// (c, weights3, bias3)->conv mkldnn->d
// (d, weights3, bias3)->conv no mkldnn->e
ProgramDesc BuildProgramDesc() {
  ProgramDesc prog;
  for (auto& v : std::vector<std::string>(
           {"a", "b", "c", "d", "e", "weights", "bias", "weights2", "bias2",
            "weights3", "bias3", "weights4", "bias4"})) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::SELECTED_ROWS);
    if (v == "weights" || v == "bias" || v == "weights2" || v == "bias2" ||
        v == "weights3" || v == "bias3" || v == "weights4" || v == "bias4") {
      var->SetPersistable(true);
    }
  }

  // depthwise conv with MKL-DNN
  SetOp(&prog, "depthwise_conv2d", "conv1",
        std::vector<std::string>({"a", "weights", "bias"}),
        std::vector<std::string>({"b"}), true);
  // depthwise conv without MKL-DNN
  SetOp(&prog, "depthwise_conv2d", "conv2",
        std::vector<std::string>({"b", "weights2", "bias2"}),
        std::vector<std::string>({"c"}), false);
  // conv with MKL-DNN
  SetOp(&prog, "conv2d", "conv3",
        std::vector<std::string>({"c", "weights3", "bias3"}),
        std::vector<std::string>({"d"}), true);
  // conv without MKL-dNN
  SetOp(&prog, "conv2d", "conv4",
        std::vector<std::string>({"d", "weights4", "bias4"}),
        std::vector<std::string>({"e"}), false);

  return prog;
}

TEST(DepthwiseConvMKLDNNPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("depthwise_conv_mkldnn_pass"));
}

TEST(DepthwiseConvMKLDNNPass, basic) {
  auto prog = BuildProgramDesc();

  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));

  auto pass = PassRegistry::Instance().Get("depthwise_conv_mkldnn_pass");

  struct counters {
    int mkldnn_depthwise_conv_nodes;
    int other_depthwise_conv_nodes;
    int mkldnn_conv_nodes;
    int other_conv_nodes;
  };

  counters before{1, 1, 1, 1};

  graph.reset(pass->Apply(graph.release()));

  // initialize counters before loop
  counters after{0, 0, 0, 0};

  for (auto* node : graph->Nodes()) {
    if (node->IsOp()) {
      auto* op = node->Op();
      if (op->Type() == "conv2d") {
        if (BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")))
          after.mkldnn_conv_nodes++;
        else
          after.other_conv_nodes++;
      } else if (op->Type() == "depthwise_conv2d") {
        if (BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")))
          after.mkldnn_depthwise_conv_nodes++;
        else
          after.other_depthwise_conv_nodes++;
      }
    }
  }

  EXPECT_EQ(after.other_depthwise_conv_nodes,
            before.other_depthwise_conv_nodes);
  EXPECT_EQ(after.other_conv_nodes, before.other_conv_nodes);
  EXPECT_EQ(after.mkldnn_depthwise_conv_nodes,
            before.mkldnn_depthwise_conv_nodes - 1);
  EXPECT_EQ(after.mkldnn_conv_nodes, before.mkldnn_conv_nodes + 1);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(depthwise_conv_mkldnn_pass);
