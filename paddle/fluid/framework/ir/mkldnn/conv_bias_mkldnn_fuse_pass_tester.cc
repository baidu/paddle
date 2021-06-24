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

#include "paddle/fluid/framework/ir/mkldnn/conv_bias_mkldnn_fuse_pass.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/platform/place.h"

#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/imperative/type_defs.h"

namespace paddle {
namespace framework {
namespace ir {

void SetOp(ProgramDesc* prog, const std::string& type, const std::string& name,
           const std::vector<std::string>& inputs,
           const std::vector<std::string>& outputs) {
  auto* op = prog->MutableBlock(0)->AppendOp();
  op->SetType(type);
  if (type == "conv2d") {
    const std::vector<int> strides({1, 1});
    const std::vector<int> paddings({0, 0});
    const std::vector<int> dilations({1, 1});
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("name", name);
    op->SetAttr("strides", strides);
    op->SetAttr("groups", 1);
    op->SetAttr("paddings", paddings);
    op->SetAttr("padding_algorithm", std::string("EXPLICIT"));
    op->SetAttr("dilations", dilations);
    op->SetAttr("data_format", std::string("NCHW"));

    op->SetOutput("Output", outputs);
    op->SetInput("Input", {inputs[0]});
    op->SetInput("Filter", {inputs[1]});
    if (inputs.size() > 2)
      op->SetInput("Bias", {inputs[2]});
    else
      op->SetInput("Bias", {});
  } else if (type == "elementwise_add") {
    op->SetAttr("use_mkldnn", true);
    op->SetAttr("axis", -1);
    op->SetInput("X", {inputs[0]});
    op->SetInput("Y", {inputs[1]});
    op->SetOutput("Out", outputs);
  }
  op->SetAttr(OpProtoAndCheckerMaker::OpRoleAttrName(),
              static_cast<int>(OpRole::kForward));
}

// (c, weights)->conv->f
// (f)->elementwise_add->g
ProgramDesc BuildProgramDesc(bool convWithExistingBias) {
  ProgramDesc prog;
  std::vector<std::string> nodes{"c", "weights", "f", "eltwise_bias", "g"};
  if (convWithExistingBias) nodes.push_back("conv_bias");
  for (auto& v : nodes) {
    auto* var = prog.MutableBlock(0)->Var(v);
    var->SetType(proto::VarType::LOD_TENSOR);
    if (v == "weights" || v == "conv_bias" || v == "eltwise_bias") {
      var->SetPersistable(true);
    }
  }

  // conv+bias, both with MKL-DNN
  if (convWithExistingBias) {
    SetOp(&prog, "conv2d", "conv",
          std::vector<std::string>({"c", "weights", "conv_bias"}),
          std::vector<std::string>({"f"}));
  } else {
    SetOp(&prog, "conv2d", "conv", std::vector<std::string>({"c", "weights"}),
          std::vector<std::string>({"f"}));
  }
  SetOp(&prog, "elementwise_add", "eltwise",
        std::vector<std::string>({"f", "eltwise_bias"}),
        std::vector<std::string>({"g"}));

  return prog;
}

void InitTensorHolder(Scope* scope, const paddle::platform::Place& place,
                      const char* var_name) {
  auto x = scope->Var(var_name);
  auto tensor = x->GetMutable<LoDTensor>();
  tensor->mutable_data(place, proto::VarType::FP32, 1);
}

void MainTest(bool convWithExistingBias) {
  auto prog = BuildProgramDesc(convWithExistingBias);
  std::unique_ptr<ir::Graph> graph(new ir::Graph(prog));
  auto place = paddle::platform::CPUPlace();
  NaiveExecutor exe{place};
  Scope scope;
  // Init scope, as it is used in pass
  exe.CreateVariables(prog, 0, true, &scope);
  if (convWithExistingBias) {
    InitTensorHolder(&scope, place, "conv_bias");
    InitTensorHolder(&scope, place, "eltwise_bias");
  }
  graph->SetNotOwned(kParamScopeAttr, &scope);

  auto pass = PassRegistry::Instance().Get("conv_bias_mkldnn_fuse_pass");

  int original_nodes_num = graph->Nodes().size();

  graph.reset(pass->Apply(graph.release()));

  int current_nodes_num = graph->Nodes().size();

  // Remove 3 Nodes: Conv, Bias, conv_out
  // Add 1 Node: ConvBias
  EXPECT_EQ(original_nodes_num - 2, current_nodes_num);

  // Assert conv_bias op in newly generated graph
  int conv_bias_count = 0;

  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "conv2d") {
      auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      // check if "conv" convolution is fused
      auto op_name = BOOST_GET_CONST(std::string, op->GetAttr("name"));
      if (op_name == "conv") {
        auto input_names = op->InputNames();
        ASSERT_TRUE(std::find(input_names.begin(), input_names.end(), "Bias") !=
                    input_names.end());
        auto bias = op->Input("Bias");
        if (bias.size()) {
          ++conv_bias_count;
        }
      }
    }
  }
  EXPECT_EQ(conv_bias_count, 1);
}

TEST(ConvBiasFusePass, bias_free_conv) { MainTest(false); }

TEST(ConvBiasFusePass, conv_with_existing_bias) { MainTest(true); }

TEST(ConvBiasFusePass, conv3d) {
  Conv3DBiasFusePass pass;
  ASSERT_EQ(pass.type(), std::string("conv3d"));
}

TEST(ConvBiasFusePass, conv2d_transpose) {
  Conv2DTransposeBiasFusePass pass;
  ASSERT_EQ(pass.type(), std::string("conv2d_transpose"));
}

TEST(ConvBiasFusePass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("conv_bias_mkldnn_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(conv_bias_mkldnn_fuse_pass);
