/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/op_compat_sensible_pass.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {
namespace ir {

TEST(OpCompatSensiblePass, compatOp) {
  auto lambda = [](const std::string& str) { return str == "tanh"; };
  OpCompat compat("fc");
  compat.AddAttr("in_num_col_dims")
      .IsIntIn({1, 2})
      .IsNumLE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"tanh", "sigmoid"})
      .IsStringMatch(lambda)
      .End()
      .AddAttr("test_attr")
      .IsBoolEQ(true)
      .End()
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddInput("Test")
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();

  OpDesc fc_op;

  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");
  attr_map["test_attr"] = true;

  fc_op.SetAttrMap(attr_map);

  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});
  fc_op.SetInput("W", std::vector<std::string>{"test_input_0"});
  fc_op.SetInput("Bias", std::vector<std::string>{"test_input_1"});
  fc_op.SetOutput("Out", std::vector<std::string>{"test_output"});

  EXPECT_STREQ(compat.Name().c_str(), "fc");
  EXPECT_TRUE(compat.Judge(fc_op));
}

TEST(OpCompatSensiblePass, compatOpAttribute) {
  OpCompat compat("fc");

  OpDesc fc_op;

  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  fc_op.SetAttrMap(attr_map);

  OpInfo info;
  info.checker_ = new OpAttrChecker();
  OpInfoMap::Instance().Insert("fc", info);

  EXPECT_FALSE(compat.Judge(fc_op));

  info.checker_->AddAttrChecker<int>("in_num_col_dims").SetDefault(1);

  EXPECT_TRUE(compat.Judge(fc_op));
  delete info.checker_;
}

TEST(OpCompatSensiblePass, opDefNotFound) {
  OpCompat compat("fc_1");

  OpDesc fc_op;

  compat.Judge(fc_op);

  OpCompat compat_1("");

  compat_1.Judge(fc_op);
}

TEST(OpCompatSensiblePass, compatOpAttributeOptional) {
  OpCompat compat("fc");
  compat.AddAttr("activation_type")
      .IsOptional()
      .IsStringIn({"tanh", "sigmoid"});
  OpDesc fc_op;
  EXPECT_TRUE(compat.Judge(fc_op));
}

TEST(OpCompatSensiblePass, compatOpInput) {
  OpCompat compat("fc");

  OpDesc fc_op;
  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});

  EXPECT_FALSE(compat.Judge(fc_op));

  compat.AddInput("Input").IsTensor().End().AddInput("Bias").IsTensor().End();
  EXPECT_FALSE(compat.Judge(fc_op));

  fc_op.SetInput("Bias", std::vector<std::string>{"test_input", ""});
  EXPECT_FALSE(compat.Judge(fc_op));
}

TEST(OpCompatSensiblePass, compatOutput) {
  OpCompat compat("fc");

  OpDesc fc_op;
  fc_op.SetOutput("Output", std::vector<std::string>{"test_output"});

  EXPECT_FALSE(compat.Judge(fc_op));

  compat.AddOutput("Output")
      .IsTensor()
      .End()
      .AddOutput("Output_2")
      .IsTensor()
      .End();
  EXPECT_FALSE(compat.Judge(fc_op));

  fc_op.SetOutput("Output_2", std::vector<std::string>{"test_output", ""});
  EXPECT_FALSE(compat.Judge(fc_op));
}

class OpCompatSensiblePassTest : public OpCompatSensiblePass {
 public:
  OpCompatSensiblePassTest();
  bool TestIsCompat(const OpDesc& op_desc) { return IsCompat(op_desc); }
  bool TestIsCompat(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g) {
    return IsCompat(subgraph, g);
  }
};

OpCompatSensiblePassTest::OpCompatSensiblePassTest() {
  AddOpCompat(OpCompat("fc"))
      .AddAttr("in_num_col_dims")
      .IsNumLE(1)
      .End()
      .AddAttr("activation_type")
      .IsStringIn({"tanh", "sigmoid"})
      .End()
      .AddInput("Input")
      .IsTensor()
      .End()
      .AddInput("W")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .IsOptional()
      .End()
      .AddOutput("Out")
      .IsTensor();
}

TEST(OpCompatSensiblePass, IsCompat) {
  OpCompatSensiblePassTest test;
  OpDesc fc_op;
  fc_op.SetType("fc");
  std::unordered_map<std::string, Attribute> attr_map;
  attr_map["in_num_col_dims"] = 1;
  attr_map["activation_type"] = std::string("tanh");

  fc_op.SetAttrMap(attr_map);
  fc_op.SetInput("Input", std::vector<std::string>{"test_input"});
  fc_op.SetInput("W", std::vector<std::string>{"test_input_0"});
  fc_op.SetInput("Bias", std::vector<std::string>{"test_input_1"});
  fc_op.SetOutput("Out", std::vector<std::string>{"test_output"});

  EXPECT_TRUE(test.TestIsCompat(fc_op));
}

TEST(OpCompatSensiblePass, IsCompatFail) {
  OpCompatSensiblePassTest test;
  GraphPatternDetector::subgraph_t subgraph;
  PDPattern pattern;
  PDNode* pd_node = pattern.NewNode();
  ProgramDesc prog;
  Graph g(prog);
  OpDesc fc_op;
  fc_op.SetType("op1");
  subgraph[pd_node] = g.CreateOpNode(&fc_op);
  EXPECT_TRUE(test.TestIsCompat(subgraph, &g));

  fc_op.SetType("mul");
  subgraph[pd_node] = g.CreateOpNode(&fc_op);
  EXPECT_FALSE(test.TestIsCompat(subgraph, &g));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
