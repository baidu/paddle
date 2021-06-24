// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include "paddle/fluid/framework/ir/mkldnn/batch_norm_act_fuse_pass.h"
#include "paddle/fluid/framework/ir/pass_test_util.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace framework {
namespace ir {

namespace {

void SetBatchNormAttrs(OpDesc* bn_op, bool is_test = true,
                       bool trainable_stats = true) {
  bn_op->SetAttr("is_test", is_test);
  bn_op->SetAttr("trainable_statistics", trainable_stats);
  bn_op->SetAttr("fuse_with_relu", false);
  bn_op->SetAttr("epsilon", 0.001f);
}
}

// ------------------------------ Test cases -----------------------------------

// The below test cases are distinguished by whether following attributes have
// true or false value:
// - is_test
// - trainable_statistics
// The test case name would have only attributes with true value in its name.

TEST(FuseBatchNormActOneDNNPass, ThrowIsTestTrainableStats) {
  auto prog = test::BuildProgramDesc(
      {"x", "m", "v", "bn_y", "act_y", "m_out", "var_out", "sm", "sv"},
      {"scale", "bias"});
  auto* bn_op = test::CreateOp(&prog, "batch_norm", {{"X", "x"},
                                                     {"Scale", "scale"},
                                                     {"Bias", "bias"},
                                                     {"Mean", "m"},
                                                     {"Variance", "v"}},
                               {{"Y", "bn_y"},
                                {"MeanOut", "m_out"},
                                {"VarianceOut", "var_out"},
                                {"SavedMean", "sm"},
                                {"SavedVariance", "sv"}});
  SetBatchNormAttrs(bn_op, true, true);
  test::CreateOp(&prog, "relu", {{"X", "bn_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(test::RunPassAndAssert(&graph, "batch_norm_act_fuse_pass", "x",
                                      "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseBatchNormActOneDNNPass, FuseIsTest) {
  auto prog = test::BuildProgramDesc({"x", "m", "v", "bn_y", "act_y"},
                                     {"scale", "bias"});
  auto* bn_op = test::CreateOp(&prog, "batch_norm", {{"X", "x"},
                                                     {"Scale", "scale"},
                                                     {"Bias", "bias"},
                                                     {"Mean", "m"},
                                                     {"Variance", "v"}},
                               {{"Y", "bn_y"}});
  SetBatchNormAttrs(bn_op, true, false);
  test::CreateOp(&prog, "relu", {{"X", "bn_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  constexpr int removed_nodes_count = 2;

  EXPECT_TRUE(test::RunPassAndAssert(&graph, "batch_norm_act_fuse_pass", "x",
                                     "act_y", removed_nodes_count));
  EXPECT_TRUE(test::AssertOpsCount(graph, {{"batch_norm", 1}, {"relu", 0}}));

  for (const auto* node : graph.Nodes()) {
    if (node->IsOp() && node->Op()->Type() == "batch_norm") {
      const auto* op = node->Op();
      ASSERT_TRUE(op->HasAttr("use_mkldnn"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("use_mkldnn")));
      ASSERT_TRUE(op->HasAttr("fuse_with_relu"));
      EXPECT_TRUE(BOOST_GET_CONST(bool, op->GetAttr("fuse_with_relu")));
      ASSERT_TRUE(op->HasAttr("trainable_statistics"));
      EXPECT_FALSE(BOOST_GET_CONST(bool, op->GetAttr("trainable_statistics")));
    }
  }
}

TEST(FuseBatchNormActOneDNNPass, ThrowTrainableStats) {
  auto prog = test::BuildProgramDesc(
      {"x", "m", "v", "bn_y", "act_y", "m_out", "var_out", "sm", "sv"},
      {"scale", "bias"});
  auto* bn_op = test::CreateOp(&prog, "batch_norm", {{"X", "x"},
                                                     {"Scale", "scale"},
                                                     {"Bias", "bias"},
                                                     {"Mean", "m"},
                                                     {"Variance", "v"}},
                               {{"Y", "bn_y"},
                                {"MeanOut", "m_out"},
                                {"VarianceOut", "var_out"},
                                {"SavedMean", "sm"},
                                {"SavedVariance", "sv"}});
  SetBatchNormAttrs(bn_op, false, true);
  test::CreateOp(&prog, "relu", {{"X", "bn_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(test::RunPassAndAssert(&graph, "batch_norm_act_fuse_pass", "x",
                                      "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseBatchNormActOneDNNPass, AllAttrsFalse) {
  auto prog = test::BuildProgramDesc(
      {"x", "m", "v", "bn_y", "act_y", "m_out", "var_out", "sm", "sv"},
      {"scale", "bias"});
  auto* bn_op = test::CreateOp(&prog, "batch_norm", {{"X", "x"},
                                                     {"Scale", "scale"},
                                                     {"Bias", "bias"},
                                                     {"Mean", "m"},
                                                     {"Variance", "v"}},
                               {{"Y", "bn_y"},
                                {"MeanOut", "m_out"},
                                {"VarianceOut", "var_out"},
                                {"SavedMean", "sm"},
                                {"SavedVariance", "sv"}});
  SetBatchNormAttrs(bn_op, false, false);
  test::CreateOp(&prog, "relu", {{"X", "bn_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(test::RunPassAndAssert(&graph, "batch_norm_act_fuse_pass", "x",
                                      "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseBatchNormActOneDNNPass, ThrowUseMkldnn) {
  auto prog = test::BuildProgramDesc(
      {"x", "m", "v", "bn_y", "act_y", "m_out", "var_out", "sm", "sv"},
      {"scale", "bias"});
  auto* bn_op = test::CreateOp(&prog, "batch_norm", {{"X", "x"},
                                                     {"Scale", "scale"},
                                                     {"Bias", "bias"},
                                                     {"Mean", "m"},
                                                     {"Variance", "v"}},
                               {{"Y", "bn_y"},
                                {"MeanOut", "m_out"},
                                {"VarianceOut", "var_out"},
                                {"SavedMean", "sm"},
                                {"SavedVariance", "sv"}},
                               false);
  SetBatchNormAttrs(bn_op, false, false);
  test::CreateOp(&prog, "relu", {{"X", "bn_y"}}, {{"Out", "act_y"}}, false);

  Graph graph(prog);
  // No fusion in this attribute configuration
  constexpr int removed_nodes_count = 0;

  EXPECT_THROW(test::RunPassAndAssert(&graph, "batch_norm_act_fuse_pass", "x",
                                      "act_y", removed_nodes_count),
               paddle::platform::EnforceNotMet);
}

TEST(FuseBatchNormActOneDNNPass, pass_op_version_check) {
  ASSERT_TRUE(
      paddle::framework::compatible::PassVersionCheckerRegistrar::GetInstance()
          .IsPassCompatible("batch_norm_act_fuse_pass"));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(batch_norm_act_fuse_pass);
