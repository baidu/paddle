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

#include "paddle/fluid/framework/ir/mkldnn/batch_norm_act_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace framework {
namespace ir {

using string::PrettyLogDetail;

void FuseBatchNormActOneDNNPass::ApplyImpl(Graph *graph) const {
  std::string act_type("relu");
  FuseBatchNormAct(graph, act_type);
}

FuseBatchNormActOneDNNPass::FuseBatchNormActOneDNNPass() {
  AddOpCompat(OpCompat("batch_norm"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddInput("Scale")
      .IsTensor()
      .End()
      .AddInput("Bias")
      .IsTensor()
      .End()
      .AddInput("Mean")
      .IsTensor()
      .End()
      .AddInput("Variance")
      .IsTensor()
      .End()
      .AddOutput("Y")
      .IsTensor()
      .End()
      .AddOutput("MeanOut")
      .IsOptional()
      .End()
      .AddOutput("VarianceOut")
      .IsOptional()
      .End()
      .AddOutput("SavedMean")
      .IsOptional()
      .End()
      .AddOutput("SavedVariance")
      .IsOptional()
      .End()
      .AddOutput("ReserveSpace")
      .IsOptional()
      .End()
      .AddAttr("epsilon")
      .IsNumGE(0.0f)
      .IsNumLE(0.001f)
      .End();

  AddOpCompat(OpCompat("relu"))
      .AddInput("X")
      .IsTensor()
      .End()
      .AddOutput("Out")
      .IsTensor()
      .End();
}

void FuseBatchNormActOneDNNPass::FuseBatchNormAct(
    Graph *graph, const std::string &act_type) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument(
                 "The input graph of "
                 "FuseBatchNormActOneDNNPass should not be nullptr."));
  FusePassBase::Init("bn_act", graph);

  GraphPatternDetector gpd;
  patterns::BatchNormActOneDNN bn_act_pattern(gpd.mutable_pattern(), "bn_act");
  bn_act_pattern(act_type);

  int found_bn_act_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t &subgraph,
                     Graph *g) {
    VLOG(4) << "Fuse BatchNorm with ReLU activation op.";

    if (!IsCompat(subgraph, g)) {
      LOG(WARNING) << "Pass in op compat failed.";
      return;
    }
    // BN output
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, bn_act_pattern);
    // ACT output
    GET_IR_NODE_FROM_SUBGRAPH(act_out, act_out, bn_act_pattern);
    // ops
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, bn_act_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(act, act, bn_act_pattern);

    auto *bn_op = batch_norm->Op();

    if (bn_op->HasAttr("use_mkldnn")) {
      PADDLE_ENFORCE(
          BOOST_GET_CONST(bool, bn_op->GetAttr("use_mkldnn")),
          platform::errors::PreconditionNotMet(
              "The BatchNorm+Act fusion may happen only when oneDNN library "
              "is used."));
    }

    if (bn_op->HasAttr("trainable_statistics")) {
      PADDLE_ENFORCE(
          !BOOST_GET_CONST(bool, bn_op->GetAttr("trainable_statistics")),
          platform::errors::PreconditionNotMet(
              "The BatchNorm+Act fusion may happen only when mean and variance "
              "are not calculated by current batch statistics."));
    }

    if (bn_op->HasAttr("is_test")) {
      PADDLE_ENFORCE(
          BOOST_GET_CONST(bool, bn_op->GetAttr("is_test")),
          platform::errors::PreconditionNotMet(
              "The BatchNorm+Act fusion may happen only during inference."));
    }

    bn_op->SetAttr("use_mkldnn", true);
    bn_op->SetAttr("is_test", true);
    bn_op->SetAttr("fuse_with_relu", true);
    bn_op->SetAttr("trainable_statistics", false);
    bn_op->SetOutput("Y", {act_out->Name()});

    if (!IsCompat(*bn_op)) {
      LOG(WARNING) << "Fc fuse pass in out fc op compat failed.";
      return;
    }

    IR_OP_VAR_LINK(batch_norm, act_out);
    GraphSafeRemoveNodes(g, {act, bn_out});
    found_bn_act_count++;
  };

  gpd(graph, handler);
  AddStatis(found_bn_act_count);
  PrettyLogDetail("---    fused %d batch norm with relu activation",
                  found_bn_act_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(batch_norm_act_fuse_pass,
              paddle::framework::ir::FuseBatchNormActOneDNNPass);
REGISTER_PASS_CAPABILITY(batch_norm_act_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("batch_norm", 0)
            .EQ("relu", 0));
