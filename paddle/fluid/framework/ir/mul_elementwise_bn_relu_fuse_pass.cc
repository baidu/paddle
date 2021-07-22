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

#include "paddle/fluid/framework/ir/mul_elementwise_bn_relu_fuse_pass.h"

#include <cmath>
#include <string>

#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Node;

void MulElementwiseBnReluPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  std::string name_scope = "mul_elementwise_bn_relu_fuse_pass";
  FusePassBase::Init(name_scope, graph);
  auto* scope = param_scope();

  GraphPatternDetector gpd;
  auto* x = gpd.mutable_pattern()
                ->NewNode("mul_elementwise_bn_relu_fuse/x")
                ->AsInput()
                ->assert_is_op_input("mul", "X");
  patterns::MulElementwiseBnRelu mul_elementwise_bn_relu_pattern(
      gpd.mutable_pattern(), name_scope);
  mul_elementwise_bn_relu_pattern(x);

  int found_count = 0;
  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "mul_elementwise_bn_relu_fuse";
    // mul
    GET_IR_NODE_FROM_SUBGRAPH(mul, mul, mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(w, w, mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(mul_out, mul_out,
                              mul_elementwise_bn_relu_pattern);
    // elementwise_add
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add, elementwise_add,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bias, bias, mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(elementwise_add_out, elementwise_add_out,
                              mul_elementwise_bn_relu_pattern);
    // BN
    GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean,
                              mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance,
                              mul_elementwise_bn_relu_pattern)
    // relu
    GET_IR_NODE_FROM_SUBGRAPH(relu, relu, mul_elementwise_bn_relu_pattern);
    GET_IR_NODE_FROM_SUBGRAPH(relu_out, relu_out,
                              mul_elementwise_bn_relu_pattern);

    // recompute weight
    auto* weight_tensor = scope->FindVar(w->Name())->GetMutable<LoDTensor>();
    auto* weight_data = weight_tensor->data<float>();
    auto weight_dims = weight_tensor->dims();
    int weight_num = product(weight_dims);
    int w_h = weight_dims[0];
    int w_w = weight_dims[1];

    auto* bias_tensor = scope->FindVar(bias->Name())->GetMutable<LoDTensor>();
    auto* bias_data = bias_tensor->data<float>();
    auto bias_dims = bias_tensor->dims();
    int bias_num = weight_dims[0];

    auto* bn_scale_tensor =
        scope->FindVar(bn_scale->Name())->GetMutable<LoDTensor>();
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();
    auto* bn_mean_tensor =
        scope->FindVar(bn_mean->Name())->GetMutable<LoDTensor>();
    auto* bn_variance_tensor =
        scope->FindVar(bn_variance->Name())->GetMutable<LoDTensor>();

    auto* bn_scale_data = bn_scale_tensor->data<float>();
    auto* bn_bias_data = bn_bias_tensor->data<float>();
    auto* bn_mean_data = bn_mean_tensor->data<float>();
    auto* bn_variance_data = bn_variance_tensor->data<float>();

    int bn_num = bn_scale_tensor->numel();

    auto print_tensor = [&](std::string name, float* d) {
      std::cout << "Tensor Name: " << name << "\n";
      std::cout << "Tensor data: [";
      for (int p = 0; p < d->numel(); p++) {
        std::cout << d[p] << ", ";
      }
      std::cout << "]\n";
    };
    print_tensor("weight_data", weight_data);
    print_tensor("bias_data", bias_data);
    print_tensor("bn_scale_data", bn_scale_data);
    print_tensor("bn_bias_data", bn_bias_data);
    print_tensor("bn_mean_data", bn_mean_data);
    print_tensor("bn_variance_data", bn_variance_data);

    OpDesc desc;
    desc.SetType("fc");

    // Set inputs of fc
    desc.SetInput("Input", {subgraph.at(x)->Name()});
    desc.SetInput("W", {w->Name()});
    desc.SetInput("Bias", {bias->Name()});

    // Set output of fc
    std::string out_name = relu_out->Name();
    desc.SetOutput("Out", std::vector<std::string>({relu_out->Name()}));

    // Set attrs of fc
    desc.SetAttr("in_num_col_dims", mul->Op()->GetAttr("x_num_col_dims"));
    std::string activation_type = "relu";
    desc.SetAttr("activation_type", activation_type);

    auto fc_node = g->CreateOpNode(&desc);

    const std::string& w_name = patterns::UniqueKey(w->Name());
    VarDesc w_key(w_name);
    w_key.SetPersistable(true);
    auto* w_node = g->CreateVarNode(&w_key);

    const std::string& bias_name = patterns::UniqueKey(bias->Name());
    VarDesc bias_key(bias_name);
    bias_key.SetPersistable(true);
    auto* bias_node = g->CreateVarNode(&bias_key);

    IR_NODE_LINK_TO(subgraph.at(x), fc_node);
    IR_NODE_LINK_TO(w_node, fc_node);
    IR_NODE_LINK_TO(bias_node, fc_node);
    IR_NODE_LINK_TO(fc_node, relu_out);
    GraphSafeRemoveNodes(
        graph,
        {mul, elementwise_add, batch_norm, relu, mul_out, elementwise_add_out,
         bn_scale, bn_bias, bn_mean, bn_variance, bn_out, bn_mean_out,
         bn_variance_out, bn_saved_mean, bn_saved_variance});
    ++found_count;
  };

  gpd(graph, handler);
  AddStatis(found_count);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(mul_elementwise_bn_relu_fuse_pass,
              paddle::framework::ir::MulElementwiseBnReluPass);
REGISTER_PASS_CAPABILITY(mul_elementwise_bn_relu_fuse_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("mul", 0)
            .LE("elementwise_add", 1)
            .EQ("batch_norm", 0)
            .EQ("relu", 0));
