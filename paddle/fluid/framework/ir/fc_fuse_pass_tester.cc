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

#include "paddle/fluid/framework/ir/fc_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

void AddVarToScope(Scope* param_scope, const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "conv2d_filters_0", {});
  AddVarToScope(param_scope, "conv2d_bias_0", {});
  AddVarToScope(param_scope, "weights_0", {});
  AddVarToScope(param_scope, "weights_1", {});
  AddVarToScope(param_scope, "bias_1", {});
  AddVarToScope(param_scope, "bias_2", {});
  return param_scope;
}

TEST(FCFusePass, basic) {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, filters_0 bias_0)      conv2d           -> conv2d_out
  // conv2d_out                 relu             -> relu_out_0
  // (relu_out_0, weights_0)    mul              -> mul_out_0
  // (mul_out_0, bias_1)        elementwise_add  -> add_out_0
  // add_out_0                  relu             -> relu_out_1
  // (relu_out_1, weights_1)    mul              -> mul_out_1
  // (mul_out_1, bias_2)        elementwise_add  -> add_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* filters_0 = layers.data("conv2d_filters_0", {}, true);
  auto* bias_0 = layers.data("conv2d_bias_0", {}, true);
  auto* conv2d_out = layers.conv2d(a, filters_0, bias_0, false);
  auto* relu_out_0 = layers.relu(conv2d_out);
  auto* weights_0 = layers.data("weights_0", {}, true);
  auto* mul_out_0 = layers.mul(relu_out_0, weights_0);
  auto* bias_1 = layers.data("bias_1", {}, true);
  auto* add_out_0 = layers.elementwise_add(mul_out_0, bias_1, nullptr, 1);
  auto* relu_out_1 = layers.relu(add_out_0);
  auto* weights_1 = layers.data("weights_1", {}, true);
  auto* mul_out_1 = layers.mul(relu_out_1, weights_1);
  auto* bias_2 = layers.data("bias_2", {}, true);
  auto* add_out_1 = layers.elementwise_add(mul_out_1, bias_2, nullptr, 1);
  VLOG(4) << add_out_1;

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  auto pass = PassRegistry::Instance().Get("fc_fuse_pass");
  pass->Set("use_gpu", new bool(true));
  graph->Set("__param_scope__", CreateParamScope());
  int num_nodes_before = graph->Nodes().size();
  int num_mul_nodes_before = GetNumOpNodes(graph, "mul");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fc_nodes_after = GetNumOpNodes(graph, "fc");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 6,
                    platform::errors::InvalidArgument(
                        "num_nodes_before=%d, num_nodes_after=%d.",
                        num_nodes_before, num_nodes_after));
  PADDLE_ENFORCE_EQ(num_fc_nodes_after, 2,
                    platform::errors::InvalidArgument("num_fc_nodes_after=%d.",
                                                      num_fc_nodes_after));
  PADDLE_ENFORCE_EQ(num_mul_nodes_before, num_fc_nodes_after,
                    platform::errors::InvalidArgument(
                        "num_mul_nodes_before=%d, num_fc_nodes_after=%d.",
                        num_mul_nodes_before, num_fc_nodes_after));
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_fuse_pass);
