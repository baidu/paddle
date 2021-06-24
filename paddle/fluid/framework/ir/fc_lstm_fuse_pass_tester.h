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

#pragma once

#include "paddle/fluid/framework/ir/fc_lstm_fuse_pass.h"

#include <gtest/gtest.h>
#include "paddle/fluid/framework/ir/pass_tester_helper.h"

namespace paddle {
namespace framework {
namespace ir {

namespace fc_lstm_test {

void AddVarToScope(Scope* param_scope, const std::string& name,
                   const DDim& dims) {
  auto* tensor = param_scope->Var(name)->GetMutable<LoDTensor>();
  tensor->Resize(dims);
  tensor->mutable_data<float>(platform::CPUPlace());
}

Scope* CreateParamScope() {
  auto param_scope = new Scope();
  AddVarToScope(param_scope, "lstm_fc_w", {});
  AddVarToScope(param_scope, "lstm_fc_b", {});
  AddVarToScope(param_scope, "lstm_w", {});
  AddVarToScope(param_scope, "lstm_b", {});
  AddVarToScope(param_scope, "lstm_cell_0", {});
  AddVarToScope(param_scope, "lstm_batch_gate_0", {});
  AddVarToScope(param_scope, "lstm_batch_cell_pre_gate_0", {});
  AddVarToScope(param_scope, "lstm_hidden_0", {});
  AddVarToScope(param_scope, "lstm_cell_1", {});
  AddVarToScope(param_scope, "lstm_batch_gate_1", {});
  AddVarToScope(param_scope, "lstm_batch_cell_pre_gate_1", {});
  AddVarToScope(param_scope, "lstm_hidden_1", {});
  return param_scope;
}

std::unique_ptr<ir::Graph> PrepareGraph(
    std::string gate_activation = "sigmoid",
    std::string cell_activation = "tanh",
    std::string candidate_activation = "tanh") {
  // inputs                     operator            output
  // --------------------------------------------------------
  // (a, lstm_fc_w)                mul         ->   fc_0_tmp_0
  // (fc_0_tmp_0, lstm_fc_b)  elementwise_add  ->   fc_0_tmp_1
  // fc_0_tmp_1,lstm_w,lstm_b     lstm         ->   lstm_out_0

  // (b, lstm_fc_w)                mul         ->   fc_1_tmp_0
  // (fc_1_tmp_0, lstm_fc_b)  elementwise_add  ->   fc_1_tmp_1
  // (fc_1_tmp_1,lstm_w,lstm_b)   lstm         ->   lstm_out_1
  Layers layers;
  auto* a = layers.data("a");
  auto* b = layers.data("b");
  auto* fc_w = layers.data("lstm_fc_w", {}, true);
  auto* fc_b = layers.data("lstm_fc_b", {}, true);
  auto* lstm_w = layers.data("lstm_w", {}, true);
  auto* lstm_b = layers.data("lstm_b", {}, true);
  auto* fc_0_tmp0 = layers.mul(a, fc_w);
  auto* fc_0_tmp1 = layers.elementwise_add(fc_0_tmp0, fc_b);
  auto* lstm_cell_0 = layers.data("lstm_cell_0", {}, false);
  auto* lstm_batch_gate_0 = layers.data("lstm_batch_gate_0", {}, false);
  auto* lstm_batch_cell_pre_gate_0 =
      layers.data("lstm_batch_cell_pre_gate_0", {}, false);
  auto* lstm_hidden_0 = layers.data("lstm_hidden_0", {}, false);
  layers.lstm(fc_0_tmp1, lstm_w, lstm_b, lstm_cell_0, lstm_batch_gate_0,
              lstm_hidden_0, lstm_batch_cell_pre_gate_0, nullptr, nullptr, true,
              false, gate_activation, cell_activation, candidate_activation);
  auto* fc_1_tmp0 = layers.mul(b, fc_w);
  auto* fc_1_tmp1 = layers.elementwise_add(fc_1_tmp0, fc_b);
  auto* lstm_cell_1 = layers.data("lstm_cell_1", {}, false);
  auto* lstm_batch_gate_1 = layers.data("lstm_batch_gate_1", {}, false);
  auto* lstm_batch_cell_pre_gate_1 =
      layers.data("lstm_batch_cell_pre_gate_1", {}, false);
  auto* lstm_hidden_1 = layers.data("lstm_hidden_1", {}, false);
  layers.lstm(fc_1_tmp1, lstm_w, lstm_b, lstm_cell_1, lstm_batch_gate_1,
              lstm_hidden_1, lstm_batch_cell_pre_gate_1, nullptr, nullptr, true,
              false, gate_activation, cell_activation, candidate_activation);

  std::unique_ptr<ir::Graph> graph(new ir::Graph(layers.main_program()));
  return std::move(graph);
}

}  // namespace fc_lstm_test
}  // namespace ir
}  // namespace framework
}  // namespace paddle
