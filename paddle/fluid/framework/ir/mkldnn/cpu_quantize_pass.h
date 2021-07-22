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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Map variable name to tensor of scaling factors scaling it to MAX=1.0.
 * bool denotes whether quantization of the variable should be done to unsigned
 * type.
 */
class Graph;
class Node;

using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, LoDTensor>>;

/*
 * Quantize all supported operators.
 */
class CPUQuantizePass : public FusePassBase {
 public:
  virtual ~CPUQuantizePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  void QuantizeConv(Graph* graph, bool with_residual_data = false) const;
  void QuantizeFc(Graph* graph) const;
  void QuantizePool(Graph* graph) const;
  void QuantizeConcat(Graph* graph) const;
  void QuantizePriorBox(Graph* graph) const;
  void QuantizeTranspose(Graph* graph) const;
  void QuantizeReshape(Graph* graph) const;
  void QuantizeMatmul(Graph* graph) const;
  void QuantizeElementwiseAdd(Graph* graph) const;
  void QuantizeFusionLSTM(Graph* graph) const;
  void QuantizeFusionGru(Graph* graph) const;
  void QuantizeMultiGru(Graph* graph) const;

  void QuantizeInput(Graph* g, Node* op, Node* input, std::string input_name,
                     double scale_to_one, bool is_input_unsigned,
                     std::string scale_attr_name = "", float shift = 0.0,
                     std::string shift_attr_name = "") const;

  // quantize all inputs of given name with the same (minimum) scale
  void QuantizeInputs(Graph* g, Node* op, std::string input_name,
                      bool are_inputs_unsigned,
                      std::string scale_attr_name = "", float shift = 0.0,
                      std::string shift_attr_name = "") const;

  void DequantizeOutput(Graph* g, Node* op, Node* output,
                        std::string output_name, double scale_to_one,
                        bool is_unsigned,
                        std::string scale_attr_name = "") const;

  bool AreScalesPresentForVarNames(std::vector<std::string> names) const;
  bool AreScalesPresentForNodes(std::initializer_list<Node*> nodes) const;
  std::pair<bool, LoDTensor> GetScaleDataByName(const std::string& name) const;
  std::pair<bool, LoDTensor> GetScaleDataForNode(const Node* node) const;
  LoDTensor GetScaleTensorByName(const std::string& name) const;
  LoDTensor GetScaleTensorForNode(const Node* node) const;
  double GetScaleValueByName(const std::string& name,
                             bool* is_unsigned = nullptr) const;
  double GetScaleValueForNode(const Node* node,
                              bool* is_unsigned = nullptr) const;
  bool IsOpDequantized(const Node* node) const;
  bool IsOpQuantized(const Node* node) const;

  const std::string name_scope_{"quantize"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
