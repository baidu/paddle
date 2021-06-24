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

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

/*
 * Squash dequantize->quantize pair pattern into requantize op
 */

class CPUQuantizeSquashPass : public FusePassBase {
 public:
  CPUQuantizeSquashPass();
  virtual ~CPUQuantizeSquashPass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;

  /*
   * For each dequantize's output find the number of operators it is an input to
   */
  void FindNodesToKeep(
      Graph* graph,
      std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  /*
   * Squash dequantize-quantize ops pairs into requantize or nothing
   */
  void DequantQuantSquash(
      Graph* graph,
      std::unordered_map<const Node*, int>* nodes_keep_counter) const;

  /*
   * Squash requantize op into conv with scale_out like requantize scale_out
   */
  void OpRequantSquash(Graph* graph) const;

  /*
   * Squash requantize op if the next operator's input scale can be updated
   */
  void RequantOpSquash(Graph* graph) const;

  /*
   * Squash dequant if the previous operator has force_fp32_output attribute
   */
  void OpDequantSquash(Graph* graph) const;

  /*
   * Squash quantize if several quatize ops have the same scale
   */
  void MultipleQuantizeSquash(Graph* graph) const;

  /*
   * Squash scale if dequantize is before scale
   */
  void DequantScaleSquash(Graph* graph) const;

  /*
   * Squash scale if scale is before quantize
   */
  void ScaleQuantSquash(Graph* graph) const;

  /*
   * Squash quantize if is before bfloat16 conv2d
   */
  void QuantizeBf16Conv(Graph* graph) const;

  const std::string name_scope_{"squash"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
