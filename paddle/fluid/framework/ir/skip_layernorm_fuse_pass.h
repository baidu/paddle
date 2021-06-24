/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/fluid/framework/ir/fuse_pass_base.h"

namespace paddle {
namespace framework {
namespace ir {

//     |           |                            |            |
// other_op1   other_op2                    other_op1    other_op2
//     |           |              fuse           \          /
//     |------elementwise_add      ->           skip_layernorm
//                 |                                   |
//             layer_norm                          other_op3
//                 |                                   |
//             other_op3
//                 |
class Graph;

class SkipLayerNormFusePass : public FusePassBase {
 public:
  SkipLayerNormFusePass() {
    AddOpCompat(OpCompat("elementwise_add"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Y")
        .IsTensor()
        .End()
        .AddOutput("Out")
        .IsTensor()
        .End()
        .AddAttr("axis")
        .IsIntIn({0, -1})
        .End();

    AddOpCompat(OpCompat("layer_norm"))
        .AddInput("X")
        .IsTensor()
        .End()
        .AddInput("Scale")
        .IsTensor()
        .End()
        .AddInput("Bias")
        .IsTensor()
        .End()
        .AddOutput("Y")
        .IsTensor()
        .End()
        .AddOutput("Mean")
        .IsTensor()
        .End()
        .AddOutput("Variance")
        .IsTensor()
        .End()
        .AddAttr("epsilon")
        .IsNumGE(0.0f)
        .IsNumLE(0.001f)
        .End()
        .AddAttr("begin_norm_axis")
        .IsNumGT(0)
        .End();
  }

  virtual ~SkipLayerNormFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
