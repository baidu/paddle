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
#pragma once
#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/pass.h"

namespace paddle {
namespace framework {
namespace ir {
/*
* Fuse the Conv and Elementwise_add to a ConvBiasOp.
*/
class Graph;

class ConvBiasFusePass : public FusePassBase {
 public:
  ConvBiasFusePass();
  virtual ~ConvBiasFusePass() {}
  virtual std::string type() const { return "conv2d"; }

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"conv_bias_mkldnn_fuse"};
};
/*
* Fuse the Conv3D and Elementwise_add to a Conv3DBiasOp.
*/
class Conv2DTransposeBiasFusePass : public ConvBiasFusePass {
 public:
  Conv2DTransposeBiasFusePass();
  std::string type() const override { return "conv2d_transpose"; }
};

class Conv3DBiasFusePass : public ConvBiasFusePass {
 public:
  std::string type() const override { return "conv3d"; }
};
}  // namespace ir
}  // namespace framework
}  // namespace paddle
