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

namespace paddle {
namespace framework {
namespace ir {

/*
 * Fuse the Conv and BatchNorm to a ConvBNMKLDNNOp.
 */
class Graph;

class ConvBNFusePass : public FusePassBase {
 public:
  ConvBNFusePass();
  virtual ~ConvBNFusePass() {}
  virtual std::string conv_type() const { return "conv2d"; }

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"conv_bn_fuse"};
};

class ConvEltwiseAddBNFusePass : public FusePassBase {
 public:
  ConvEltwiseAddBNFusePass();
  virtual ~ConvEltwiseAddBNFusePass() {}
  virtual std::string conv_type() const { return "conv2d"; }

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"conv_eltwiseadd_bn_fuse"};
};

class ConvTransposeBNFusePass : public ConvBNFusePass {
 public:
  ConvTransposeBNFusePass();
  virtual ~ConvTransposeBNFusePass() {}
  std::string conv_type() const { return "conv2d_transpose"; }
};

class ConvTransposeEltwiseAddBNFusePass : public ConvEltwiseAddBNFusePass {
 public:
  ConvTransposeEltwiseAddBNFusePass();
  virtual ~ConvTransposeEltwiseAddBNFusePass() {}
  std::string conv_type() const { return "conv2d_transpose"; }
};

class DepthwiseConvBNFusePass : public ConvBNFusePass {
 public:
  std::string conv_type() const { return "depthwise_conv2d"; }
};

class DepthwiseConvEltwiseAddBNFusePass : public ConvEltwiseAddBNFusePass {
 public:
  std::string conv_type() const { return "depthwise_conv2d"; }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
