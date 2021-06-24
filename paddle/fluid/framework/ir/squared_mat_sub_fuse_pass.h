/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#pragma once

#include <string>

#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"

namespace paddle {
namespace framework {
namespace ir {

/**
 * Fuse ( (A * B).^2 - (A.^2 * B.^2) ) .* scalar
 */
class Graph;

class SquaredMatSubFusePass : public FusePassBase {
 public:
  SquaredMatSubFusePass();
  bool IsAcceptable(const GraphPatternDetector::subgraph_t& subgraph,
                    Graph* g) const;
  virtual ~SquaredMatSubFusePass() {}

 protected:
  void ApplyImpl(ir::Graph* graph) const override;
  const std::string name_scope_{"squared_mat_sub_fuse"};
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle
