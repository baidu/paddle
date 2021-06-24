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

#include "paddle/fluid/framework/ir/fc_gru_fuse_pass_tester.h"

namespace paddle {
namespace framework {
namespace ir {

namespace fc_gru_test {
TEST(FcGruFusePass, basic) {
  std::unique_ptr<ir::Graph> graph = PrepareGraph();
  auto pass = PassRegistry::Instance().Get("fc_gru_fuse_pass");
  pass->Set("use_gpu", new bool(true));
  graph->Set("__param_scope__", CreateParamScope());
  int num_nodes_before = graph->Nodes().size();
  int num_gru_nodes_before = GetNumOpNodes(graph, "gru");
  VLOG(3) << DebugString(graph);

  graph.reset(pass->Apply(graph.release()));
  int num_nodes_after = graph->Nodes().size();
  int num_fuse_gru_nodes_after = GetNumOpNodes(graph, "fusion_gru");
  VLOG(3) << DebugString(graph);

  PADDLE_ENFORCE_EQ(num_nodes_before, num_nodes_after + 6,
                    platform::errors::PreconditionNotMet(
                        "The number of nodes before and after "
                        "the fuse does not meet expectations"));
  PADDLE_ENFORCE_EQ(
      num_fuse_gru_nodes_after, 2,
      platform::errors::PreconditionNotMet("The number of gru nodes before the "
                                           "fuse does not meet expectations"));
  PADDLE_ENFORCE_EQ(num_gru_nodes_before, num_fuse_gru_nodes_after,
                    platform::errors::PreconditionNotMet(
                        "The number of fusion_gru nodes does not meet "
                        "expectations after fuse"));
}

}  // namespace fc_gru_test
}  // namespace ir
}  // namespace framework
}  // namespace paddle

USE_PASS(fc_gru_fuse_pass);
