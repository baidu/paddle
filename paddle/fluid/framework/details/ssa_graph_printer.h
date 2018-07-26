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

#include <fstream>
#include <iosfwd>
#include <ostream>
#include <string>
#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace framework {
namespace details {

class SSAGraphPrinter {
 public:
  virtual ~SSAGraphPrinter() {}
  virtual void Print(const ir::Graph& graph, std::ostream& sout) const = 0;
};

class GraphvizSSAGraphPrinter : public SSAGraphPrinter {
 public:
  void Print(const ir::Graph& graph, std::ostream& sout) const override;
};

class SSAGraghBuilderWithPrinter : public SSAGraphBuilder {
 public:
  std::unique_ptr<ir::Graph> Apply(
      std::unique_ptr<ir::Graph> graph) const override {
    auto new_graph = Get<ir::Pass>("previous_pass").Apply(std::move(graph));

    std::unique_ptr<std::ostream> fout(
        new std::ofstream(Get<std::string>("debug_graphviz_path")));
    PADDLE_ENFORCE(fout->good());
    Get<GraphvizSSAGraphPrinter>("graph_printer").Print(*new_graph, *fout);
    return new_graph;
  }
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
