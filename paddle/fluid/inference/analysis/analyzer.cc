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

#include "paddle/fluid/inference/analysis/analyzer.h"
#include <string>
#include <vector>
#include "paddle/fluid/inference/analysis/passes/ir_analysis_compose_pass.h"
#include "paddle/fluid/inference/analysis/passes/passes.h"

namespace paddle {
namespace inference {
namespace analysis {

Analyzer::Analyzer() {}

void Analyzer::Run(Argument *argument) { RunIrAnalysis(argument); }

void Analyzer::RunIrAnalysis(Argument *argument) {
  std::vector<std::string> passes({"ir_analysis_compose_pass"});

  for (auto &pass : passes) {
    PassRegistry::Global().Retreive(pass)->Run(argument);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
