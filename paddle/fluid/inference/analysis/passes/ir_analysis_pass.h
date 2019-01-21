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
#include "paddle/fluid/inference/analysis/analysis_pass.h"

namespace paddle {
namespace inference {
namespace analysis {

/*
 * Perform IR analysis passes.
 *
 * It is used to fuse some
 */
class IrAnalysisPass : public AnalysisPass {
 public:
  void RunImpl(Argument* argument) override;

  void CollectFusionStatis(Argument* argument);

  std::string repr() const override;
};

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
