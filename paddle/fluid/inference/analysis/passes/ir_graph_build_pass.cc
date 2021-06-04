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

#include "paddle/fluid/inference/analysis/passes/ir_graph_build_pass.h"

#include <memory>
#include <string>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace inference {

extern void ReadBinaryFile(const std::string &filename, std::string *contents);

namespace analysis {

void IrGraphBuildPass::RunImpl(Argument *argument) {
  if (!argument->scope_valid()) {
    argument->SetScope(new framework::Scope);
  }
  PADDLE_ENFORCE_EQ(argument->use_gpu_valid(), true,
                    platform::errors::PreconditionNotMet(
                        "The use_gpu field should be valid"));

  // The load program should run on the same device with the inference program,
  // so that the parameters will on the same device, or they will keep copying
  // between difference devices.
  platform::Place place;
  place = platform::CPUPlace();

  if (argument->model_dir_valid()) {
    auto program =
        LoadModel(argument->model_dir(), argument->scope_ptr(), place);
    argument->SetMainProgram(program.release());
  } else if (argument->model_program_path_valid() &&
             argument->model_params_path_valid()) {
    auto program = LoadModel(
        argument->model_program_path(), argument->model_params_path(),
        argument->scope_ptr(), place,
        argument->model_from_memory_valid() && argument->model_from_memory());
    argument->SetMainProgram(program.release());
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "either model_dir or (program path and parameter path) should be "
        "set."));
  }

  auto graph = std::make_unique<Graph>(argument->main_program());
  argument->SetMainGraph(graph.release());
  auto *scope_ptr = argument->scope_ptr();
  PADDLE_ENFORCE_NOT_NULL(scope_ptr,
                          platform::errors::PreconditionNotMet(
                              "The scope ptr should not be nullptr."));
  argument->main_graph().SetNotOwned(framework::ir::kParamScopeAttr, scope_ptr);
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &path, framework::Scope *scope,
    const platform::Place &place) {
  framework::Executor exe(place);
  return Load(&exe, scope, path);
}

std::unique_ptr<framework::ProgramDesc> IrGraphBuildPass::LoadModel(
    const std::string &program_path, const std::string &params_path,
    framework::Scope *scope, const platform::Place &place,
    bool model_from_memory) {
  framework::Executor exe(place);
  if (!model_from_memory) {
    return Load(&exe, scope, program_path, params_path);
  } else {
    return LoadFromMemory(&exe, scope, program_path, params_path);
  }
}

std::string IrGraphBuildPass::repr() const { return "ir-graph-build-pass"; }

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
