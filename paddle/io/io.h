/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/framework/executor.h"
#include "paddle/framework/init.h"
#include "paddle/framework/program_desc.h"
#include "paddle/framework/scope.h"
#include "paddle/framework/var_desc.h"

namespace paddle {

namespace io {

bool IsParameter(const framework::VarDesc* var,
                 const framework::ProgramDesc* main_program);

void LoadPersistables(framework::Executor& executor,
                      framework::Scope& scope,
                      const std::string& dirname,
                      framework::ProgramDesc* main_program);

framework::ProgramDesc* Load(framework::Executor& executor,
                             framework::Scope& scope,
                             const std::string& dirname);

std::vector<std::string> GetFeedVarNames(const ProgramDesc* main_program);
std::vector<std::string> GetFetchVarNames(const ProgramDesc* main_program);

}  // namespace io
}  // namespace paddle
