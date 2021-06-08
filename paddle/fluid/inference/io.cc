/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/inference/io.h"

#include <algorithm>
#include <fstream>
#include <vector>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/pybind/pybind.h"

DEFINE_string(devices, "", "The devices to be used which is joined by comma.");
DEFINE_int32(math_num_threads, 1,
             "Number of threads used to run math functions.");

namespace paddle {
namespace inference {

void Init(const std::vector<std::string> argv) {
  framework::InitGflags(argv);
  platform::SetNumThreads(FLAGS_math_num_threads);
  // init devices
  std::vector<int> devices;
  std::string token;
  std::istringstream tokenStream(FLAGS_devices);
  while (std::getline(tokenStream, token, ',')) {
    devices.push_back(std::stoi(token));
  }
  framework::InitDevices(devices);
}

void ReadBinaryFile(const std::string& filename, std::string* contents) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      fin.is_open(), true,
      platform::errors::Unavailable("Failed to open file %s.", filename));
  fin.seekg(0, std::ios::end);
  contents->clear();
  contents->resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(contents->at(0)), contents->size());
  fin.close();
}

bool IsPersistable(const framework::VarDesc* var) {
  if (var->Persistable() &&
      var->GetType() != framework::proto::VarType::FEED_MINIBATCH &&
      var->GetType() != framework::proto::VarType::FETCH_LIST &&
      var->GetType() != framework::proto::VarType::RAW) {
    return true;
  }
  return false;
}

void LoadPersistables(framework::Executor* executor, framework::Scope* scope,
                      const framework::ProgramDesc& main_program,
                      const std::string& dirname,
                      const std::string& param_filename,
                      bool model_from_memory = false) {
  const framework::BlockDesc& global_block = main_program.Block(0);

  auto* load_program = new framework::ProgramDesc();
  framework::BlockDesc* load_block = load_program->MutableBlock(0);
  std::vector<std::string> paramlist;

  for (auto* var : global_block.AllVars()) {
    if (IsPersistable(var)) {
      VLOG(4) << "persistable variable's name: " << var->Name();

      framework::VarDesc* new_var = load_block->Var(var->Name());
      new_var->SetShape(var->GetShape());
      new_var->SetDataType(var->GetDataType());
      new_var->SetType(var->GetType());

      if (var->GetType() !=
          framework::proto::VarType::Type::VarType_Type_SELECTED_ROWS) {
        new_var->SetLoDLevel(var->GetLoDLevel());
      }

      new_var->SetPersistable(true);

      if (!param_filename.empty()) {
        paramlist.push_back(new_var->Name());
      } else {
        // append_op
        framework::OpDesc* op = load_block->AppendOp();
        op->SetType("load");
        op->SetOutput("Out", {new_var->Name()});
        op->SetAttr("file_path", {dirname + "/" + new_var->Name()});
        op->CheckAttrs();
      }
    }
  }

  if (!param_filename.empty()) {
    // sort paramlist to have consistent ordering
    std::sort(paramlist.begin(), paramlist.end());
    // append just the load_combine op
    framework::OpDesc* op = load_block->AppendOp();
    op->SetType("load_combine");
    op->SetOutput("Out", paramlist);
    op->SetAttr("file_path", {param_filename});
    op->SetAttr("model_from_memory", {model_from_memory});
    op->CheckAttrs();
  }

  executor->Run(*load_program, scope, 0, true, true);

  delete load_program;
}

std::unique_ptr<framework::ProgramDesc> Load(framework::Executor* executor,
                                             framework::Scope* scope,
                                             const std::string& dirname) {
  std::string model_filename = dirname + "/__model__";
  std::string program_desc_str;
  VLOG(3) << "loading model from " << model_filename;
  ReadBinaryFile(model_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));
  PADDLE_ENFORCE_EQ(
      framework::IsProgramVersionSupported(main_program->Version()), true,
      platform::errors::Unavailable("Model version %ld is not supported.",
                                    main_program->Version()));

  // model_from_memory is false in separate parameters.
  LoadPersistables(executor, scope, *main_program, dirname, "",
                   false /* model_from_memory */);
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> Load(
    framework::Executor* executor, framework::Scope* scope,
    const std::string& prog_filename, const std::string& param_filename) {
  std::string program_desc_str;
  ReadBinaryFile(prog_filename, &program_desc_str);

  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(program_desc_str));
  PADDLE_ENFORCE_EQ(
      framework::IsProgramVersionSupported(main_program->Version()), true,
      platform::errors::Unavailable("Model version %ld is not supported.",
                                    main_program->Version()));

  LoadPersistables(executor, scope, *main_program, "", param_filename,
                   false /* model_from_memory */);
  return main_program;
}

std::unique_ptr<framework::ProgramDesc> LoadFromMemory(
    framework::Executor* executor, framework::Scope* scope,
    const std::string& prog_buffer, const std::string& param_buffer) {
  std::unique_ptr<framework::ProgramDesc> main_program(
      new framework::ProgramDesc(prog_buffer));
  PADDLE_ENFORCE_EQ(
      framework::IsProgramVersionSupported(main_program->Version()), true,
      platform::errors::Unavailable("Model version %ld is not supported.",
                                    main_program->Version()));

  LoadPersistables(executor, scope, *main_program, "", param_buffer,
                   true /* model_filename */);
  return main_program;
}

void SaveVars(const framework::Scope& scope,
              const std::vector<std::string>& vars, const std::string& dirname,
              bool predicate) {
  framework::ProgramDesc prog;
  auto* block = prog.MutableBlock(0);
  auto* op = block->AppendOp();
  op->SetType("save_combine");
  op->SetInput("X", vars);
  op->SetAttr("file_path", dirname + "/param");
  op->CheckAttrs();

  platform::CPUPlace place;
  framework::Executor exe(place);
  exe.Run(prog, const_cast<framework::Scope*>(&scope), 0, true, true);
}

}  // namespace inference
}  // namespace paddle
