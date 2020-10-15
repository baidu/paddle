// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <fstream>
#include <iostream>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/lite/op_teller.h"
#include "paddle/fluid/inference/utils/singleton.h"

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/inference/analysis/ir_passes/lite_subgraph_pass.h"
#include "paddle/fluid/string/pretty_log.h"

#include "paddle/fluid/inference/lite/engine.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;
using framework::ir::Agent;
using framework::ir::SubGraphFuser;
using framework::ir::Graph;

namespace lite {

std::string UniqueKey(const std::vector<std::string>& engine_inputs,
                      const std::vector<std::string>& engine_outputs,
                      const std::string& id) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += id;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

std::vector<std::string> IOVarsFilter(const std::vector<Node*>& nodes) {
  std::set<std::string> names;
  for (const auto& node : nodes) {
    if (node->IsVar() && !node->Var()->Persistable()) {
      names.insert(node->Name());
    }
  }
  return std::vector<std::string>(names.begin(), names.end());
}

void StrToBinaryFile(const std::string& path, const std::string& str) {
  std::ofstream file(path.c_str(), std::ios::binary);
  file.write(str.c_str(), str.size());
  file.close();
}

void ModifyHostSubgraphOps(
    framework::ProgramDesc* host_program, framework::BlockDesc* host_sub_block,
    const std::vector<framework::OpDesc*>& subgraph_ops) {
  for (auto* op_desc : subgraph_ops) {
    auto* sub_block_op = host_sub_block->AppendOp();
    sub_block_op->CopyFrom(*op_desc);
    if (op_desc->HasAttr("sub_block")) {
      int32_t global_sub_id = host_sub_block->ID();
      auto* op_sub_block =
          host_program->MutableBlock(op_desc->GetBlockAttrId("sub_block"));
      op_sub_block->Proto()->set_parent_idx(global_sub_id);
    }
  }
}

void ModifyHostProgram(framework::ProgramDesc* host_program,
                       framework::BlockDesc* host_sub_block,
                       const std::unordered_set<Node*>& io_var_nodes,
                       const std::vector<framework::OpDesc*>& subgraph_ops) {
  for (auto* var_node : io_var_nodes) {
    auto* sub_block_var = host_sub_block->Var(var_node->Name());
    sub_block_var->Proto()->CopyFrom(*var_node->Var()->Proto());
  }
  ModifyHostSubgraphOps(host_program, host_sub_block, subgraph_ops);
}

void AppendLiteSubBlocks(const std::vector<framework::OpDesc*>& subgraph_ops,
                         framework::ProgramDesc* engine_program,
                         framework::ProgramDesc* host_program,
                         const int32_t host_sub_id) {
  std::unordered_map<int32_t, int32_t> sub_blocks_map;
  std::unordered_set<int32_t> copied_host_ids;
  sub_blocks_map[host_sub_id] = framework::kRootBlockIndex;
  std::function<void(const std::vector<framework::OpDesc*>&)> append_sub_blocks;
  append_sub_blocks = [&](const std::vector<framework::OpDesc*>& ops) {
    for (auto* op_desc : ops) {
      if (op_desc->HasAttr("sub_block")) {
        int32_t host_op_sub_id = op_desc->GetBlockAttrId("sub_block");
        if (copied_host_ids.count(host_op_sub_id)) continue;
        size_t engine_block_size = engine_program->Size();
        auto* host_op_sub_block = host_program->MutableBlock(host_op_sub_id);
        auto* engine_op_sub_block =
            engine_program->AppendBlock(*(op_desc->Block()));
        for (auto* var : host_op_sub_block->AllVars()) {
          auto* engine_var = engine_op_sub_block->Var(var->Name());
          engine_var->Proto()->CopyFrom(*var->Proto());
        }
        for (auto* op : host_op_sub_block->AllOps()) {
          auto* engine_op = engine_op_sub_block->AppendOp();
          engine_op->Proto()->CopyFrom(*op->Proto());
        }
        sub_blocks_map[host_op_sub_id] = engine_block_size;
        append_sub_blocks(host_op_sub_block->AllOps());
      }
    }
  };
  append_sub_blocks(subgraph_ops);
  for (size_t i = 0; i < engine_program->Size(); i++) {
    for (auto* op_desc : engine_program->Block(i).AllOps()) {
      if (op_desc->HasAttr("sub_block")) {
        int32_t id = op_desc->GetBlockAttrId("sub_block");
        op_desc->SetAttr("sub_block", sub_blocks_map[id]);
      }
    }
  }
}

// The modification of pass should be a process of framework::desc
// (initial) -> proto::desc (flush) -> framework::desc (final).
// Ir::Graph is limited to changing the main block, so the sub block
// needs to be processed here.
void ModifyEngineProgram(Node* merged_node,
                         framework::ProgramDesc* host_program,
                         framework::ProgramDesc* engine_program,
                         const int32_t host_sub_block_id,
                         const std::unordered_set<Node*>& io_var_nodes,
                         const std::vector<framework::OpDesc*>& subgraph_ops) {
  // 1. Fill the main block of lite program.
  framework::BlockDesc* engine_global_block =
      engine_program->MutableBlock(framework::kRootBlockIndex);
  PrependFeedOps(engine_global_block, IOVarsFilter(merged_node->inputs));
  for (auto* var_node : io_var_nodes) {
    framework::VarDesc* sub_block_var =
        engine_global_block->Var(var_node->Name());
    sub_block_var->Proto()->CopyFrom(*var_node->Var()->Proto());
  }
  for (auto* op_desc : subgraph_ops) {
    auto* sub_block_op = engine_global_block->AppendOp();
    sub_block_op->CopyFrom(*op_desc);
  }
  PrependFetchOps(engine_global_block, IOVarsFilter(merged_node->outputs));

  // 2. Append sub blocks in the lite program.
  AppendLiteSubBlocks(subgraph_ops, engine_program, host_program,
                      host_sub_block_id);
}

void OrganizeProgram(Node* merged_node, framework::ProgramDesc* host_program,
                     framework::ProgramDesc* engine_program,
                     std::vector<std::string>* repetitive_params) {
  std::vector<framework::ir::Node*>& subgraph = *Agent(merged_node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(), false,
                    platform::errors::NotFound(
                        "No subgraph found in lite subgraph pass. Please use "
                        "the full model call from Analysis Predictor."));

  const framework::BlockDesc& host_global_block =
      host_program->Block(framework::kRootBlockIndex);
  framework::BlockDesc* host_sub_block =
      host_program->AppendBlock(host_global_block);

  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  std::unordered_set<Node*> io_var_nodes = GetRelatedIOVarNodes(subgraph);
  for (const auto* node : io_var_nodes) {
    VLOG(3) << "IO Variable Name: " << node->Name();
  }

  std::vector<framework::OpDesc*> subgraph_ops;
  for (auto* op_node : subgraph) {
    subgraph_ops.push_back(op_node->Op());
  }

  ModifyHostProgram(host_program, host_sub_block, io_var_nodes, subgraph_ops);
  ModifyEngineProgram(merged_node, host_program, engine_program,
                      host_sub_block->ID(), io_var_nodes, subgraph_ops);
  *repetitive_params = ExtractParameters(io_var_nodes, true);
  for (const auto& param : *repetitive_params) {
    VLOG(3) << "Repetitive param: " << param;
  }
  host_program->Flush();
  engine_program->Flush();
}
}  // namespace lite

void LiteSubgraphPass::SetUpEngine(
    framework::ProgramDesc* program,
    const std::vector<std::string>& repetitive_params,
    const std::string& unique_key, bool dump_model) const {
  inference::lite::EngineConfig config;
  auto* scope = param_scope();

  // When the pass is started, only the persistent variables of the
  // main block are read. Fluid seems to allow persistence variables
  // in the sub block, but they are controlled by context, so the
  // support is suspended here.
  auto serialize_params = [](std::string* str, framework::Scope* scope,
                             const std::vector<std::string>& params) {
    std::ostringstream os;
    platform::CPUDeviceContext ctx;
    for (const auto& param : params) {
      VLOG(3) << "Serialize param: " << param;
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(param),
          platform::errors::NotFound(
              "Block should already have a '%s' variable", param));
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    *str = os.str();
  };

  bool use_gpu = Get<bool>("use_gpu");
  bool enable_int8 = Get<bool>("enable_int8");
  bool use_xpu = Get<bool>("use_xpu");
  int xpu_l3_workspace_size = Get<int>("xpu_l3_workspace_size");
  int cpu_math_library_num_threads = Get<int>("cpu_math_library_num_threads");

  lite_api::TargetType target_type;
  if (use_gpu) {
    target_type = TARGET(kCUDA);
  } else if (use_xpu) {
    target_type = TARGET(kXPU);
  } else {
#ifdef PADDLE_WITH_ARM
    target_type = TARGET(kARM);
#else
    target_type = TARGET(kX86);
#endif
  }

  paddle::lite_api::PrecisionType precision_type =
      enable_int8 ? PRECISION(kInt8) : PRECISION(kFloat);

  serialize_params(&config.param, scope, repetitive_params);
  config.model = program->Proto()->SerializeAsString();
  config.valid_places = {
      // Notice: The ordering here determines the device where the
      // input tensor of the Lite engine is located, and then affects
      // whether tensor sharing is feasible.
      paddle::lite_api::Place({target_type, precision_type}),
      paddle::lite_api::Place({target_type, PRECISION(kInt64)}),
      paddle::lite_api::Place({target_type, PRECISION(kFloat)}),
      paddle::lite_api::Place({TARGET(kHost), PRECISION(kFloat)}),
  };
  config.cpu_math_library_num_threads = cpu_math_library_num_threads;
  config.xpu_l3_workspace_size = xpu_l3_workspace_size;
  if (dump_model) {
    lite::StrToBinaryFile("./model.bin", config.model);
    lite::StrToBinaryFile("./param.bin", config.param);
  }
  inference::Singleton<inference::lite::EngineManager>::Global().Create(
      unique_key, config);
}

void LiteSubgraphPass::BuildOperator(
    Node* merged_node, framework::ProgramDesc* global_program,
    std::vector<std::string>* repetitive_params) const {
  framework::ProgramDesc engine_program;

  const std::string id = std::to_string(Get<int>("predictor_id"));
  const std::vector<std::string> input_names =
      lite::IOVarsFilter(merged_node->inputs);
  const std::vector<std::string> output_names =
      lite::IOVarsFilter(merged_node->outputs);
  const std::string unique_key = lite::UniqueKey(input_names, output_names, id);

  lite::OrganizeProgram(merged_node, global_program, &engine_program,
                        repetitive_params);
  SetUpEngine(&engine_program, *repetitive_params, unique_key);

  auto* op_desc = merged_node->Op();
  op_desc->SetInput("Xs", input_names);
  op_desc->SetOutput("Ys", output_names);
  op_desc->SetType("lite_engine");
  op_desc->SetAttr("engine_key", unique_key);
  op_desc->SetAttr("enable_int8", Get<bool>("enable_int8"));
  op_desc->SetAttr("use_gpu", Get<bool>("use_gpu"));
  op_desc->SetAttr("zero_copy", Get<bool>("zero_copy"));
}

void LiteSubgraphPass::ApplyImpl(framework::ir::Graph* graph) const {
  framework::ir::FusePassBase::Init("lite_subgraph_pass", graph);
  framework::ProgramDesc* global_program =
      Get<framework::ProgramDesc*>("program");

  auto& lite_ops_filter = Get<std::vector<std::string>>("lite_ops_filter");

  auto teller = [&lite_ops_filter](const Node* node) {
    if (!node->IsOp() || !node->Op())
      return false;
    else if (node->Op()->Type() == "feed" || node->Op()->Type() == "fetch")
      return false;
    else if (std::find(lite_ops_filter.begin(), lite_ops_filter.end(),
                       node->Op()->Type()) != lite_ops_filter.end())
      return false;
    return inference::lite::OpTeller::Global().Tell(node->Op()->Type(),
                                                    *node->Op());
  };

  SubGraphFuser fuser(graph, teller, 0 /* min_subgraph_size */, "lite_engine");
  fuser();

  std::vector<std::string> repetitive_params;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      BuildOperator(node, global_program, &repetitive_params);
      std::unordered_set<const Node*> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node*> nodes2remove;
  for (auto* node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(lite_subgraph_pass,
              paddle::inference::analysis::LiteSubgraphPass);
