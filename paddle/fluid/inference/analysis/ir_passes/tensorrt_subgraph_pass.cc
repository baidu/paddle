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

#include <algorithm>
#include <map>
#include <set>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

void analysis::TensorRtSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph);
  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  bool no_calib_int8 = enable_int8 && !(use_calib_mode);
  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    return tensorrt::OpTeller::Global().Tell(node->Op()->Type(), *node->Op(),
                                             no_calib_int8);
  };

  framework::ir::SubGraphFuser fuser(
      graph, teller, Get<int>("min_subgraph_size") /*min subgraph size*/,
      "tensorrt_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in trt, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      CreateTensorRTOp(node, graph, graph_param_names, &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          framework::ir::Agent(node).subgraph()->begin(),
          framework::ir::Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += predictor_id;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

void TensorRtSubgraphPass::CreateTensorRTOp(
    framework::ir::Node *node, framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "The subgraph should not be empty."));

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for TensorRTEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  // const framework::BlockDesc& main_block = program_desc->Block(0);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  // if we delete fluid copy of params shared by more than 1 ops, there will be
  // problem, so we filter them out.
  std::vector<std::string> params_not_shared;

  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0 &&
        x->outputs.size() <= 1) {
      params_not_shared.push_back(x->Name());
    }
  }

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }

  std::unordered_map<std::string, std::string> output_name_map;
  std::unordered_map<std::string, framework::ir::Node *> graph_var_map;

  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      graph_var_map[node->Name()] = node;
    }
  }
  auto precision_mode = Get<AnalysisConfig::Precision>("precision_mode");
  bool enable_fp16 = false;
  if (precision_mode == AnalysisConfig::Precision::kHalf) enable_fp16 = true;
  auto enable_int8 = Get<bool>("enable_int8");
  auto use_calib_mode = Get<bool>("use_calib_mode");
  auto &subgraph_nodes = *framework::ir::Agent(node).subgraph();
  auto min_input_shape =
      Get<std::map<std::string, std::vector<int>>>("min_input_shape");
  auto max_input_shape =
      Get<std::map<std::string, std::vector<int>>>("max_input_shape");
  auto opt_input_shape =
      Get<std::map<std::string, std::vector<int>>>("optim_input_shape");

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.
  RenameAndGetOutputs(subgraph_nodes, &block_desc, input_names_with_id,
                      &output_names_with_id, &output_names, &output_name_map,
                      graph_var_map, !enable_int8);

  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  for (auto name : output_names) {
    PADDLE_ENFORCE_NE(output_name_map.count(name), 0,
                      platform::errors::PreconditionNotMet(
                          "The output_name_map should have %s", name));
    output_mapping.push_back(output_name_map[name]);
  }
  PADDLE_ENFORCE_EQ(output_mapping.empty(), false,
                    platform::errors::PreconditionNotMet(
                        "The output_mapping should not be empty."));
  PADDLE_ENFORCE_EQ(
      !block_desc.Proto()->vars().empty(), true,
      platform::errors::PreconditionNotMet("the block has no var-desc"));

  // Set attrs
  op_desc->SetType("tensorrt_engine");
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));

  op_desc->SetBlockAttr("sub_block", new_block);
  op_desc->SetAttr("subgraph", block_desc.Proto()->SerializeAsString());
  op_desc->SetAttr("max_batch_size", Get<int>("max_batch_size"));
  op_desc->SetAttr("workspace_size", Get<int>("workspace_size"));
  op_desc->SetAttr("gpu_id", Get<int>("gpu_device_id"));
  op_desc->SetAttr("output_name_mapping", output_mapping);
  op_desc->SetAttr("parameters", params);

  // we record all inputs' shapes in attr to check if they are consistent
  // with the real inputs' shapes retrieved from scope when trt runs.
  for (auto *x : node->inputs) {
    if (x->IsVar() && x->Var()) {
      framework::VarDesc *var = x->Var();
      op_desc->SetAttr(var->Name() + "_shape", var->GetShape());
    }
  }

  auto use_static_engine = Get<bool>("use_static_engine");
  // TODO(NHZlX)
  // There are models with the same structure but the different parameters,
  // when running in the 'use_serialize' mode, there is a bug.
  auto engine_key = GenerateEngineKey(input_names_with_id, output_names_with_id,
                                      std::to_string(0));
  auto predictor_id = Get<int>("predictor_id");

  // Get "" when there is no cached calibration table data.
  bool load_from_memory = Get<bool>("model_from_memory");
  std::string calibration_data = "";
  if (enable_int8 && use_calib_mode) {
    calibration_data = GetTrtCalibTableData(
        Get<std::string>("model_opt_cache_dir"), engine_key, enable_int8);
  }
  op_desc->SetAttr("calibration_data", calibration_data);
  op_desc->SetAttr("enable_int8", enable_int8);
  op_desc->SetAttr("enable_fp16", enable_fp16);
  op_desc->SetAttr("use_calib_mode", use_calib_mode);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("predictor_id", predictor_id);

  std::string trt_engine_serialized_data = "";
  op_desc->SetAttr("engine_serialized_data", trt_engine_serialized_data);
  op_desc->Flush();

  std::unique_ptr<tensorrt::TRTInt8Calibrator> calibrator;
  if (enable_int8 && calibration_data.size() != 0) {
    calibrator.reset(new tensorrt::TRTInt8Calibrator(calibration_data));
    LOG(INFO) << "RUN Paddle TRT int8 calibration mode...";
  }
  // When in int8 mode and calibration_mode, the program just produce the
  // calibration table data.
  bool calibration_mode =
      (enable_int8 && calibration_data.size() == 0 && use_calib_mode);
  if (calibration_mode) {
    // calibraion mode means generate int8 calibration table data process.
    return;
  }

  std::copy(params_not_shared.begin(), params_not_shared.end(),
            std::back_inserter(*repetitive_params));

  // Check trt version for dynamic shape input.

  if (min_input_shape.size() > 0 && TRT_VERSION < 6000) {
    LOG_FIRST_N(WARNING, 1) << "You are using the dynamic size input mode of "
                               "Paddle-TRT, but we found that the version of "
                               "the TensorRT is less than 6.0, so we use the "
                               "static shape mode instead.";
    min_input_shape = {};
    max_input_shape = {};
    opt_input_shape = {};
  }

  if (min_input_shape.size() > 0 && TRT_VERSION > 6000) {
    LOG_FIRST_N(WARNING, 1)
        << "The Paddle lib links the " << TRT_VERSION << " version TensorRT, "
        << "make sure the runtime TensorRT you are using is no less than this "
           "version, otherwise, there might be Segfault!";
  }

  // Setting the disable_trt_plugin_fp16 to true means that TRT plugin will not
  // run fp16.
  // When running fp16, the output accuracy of the model will be affected,
  // closing the plugin fp16 may bring some improvement on accuracy.
  bool disable_trt_plugin_fp16 = Get<bool>("disable_trt_plugin_fp16");
  tensorrt::TensorRTEngine *trt_engine =
      inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
          .Create(engine_key + std::to_string(predictor_id),
                  Get<int>("max_batch_size"), Get<int>("workspace_size"),
                  precision_mode, calibrator.get(), Get<int>("gpu_device_id"),
                  min_input_shape, max_input_shape, opt_input_shape,
                  disable_trt_plugin_fp16);

  bool need_serialize = (use_static_engine && !load_from_memory);
  if (need_serialize) {
    trt_engine_serialized_data = GetTrtEngineSerializedData(
        Get<std::string>("model_opt_cache_dir"), engine_key);
    // we can load the engine info serialized before from the disk.
    if (!trt_engine_serialized_data.empty()) {
      trt_engine->Deserialize(trt_engine_serialized_data);
      LOG(INFO) << "Load TRT Optimized Info from "
                << GetTrtEngineSerializedPath(
                       Get<std::string>("model_opt_cache_dir"), engine_key);
      return;
    }
  }

  // the following code will NOT run in following situation:
  // 1. calibraion mode (generate trt int8 calibraiton table data)
  // 2. already load serialized trt engine info.
  LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
               "kernel etc). This process may cost a lot of time.";

  auto *scope = param_scope();
  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());
  std::unordered_set<std::string> param_set(params.begin(), params.end());
  inference::Singleton<inference::tensorrt::OpConverter>::Global()
      .ConvertBlockToTRTEngine(
          &block_desc_temp, *scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          param_set, output_mapping, trt_engine);

  if (need_serialize) {
    nvinfer1::IHostMemory *serialized_engine_data = trt_engine->Serialize();
    trt_engine_serialized_data =
        std::string((const char *)serialized_engine_data->data(),
                    serialized_engine_data->size());
    SaveTrtEngineSerializedDataToFile(
        GetTrtEngineSerializedPath(Get<std::string>("model_opt_cache_dir"),
                                   engine_key),
        trt_engine_serialized_data);
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");

REGISTER_PASS_CAPABILITY(tensorrt_subgraph_pass)
    .AddCombination(
        paddle::framework::compatible::OpVersionComparatorCombination()
            .EQ("conv2d", 0)
            .EQ("pool2d", 0)
            .EQ("relu", 0)
            .EQ("softmax", 0)
            .EQ("sigmoid", 0)
            .EQ("hard_swish", 0)
            .EQ("depthwise_conv2d", 0)
            .EQ("batch_norm", 0)
            .EQ("concat", 0)
            .EQ("tanh", 0)
            .EQ("pad", 0)
            .EQ("elementwise_add", 0)
            .EQ("elementwise_mul", 0)
            .EQ("prelu", 0)
            .LE("conv2d_transpose", 1)
            .LE("leaky_relu", 1)
            .EQ("fc", 0)
            .EQ("shuffle_channel", 0)
            .EQ("swish", 0)
            .EQ("split", 0)
            .EQ("instance_norm", 0)
            .EQ("gelu", 0)
            .EQ("layer_norm", 0)
            .EQ("scale", 0));
