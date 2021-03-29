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

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_bool(disable_mkldnn_fc, false, "Disable usage of MKL-DNN's FC op");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  SetFakeImageInput(inputs, FLAGS_infer_model);
}

void SetOptimConfig(AnalysisConfig *cfg) {
  std::string optimModelPath = FLAGS_infer_model + "/saved_optim_model";
  cfg->SetModel(optimModelPath + "/model", optimModelPath + "/params");
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false,
             bool onednn_use_input_mem_format = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    if (onednn_use_input_mem_format) cfg.EnableOneDNNUseInputMemFormat();
    if (!FLAGS_disable_mkldnn_fc) {
      cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
      cfg.pass_builder()->AppendPass("fc_act_mkldnn_fuse_pass");
    }
  }
  std::vector<std::vector<PaddleTensor>> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
}

TEST(Analyzer_resnet50, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_resnet50, profile_mkldnn) { profile(true /* use_mkldnn */); }
TEST(Analyzer_resnet50, profile_mkldnn_input_mem_format) {
  profile(true /* use_mkldnn */, true /* onednn_use_input_mem_format */);
}
#endif

// Check the fuse status
TEST(Analyzer_resnet50, fuse_statis) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  int num_ops;
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  auto fuse_statis = GetFuseStatis(
      static_cast<AnalysisPredictor *>(predictor.get()), &num_ops);
  LOG(INFO) << "num_ops: " << num_ops;
}

// Compare result of NativeConfig and AnalysisConfig
void compare(bool use_mkldnn = false,
             bool onednn_use_input_mem_format = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  if (use_mkldnn) {
    cfg.EnableMKLDNN();
    if (onednn_use_input_mem_format) cfg.EnableOneDNNUseInputMemFormat();
    if (!FLAGS_disable_mkldnn_fc) {
      cfg.pass_builder()->AppendPass("fc_mkldnn_pass");
      cfg.pass_builder()->AppendPass("fc_act_mkldnn_fuse_pass");
    }
  }

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareNativeAndAnalysis(
      reinterpret_cast<const PaddlePredictor::Config *>(&cfg), input_slots_all);
}

TEST(Analyzer_resnet50, compare) { compare(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_resnet50, compare_mkldnn) { compare(true /* use_mkldnn */); }
TEST(Analyzer_resnet50, compare_mkldnn_input_mem_format) {
  compare(true /* use_mkldnn */, true /*onednn_use_input_mem_format*/);
}
#endif

// Compare Deterministic result
TEST(Analyzer_resnet50, compare_determine) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareDeterministic(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                       input_slots_all);
}

// Save optim model
TEST(Analyzer_resnet50, save_optim_model) {
  AnalysisConfig cfg;
  std::string optimModelPath = FLAGS_infer_model + "/saved_optim_model";
#ifdef _WIN32
  _mkdir(optimModelPath.c_str());
#else
  mkdir(optimModelPath.c_str(), 0777);
#endif
  SetConfig(&cfg);
  SaveOptimModel(&cfg, optimModelPath);
}

void CompareOptimAndOrig(const PaddlePredictor::Config *orig_config,
                         const PaddlePredictor::Config *optim_config,
                         const std::vector<std::vector<PaddleTensor>> &inputs) {
  PrintConfig(orig_config, true);
  PrintConfig(optim_config, true);
  std::vector<std::vector<PaddleTensor>> orig_outputs, optim_outputs;
  TestOneThreadPrediction(orig_config, inputs, &orig_outputs, false);
  TestOneThreadPrediction(optim_config, inputs, &optim_outputs, false);
  CompareResult(orig_outputs.back(), optim_outputs.back());
}

TEST(Analyzer_resnet50, compare_optim_orig) {
  AnalysisConfig orig_cfg;
  AnalysisConfig optim_cfg;
  SetConfig(&orig_cfg);
  SetOptimConfig(&optim_cfg);
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  CompareOptimAndOrig(
      reinterpret_cast<const PaddlePredictor::Config *>(&orig_cfg),
      reinterpret_cast<const PaddlePredictor::Config *>(&optim_cfg),
      input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
