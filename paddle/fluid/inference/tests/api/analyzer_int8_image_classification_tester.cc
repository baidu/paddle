/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim();
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
  cfg->EnableMKLDNN();
}

TEST(Analyzer_int8_image_classification, quantization) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  AnalysisConfig q_cfg;
  SetConfig(&q_cfg);

  // read data from file and prepare batches with test data
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInputs(&input_slots_all);

  if (FLAGS_enable_int8) {
    // prepare warmup batch from input data read earlier
    // warmup batch size can be different than batch size
    std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
        paddle::inference::GetWarmupData(input_slots_all);

    // configure quantizer
    q_cfg.EnableMkldnnQuantizer();
    q_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
    q_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(
        FLAGS_warmup_batch_size);
  }

  CompareQuantizedAndAnalysis(&cfg, &q_cfg, input_slots_all);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
