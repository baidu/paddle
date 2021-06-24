/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/place.h"

DEFINE_string(infer_shape, "", "data shape file");
DEFINE_int32(sample, 20, "number of sample");

namespace paddle {
namespace inference {
namespace analysis {

struct Record {
  std::vector<float> data;
  std::vector<int32_t> shape;
};

Record ProcessALine(const std::string &line, const std::string &shape_line) {
  VLOG(3) << "process a line";

  Record record;
  std::vector<std::string> data_strs;
  split(line, ' ', &data_strs);
  for (auto &d : data_strs) {
    record.data.push_back(std::stof(d));
  }

  std::vector<std::string> shape_strs;
  split(shape_line, ' ', &shape_strs);
  for (auto &s : shape_strs) {
    record.shape.push_back(std::stoi(s));
  }
  return record;
}

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  cfg->DisableGpu();
  // cfg->SwitchIrDebug(); // Enable to have graphs dumped
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(FLAGS_cpu_num_threads);
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              const std::string &line, const std::string &shape_line) {
  auto record = ProcessALine(line, shape_line);

  PaddleTensor input;
  input.shape = record.shape;
  input.dtype = PaddleDType::FLOAT32;
  size_t input_size = record.data.size() * sizeof(float);
  input.data.Resize(input_size);
  memcpy(input.data.data(), record.data.data(), input_size);
  std::vector<PaddleTensor> input_slots;
  input_slots.assign({input});
  (*inputs).emplace_back(input_slots);
}

#ifdef PADDLE_WITH_MKLDNN
int GetNumCachedObjects(void) {
  auto &pool = platform::DeviceContextPool::Instance();
  platform::CPUPlace place;
  auto onednn_dev_ctx =
      dynamic_cast<platform::MKLDNNDeviceContext *>(pool.Get(place));
  return onednn_dev_ctx->GetCachedObjectsNumber();
}

void validate_cache_onednn(int cache_capacity = 1) {
  AnalysisConfig cfg;
  SetConfig(&cfg);
  cfg.EnableMKLDNN();
  cfg.SetMkldnnCacheCapacity(cache_capacity);

  auto predictor = CreatePaddlePredictor<AnalysisConfig>(cfg);
  std::vector<std::vector<PaddleTensor>> ref_outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;

  std::ifstream file(FLAGS_infer_data);
  std::ifstream infer_file(FLAGS_infer_shape);
  std::vector<std::string> lines;
  std::vector<std::string> shape_lines;

  // Let's work with 4 samples
  auto num_samples = 4;
  ref_outputs.resize(num_samples);
  lines.resize(num_samples);
  shape_lines.resize(num_samples);

  // Let's remember number of cached objects before
  // execution and after every single execution
  std::vector<int> cache_filling;
  cache_filling.push_back(GetNumCachedObjects());

  // compute sequentially prediction
  for (int i = 0; i < num_samples; ++i) {
    std::getline(file, lines[i]);
    std::getline(infer_file, shape_lines[i]);
    SetInput(&input_slots_all, lines[i], shape_lines[i]);
    predictor->Run(input_slots_all[i], &ref_outputs[i], FLAGS_batch_size);
    // record number of cached objects
    cache_filling.push_back(GetNumCachedObjects());
  }

  file.close();
  infer_file.close();

  // Pick first output tensor from model
  // as internally reorders may be called
  // so it will impact cache size
  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                   std::multiplies<int>());
  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  // Release predictor (relevant cache should be emptied)
  predictor.reset(nullptr);
  cache_filling.push_back(GetNumCachedObjects());

  // Compare results
  // First and last value should be equal e.g. before using cache (empty) and
  // after releasing executor
  PADDLE_ENFORCE_EQ(
      cache_filling[0], cache_filling[cache_filling.size() - 1],
      platform::errors::Fatal("Cache size before execution and after "
                              "releasing Executor do not match"));

  // Iterate to check if cache is not increasing
  // over exceeding cache capacity
  if (cache_capacity != 0) {
    for (int i = cache_capacity + 1; i < num_samples + 1; ++i) {
      PADDLE_ENFORCE_EQ(
          cache_filling[cache_capacity], cache_filling[i],
          platform::errors::Fatal("Cache capacity should not increase "
                                  "after full capacity is used"));
    }
  }
}

TEST(Analyzer_detect, validate_cache_onednn) {
  validate_cache_onednn(2 /*cache_capacity */);
}
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
