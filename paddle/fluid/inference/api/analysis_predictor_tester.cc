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

#include "paddle/fluid/inference/api/analysis_predictor.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <thread>  // NOLINT
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"
#include "paddle/fluid/platform/cpu_info.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {

TEST(AnalysisPredictor, analysis_off) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(false);
  LOG(INFO) << config.Summary();

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  // Without analysis, the scope_ and sub_scope_ are created by predictor
  // itself.
  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // ir is turned off, so program shouldn't be optimized.
  LOG(INFO) << "scope parameters " << predictor->scope_->LocalVarNames().size();

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));
}

TEST(AnalysisPredictor, analysis_on) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(true);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  config.EnableUseGpu(100, 0);
#else
  config.DisableGpu();
#endif
  LOG(INFO) << config.Summary();

  auto _predictor = CreatePaddlePredictor<AnalysisConfig>(config);
  auto* predictor = static_cast<AnalysisPredictor*>(_predictor.get());

  ASSERT_TRUE(predictor->scope_);
  ASSERT_TRUE(predictor->sub_scope_);
  ASSERT_EQ(predictor->scope_->parent(), nullptr);
  ASSERT_EQ(predictor->sub_scope_->parent(), predictor->scope_.get());
  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  for (auto& output : outputs) {
    LOG(INFO) << inference::DescribeTensor(output);
  }

  // compare with NativePredictor
  auto naive_predictor =
      CreatePaddlePredictor<NativeConfig>(config.ToNativeConfig());
  std::vector<PaddleTensor> naive_outputs;
  ASSERT_TRUE(naive_predictor->Run(inputs, &naive_outputs));
  ASSERT_EQ(naive_outputs.size(), 1UL);
  inference::CompareTensor(outputs.front(), naive_outputs.front());
}

TEST(AnalysisPredictor, ZeroCopy) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchUseFeedFetchOps(false);
  LOG(INFO) << config.Summary();
  auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

  auto w0 = predictor->GetInputTensor("firstw");
  auto w1 = predictor->GetInputTensor("secondw");
  auto w2 = predictor->GetInputTensor("thirdw");
  auto w3 = predictor->GetInputTensor("forthw");

  w0->Reshape({4, 1});
  w1->Reshape({4, 1});
  w2->Reshape({4, 1});
  w3->Reshape({4, 1});

  auto* w0_data = w0->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w1_data = w1->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w2_data = w2->mutable_data<int64_t>(PaddlePlace::kCPU);
  auto* w3_data = w3->mutable_data<int64_t>(PaddlePlace::kCPU);

  for (int i = 0; i < 4; i++) {
    w0_data[i] = i;
    w1_data[i] = i;
    w2_data[i] = i;
    w3_data[i] = i;
  }

  predictor->ZeroCopyRun();

  auto out = predictor->GetOutputTensor("fc_1.tmp_2");
  PaddlePlace place;
  int size = 0;
  auto* out_data = out->data<float>(&place, &size);
  LOG(INFO) << "output size: " << size / sizeof(float);
  LOG(INFO) << "output_data: " << out_data;
  predictor->TryShrinkMemory();
}

TEST(AnalysisPredictor, Clone) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchUseFeedFetchOps(true);
  config.SwitchIrOptim(true);
  LOG(INFO) << config.Summary();

  std::vector<std::unique_ptr<PaddlePredictor>> predictors;
  predictors.emplace_back(CreatePaddlePredictor(config));

  LOG(INFO) << "************** to clone ************************";
  const int num_threads = 3;
  for (int i = 1; i < num_threads; i++) {
    predictors.emplace_back(predictors.front()->Clone());
  }

  auto* root_scope =
      static_cast<AnalysisPredictor*>(predictors[0].get())->scope();
  ASSERT_FALSE(root_scope->kids().empty());
  LOG(INFO) << "***** scope ******\n"
            << framework::GenScopeTreeDebugInfo(root_scope);

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> outputs;
  predictors[0]->Run(inputs, &outputs);

  LOG(INFO) << "Run with single thread";
  for (int i = 0; i < num_threads; i++) {
    LOG(INFO) << "run predictor " << i;
    ASSERT_TRUE(predictors[i]->Run(inputs, &outputs));
  }

  LOG(INFO) << "Run with multiple threads";
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([&predictors, &inputs, i] {
      LOG(INFO) << "thread #" << i << " running";
      std::vector<PaddleTensor> outputs;
      auto predictor = predictors.front()->Clone();
      for (int j = 0; j < 10; j++) {
        ASSERT_TRUE(predictor->Run(inputs, &outputs));
      }
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

// This function is not released yet, will fail on some machine.
// TODO(Superjomn) Turn on it latter.
/*
TEST(AnalysisPredictor, memory_optim) {
  AnalysisConfig config(FLAGS_dirname);
  config.DisableGpu();
  config.EnableMemoryOptim(true);
  config.SwitchIrDebug();

  auto native_predictor =
      CreatePaddlePredictor<NativeConfig>(config.ToNativeConfig());

  // 2. Dummy Input Data
  int64_t data[4] = {1, 2, 3, 4};
  PaddleTensor tensor;
  tensor.shape = std::vector<int>({4, 1});
  tensor.data.Reset(data, sizeof(data));
  tensor.dtype = PaddleDType::INT64;

  std::vector<PaddleTensor> inputs(4, tensor);
  std::vector<PaddleTensor> output, output1;

  {
    // The first predictor help to cache the memory optimize strategy.
    auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);
    LOG(INFO) << "serialized program: " << predictor->GetSerializedProgram();
    ASSERT_FALSE(predictor->GetSerializedProgram().empty());

    // Run several times to check the parameters are not reused by mistake.
    for (int i = 0; i < 5; i++) {
      ASSERT_TRUE(predictor->Run(inputs, &output));
    }
  }

  {
    output.clear();
    // The second predictor to perform memory optimization.
    config.EnableMemoryOptim(false);
    auto predictor = CreatePaddlePredictor<AnalysisConfig>(config);

    // Run with memory optimization
    ASSERT_TRUE(predictor->Run(inputs, &output));
  }

  // Run native
  ASSERT_TRUE(native_predictor->Run(inputs, &output1));

  LOG(INFO) << "the output " << inference::DescribeTensor(output.front());
  LOG(INFO) << "the native output "
            << inference::DescribeTensor(output1.front());

  inference::CompareResult(output, output1);
}
*/

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(AnalysisPredictor, bf16_gpu_pass_strategy) {
  AnalysisConfig config;
  config.SetModel(FLAGS_dirname);
  config.SwitchIrOptim(true);
  config.EnableUseGpu(100, 0);
  config.EnableMkldnnBfloat16();
#ifdef PADDLE_WITH_MKLDNN
  if (platform::MayIUse(platform::cpu_isa_t::avx512_core))
    ASSERT_EQ(config.mkldnn_bfloat16_enabled(), true);
  else
    ASSERT_EQ(config.mkldnn_bfloat16_enabled(), false);
#else
  ASSERT_EQ(config.mkldnn_bfloat16_enabled(), false);
#endif
}
#endif

TEST(AnalysisPredictor, bf16_pass_strategy) {
  std::vector<std::string> passes;
  PassStrategy passStrategy(passes);
  passStrategy.EnableMkldnnBfloat16();
}

}  // namespace paddle

namespace paddle_infer {

TEST(Predictor, Run) {
  Config config;
  config.SetModel(FLAGS_dirname);

  auto predictor = CreatePredictor(config);

  auto w0 = predictor->GetInputHandle("firstw");
  auto w1 = predictor->GetInputHandle("secondw");
  auto w2 = predictor->GetInputHandle("thirdw");
  auto w3 = predictor->GetInputHandle("forthw");

  w0->Reshape({4, 1});
  w1->Reshape({4, 1});
  w2->Reshape({4, 1});
  w3->Reshape({4, 1});

  auto* w0_data = w0->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w1_data = w1->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w2_data = w2->mutable_data<int64_t>(PlaceType::kCPU);
  auto* w3_data = w3->mutable_data<int64_t>(PlaceType::kCPU);

  for (int i = 0; i < 4; i++) {
    w0_data[i] = i;
    w1_data[i] = i;
    w2_data[i] = i;
    w3_data[i] = i;
  }

  predictor->Run();

  auto out = predictor->GetOutputHandle("fc_1.tmp_2");
  PlaceType place;
  int size = 0;
  out->data<float>(&place, &size);
  LOG(INFO) << "output size: " << size / sizeof(float);
  predictor->TryShrinkMemory();
}

}  // namespace paddle_infer
