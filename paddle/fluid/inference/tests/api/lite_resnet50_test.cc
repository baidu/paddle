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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cmath>

#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {

TEST(AnalysisPredictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  AnalysisConfig config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableLiteEngine(paddle::AnalysisConfig::Precision::kFloat32, true);

  std::vector<PaddleTensor> inputs;
  auto predictor = CreatePaddlePredictor(config);
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  PaddleTensor in;
  in.shape = {1, 3, 318, 318};
  in.data =
      PaddleBuf(static_cast<void*>(input.data()), input_num * sizeof(float));
  in.dtype = PaddleDType::FLOAT32;
  inputs.emplace_back(in);

  std::vector<PaddleTensor> outputs;
  ASSERT_TRUE(predictor->Run(inputs, &outputs));

  const std::vector<float> truth_values = {
      127.779f,  738.165f,  1013.22f,  -438.17f,  366.401f,  927.659f,
      736.222f,  -633.684f, -329.927f, -430.155f, -633.062f, -146.548f,
      -1324.28f, -1349.36f, -242.675f, 117.448f,  -801.723f, -391.514f,
      -404.818f, 454.16f,   515.48f,   -133.031f, 69.293f,   590.096f,
      -1434.69f, -1070.89f, 307.074f,  400.525f,  -316.12f,  -587.125f,
      -161.056f, 800.363f,  -96.4708f, 748.706f,  868.174f,  -447.938f,
      112.737f,  1127.2f,   47.4355f,  677.72f,   593.186f,  -336.4f,
      551.362f,  397.823f,  78.3979f,  -715.398f, 405.969f,  404.256f,
      246.019f,  -8.42969f, 131.365f,  -648.051f};

  const size_t expected_size = 1;
  EXPECT_EQ(outputs.size(), expected_size);
  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); j += 10) {
    EXPECT_NEAR((data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0.,
                12e-5);
  }
}

}  // namespace inference
}  // namespace paddle

namespace paddle_infer {

TEST(Predictor, use_gpu) {
  std::string model_dir = FLAGS_infer_model + "/" + "model";
  Config config;
  config.EnableUseGpu(100, 0);
  config.SetModel(model_dir + "/model", model_dir + "/params");
  config.EnableLiteEngine(PrecisionType::kFloat32);

  auto predictor = CreatePredictor(config);
  const int batch = 1;
  const int channel = 3;
  const int height = 318;
  const int width = 318;
  const int input_num = batch * channel * height * width;
  std::vector<float> input(input_num, 1);

  auto input_names = predictor->GetInputNames();
  auto input_t = predictor->GetInputHandle(input_names[0]);

  input_t->Reshape({1, 3, 318, 318});
  input_t->CopyFromCpu(input.data());
  predictor->Run();

  auto output_names = predictor->GetOutputNames();
  auto output_t = predictor->GetOutputHandle(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  size_t out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                   std::multiplies<int>());

  std::vector<float> out_data;
  out_data.resize(out_num);
  output_t->CopyToCpu(out_data.data());

  const std::vector<float> truth_values = {
      127.779f,  738.165f,  1013.22f,  -438.17f,  366.401f,  927.659f,
      736.222f,  -633.684f, -329.927f, -430.155f, -633.062f, -146.548f,
      -1324.28f, -1349.36f, -242.675f, 117.448f,  -801.723f, -391.514f,
      -404.818f, 454.16f,   515.48f,   -133.031f, 69.293f,   590.096f,
      -1434.69f, -1070.89f, 307.074f,  400.525f,  -316.12f,  -587.125f,
      -161.056f, 800.363f,  -96.4708f, 748.706f,  868.174f,  -447.938f,
      112.737f,  1127.2f,   47.4355f,  677.72f,   593.186f,  -336.4f,
      551.362f,  397.823f,  78.3979f,  -715.398f, 405.969f,  404.256f,
      246.019f,  -8.42969f, 131.365f,  -648.051f};

  float* data_o = out_data.data();
  for (size_t j = 0; j < out_num; j += 10) {
    EXPECT_NEAR((data_o[j] - truth_values[j / 10]) / truth_values[j / 10], 0.,
                10e-5);
  }
}

}  // namespace paddle_infer
