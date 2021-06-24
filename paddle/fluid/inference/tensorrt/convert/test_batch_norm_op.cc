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

#include <gtest/gtest.h>
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(batch_norm_op, test) {
  std::unordered_set<std::string> parameters(
      {"batch_norm_scale", "batch_norm_bias", "batch_norm_mean",
       "batch_norm_variance"});
  framework::Scope scope;
  TRTConvertValidation validator(5, parameters, scope, 1 << 15);
  std::vector<int> param_shape{2};

  validator.DeclInputVar("batch_norm_X", nvinfer1::Dims3(2, 5, 5));
  validator.DeclParamVar("batch_norm_scale", param_shape);
  validator.DeclParamVar("batch_norm_bias", param_shape);
  validator.DeclParamVar("batch_norm_mean", param_shape);
  validator.DeclParamVar("batch_norm_variance", param_shape);
  validator.DeclOutputVar("batch_norm_Y", nvinfer1::Dims3(2, 5, 5));
  validator.DeclOutputVar("batch_norm_save_mean", param_shape);
  validator.DeclOutputVar("batch_norm_save_variance", param_shape);

  // Prepare Op description
  framework::OpDesc desc;

  desc.SetType("batch_norm");
  desc.SetInput("X", {"batch_norm_X"});
  desc.SetInput("Scale", {"batch_norm_scale"});
  desc.SetInput("Bias", {"batch_norm_bias"});
  desc.SetInput("Mean", {"batch_norm_mean"});
  desc.SetInput("Variance", {"batch_norm_variance"});
  desc.SetOutput("Y", {"batch_norm_Y"});
  desc.SetOutput("MeanOut", {"batch_norm_mean"});
  desc.SetOutput("VarianceOut", {"batch_norm_variance"});
  desc.SetOutput("SavedMean", {"batch_norm_save_mean"});
  desc.SetOutput("SavedVariance", {"batch_norm_save_variance"});

  float eps = 1e-5f;
  bool is_test = true;
  desc.SetAttr("epsilon", eps);
  desc.SetAttr("is_test", is_test);

  validator.SetOp(*desc.Proto());

  std::unordered_set<std::string> neglected_output = {
      "batch_norm_save_mean", "batch_norm_save_variance", "batch_norm_mean",
      "batch_norm_variance"};
  validator.Execute(3, neglected_output);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
USE_OP(batch_norm);
