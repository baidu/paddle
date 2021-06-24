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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/inference/tensorrt/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace tensorrt {

TEST(DropoutOpConverter, main) {
  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  TRTConvertValidation validator(8, parameters, scope, 1000);

  std::vector<int> tensor_shape{8, 10};
  validator.DeclInputVar("dropout-X", tensor_shape, nvinfer1::Dims3(10, 1, 1));
  validator.DeclOutputVar("dropout-Out", nvinfer1::Dims3(10, 1, 1));
  validator.DeclOutputVar("mask-Out", nvinfer1::Dims3(10, 1, 1));

  // Prepare Op description
  framework::OpDesc desc;
  int is_test = 1;
  float dropout_prob = 0.4;
  std::string dropout_implementation = "upscale_in_train";

  desc.SetType("dropout");
  desc.SetInput("X", {"dropout-X"});
  desc.SetOutput("Mask", {"mask-Out"});
  desc.SetOutput("Out", {"dropout-Out"});
  desc.SetAttr("is_test", is_test);
  desc.SetAttr("dropout_prob", dropout_prob);

  desc.SetAttr("dropout_implementation", dropout_implementation);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  std::unordered_set<std::string> neglected_output = {"mask-Out"};

  validator.Execute(8, neglected_output);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(dropout);
