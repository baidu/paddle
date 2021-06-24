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

TEST(leaky_relu_op, test_leaky_relu) {
  std::unordered_set<std::string> parameters;
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("sc_input", nvinfer1::Dims3(4, 2, 2));
  validator.DeclOutputVar("sc_out", nvinfer1::Dims3(4, 2, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("shuffle_channel");
  desc.SetInput("X", {"sc_input"});
  desc.SetOutput("Out", {"sc_out"});
  int group = 2;
  desc.SetAttr("group", group);

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

// USE_OP(leaky_relu);
USE_OP(shuffle_channel);
