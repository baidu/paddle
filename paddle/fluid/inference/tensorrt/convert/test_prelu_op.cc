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

TEST(prelu_op, test_channel_wise) {
  std::unordered_set<std::string> parameters({"prelu_alpha"});
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("prelu_input", nvinfer1::Dims3(3, 2, 2));
  validator.DeclParamVar("prelu_alpha", nvinfer1::Dims3(3, 1, 1));
  validator.DeclOutputVar("prelu_out", nvinfer1::Dims3(3, 2, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("prelu");
  desc.SetInput("X", {"prelu_input"});
  desc.SetInput("Alpha", {"prelu_alpha"});
  desc.SetOutput("Out", {"prelu_out"});

  desc.SetAttr("mode", std::string("channel"));

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

TEST(prelu_op, test_element_wise) {
  std::unordered_set<std::string> parameters({"prelu_alpha"});
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("prelu_input", nvinfer1::Dims3(3, 2, 2));
  validator.DeclParamVar("prelu_alpha", nvinfer1::Dims4(10, 3, 2, 2));
  validator.DeclOutputVar("prelu_out", nvinfer1::Dims3(3, 2, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("prelu");
  desc.SetInput("X", {"prelu_input"});
  desc.SetInput("Alpha", {"prelu_alpha"});
  desc.SetOutput("Out", {"prelu_out"});

  desc.SetAttr("mode", std::string("element"));

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

TEST(prelu_op, test_scalar) {
  std::unordered_set<std::string> parameters({"prelu_alpha"});
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1000);
  validator.DeclInputVar("prelu_input", nvinfer1::Dims3(3, 2, 2));
  validator.DeclParamVar("prelu_alpha", nvinfer1::Dims3(1, 1, 1));
  validator.DeclOutputVar("prelu_out", nvinfer1::Dims3(3, 2, 2));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("prelu");
  desc.SetInput("X", {"prelu_input"});
  desc.SetInput("Alpha", {"prelu_alpha"});
  desc.SetOutput("Out", {"prelu_out"});

  desc.SetAttr("mode", std::string("all"));

  validator.SetOp(*desc.Proto());

  validator.Execute(1);
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(prelu);
