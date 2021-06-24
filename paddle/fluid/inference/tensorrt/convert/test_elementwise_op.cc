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

TEST(elementwise_op, add_weight) {
  std::unordered_set<std::string> parameters({"elementwise_add-Y"});
  framework::Scope scope;
  TRTConvertValidation validator(10, parameters, scope, 1 << 15);
  validator.DeclInputVar("elementwise_add-X", nvinfer1::Dims3(10, 3, 3));
  validator.DeclParamVar("elementwise_add-Y", nvinfer1::Dims3(10, 1, 1));
  validator.DeclOutputVar("elementwise_add-Out", nvinfer1::Dims3(10, 3, 3));

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("elementwise_add");
  desc.SetInput("X", {"elementwise_add-X"});
  desc.SetInput("Y", {"elementwise_add-Y"});
  desc.SetOutput("Out", {"elementwise_add-Out"});

  int axis = 1;
  desc.SetAttr("axis", axis);

  validator.SetOp(*desc.Proto());

  validator.Execute(8);
}

TEST(elementwise_op, native) {
  for (std::string type : {"add", "mul"}) {
    int batch_size = 8;
    std::unordered_set<std::string> parameters;
    framework::Scope scope;
    TRTConvertValidation validator(batch_size, parameters, scope, 1 << 15);
    validator.DeclInputVar("elementwise_" + type + "-X",
                           nvinfer1::Dims3(10, 3, 3));
    validator.DeclInputVar("elementwise_" + type + "-Y",
                           nvinfer1::Dims3(10, 3, 3));
    validator.DeclOutputVar("elementwise_" + type + "-Out",
                            nvinfer1::Dims3(10, 3, 3));

    // Prepare Op description
    framework::OpDesc desc;
    desc.SetType("elementwise_" + type);
    desc.SetInput("X", {"elementwise_" + type + "-X"});
    desc.SetInput("Y", {"elementwise_" + type + "-Y"});
    desc.SetOutput("Out", {"elementwise_" + type + "-Out"});

    int axis = -1;
    desc.SetAttr("axis", axis);

    validator.SetOp(*desc.Proto());
    validator.Execute(batch_size);
  }
}

TEST(elementwise_op, plugin) {
  for (std::string type : {"add", "mul"}) {
    int batch_size = 8;
    std::unordered_set<std::string> parameters;
    framework::Scope scope;
    TRTConvertValidation validator(batch_size, parameters, scope, 1 << 15);
    validator.DeclInputVar("elementwise_" + type + "-X",
                           nvinfer1::Dims3(10, 3, 3));
    validator.DeclInputVar("elementwise_" + type + "-Y",
                           nvinfer1::Dims3(10, 1, 1));
    validator.DeclOutputVar("elementwise_" + type + "-Out",
                            nvinfer1::Dims3(10, 3, 3));

    // Prepare Op description
    framework::OpDesc desc;
    desc.SetType("elementwise_" + type);
    desc.SetInput("X", {"elementwise_" + type + "-X"});
    desc.SetInput("Y", {"elementwise_" + type + "-Y"});
    desc.SetOutput("Out", {"elementwise_" + type + "-Out"});

    int axis = -1;
    desc.SetAttr("axis", axis);

    validator.SetOp(*desc.Proto());
    validator.Execute(batch_size);
  }
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(elementwise_add);
USE_OP(elementwise_mul);
