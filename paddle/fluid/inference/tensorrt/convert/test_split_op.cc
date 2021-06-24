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

template <int BatchSize, int Axis>
void TensorRTSplitTest(const std::vector<int> &in_shape,
                       const std::vector<int> &sections) {
  std::unordered_set<std::string> parameters({""});
  framework::Scope scope;
  TRTConvertValidation validator(BatchSize + 1, parameters, scope, 10000);

  auto make_dim = [](const std::vector<int> &shape) {
    nvinfer1::Dims3 dim;
    dim.c() = shape[0];
    dim.h() = shape[1];
    dim.w() = shape[2];
    return dim;
  };
  validator.DeclInputVar("split_input", make_dim(in_shape));
  std::vector<std::string> output_vars;
  for (size_t i = 0; i < sections.size(); ++i) {
    auto out_shape = in_shape;
    out_shape[Axis - 1] = sections[i];
    std::string output_name = "split_out" + std::to_string(i);
    validator.DeclOutputVar(output_name, make_dim(out_shape));
    output_vars.push_back(output_name);
  }

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("split");
  desc.SetInput("X", {"split_input"});
  desc.SetOutput("Out", output_vars);

  desc.SetAttr("axis", Axis);
  desc.SetAttr("num", 0);
  desc.SetAttr("sections", sections);

  validator.SetOp(*desc.Proto());

  validator.Execute(BatchSize);
}

// batch = 0, axis = 1, same shape
TEST(split_op, test_same_shape_axis1_batch1) {
  TensorRTSplitTest<1, 1>({4, 2, 2}, {2, 2});
}
// batch = 0, axis = 1, different shape
TEST(split_op, test_different_shape_axis1_batch1) {
  TensorRTSplitTest<1, 1>({3, 2, 2}, {2, 1});
}
// batch = 10, axis = 1, same shape
TEST(split_op, test_same_shape_axis1_batch10) {
  TensorRTSplitTest<10, 1>({4, 2, 2}, {2, 2});
}
// batch = 10, axis = 1, different shape
TEST(split_op, test_different_shape_axis1_batch10) {
  TensorRTSplitTest<10, 1>({3, 2, 2}, {2, 1});
}
// batch = 0, axis = 2, same shape
TEST(split_op, test_same_shape_axis2_batch1) {
  TensorRTSplitTest<1, 2>({3, 4, 2}, {2, 2});
}
// batch = 0, axis = 2, different shape
TEST(split_op, test_different_shape_axis2_batch1) {
  TensorRTSplitTest<1, 2>({3, 3, 2}, {2, 1});
}
// batch = 10, axis = 2, same shape
TEST(split_op, test_same_shape_axis2_batch10) {
  TensorRTSplitTest<10, 2>({3, 4, 2}, {2, 2});
}
// batch = 10, axis = 2, different shape
TEST(split_op, test_different_shape_axis2_batch10) {
  TensorRTSplitTest<10, 2>({3, 3, 2}, {2, 1});
}
// batch = 0, axis = 3, same shape
TEST(split_op, test_same_shape_axis3_batch1) {
  TensorRTSplitTest<1, 3>({3, 2, 4}, {2, 2});
}
// batch = 0, axis = 3, different shape
TEST(split_op, test_different_shape_axis3_batch1) {
  TensorRTSplitTest<1, 3>({3, 2, 3}, {2, 1});
}
// batch = 10, axis = 3, same shape
TEST(split_op, test_same_shape_axis3_batch10) {
  TensorRTSplitTest<10, 3>({3, 2, 4}, {2, 2});
}
// batch = 10, axis = 3, different shape
TEST(split_op, test_different_shape_axis3_batch10) {
  TensorRTSplitTest<10, 3>({3, 2, 3}, {2, 1});
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

USE_OP(split);
