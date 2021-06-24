/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/operators/load_op.h"

namespace paddle {
namespace operators {

class LoadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = framework::OpKernelType(
        framework::proto::VarType::FP32, ctx.GetPlace());
    return kt;
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The LoDTensor / SelectedRows need to be loaded");
    AddAttr<bool>(
        "load_as_fp16",
        "If true, the tensor will be first loaded and then "
        "converted to float16 data type. Otherwise, the tensor will be "
        "directly loaded without data type conversion. Default is false.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddAttr<int64_t>("seek", "(int64_t) Starting for load tensor from seek pos")
        .SetDefault(-1);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddComment(
        "Load operator will load a LoDTensor / SelectedRows variable from "
        "disk "
        "file.");
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);

REGISTER_OP_CPU_KERNEL(
    load, ops::LoadOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LoadOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::LoadOpKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::bfloat16>,
    ops::LoadOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::LoadOpKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::LoadOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
