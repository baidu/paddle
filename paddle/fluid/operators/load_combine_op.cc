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

#include <string>
#include <vector>

#include "paddle/fluid/operators/load_combine_op.h"

namespace paddle {
namespace operators {

class LoadCombineOp : public framework::OperatorWithKernel {
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

class LoadCombineOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput(
        "Out",
        "(vector) The output LoDTensors that will be read from the input file.")
        .AsDuplicable();
    AddAttr<bool>(
        "load_as_fp16",
        "(boolean, default false)"
        "If true, the tensor will be first loaded and then "
        "converted to float16 data type. Otherwise, the tensor will be "
        "directly loaded without data type conversion.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         "(string) "
                         "LoDTensors will be loaded from \"file_path\".")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddAttr<bool>("model_from_memory",
                  "(boolean, default false)"
                  "If true, file_path is in memory, and LoDTensors will be "
                  "loaded directly from memory")
        .SetDefault(false);
    AddComment(R"DOC(
LoadCombine Operator.

LoadCombine operator loads LoDTensor variables from a file, which could be
loaded in memory already. The file should contain one or more LoDTensors
serialized using the SaveCombine operator. The
LoadCombine operator applies a deserialization strategy to appropriately load
the LodTensors, and this strategy complements the serialization strategy used
in the SaveCombine operator. Hence, the LoadCombine operator is tightly coupled
with the SaveCombine operator, and can only deserialize one or more LoDTensors
that were saved using the SaveCombine operator.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(load_combine, ops::LoadCombineOp,
                  ops::LoadCombineOpProtoMaker);

REGISTER_OP_CPU_KERNEL(
    load_combine,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext,
                             paddle::platform::bfloat16>,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::LoadCombineOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
