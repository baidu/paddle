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

#include "paddle/fluid/operators/save_op.h"

#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
class SaveOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

class SaveOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor ) Input LoDTensor and SelectedRows to be saved");
    AddComment(R"DOC(
Save operator

This operator will serialize and write LoDTensor / SelectedRows variable to file on disk.
)DOC");
    AddAttr<bool>("overwrite",
                  "(boolean, default true)"
                  "Overwrite the output file if exist")
        .SetDefault(true);
    AddAttr<bool>("save_as_fp16",
                  "(boolean, default false)"
                  "If true, the tensor will be converted to float16 data "
                  "type and then saved. Otherwise, the tensor will be "
                  "directly saved without data type conversion.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         "(string)"
                         "The \"file_path\" where the variable will be saved.")
        .AddCustomChecker(
            [](const std::string &path) { return !path.empty(); });
    AddOutput(LOOKUP_TABLE_PATH,
              "(string)"
              "for pserver: The \"kLookupTablePath\" where checkpoint notify "
              "to save lookup table variables"
              " to directory specified.")
        .AsDispensable();
  }
};

class SaveOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_type = framework::proto::VarType::RAW;
    ctx->InsertVar(LOOKUP_TABLE_PATH, var_type);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(save, ops::SaveOp, ops::SaveOpProtoMaker,
                  ops::SaveOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    save, ops::SaveOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext,
                      paddle::platform::float16>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, uint8_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int8_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int16_t>,
    ops::SaveOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
