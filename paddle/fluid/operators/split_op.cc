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

#include "paddle/fluid/operators/split_op.h"
#include <string>

namespace paddle {
namespace operators {
using framework::Tensor;

class SplitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of SplitOp should not be null."));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Outputs(Out) of SplitOp should not be empty."));
    auto in_dims = ctx->GetInputDim("X");
    auto outs_names = ctx->Outputs("Out");
    size_t axis = static_cast<size_t>(ctx->Attrs().Get<int>("axis"));
    size_t num = static_cast<size_t>(ctx->Attrs().Get<int>("num"));
    std::vector<int> sections = static_cast<std::vector<int>>(
        ctx->Attrs().Get<std::vector<int>>("sections"));
    const size_t outs_number = outs_names.size();

    if (sections.size() > 0) {
      PADDLE_ENFORCE_EQ(
          sections.size(), outs_number,
          platform::errors::InvalidArgument("tensor split sections size "
                                            "should be equal to output size."));
    }

    if (ctx->HasInput("AxisTensor")) {
      auto out_dims =
          framework::make_ddim(std::vector<int>(in_dims.size(), -1));
      std::vector<framework::DDim> outs_dims(outs_number, out_dims);
      ctx->SetOutputsDim("Out", outs_dims);
      for (size_t i = 0; i < outs_number; ++i) {
        ctx->ShareLoD("X", "Out", 0, i);
      }
      return;
    }

    bool each_section_is_known =
        (sections.size() > 0 && !ctx->HasInputs("SectionsTensorList"));

    auto outs_dims = UpdateOutsDims(ctx->IsRuntime(), each_section_is_known,
                                    in_dims, num, sections, axis, outs_number);
    ctx->SetOutputsDim("Out", outs_dims);
    if (axis != 0) {
      // Only pass LoD when not spliting along the first dim.
      for (size_t i = 0; i < outs_number; ++i) {
        ctx->ShareLoD("X", "Out", 0, i);
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "AxisTensor" || var_name == "SectionsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class SplitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of the split operator.");
    AddInput("AxisTensor",
             "(Tensor) The axis which the input will be split on. "
             "It has higher priority than Attr(axis). "
             "The shape of AxisTensor must be [1]")
        .AsDispensable();
    AddInput("SectionsTensorList",
             "(vector<Tensor<int>>, optional). "
             "The length of each output along the specified axis. "
             "It has a higher priority than Attr(sections)."
             "The shape of the element in vector must be [1].")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "(Tensor) Output tensors of the split operator.")
        .AsDuplicable();
    AddComment(R"DOC(
Split operator

This operator splits the input tensor into multiple sub-tensors.

Example:
  Input = [[1,2],
           [3,4],
           [5,6]]
  sections = [2,1]
  axis = 0
  Output[0] = [[1,2],
               [3,4]]
  Output[1] = [[5,6]]

    )DOC");
    AddAttr<std::vector<int>>("sections",
                              "(vector<int>) "
                              "the length of each output along the "
                              "specified axis.")
        .SetDefault(std::vector<int>{});
    AddAttr<int>("num",
                 "(int, default 0)"
                 "Number of sub-tensors. This must evenly divide "
                 "Input.dims()[axis]")
        .SetDefault(0);
    AddAttr<int>("axis",
                 "(int, default 0) "
                 "The axis which the input will be split on.")
        .SetDefault(0);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(split, ops::SplitOp, ops::SplitOpMaker,
                  ops::SplitGradMaker<paddle::framework::OpDesc>,
                  ops::SplitGradMaker<paddle::imperative::OpBase>);
namespace plat = paddle::platform;
REGISTER_OP_CPU_KERNEL(
    split, ops::SplitOpKernel<plat::CPUDeviceContext, double>,
    ops::SplitOpKernel<plat::CPUDeviceContext, float>,
    ops::SplitOpKernel<plat::CPUDeviceContext, int64_t>,
    ops::SplitOpKernel<plat::CPUDeviceContext, int>,
    ops::SplitOpKernel<plat::CPUDeviceContext, bool>,
    ops::SplitOpKernel<plat::CPUDeviceContext, plat::float16>);
