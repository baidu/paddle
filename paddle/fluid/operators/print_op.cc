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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class LoDTensor;
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {
using framework::GradVarName;

#define CLOG std::cout

const char kForward[] = "FORWARD";
const char kBackward[] = "BACKWARD";
const char kBoth[] = "BOTH";

// TODO(ChunweiYan) there should be some other printers for TensorArray
class PrintOp : public framework::OperatorBase {
 public:
  PrintOp(const std::string &type, const framework::VariableNameMap &inputs,
          const framework::VariableNameMap &outputs,
          const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    const auto in_var = scope.FindVar(Input("In"));
    auto out_var = scope.FindVar(Output("Out"));

    PADDLE_ENFORCE_NOT_NULL(
        in_var, platform::errors::NotFound("The input:%s not found in scope",
                                           Input("In")));
    PADDLE_ENFORCE_NOT_NULL(
        out_var, platform::errors::NotFound("The output:%s not found in scope",
                                            Output("Out")));

    auto &in_tensor = in_var->Get<framework::LoDTensor>();
    auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

    PrintValue(place, Inputs("In").front(), in_tensor);
    framework::TensorCopy(in_tensor, place, out_tensor);
    out_tensor->set_lod(in_tensor.lod());
  }

  void PrintValue(const platform::Place &place,
                  const std::string &printed_var_name,
                  const framework::LoDTensor &in_tensor) const {
    std::string print_phase = Attr<std::string>("print_phase");
    bool is_forward = Attr<bool>("is_forward");

    if ((is_forward && print_phase == kBackward) ||
        (!is_forward && print_phase == kForward)) {
      return;
    }

    int first_n = Attr<int>("first_n");
    if (first_n > 0 && ++times_ > first_n) return;

    TensorFormatter formatter;
    const std::string &name =
        Attr<bool>("print_tensor_name") ? printed_var_name : "";
    formatter.SetPrintTensorType(Attr<bool>("print_tensor_type"));
    formatter.SetPrintTensorShape(Attr<bool>("print_tensor_shape"));
    formatter.SetPrintTensorLod(Attr<bool>("print_tensor_lod"));
    formatter.SetPrintTensorLayout(Attr<bool>("print_tensor_layout"));
    formatter.SetSummarize(static_cast<int64_t>(Attr<int>("summarize")));
    formatter.Print(in_tensor, name, Attr<std::string>("message"));
  }

 private:
  mutable int times_{0};
};

class PrintOpProtoAndCheckMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("In", "Input tensor to be displayed.");
    AddOutput("Out", "The output tensor.");
    AddAttr<int>("first_n", "Only log `first_n` number of times.");
    AddAttr<std::string>("message", "A string message to print as a prefix.");
    AddAttr<int>("summarize", "Number of elements printed.");
    AddAttr<bool>("print_tensor_name", "Whether to print the tensor name.")
        .SetDefault(true);
    AddAttr<bool>("print_tensor_type", "Whether to print the tensor's dtype.")
        .SetDefault(true);
    AddAttr<bool>("print_tensor_shape", "Whether to print the tensor's shape.")
        .SetDefault(true);
    AddAttr<bool>("print_tensor_layout",
                  "Whether to print the tensor's layout.")
        .SetDefault(true);
    AddAttr<bool>("print_tensor_lod", "Whether to print the tensor's lod.")
        .SetDefault(true);
    AddAttr<std::string>("print_phase",
                         "(string, default 'FORWARD') Which phase to display "
                         "including 'FORWARD' "
                         "'BACKWARD' and 'BOTH'.")
        .SetDefault(std::string(kBoth))
        .InEnum({std::string(kForward), std::string(kBackward),
                 std::string(kBoth)});
    AddAttr<bool>("is_forward", "Whether is forward or not").SetDefault(true);
    AddComment(R"DOC(
Creates a print op that will print when a tensor is accessed.

Wraps the tensor passed in so that whenever that a tensor is accessed,
the message `message` is printed, along with the current value of the
tensor `t`.)DOC");
  }
};

class PrintOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    VLOG(10) << "PrintOpInferShape";
    OP_INOUT_CHECK(ctx->HasInput("In"), "Input", "In", "Print");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Print");
    ctx->ShareDim("In", /*->*/ "Out");
    ctx->ShareLoD("In", /*->*/ "Out");
  }
};

class PrintOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("Out", ctx->GetInputType("In"));
  }
};

template <typename T>
class PrintOpGradientMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op_desc_ptr) const override {
    op_desc_ptr->SetType("print");
    op_desc_ptr->SetInput("In", this->OutputGrad("Out"));
    op_desc_ptr->SetOutput("Out", this->InputGrad("In"));
    op_desc_ptr->SetAttrMap(this->Attrs());
    op_desc_ptr->SetAttr("is_forward", false);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(print, ops::PrintOp, ops::PrintOpProtoAndCheckMaker,
                  ops::PrintOpGradientMaker<paddle::framework::OpDesc>,
                  ops::PrintOpGradientMaker<paddle::imperative::OpBase>,
                  ops::PrintOpInferShape, ops::PrintOpVarTypeInference);

REGISTER_OP_VERSION(print).AddCheckpoint(
    R"ROC(Upgrade print add a new attribute [print_tensor_layout] to "
             "contorl whether to print tensor's layout.)ROC",
    paddle::framework::compatible::OpVersionDesc().NewAttr(
        "print_tensor_layout", "Whether to print the tensor's layout.", true));
