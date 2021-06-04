/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/select_op_helper.h"

namespace paddle {
namespace operators {

// SelectInputOp takes multiple inputs and uses an integer mask to select
// one input to output. It is used in control flow.
class SelectInputOp : public framework::OperatorBase {
 public:
  SelectInputOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    auto output_branch = static_cast<size_t>(GetBranchNumber(mask));

    const std::vector<std::string> &x_names = Inputs("X");
    PADDLE_ENFORCE_LT(
        output_branch, x_names.size(),
        platform::errors::InvalidArgument(
            "Input 'Mask' in SelectInputOp is invalid. "
            "'Mask' must be less than the size of input vector 'X'. "
            "But received Mask = %d, X's size = %d.",
            output_branch, x_names.size()));

    const framework::Variable *selected_x =
        scope.FindVar(x_names[output_branch]);
    framework::Variable *out = scope.FindVar(Output("Out"));
    framework::VisitVarType(*selected_x, AssignFunctor(out, dev_ctx));
  }
};

class SelectInputOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input LoDTensors or LoDTensorArray or SelectedRows. All "
             "inputs must have same variable type")
        .AsDuplicable();
    AddInput("Mask",
             "A integer tensor with numel 1 specifying which input to output");
    AddOutput(
        "Out",
        "The merged output. The variable type of output must be same as X");
    // TODO(huihuangzheng): decide whether to add support for lod level
    // Because this op is blocking whole control flow. I am implementing MVP
    // (minimal viable product) here.
    AddComment(R"DOC(
Merge branches of LoDTensor into a single Output with a mask integer
specifying the output branchi.
)DOC");
  }
};

class SelectInputInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInputs("X"), "Input", "X", "SelectInput");
    OP_INOUT_CHECK(context->HasInput("Mask"), "Input", "Mask", "SelectInput");
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out", "SelectInput");
  }
};

template <typename T>
class SelectInputGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("select_output");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetOutput("Out",
                       this->InputGrad("X", /* drop_empty_grad */ false));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(select_input, ops::SelectInputOp,
                  ops::SelectInputOpProtoMaker, ops::SelectInputInferShape,
                  ops::SelectInputGradMaker<paddle::framework::OpDesc>,
                  ops::SelectInputGradMaker<paddle::imperative::OpBase>);
