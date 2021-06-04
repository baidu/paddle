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
#include "paddle/fluid/operators/assign_op.h"
#include "paddle/fluid/operators/select_op_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
class Variable;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

// SelectOutputOp has one input, one integer mask and multiple outputs. It
// selects one output specified by the mask and copy the input to it.
class SelectOutputOp : public framework::OperatorBase {
 public:
  SelectOutputOp(const std::string &type,
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

    const std::vector<std::string> &out_names = Outputs("Out");
    PADDLE_ENFORCE_LT(
        output_branch, out_names.size(),
        platform::errors::InvalidArgument(
            "Input 'Mask' in SelectOutputOp is invalid. "
            "'Mask' must be less than the size of output vector 'Out'. "
            "But received Mask = %d, Out's size = %d.",
            output_branch, out_names.size()));

    const framework::Variable *x = scope.FindVar(Input("X"));
    framework::Variable *selected_out = scope.FindVar(out_names[output_branch]);
    framework::VisitVarType(*x, AssignFunctor(selected_out, dev_ctx));
  }
};

class SelectOutputOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input LoDTensor or LoDTensorArray or SelectedRows.");
    AddInput("Mask", "Tensor with numel 1 specifying which branch to output");
    AddOutput("Out",
              "The output can contains multiple variables. The output of "
              "selected branch will be same as input. We do nothing for "
              "variables in other branch")
        .AsDuplicable();
    // TODO(huihuangzheng): decide whether to add support for lod level
    // Because this op is blocking whole control flow. I am implementing MVP
    // (minimal viable product) here.
    AddComment(R"DOC(
Split input variable into one output branch. The mask is an integer tensor to
specify which output branch should copy the input. 
)DOC");
  }
};

class SelectOutputInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "SelectOutput");
    OP_INOUT_CHECK(context->HasInput("Mask"), "Input", "Mask", "SelectOutput");
    OP_INOUT_CHECK(context->HasOutputs("Out"), "Output", "Out", "SelectOutput");
  }
};

template <typename T>
class SelectOutputGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("select_input");
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(select_output, ops::SelectOutputOp,
                  ops::SelectOutputOpProtoMaker, ops::SelectOutputInferShape,
                  ops::SelectOutputGradMaker<paddle::framework::OpDesc>,
                  ops::SelectOutputGradMaker<paddle::imperative::OpBase>);
