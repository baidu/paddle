/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include "paddle/fluid/operators/controlflow/while_op_helper.h"
#include "paddle/fluid/operators/tensor_formatter.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class LoDTensor;
class OpDesc;
class Scope;
class Variable;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

const char kCond[] = "Cond";
const char kData[] = "Data";
const char kSummarize[] = "summarize";

namespace paddle {
namespace operators {

using framework::LoDTensor;

class AssertOp : public framework::OperatorBase {
 public:
  AssertOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    const framework::Variable *cond_var_ptr = scope.FindVar(Input(kCond));
    PADDLE_ENFORCE_NOT_NULL(cond_var_ptr,
                            platform::errors::NotFound(
                                "Input(Condition) of AssertOp is not found."));
    const auto &cond = cond_var_ptr->Get<LoDTensor>();
    PADDLE_ENFORCE_EQ(
        cond.dims(), paddle::framework::make_ddim({1}),
        platform::errors::InvalidArgument(
            "The numel of Input(Condition) of AssertOp must be 1. But now "
            "the Condition's shape is %s.",
            cond.dims().to_str()));

    bool cond_data = GetCondData(cond);
    if (cond_data) {
      return;
    }

    TensorFormatter formatter;
    formatter.SetSummarize(Attr<int64_t>(kSummarize));

    const std::vector<std::string> &x_names = Inputs(kData);
    for (const std::string &name : x_names) {
      const framework::Variable *x_var_ptr = scope.FindVar(name);
      const auto &x_tensor = x_var_ptr->Get<LoDTensor>();
      formatter.Print(x_tensor, name);
    }

    PADDLE_THROW(platform::errors::InvalidArgument(
        "The condition variable '%s' of AssertOp must be "
        "true, but received false",
        Input(kCond)));
  }
};

class AssertOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        kCond,
        "The boolean scalar condition tensor which is asserted to be true.");
    AddInput(kData,
             "The tensors to print when the assert condition is not true.")
        .AsDuplicable();
    AddAttr<int64_t>(
        kSummarize,
        "The number of entries of each tensor to print when the "
        "assert condition is not true. -1 means print all entries. If "
        "the number of entries of a tensor is less then "
        "summarize_num, this OP will print all entries of the tensor.")
        .SetDefault(-1);
    AddComment(
        R"DOC(Assert the input Condition Tensor is true and print Tensors if the Condition Tensor is false.)DOC");
  }
};

class AssertOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInputs(kCond), "Input", "Condition", "AssertOp");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    assert, ops::AssertOp, ops::AssertOpProtoMaker, ops::AssertOpInferShape,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
