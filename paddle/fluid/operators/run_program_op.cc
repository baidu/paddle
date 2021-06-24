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

#include "paddle/fluid/operators/run_program_op.h"

#include <string>

namespace paddle {
namespace operators {

class RunProgramOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of RunProgramOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutputs("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of RunProgramOp should not be null."));
  }

 protected:
  /* [Why use single type kernel]:
   *
   * This op is similar to a control flow op, it doses not need
   * a op kernel, but in order to make it execute under dynamic
   * graph mode, implement it with op kernel.
   *
   * So whether the kernel data type is int, float or other type,
   * which has no effect on its execution logic, so directly
   * specified a data type here.
   *
   * Of course, the data type here is also not important.
   */
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

class RunProgramOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(vector<LoDTensor>)"
             "The input tensors of RunProgram operator, also the feed targets "
             "of loaded program.")
        .AsDuplicable();
    AddInput("Params",
             "(vector<LoDTensor or SelecetedRows>)"
             "The input parameter of RunProgram operator, also the parameters "
             "of the loaded program.")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out",
              "(vector<LoDTensor>)"
              "The output tensors of RunProgram operator, also the fetch "
              "targets of the loaded program.")
        .AsDuplicable();
    AddOutput("OutScope",
              "(StepScopeVar)"
              "A vector of execution scope in RunProgram operator, which "
              "contains at most one scope."
              "NOTE: Do not use Scope directly because Scope output is not "
              "currently supported.");
    AddOutput("DOut",
              "(vector<LoDTensor>)"
              "The output tensors for GRAD Tensors in RunProgram forward "
              "operator, the forward operator contains GRAD Tensors when it "
              "computes double grad.")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<BlockDesc*>("global_block",
                        "(BlockDesc *)"
                        "The global block of executed program desc.");
    AddAttr<int64_t>("start_op_index",
                     "(int64_t)"
                     "The index of the op to start execution");
    AddAttr<int64_t>("end_op_index",
                     "(int64_t)"
                     "The index of the op to stop execution");
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training.")
        .SetDefault(false);
    AddComment(R"DOC(
RunProgram operator.

The RunProgram operator receives a program's feed targets, fetch targets, 
and parameters, and receives the forward and backward program desc 
as attributes, and then executes the program by executor.

NOTE: This operator is added so that the inference model stored by 
`fluid.io.save_inference_model` under the static graph mode can be loaded 
under the dynamic graph mode for fine-tuning or inferencing.
      
)DOC");
  }
};

class RunProgramGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInputs("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of RunProgramGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInputs(framework::GradVarName("Out")), true,
        platform::errors::NotFound(
            "Input(Out@GRAD) of RunProgramGradOp should not be null."));
    // NOTE: The X@GRAD and Params@GRAD may not exist,
    // because they can be set stop_gradient = True
  }

 protected:
  /* see [Why use single type kernel] */
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return expected_kernel_type;
  }
};

template <typename T>
class RunProgramGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("run_program_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput("Params", this->Input("Params"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetInput("OutScope", this->Output("OutScope"));
    grad_op->SetInput("DOut", this->Output("DOut"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetOutput(framework::GradVarName("Params"),
                       this->InputGrad("Params"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(run_program, ops::RunProgramOp, ops::RunProgramOpMaker,
                  ops::RunProgramGradOpMaker<paddle::framework::OpDesc>,
                  ops::RunProgramGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(run_program_grad, ops::RunProgramGradOp);

/* see [Why use single type kernel] */
REGISTER_OP_CPU_KERNEL(
    run_program,
    ops::RunProgramOpKernel<paddle::platform::CPUDeviceContext, float>)
REGISTER_OP_CPU_KERNEL(
    run_program_grad,
    ops::RunProgramGradOpKernel<paddle::platform::CPUDeviceContext, float>)
