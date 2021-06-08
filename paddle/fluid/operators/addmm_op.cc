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

#include "paddle/fluid/operators/addmm_op.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class AddMMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      platform::errors::NotFound(
                          "Input(Input) of AddMMOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of AddMMOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) of AddMMOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of AddMMOp should not be null."));

    auto input_dims = ctx->GetInputDim("Input");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto ndim_input = input_dims.size();
    auto ndim_x = x_dims.size();
    auto ndim_y = y_dims.size();

    float alpha = ctx->Attrs().Get<float>("Alpha");
    float beta = ctx->Attrs().Get<float>("Beta");

    VLOG(3) << "addmm operator input.shape=" << input_dims
            << " x.shape=" << x_dims << " y.shape=" << y_dims
            << " beta=" << beta << " alpha=" << alpha
            << " ndim_input=" << ndim_input << " ndim_x=" << ndim_x
            << " ndim_y=" << ndim_y;

    PADDLE_ENFORCE_NE(framework::product(input_dims), 0,
                      platform::errors::PreconditionNotMet(
                          "The Input variable Input(%s) has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function.",
                          ctx->Inputs("Input").front()));

    PADDLE_ENFORCE_NE(framework::product(x_dims), 0,
                      platform::errors::PreconditionNotMet(
                          "The Input variable X(%s) has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function.",
                          ctx->Inputs("X").front()));

    PADDLE_ENFORCE_NE(framework::product(y_dims), 0,
                      platform::errors::PreconditionNotMet(
                          "The Input variable Y(%s) has not "
                          "been initialized. You may need to confirm "
                          "if you put exe.run(startup_program) "
                          "after optimizer.minimize function.",
                          ctx->Inputs("Y").front()));
    // dim check
    PADDLE_ENFORCE_EQ(ndim_input, 2,
                      platform::errors::InvalidArgument(
                          "The input tensor input's dimension must be 2. "
                          "But received input's dimension = [%s].",
                          ndim_input));
    PADDLE_ENFORCE_EQ(ndim_x, 2,
                      platform::errors::InvalidArgument(
                          "The input tensor x's dimension must be 2. "
                          "But received x's dimension = [%s].",
                          ndim_x));
    PADDLE_ENFORCE_EQ(ndim_y, 2,
                      platform::errors::InvalidArgument(
                          "The input tensor y's dimension must be 2. "
                          "But received y's dimension = [%s].",
                          ndim_y));

    std::vector<int64_t> output_dims;
    output_dims.push_back(x_dims[0]);
    output_dims.push_back(y_dims[1]);

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;

      if (input_data_type == framework::DataTypeTrait<int8_t>::DataType() ||
          input_data_type == framework::DataTypeTrait<uint8_t>::DataType()) {
        customized_type_value = kMULMKLDNNINT8;
      }
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class AddMMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor), tensor to be added to the final result.");
    AddInput("X", "(Tensor), The first input tensor for mul.");
    AddInput("Y", "(Tensor), The second input tensor for mul.");
    AddOutput("Out", "(Tensor), The output tensor of addmm op.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<float>("Alpha", "coefficient of x*y.").SetDefault(1.0f);
    AddAttr<float>("Beta", "coefficient of input.").SetDefault(1.0f);
    AddComment(R"DOC(
AddMM Operator.
This operator is used to perform matrix multiplication for input $x$ and $y$ with coefficient $alpha$.
$input$ with coefficient $beta$ is added to the final result. 
The equation is:

$$Out = alpha * x * y + beta * input$$

$x$ and $y$ must be two-dimensional, and $input$ can be broadcastable.
)DOC");
  }
};

class AddMMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::NotFound("Input(Input) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Y"), true,
        platform::errors::NotFound("Input(Y) should not be null"));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")), true,
        platform::errors::NotFound("Input(Out@GRAD) should not be null"));
    const auto& input_dims = ctx->GetInputDim("Input");
    const auto& x_dims = ctx->GetInputDim("X");
    const auto& y_dims = ctx->GetInputDim("Y");

    auto input_grad_name = framework::GradVarName("Input");
    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(input_grad_name)) {
      ctx->SetOutputDim(input_grad_name, input_dims);
    }
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class AddMMOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("addmm_grad");
    retv->SetInput("Input", this->Input("Input"));
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Y", this->Input("Y"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(addmm, ops::AddMMOp, ops::AddMMOpMaker,
                  ops::AddMMOpGradMaker<paddle::framework::OpDesc>,
                  ops::AddMMOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(addmm_grad, ops::AddMMGradOp);

REGISTER_OP_CPU_KERNEL(
    addmm, ops::AddMMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddMMKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    addmm_grad, ops::AddMMGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AddMMGradKernel<paddle::platform::CPUDeviceContext, double>);
