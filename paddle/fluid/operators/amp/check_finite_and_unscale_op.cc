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

#include "paddle/fluid/operators/amp/check_finite_and_unscale_op.h"

namespace paddle {
namespace operators {

class CheckFiniteAndUnscaleOp : public framework::OperatorWithKernel {
 public:
  CheckFiniteAndUnscaleOp(const std::string& type,
                          const framework::VariableNameMap& inputs,
                          const framework::VariableNameMap& outputs,
                          const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X",
                   "check_finite_and_unscale");
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out",
                   "check_finite_and_unscale");
    PADDLE_ENFORCE_EQ(
        ctx->Inputs("X").size(), ctx->Outputs("Out").size(),
        platform::errors::InvalidArgument(
            "The input(X) and output(Out) should have same size in "
            "Operator(check_finite_and_unscale), size of input(X) is %d "
            "and size of output(Out) is %d.",
            ctx->Inputs("X").size(), ctx->Outputs("Out").size()));
    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
    ctx->SetOutputDim("FoundInfinite", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class CheckFiniteAndUnscaleOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensors) The input tensors of check_finite_and_unscale operator.")
        .AsDuplicable();
    AddInput("Scale",
             "(Tensor) 1-dim tensor, the scale of check_finite_and_unscale "
             "operator.");
#ifdef PADDLE_WITH_ASCEND_CL
    AddInput("FloatStatus",
             "(Tensor) 1-dim tensor of shape [8], allocated by "
             "alloc_float_status op")
        .AsDispensable();
#endif
    AddOutput("Out",
              "(Tensors) The scaled output tensor of "
              "check_finite_and_unscale operator.")
        .AsDuplicable();
    AddOutput("FoundInfinite",
              "(Tensor) 1-dim tensor, contains a bool scalar, which indicates "
              "if there there is infinite or nan item in input X.");
    AddComment(R"DOC(
check_finite_and_unscale operator.
Check if input X contains all finite data, if yes, scale it by input Scale.

$$Out = X / scale$$

If any tensor in X contains Inf or Nan, the Out will generate a indicator.
FoundInfinite will be 1 (True), and Out will not be scaled. In this case, the data of 
Out should not be used, and its data may not be deterministic. 
Otherwise, FoundInfinite will be 0 (False).

)DOC");
  }
};

template <typename T>
class CheckFiniteAndUnscaleCpuKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    const auto xs = ctx.MultiInput<framework::Tensor>("X");
    const auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto outs = ctx.MultiOutput<framework::Tensor>("Out");
    auto* found_inf = ctx.Output<framework::Tensor>("FoundInfinite");

    const T* scale_data = scale->data<T>();
    bool* found_inf_data = found_inf->mutable_data<bool>(dev_ctx.GetPlace());

    *found_inf_data = false;
    framework::Tensor is_finite =
        ctx.AllocateTmpTensor<bool, platform::CPUDeviceContext>({1}, dev_ctx);
    bool* is_finite_data = is_finite.template data<bool>();

    auto& dev = *ctx.template device_context<platform::CPUDeviceContext>()
                     .eigen_device();

    T inverse_scale = Inverse<T>(*scale_data);
    for (size_t i = 0; i < xs.size(); ++i) {
      const auto* x = xs[i];
      auto* out = outs[i];
      out->mutable_data<T>(dev_ctx.GetPlace());
      if (!(*found_inf_data)) {
        framework::TensorIsfinite(*x, &is_finite);
        *found_inf_data = !(*is_finite_data);
      }
      auto eigen_out = framework::EigenVector<T>::Flatten(*out);
      auto eigen_in = framework::EigenVector<T>::Flatten(*x);
      if (!(*found_inf_data)) {
        eigen_out.device(dev) = eigen_in * inverse_scale;
      } else {
        eigen_out.device(dev) = eigen_in * static_cast<T>(0);
      }
    }
    return;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    check_finite_and_unscale, ops::CheckFiniteAndUnscaleOp,
    ops::CheckFiniteAndUnscaleOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(check_finite_and_unscale,
                       ops::CheckFiniteAndUnscaleCpuKernel<float>,
                       ops::CheckFiniteAndUnscaleCpuKernel<double>);
