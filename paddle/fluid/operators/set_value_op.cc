//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/set_value_op.h"
#include <string>
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
struct CPUPlace;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class SetValue : public framework::OperatorWithKernel {
 public:
  SetValue(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "SetValue");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SetValue");
    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_LT(
        in_dims.size(), 7,
        platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.",
            in_dims.size()));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StepsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class SetValueMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // Input
    AddInput("Input", "(Tensor) Input tensor of set_value operator.");
    AddInput("ValueTensor", "(Tensor) Value tensor of set_value operator.")
        .AsDispensable();
    AddInput("StartsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must be [1]."
             "It has higher priority compare with attr(starts).")
        .AsDuplicable()
        .AsDispensable();
    AddInput("EndsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must BE [1]."
             "It has higher priority compare with attr(ends).")
        .AsDuplicable()
        .AsDispensable();

    AddInput("StepsTensorList",
             "(vector<Tensor<int32>>, optional) If provided, set_value will "
             "use this. The shape of the tensor in vector must BE [1]."
             "It has higher priority compare with attr(steps).")
        .AsDuplicable()
        .AsDispensable();

    // Output
    AddOutput("Out",
              "(Tensor) Output tensor of set_value operator. The output is the "
              "same Tensor as input");

    // Attr
    AddAttr<int>("dtype", "data type of input.")
        .InEnum(
            {framework::proto::VarType::BOOL, framework::proto::VarType::INT32,
             framework::proto::VarType::INT64, framework::proto::VarType::FP32,
             framework::proto::VarType::FP64})
        .SetDefault(framework::proto::VarType::FP32);
    AddAttr<std::vector<int64_t>>(
        "axes", "(list<int64_t>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int64_t>>(
        "starts",
        "(list<int64_t>) Starting indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>(
        "ends",
        "(list<int64_t>) Ending indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>(
        "steps", "(list<int64_t>) Stride step from the start to the end.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("decrease_axes",
                                  "(list<int>) The axes to decrease.")
        .SetDefault({});

    AddAttr<std::vector<int>>("bool_values", "Store the bool values.")
        .SetDefault({});
    AddAttr<std::vector<float>>("fp32_values", "Store the float32 values.")
        .SetDefault({});
    AddAttr<std::vector<int>>("int32_values", "Store the int32 values.")
        .SetDefault({});
    AddAttr<std::vector<int64_t>>("int64_values", "Store the int64 values.")
        .SetDefault({});
    AddAttr<std::vector<double>>("fp64_values", "Store the float64 values.")
        .SetDefault({});

    AddAttr<std::vector<int64_t>>("shape", "(vector<int64_t>) Shape of values.")
        .SetDefault({});
    AddComment(R"DOC(SetValue operator.
Assignment to a Tensor in static mode.
)DOC");
  }
};

template <typename T>
class SetValueGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    if (this->HasInput("ValueTensor")) {
      op->SetType("set_value_grad");
      op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
      if (this->HasInput("StartsTensorList")) {
        op->SetInput("StartsTensorList", this->Input("StartsTensorList"));
      }
      if (this->HasInput("EndsTensorList")) {
        op->SetInput("EndsTensorList", this->Input("EndsTensorList"));
      }

      // convert std::vector<int64_t > to std::vector<int >
      std::vector<int64_t> axes_int64 = static_cast<std::vector<int64_t>>(
          BOOST_GET_CONST(std::vector<int64_t>, this->GetAttr("axes")));
      std::vector<int64_t> starts_int64 = static_cast<std::vector<int64_t>>(
          BOOST_GET_CONST(std::vector<int64_t>, this->GetAttr("starts")));
      std::vector<int64_t> ends_int64 = static_cast<std::vector<int64_t>>(
          BOOST_GET_CONST(std::vector<int64_t>, this->GetAttr("ends")));
      std::vector<int64_t> decrease_axes_int64 =
          static_cast<std::vector<int64_t>>(BOOST_GET_CONST(
              std::vector<int64_t>, this->GetAttr("decrease_axes")));

      std::vector<int> axes(axes_int64.begin(), axes_int64.end());
      std::vector<int> starts(starts_int64.begin(), starts_int64.end());
      std::vector<int> ends(ends_int64.begin(), ends_int64.end());
      std::vector<int> decrease_axes(decrease_axes_int64.begin(),
                                     decrease_axes_int64.end());

      op->SetAttr("axes", axes);
      op->SetAttr("starts", starts);
      op->SetAttr("ends", ends);
      op->SetAttr("decrease_axis", decrease_axes);
      op->SetAttr("infer_flags", std::vector<int>({}));

      op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
      op->SetOutput(framework::GradVarName("ValueTensor"), this->InputGrad("ValueTensor"));
    } else {
      op->SetType("assign");
      op->SetInput("X", this->OutputGrad("Out"));
      op->SetOutput("Out", this->InputGrad("Input"));
    }
  }
};

class SetValueGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input", framework::GradVarName("Out"), "set_value_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("ValueTensor")), "Output", framework::GradVarName("ValueTensor"), "set_value_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Input")), "Output", framework::GradVarName("Input"), "set_value_grad");

    // Case 1: Special treatment when input is a tensor array.
    auto x_var_type = ctx->GetInputsVarType(framework::GradVarName("Out"))[0];
    auto axes = ctx->Attrs().Get<std::vector<int>>("axes");
    if (x_var_type == framework::proto::VarType::LOD_TENSOR_ARRAY) {
      PADDLE_ENFORCE_EQ(axes.size(), 1,
                        platform::errors::InvalidArgument(
                            "The size of axes must be 1 when the Input of "
                            "SliceOp is LoDTensorArray, "
                            "but received %d.",
                            axes.size()));
      if (ctx->IsRuntime()) {
        // If the var type of input is LOD_TENSOR_ARRAY,
        // the output shape is determined by SliceKernel:Compute in runtime.
        return;
      } else {
        // NOTE(liym27): A better way is needed to get accurate dims of tensor
        // array.
        // The resulted dim of GetInputDim("Input") is the dim of the
        // last item written into TensorArray "Input". Maybe it's a bug to fix.
        // ctx->SetOutputDim("Out", ctx->GetInputDim("Input"));
        return;
      }
    }

    // Case 2: input is a tensor.
    auto in_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    // auto in_dims_forward = ctx->GetInputDim("ForwardInput");
    // ctx->SetOutputDim("InputGrad", in_dims_forward);
 
    PADDLE_ENFORCE_LT(in_dims.size(), 7,
                      platform::errors::InvalidArgument(
                          "The rank of input should be less than 7."));
    framework::DDim out_dims(in_dims);

    auto starts = ctx->Attrs().Get<std::vector<int>>("starts");
    auto ends = ctx->Attrs().Get<std::vector<int>>("ends");
    auto decrease_axis = ctx->Attrs().Get<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx->Attrs().Get<std::vector<int>>("infer_flags");
    if (infer_flags.empty()) {
      // Initialize infer_flags with 1.
      // To be compatible with other op tests in which infer_flags is not set.
      infer_flags = std::vector<int>(axes.size(), 1);
    }

    // 2.1 Check attrs.
    auto starts_size = starts.size();
    auto ends_size = ends.size();

    if (ctx->HasInputs("StartsTensorList")) {
      starts_size = ctx->Inputs("StartsTensorList").size();
      PADDLE_ENFORCE_GT(starts_size, 0,
                        platform::errors::InvalidArgument(
                            "StartsTensorList size can't be zero"));
    }
    if (ctx->HasInputs("EndsTensorList")) {
      ends_size = ctx->Inputs("EndsTensorList").size();
      PADDLE_ENFORCE_GT(ends_size, 0, platform::errors::InvalidArgument(
                                          "EndsTensorList size can't be zero"));
    }

    if (!ctx->HasInput("StartsTensor")) {
      PADDLE_ENFORCE_EQ(
          starts_size, axes.size(),
          platform::errors::InvalidArgument(
              "The size of starts must be equal to the size of axes."));
    }
    if (!ctx->HasInput("EndsTensor")) {
      PADDLE_ENFORCE_EQ(
          ends_size, axes.size(),
          platform::errors::InvalidArgument(
              "The size of ends must be equal to the size of axes."));
    }

    CheckAndUpdateSliceAttrs<int>(in_dims, axes, &starts, &ends, nullptr,
                                  &infer_flags);

    auto slice_dims =
        GetSliceDims<int>(in_dims, axes, starts, ends, nullptr, &infer_flags);
    if (ctx->IsRuntime()) {
      out_dims = GetDecreasedDims<int>(slice_dims, decrease_axis, &infer_flags);
    } else {
      out_dims = GetDecreasedDims<int>(slice_dims, decrease_axis, nullptr);
    }
    
    ctx->SetOutputDim(framework::GradVarName("ValueTensor"), out_dims);
    if (axes[0] != 0) {
      ctx->ShareLoD(framework::GradVarName("Out"), /*->*/ framework::GradVarName("ValueTensor"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar(framework::GradVarName("Out"));
    if (in_var->IsType<framework::LoDTensor>()) {
      auto &in_tensor = in_var->Get<framework::LoDTensor>();
      PADDLE_ENFORCE_EQ(
          in_tensor.IsInitialized(), true,
          platform::errors::InvalidArgument(
              "The tensor Input (Input) of Slice op is not initialized."));
      // NOTE: cuda pinned tensor need to copy its data to target place
      if (platform::is_cuda_pinned_place(in_tensor.place())) {
        return framework::OpKernelType(in_tensor.type(), ctx.device_context());
      }
      return framework::OpKernelType(in_tensor.type(), in_tensor.place());
    }
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, framework::GradVarName("Out")), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor") {
      return expected_kernel_type;
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

DECLARE_INPLACE_OP_INFERER(SetValueOpInplaceInferer, {"Input", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(set_value, ops::SetValue, ops::SetValueMaker,
                  ops::SetValueGradMaker<paddle::framework::OpDesc>,
                  ops::SetValueGradMaker<paddle::imperative::OpBase>,
                  ops::SetValueOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    set_value, ops::SetValueKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SetValueKernel<plat::CPUDeviceContext, int64_t>,
    ops::SetValueKernel<plat::CPUDeviceContext, float>,
    ops::SetValueKernel<plat::CPUDeviceContext, double>,
    ops::SetValueKernel<plat::CPUDeviceContext, bool>);


REGISTER_OPERATOR(set_value_grad, ops::SetValueGrad);
REGISTER_OP_CPU_KERNEL(
    set_value_grad, ops::SetValueGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SetValueGradKernel<plat::CPUDeviceContext, int64_t>,
    ops::SetValueGradKernel<plat::CPUDeviceContext, float>,
    ops::SetValueGradKernel<plat::CPUDeviceContext, double>);


REGISTER_OP_CUDA_KERNEL(
    set_value_grad, ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, int>, 
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SetValueGradKernel<paddle::platform::CUDADeviceContext, double>);


REGISTER_OP_VERSION(set_value)
    .AddCheckpoint(
        R"ROC(
Upgrade set_value, add 3 inputs [StartsTensorList, EndsTensorList, StepsTensorList] and 1 attribute [steps].
              )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput("StartsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(starts).")
            .NewInput("EndsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(ends).")
            .NewInput("StepsTensorList",
                      "If provided, set_value will use this.The shape of the "
                      "tensor in vector must be [1]. It has higher priority "
                      "compare with attr(steps).")
            .ModifyAttr("starts",
                        "Starting indices of corresponding axis in `axes`.",
                        std::vector<int64_t>{})
            .ModifyAttr("ends",
                        "Ending indices of corresponding axis in `axes`.",
                        std::vector<int64_t>{})
            .NewAttr("steps", "Stride step from the start to the end.",
                     std::vector<int64_t>{}))
    .AddCheckpoint(
        R"ROC(
Upgrade set_value, add 1 attribute [decrease_axes].
              )ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "decrease_axes", "The axes to decrease.", std::vector<int64_t>{}));
