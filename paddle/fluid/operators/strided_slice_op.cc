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

#include "paddle/fluid/operators/strided_slice_op.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class StridedSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "StridedSlice");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "StridedSlice");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_LT(
        in_dims.size(), 7,
        platform::errors::InvalidArgument(
            "The dimension of StridedSlice operator's input should be less "
            "than 7, but received dimension is %d.",
            in_dims.size()));

    auto starts_int = ctx->Attrs().Get<std::vector<int>>("starts");
    auto ends_int = ctx->Attrs().Get<std::vector<int>>("ends");
    auto strides_int = ctx->Attrs().Get<std::vector<int>>("strides");

    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());
    std::vector<int64_t> strides(strides_int.begin(), strides_int.end());

    auto axes = ctx->Attrs().Get<std::vector<int>>("axes");
    auto infer_flags = ctx->Attrs().Get<std::vector<int>>("infer_flags");
    auto decrease_axis = ctx->Attrs().Get<std::vector<int>>("decrease_axis");

    auto starts_size = starts.size();
    auto ends_size = ends.size();
    auto strides_size = strides.size();

    if (ctx->HasInputs("StartsTensorList")) {
      auto StartsTensorList = ctx->Inputs("StartsTensorList");
      PADDLE_ENFORCE_GT(
          StartsTensorList.size(), 0,
          platform::errors::InvalidArgument(
              "StridedSlice operator's StartsTensorList is empty."));
      starts_size = StartsTensorList.size();
    }
    if (ctx->HasInputs("EndsTensorList")) {
      auto EndsTensorList = ctx->Inputs("EndsTensorList");
      PADDLE_ENFORCE_GT(
          EndsTensorList.size(), 0,
          platform::errors::InvalidArgument(
              "StridedSlice operator's EndsTensorList is empty."));
      ends_size = EndsTensorList.size();
    }
    if (ctx->HasInputs("StridesTensorList")) {
      auto StridesTensorList = ctx->Inputs("StridesTensorList");
      PADDLE_ENFORCE_GT(
          StridesTensorList.size(), 0,
          platform::errors::InvalidArgument(
              "StridedSlice operator's StridesTensorList is empty."));
      strides_size = StridesTensorList.size();
    }

    auto tensor_input = false;
    if (ctx->HasInput("EndsTensor") || ctx->HasInput("StartsTensor") ||
        ctx->HasInput("StridesTensor")) {
      tensor_input = true;
    }
    if (!ctx->HasInput("EndsTensor")) {
      PADDLE_ENFORCE_EQ(
          ends_size, axes.size(),
          platform::errors::InvalidArgument(
              "The size of ends attribute in StridedSlice operator is not "
              "equal to the size of axes attribute. The ends attribute's size "
              "is %d, axes attribute's size is %d.",
              ends_size, axes.size()));
    }
    if (!ctx->HasInput("StartsTensor")) {
      PADDLE_ENFORCE_EQ(
          starts_size, axes.size(),
          platform::errors::InvalidArgument(
              "The size of starts attribute in StridedSlice operator is not "
              "equal to the size of axes attribute. The starts attribute's "
              "size is %d, axes attribute's size is %d.",
              starts_size, axes.size()));
    }
    if (!ctx->HasInput("StridesTensor")) {
      PADDLE_ENFORCE_EQ(
          strides_size, axes.size(),
          platform::errors::InvalidArgument(
              "The size of strides attribute in StridedSlice operator is not "
              "equal to the size of axes attribute. The strides attribute's "
              "size is %d, axes attribute's size is %d.",
              strides_size, axes.size()));
    }
    // we need to analysis strided slice op is valid for
    // the parameter that we get from python front
    std::vector<int64_t> out_dims_vector(in_dims.size(), -1);
    if (!tensor_input) {
      StridedSliceOutDims(starts, ends, strides, axes, infer_flags, in_dims,
                          decrease_axis, out_dims_vector.data(), axes.size(),
                          true);
    }
    framework::DDim out_dims(framework::make_ddim(out_dims_vector));
    // generate new shape
    if (decrease_axis.size() > 0) {
      std::vector<int64_t> new_out_shape;
      for (size_t i = 0; i < decrease_axis.size(); ++i) {
        if (ctx->IsRuntime() && infer_flags[i] != -1) {
          PADDLE_ENFORCE_EQ(out_dims[decrease_axis[i]], 1,
                            platform::errors::InvalidArgument(
                                "the size of decrease dimension should be 1, "
                                "but received %d.",
                                out_dims[decrease_axis[i]]));
        }
        out_dims[decrease_axis[i]] = 0;
      }

      for (int i = 0; i < out_dims.size(); ++i) {
        if (out_dims[i] != 0) {
          new_out_shape.push_back(out_dims[i]);
        }
      }
      if (new_out_shape.size() == 0) {
        new_out_shape.push_back(1);
      }

      out_dims = framework::make_ddim(new_out_shape);
    }
    ctx->SetOutputDim("Out", out_dims);
    ctx->ShareLoD("Input", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // NOTE: cuda pinned tensor need to copy its data to target place
    auto in_tensor = ctx.Input<Tensor>("Input");
    if (platform::is_cuda_pinned_place(in_tensor->place())) {
      return framework::OpKernelType(in_tensor->type(), ctx.device_context());
    }
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        in_tensor->place());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor" ||
        var_name == "StridesTensor") {
      return expected_kernel_type;
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StridesTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class StridedSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "Tensor of data to extract slices from.");
    AddOutput("Out", "Strided Sliced data tensor.");

    AddInput("StartsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of StartsTensor, StartsTensorList "
             "and attr(starts).")
        .AsDispensable();
    AddInput("EndsTensor",
             "(Tensor<int32>, optional) If provided, slice will use this."
             "It has the highest priority of EndsTensor, EndsTensorList and "
             "attr(ends).")
        .AsDispensable();
    AddInput(
        "StridesTensor",
        "(Tensor<int32>, optional) If provided, slice will use this."
        "It has the highest priority of StridesTensor, StridesTensorList and "
        "attr(ends).")
        .AsDispensable();
    AddInput(
        "StartsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(starts).")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "EndsTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(ends).")
        .AsDuplicable()
        .AsDispensable();
    AddInput(
        "StridesTensorList",
        "(vector<Tensor<int32>>, optional) If provided, slice will use this."
        "The shape of the tensor in vector MUST BE [1]."
        "It has higher priority compare with attr(strides).")
        .AsDuplicable()
        .AsDispensable();
    AddAttr<std::vector<int>>(
        "axes", "(list<int>) Axes that `starts` and `ends` apply to.");
    AddAttr<std::vector<int>>(
        "starts", "(list<int>) Start indices for the strided slice start.")
        .SetDefault({});
    AddAttr<std::vector<int>>("ends",
                              "(list<int>) End indices the tensor slice end")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "strides", "(list<int> Stride step from the start to the end)")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "infer_flags", "(list<int>) Flags of inferring dims in attributes.")
        .SetDefault({});
    AddAttr<std::vector<int>>("decrease_axis", "(list<int>) decrease_axis")
        .SetDefault({});
    AddComment(R"DOC(
Strided Slice Operator.
Instead of calling this op directly most users will want to use the
NumPy-style slicing syntax.
For Example:
data = fluid.layers.fill_constant(shape=[3, 3], value=0, dtype='int64')
y = fluid.layers.strided_slice(data, [0, 1], [1,0], [2, 3], [1, 1])
)DOC");
  }
};

class StridedSliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                   "StridedSliceGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "StridedSliceGrad");

    auto x_dims = ctx->GetInputDim("Input");
    auto x_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "StartsTensor" || var_name == "EndsTensor" ||
        var_name == "StridesTensor") {
      return expected_kernel_type;
    }
    if (var_name == "StartsTensorList" || var_name == "EndsTensorList" ||
        var_name == "StridesTensorList") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class StridedSliceOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetInput("Input", this->Input("Input"));
    bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    bind->SetInput("StridesTensor", this->Input("StridesTensor"));
    bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    bind->SetInput("StridesTensorList", this->Input("StridesTensorList"));
    bind->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("strided_slice_grad");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(StridedSliceOpGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(strided_slice, ops::StridedSliceOp, ops::StridedSliceOpMaker,
                  ops::StridedSliceOpGradMaker<paddle::framework::OpDesc>,
                  ops::StridedSliceOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(strided_slice_grad, ops::StridedSliceOpGrad,
                  ops::StridedSliceOpGradNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    strided_slice,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext, double>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext,
                            paddle::platform::complex<float>>,
    ops::StridedSliceKernel<paddle::platform::CPUDeviceContext,
                            paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    strided_slice_grad,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, bool>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext,
                                paddle::platform::complex<float>>,
    ops::StridedSliceGradKernel<paddle::platform::CPUDeviceContext,
                                paddle::platform::complex<double>>);
