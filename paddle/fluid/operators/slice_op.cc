/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/slice_op.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class SliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "slice");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "slice");

    // Case 1: Special treatment when input is a tensor array.
    auto x_var_type = ctx->GetInputsVarType("Input")[0];
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
        ctx->SetOutputDim("Out", ctx->GetInputDim("Input"));
        return;
      }
    }

    // Case 2: input is a tensor.
    auto in_dims = ctx->GetInputDim("Input");
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

    ctx->SetOutputDim("Out", out_dims);
    if (axes[0] != 0) {
      ctx->ShareLoD("Input", /*->*/ "Out");
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto *in_var = ctx.InputVar("Input");
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

#ifdef PADDLE_WITH_MKLDNN
      auto input_data_type =
          framework::OperatorWithKernel::IndicateVarDataType(ctx, "Input");

      if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
        // OneDNN uses blocking format, which cannot be always
        // supported with reorders, because if blocked dimension is not
        // divisible
        // by
        // 8 or 16(depending on which blocking format is used) submemory cannot
        // be
        // created, so in that scenario a fallback is needed
        auto tmp_md = dnnl::memory::desc(
            framework::vectorize(ctx.Input<Tensor>("Input")->dims()),
            dnnl::memory::data_type::f32, ctx.Input<Tensor>("Input")->format());
        if (tmp_md.data.format_desc.blocking.inner_nblks == 0)
          return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                         framework::DataLayout::kMKLDNN,
                                         framework::LibraryType::kMKLDNN);
      }
#endif

      return framework::OpKernelType(in_tensor.type(), in_tensor.place());
    }
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace());
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

class SliceOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x_name = "Input";
    auto out_name = "Out";
    auto decrease_axis = ctx->GetAttr("decrease_axis");
    auto not_decrease = boost::get<std::vector<int>>(decrease_axis).size() == 0;
    if (not_decrease) {
      // The default type of out is LoDTensor.
      // However, if no axis is decreased and the type of input is not
      // LoDTensor, the type of out should be the same as input.
      // For example, input is a LoDTensorArray and no axis is decreased, the
      // output should be a LoDTensorArray.
      ctx->SetOutputType(out_name, ctx->GetInputType(x_name));
      ctx->SetOutputDataType(out_name, ctx->GetInputDataType(x_name));
    }
  }
};

class SliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input", "(Tensor) Tensor of data to extract slices from.");
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
    AddOutput("Out", "Sliced data tensor.");
    AddAttr<std::vector<int>>(
        "axes",
        "(list<int>) Axes that `starts` and `ends` apply to. It's optional."
        "If not present, will be treated as [0, 1, ..., len(`starts`) - 1].");
    AddAttr<std::vector<int>>(
        "starts",
        "(list<int>) Starting indices of corresponding axis in `axes`")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "ends", "(list<int>) Ending indices of corresponding axis in `axes`.")
        .SetDefault({});
    AddAttr<std::vector<int>>(
        "infer_flags", "(list<int>) Flags of inferring dims in attributes.")
        .SetDefault({});
    AddAttr<std::vector<int>>("decrease_axis", "(list<int>) decrease_axis")
        .SetDefault({});
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "bfloat16"});
    AddComment(R"DOC(
Slice Operator.

Produces a slice of the input tensor along multiple axes. Similar to numpy:
https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
Slice uses `axes`, `starts` and `ends` attributes to specify the start and
end dimension for each axis in the list of axes, it uses this information
to slice the input data tensor. If a negative value is passed for any of
the start or end indices, it represents number of elements before the end
of that dimension. If the value passed to start or end is larger than
the n (the number of elements in this dimension), it represents n.
For slicing to the end of a dimension with unknown size, it is recommended
to pass in INT_MAX. The size of axes must be equal to starts\' and ends\'.
Following examples will explain how slice works:

.. code-block:: text

    Case1:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            axes = [0, 1]
            starts = [1, 0]
            ends = [2, 3]
        Then:
            result = [ [5, 6, 7], ]

    Case2:
        Given:
            data = [ [1, 2, 3, 4], [5, 6, 7, 8], ]
            starts = [0, 1]
            ends = [-1, 1000]
        Then:
            result = [ [2, 3, 4], ]
)DOC");
  }
};

class SliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Input"), true,
        platform::errors::InvalidArgument("Input should not be null"));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should not be null"));
    auto x_var_type = ctx->GetInputsVarType("Input")[0];
    if (x_var_type == framework::proto::VarType::LOD_TENSOR_ARRAY) {
      // If the var type of input is LOD_TENSOR_ARRAY,
      // the output shape is determined by SliceGradKernel:Compute in runtime.
      if (ctx->IsRuntime()) {
        return;
      }
    }
    auto x_dims = ctx->GetInputDim("Input");
    auto x_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = framework::OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));

#ifdef PADDLE_WITH_MKLDNN
    if (this->CanMKLDNNBeUsed(ctx, input_data_type)) {
      // OneDNN uses blocking format, which cannot be always
      // supported with reorders, because if blocked dimension is not divisible
      // by
      // 8 or 16(depending on which blocking format is used) submemory cannot be
      // created, so in that scenario a fallback is needed
      auto tmp_md = dnnl::memory::desc(
          framework::vectorize(
              ctx.Input<Tensor>(framework::GradVarName("Out"))->dims()),
          dnnl::memory::data_type::f32,
          ctx.Input<Tensor>(framework::GradVarName("Out"))->format());
      if (tmp_md.data.format_desc.blocking.inner_nblks == 0)
        return framework::OpKernelType(input_data_type, ctx.GetPlace(),
                                       framework::DataLayout::kMKLDNN,
                                       framework::LibraryType::kMKLDNN);
    }
#endif
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
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

class SliceOpGradVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto x = "Input";
    auto d_out = framework::GradVarName("Out");
    auto out = framework::GradVarName("Input");
    // The types of grad_input and input should always be the same.
    // The default type of out is LoDTensor, but the type of input can be
    // LoDTensor or LoDTensorArray,
    // so set the type of both to be the same.
    ctx->SetOutputType(out, ctx->GetInputType(x));
    ctx->SetOutputDataType(out, ctx->GetInputDataType(d_out));
  }
};

template <typename T>
class SliceOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    bind->SetInput("Input", this->Input("Input"));
    if (this->HasInput("StartsTensor")) {
      bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    }
    if (this->HasInput("EndsTensor")) {
      bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    }
    if (this->HasInput("StartsTensorList")) {
      bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    }
    if (this->HasInput("EndsTensorList")) {
      bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    }
    bind->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("slice_grad");
  }
};

template <typename T>
class SliceDoubleOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> bind) const override {
    if (this->HasInput("StartsTensor")) {
      bind->SetInput("StartsTensor", this->Input("StartsTensor"));
    }
    if (this->HasInput("EndsTensor")) {
      bind->SetInput("EndsTensor", this->Input("EndsTensor"));
    }
    if (this->HasInput("StartsTensorList")) {
      bind->SetInput("StartsTensorList", this->Input("StartsTensorList"));
    }
    if (this->HasInput("EndsTensorList")) {
      bind->SetInput("EndsTensorList", this->Input("EndsTensorList"));
    }
    bind->SetInput("Input", this->OutputGrad(framework::GradVarName("Input")));
    bind->SetOutput("Out", this->InputGrad(framework::GradVarName("Out")));
    bind->SetAttrMap(this->Attrs());
    bind->SetType("slice");
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SliceOpGradNoNeedBufferVarsInferer,
                                    "Input");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(slice, ops::SliceOp, ops::SliceOpMaker,
                  ops::SliceOpGradMaker<paddle::framework::OpDesc>,
                  ops::SliceOpGradMaker<paddle::imperative::OpBase>,
                  ops::SliceOpVarTypeInference);
REGISTER_OPERATOR(slice_grad, ops::SliceOpGrad,
                  ops::SliceDoubleOpGradMaker<paddle::framework::OpDesc>,
                  ops::SliceDoubleOpGradMaker<paddle::imperative::OpBase>,
                  ops::SliceOpGradNoNeedBufferVarsInferer,
                  ops::SliceOpGradVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<float>>,
    ops::SliceKernel<paddle::platform::CPUDeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OP_CPU_KERNEL(
    slice_grad, ops::SliceGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<float>>,
    ops::SliceGradKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::float16>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::complex<float>>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext,
                     paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    slice_grad,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::float16>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<float>>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex<double>>);
