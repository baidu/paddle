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

#include "paddle/fluid/operators/transpose_op.h"
#include <memory>
#include <string>
#include <vector>

#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

class TransposeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Transpose");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Transpose");
    auto x_dims = ctx->GetInputDim("X");
    std::vector<int> axis = ctx->Attrs().Get<std::vector<int>>("axis");
    size_t x_rank = x_dims.size();
    size_t axis_size = axis.size();

    PADDLE_ENFORCE_EQ(x_rank, axis_size,
                      platform::errors::InvalidArgument(
                          "The input tensor's dimension "
                          "should be equal to the axis's size. "
                          "But received input tensor's dimension is %d, "
                          "axis's size is %d",
                          x_rank, axis_size));

    std::vector<int> count(axis_size, 0);
    for (size_t i = 0; i < axis_size; i++) {
      PADDLE_ENFORCE_EQ(
          axis[i] < static_cast<int>(axis_size) && ++count[axis[i]] == 1, true,
          platform::errors::InvalidArgument(
              "Each element of Attribute axis should "
              "be a unique value range from 0 to (dims - 1), "
              "where the dims is the axis's size, "
              "unique value means this axis value can appear only once. "
              "But received axis[%d] is %d, axis_size is %d, "
              "count[axis[%d]] is %d",
              i, axis[i], axis_size, i, count[axis[i]]));
    }

    framework::DDim out_dims(x_dims);
#ifdef PADDLE_WITH_MKLDNN
    // Here we need to match dims to paddle layout
    // as we are producing non-oneDNN result
    if ((x_dims.size() >= 3) &&
        (paddle::platform::MKLDNNDeviceContext::tls()
             .get_cur_paddle_data_layout() == framework::DataLayout::kNHWC)) {
      auto dims = framework::vectorize<int>(x_dims);
      std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
      x_dims = x_dims.reshape(dims);
      VLOG(3)
          << "Rotating Shape in Transpose from: kMKLDNN to: kNHWC output_shape";
    }
#endif
    for (size_t i = 0; i < axis_size; i++) {
      out_dims[i] = x_dims[axis[i]];
    }
    ctx->SetOutputDim("Out", out_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    std::string data_format = ctx.Attr<std::string>("data_format");
    framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
      layout_ = framework::DataLayout::kMKLDNN;
    }
#endif
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout_, library_);
  }
};

class TransposeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor, tensors with rank up to 6 are supported.");
    AddOutput("Out", "(Tensor)The output tensor.");
    AddAttr<std::vector<int>>(
        "axis",
        "(vector<int>) A list of values, and the size of the list should be "
        "the same with the input tensor rank. This operator permutes the input "
        "tensor's axes according to the values given.");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "An optional string from: \"NHWC\", \"NCHW\". "
        "Defaults to \"NHWC\". Specify the data format of the output data, "
        "the input will be transformed automatically. ")
        .SetDefault("AnyLayout");
    AddAttr<bool>(
        "use_quantizer",
        "(bool, default false) "
        "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
        .SetDefault(false);
    AddAttr<std::string>(
        "mkldnn_data_type",
        "(string, default \"float32\"). Data type of mkldnn kernel")
        .SetDefault("float32")
        .InEnum({"float32", "int8", "bfloat16"});
    /* int8 parameters */
    AddComment(R"DOC(
Transpose Operator.

The input tensor will be permuted according to the axes given.
The behavior of this operator is similar to how `numpy.transpose` works.

- suppose the input `X` is a 2-D tensor:
    $$
    X = \begin{pmatrix}
    0 &1 &2 \\
    3 &4 &5
    \end{pmatrix}$$

    the given `axes` is: $[1, 0]$, and $Y$ = transpose($X$, axis)

    then the output $Y$ is:

    $$
    Y = \begin{pmatrix}
         0 &3 \\
         1 &4  \\
         2 &5
    \end{pmatrix}$$

- Given a input tensor with shape $(N, C, H, W)$ and the `axes` is
$[0, 2, 3, 1]$, then shape of the output tensor will be: $(N, H, W, C)$.

)DOC");
  }
};

class TransposeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "TransposeOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "TransposeOpGrad");
    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    std::string data_format = ctx.Attr<std::string>("data_format");
    framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
      layout_ = framework::DataLayout::kMKLDNN;
    }
#endif
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace(), layout_, library_);
  }
};

// FIXME(zcd): transpose2 adds an intermediate output(XShape) based on
// transpose, the XShape is used to carry the shape and lod of X which
// will be used in transpose_grad, in this way, the framework can reuse
// the memory of X immediately the transpose2_op is finished.
// Considering compatibility issues, we could not fix transpose2_op
class Transpose2Op : public TransposeOp {
 public:
  Transpose2Op(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : TransposeOp(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    TransposeOp::InferShape(ctx);
    OP_INOUT_CHECK(ctx->HasOutput("XShape"), "Output", "XShape", "Transpose2");
    const auto &in_dims = ctx->GetInputDim("X");
    std::vector<int64_t> x_shape_dim(in_dims.size() + 1);
    x_shape_dim[0] = 0;
    for (int i = 0; i < in_dims.size(); ++i) {
      x_shape_dim[i + 1] = in_dims[i];
    }
    ctx->SetOutputDim("XShape", framework::make_ddim(x_shape_dim));
    ctx->ShareLoD("X", /*->*/ "XShape");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    std::string data_format = ctx.Attr<std::string>("data_format");
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
      layout_ = framework::DataLayout::kMKLDNN;
      using framework::proto::VarType;
      auto input_data_type = ctx.Input<Tensor>("X")->type();
      customized_type_value = (input_data_type == VarType::INT8 ||
                               input_data_type == VarType::UINT8)
                                  ? kTransposeMKLDNNINT8
                                  : kTransposeMKLDNNFP32;
    }
#endif
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout_, library_, customized_type_value);
  }
};

class Transpose2OpMaker : public TransposeOpMaker {
 public:
  void Make() override {
    TransposeOpMaker::Make();
    AddOutput("XShape", "(Tensor)The output tensor.").AsIntermediate();
  }
};

template <typename T>
class Transpose2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("transpose2_grad");
    grad_op->SetInput("XShape", this->Output("XShape"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

class Transpose2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("XShape"), "Input", "XShape",
                   "Transpose2OpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "Transpose2OpGrad");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      auto xshape_dim = ctx->GetInputDim("XShape");
      auto x_shape_dim =
          framework::slice_ddim(xshape_dim, 1, xshape_dim.size());
      ctx->SetOutputDim(framework::GradVarName("X"), x_shape_dim);
      ctx->ShareLoD("XShape", framework::GradVarName("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::LibraryType library_{framework::LibraryType::kPlain};
    std::string data_format = ctx.Attr<std::string>("data_format");
    framework::DataLayout layout_ = framework::StringToDataLayout(data_format);
#ifdef PADDLE_WITH_MKLDNN
    if (library_ == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library_ = framework::LibraryType::kMKLDNN;
      layout_ = framework::DataLayout::kMKLDNN;
    }
#endif
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace(), layout_, library_);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    transpose, ops::TransposeOp, ops::TransposeOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(transpose_grad, ops::TransposeOpGrad);

REGISTER_OP_CPU_KERNEL(
    transpose, ops::TransposeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TransposeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    transpose_grad,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(transpose2, ops::Transpose2Op, ops::Transpose2OpMaker,
                  ops::Transpose2GradMaker<paddle::framework::OpDesc>,
                  ops::Transpose2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(transpose2_grad, ops::Transpose2OpGrad);

REGISTER_OP_CPU_KERNEL(
    transpose2, ops::TransposeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TransposeKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::TransposeKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::TransposeKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    transpose2_grad,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::TransposeGradKernel<paddle::platform::CPUDeviceContext, double>);
