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

#include "paddle/fluid/operators/conv_op.h"

#include <string>
#include <vector>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

void ConvOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("Input"),
                 "Input(Input) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Filter"),
                 "Input(Filter) of ConvOp should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Output"),
                 "Output(Output) of ConvOp should not be null.");

  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");

  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
  int groups = ctx->Attrs().Get<int>("groups");
  std::vector<int> dilations = ctx->Attrs().Get<std::vector<int>>("dilations");

  PADDLE_ENFORCE(in_dims.size() == 4 || in_dims.size() == 5,
                 "Conv intput should be 4-D or 5-D tensor, get %u",
                 in_dims.size());

  PADDLE_ENFORCE_EQ(
      in_dims.size(), filter_dims.size(),
      "Conv input dimension and filter dimension should be the same.");
  PADDLE_ENFORCE(
      in_dims.size() - strides.size() == 2U,
      "Conv input dimension and strides dimension should be consistent.");
  PADDLE_ENFORCE_EQ(
      paddings.size(), strides.size(),
      "Conv paddings dimension and Conv strides dimension should be the same.");

  PADDLE_ENFORCE_EQ(in_dims[1], filter_dims[1] * groups,
                    "The number of input channels should be equal to filter "
                    "channels * groups.");
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups, 0,
      "The number of output channels should be divided by groups.");

  std::vector<int64_t> output_shape({in_dims[0], filter_dims[0]});
  for (size_t i = 0; i < strides.size(); ++i) {
    output_shape.push_back(ConvOutputSize(in_dims[i + 2], filter_dims[i + 2],
                                          dilations[i], paddings[i],
                                          strides[i]));
  }
  ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
  ctx->ShareLoD("Input", "Output");
}

framework::OpKernelType ConvOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
    customized_type_value = kConvMKLDNNFP32;
  }
#endif

  auto input_data_type = ctx.Input<Tensor>("Input")->type();
  auto filter_data_type = ctx.Input<Tensor>("Filter")->type();
  PADDLE_ENFORCE_EQ(input_data_type, filter_data_type,
                    "input and filter data type should be consistent");

  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(library, framework::LibraryType::kCUDNN,
                      "float16 can only be used when CUDNN is used");
  }

  return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                 library, customized_type_value);
}

void Conv2DOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution operator. "
      "The format of input tensor is NCHW, where N is batch size, C is the "
      "number of channels, H is the height of the feature, "
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "H is the height of the filter, and W is the width of the filter. "
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddInput("Bias",
           "(Tensor) Bias to be added to each output of filter application."
           "The format of output tensor is X (one-dimensional) of size equal"
           "to the number of output channels. Only used with MKL-DNN.")
      .AsDispensable();
  AddInput("ResidualData",
           "(Tensor) Tensor with residual data "
           "to which convolution output will be added."
           "Used with fuse_residual_connection fusion.")
      .AsDispensable();
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator. "
            "The format of output tensor is also NCHW.");
  AddAttr<std::vector<int>>("strides",
                            "(vector<int> default:{1, 1}), the "
                            "strides(h_stride, w_stride) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(vector<int> default:{0, 0}), the "
                            "paddings(h_pad, w_pad) of "
                            "convolution operator.")
      .SetDefault({0, 0});
  AddAttr<int>(
      "groups",
      "(int default:1), the groups number of the convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1}), the "
                            "dilations(h_dilation, w_dilation) of "
                            "convolution operator.")
      .SetDefault({1, 1});
  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_residual_connection",
                "(bool, default false) Only used in mkldnn kernel. Used "
                "whenever convolution output is as an input to residual "
                "connection.")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("AnyLayout");
  // TODO(dzhwinter): need to registered layout transform function
  AddAttr<int>("workspace_size_MB",
               "Only used in cudnn kernel. Need set use_cudnn to true."
               "workspace size for cudnn, in MB, "
               "workspace is a section of GPU memory which will be "
               "allocated/freed each time the operator runs, larger "
               "workspace size can increase performance but also requires "
               "better hardware. This size should be chosen carefully.")
      .SetDefault(4096);
  AddAttr<bool>("exhaustive_search",
                "(bool, default false) cuDNN has many algorithm to calculation "
                "convolution, whether enable exhaustive search ",
                "for cuDNN convolution or not, defalut is False.")
      .SetDefault(false);
  AddComment(R"DOC(
Convolution Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and Output(Output) are in NCHW format. Where N is batch
size, C is the number of channels, H is the height of the feature, and W is
the width of the feature.
Filters(Input) is MCHW format. Where M is the number of output image channels, C is
the number of input image channels, H is the height of the filter, and W
is the width of the filter.
Parameters(strides, paddings, dilations) are two elements. These two elements represent
height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, H_{out}, W_{out})$
  Where
$$
       H_{out}= \frac{(H_{in} + 2 * paddings[0] - (dilations[0] * (H_f - 1) + 1))}{strides[0]}+ 1 \\
       W_{out}= \frac{(W_{in} + 2 * paddings[1] - (dilations[1] * (W_f - 1) + 1))}{strides[1]}+ 1
$$
)DOC");
  Apply();
}

void Conv3DOpMaker::Make() {
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);
  AddInput(
      "Input",
      "(Tensor) The input tensor of convolution operator. "
      "The format of input tensor is NCDHW. Where N is batch size, C is the "
      "number of channels, D is the depth of the feature, H is the height of "
      "the feature, "
      "and W is the width of the feature.");
  AddInput("Filter",
           "(Tensor) The filter tensor of convolution operator. "
           "The format of the filter tensor is MCDHW, where M is the number of "
           "output image channels, C is the number of input image channels, "
           "D is the depth of the filter, H is the height of the filter, and W "
           "is the width of the filter."
           "If the groups attribute is greater than 1, C equals the number of "
           "input image channels divided by the groups.");
  AddInput("ResidualData",
           "(Tensor) Tensor with residual data "
           "to which convolution output will be added."
           "Used with fuse_residual_connection fusion.")
      .AsDispensable();
  AddOutput("Output",
            "(Tensor) The output tensor of convolution operator."
            "The format of output tensor is also NCDHW.");
  AddAttr<std::vector<int>>("strides",
                            "(vector<int>, default:{1, 1, 1}), the "
                            "strides(d_stride, h_stride, w_stride) of "
                            "convolution operator.")
      .SetDefault({1, 1, 1});
  AddAttr<std::vector<int>>("paddings",
                            "(vector<int>, default:{0, 0, 0}), the "
                            "paddings(d_pad, h_pad, w_pad) of convolution "
                            "operator.")
      .SetDefault({0, 0, 0});
  AddAttr<int>(
      "groups",
      "(int default:1), the groups number of the convolution operator. "
      "According to grouped convolution in Alex Krizhevsky's Deep CNN paper: "
      "when group=2, the first half of the filters is only connected to the "
      "first half of the input channels, while the second half of the filters "
      "is only connected to the second half of the input channels.")
      .SetDefault(1);
  AddAttr<std::vector<int>>("dilations",
                            "(vector<int> default:{1, 1, 1}), the "
                            "dilations(d_dilation, h_dilation, w_dilation) of "
                            "convolution operator.")
      .SetDefault({1, 1, 1});
  AddAttr<bool>(
      "use_cudnn",
      "(bool, default false) Only used in cudnn kernel, need install cudnn")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_relu", "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<bool>("fuse_residual_connection",
                "(bool, default false) Only used in mkldnn kernel. Used "
                "whenever convolution output is as an input to residual "
                "connection.")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("AnyLayout");
  // TODO(dzhwinter): need to registered layout transform function
  AddAttr<int>("workspace_size_MB",
               "Only used in cudnn kernel. workspace size for cudnn, in MB, "
               "workspace is a section of GPU memory which will be "
               "allocated/freed each time the operator runs, larger "
               "workspace size can increase performance but also requires "
               "better hardware. This size should be chosen carefully.")
      .SetDefault(4096);
  AddAttr<bool>("exhaustive_search",
                "(bool, default false) cuDNN has many algorithm to calculation "
                "convolution, whether enable exhaustive search ",
                "for cuDNN convolution or not, defalut is False.")
      .SetDefault(false);
  AddComment(R"DOC(
Convolution3D Operator.

The convolution operation calculates the output based on the input, filter
and strides, paddings, dilations, groups parameters. The size of each dimension of the
parameters is checked in the infer-shape.
Input(Input) and output(Output) are in NCDHW format, where N is batch
size, C is the number of channels,D is the depth of the feature, H is the height of
the feature, and W is the width of the feature.
Filters(Input) is MCDHW format, where M is the number of output image channels,
C is the number of input image channels, D is the depth of the filter,
H is the height of the filter, and W is the width of the filter.
Parameters(strides, paddings, dilations) are three elements. These three elements
represent depth, height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:
  Input:
       Input shape: $(N, C_{in}, D_{in}, H_{in}, W_{in})$
       Filter shape: $(C_{out}, C_{in}, D_f, H_f, W_f)$
  Output:
       Output shape: $(N, C_{out}, D_{out}, H_{out}, W_{out})$
  Where
  $$
       D_{out}= \frac{(D_{in} + 2 * paddings[0] - (dilations[0] * (D_f - 1) + 1))}{ strides[0]}+ 1 \\
       H_{out}= \frac{(H_{in} + 2 * paddings[1] - (dilations[1] * (H_f - 1) + 1))}{ strides[1]}+ 1 \\
       W_{out}= \frac{(W_{in} + 2 * paddings[2] - (dilations[2] * (W_f - 1) + 1))}{ strides[2]}+ 1
  $$
)DOC");
  Apply();
}

void ConvOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  auto in_dims = ctx->GetInputDim("Input");
  auto filter_dims = ctx->GetInputDim("Filter");
  if (ctx->HasOutput(framework::GradVarName("Input"))) {
    ctx->SetOutputDim(framework::GradVarName("Input"), in_dims);
  }
  if (ctx->HasOutput(framework::GradVarName("Filter"))) {
    ctx->SetOutputDim(framework::GradVarName("Filter"), filter_dims);
  }
}

framework::OpKernelType ConvOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  int customized_type_value =
      framework::OpKernelType::kDefaultCustomizedTypeValue;
  framework::LibraryType library_{framework::LibraryType::kPlain};
  // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
  std::string data_format = ctx.Attr<std::string>("data_format");
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
    customized_type_value = kConvMKLDNNFP32;
  }
#endif

  return framework::OpKernelType(ctx.Input<Tensor>("Input")->type(),
                                 ctx.GetPlace(), layout_, library_,
                                 customized_type_value);
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(conv2d, ops::ConvOp, ops::Conv2DOpMaker,
                  ops::ConvOpInferVarType, ops::ConvEstimateFlops,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(conv2d_grad, ops::ConvOpGrad);

// depthwise convolution op
REGISTER_OPERATOR(depthwise_conv2d, ops::ConvOp, ops::Conv2DOpMaker,
                  ops::ConvEstimateFlops,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(depthwise_conv2d_grad, ops::ConvOpGrad);

REGISTER_OPERATOR(conv3d, ops::ConvOp, ops::Conv3DOpMaker,
                  ops::ConvOpInferVarType, ops::ConvEstimateFlops,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(conv3d_grad, ops::ConvOpGrad);

// depthwise conv kernel
// TODO(xingzhaolong): neon kernel for mobile
REGISTER_OP_CPU_KERNEL(
    depthwise_conv2d,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    depthwise_conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    conv2d, ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv2d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    conv3d, ops::GemmConvKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    conv3d_grad,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GemmConvGradKernel<paddle::platform::CPUDeviceContext, double>);
