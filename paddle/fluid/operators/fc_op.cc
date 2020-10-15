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

#include "paddle/fluid/operators/fc_op.h"
#include <vector>

namespace paddle {
namespace operators {

class FCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "FC");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "FC");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "FC");

    auto w_dims = ctx->GetInputDim("W");
    bool padding_weights = ctx->Attrs().Get<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(
        w_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The input Weight of fc is expected to be a 2-D tensor. "
            "But received the number of Weight's dimensions is %d, "
            "Weight's shape is %s.",
            w_dims.size(), w_dims));

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      auto w_dims1 = padding_weights ? w_dims[1] - 4 : w_dims[1];

      PADDLE_ENFORCE_LE(
          bias_dims.size(), 2,
          platform::errors::InvalidArgument(
              "The input Bias of fc is expected to be a 1-D or 2-D tensor. But "
              "received the number of Bias's dimensions is %d, "
              "Bias's shape is %s.",
              bias_dims.size(), bias_dims));

      PADDLE_ENFORCE_EQ(
          bias_dims[bias_dims.size() - 1], w_dims1,
          platform::errors::InvalidArgument(
              "The last dimension of input Bias is expected be equal "
              "to the actual width of input Weight. But received the last "
              "dimension of Bias is %d, Bias's shape is %s; "
              "the actual width of Weight is %d, Weight's shape is %s.",
              bias_dims[bias_dims.size() - 1], bias_dims, w_dims1, w_dims));

      if (bias_dims.size() == 2) {
        PADDLE_ENFORCE_EQ(
            bias_dims[0], 1,
            platform::errors::InvalidArgument(
                "The first dimension of input Bias is expected to be 1, "
                "but received %d, Bias's shape is %s.",
                bias_dims[0], bias_dims));
      }
    }

    auto in_dims = ctx->GetInputDim("Input");
    int in_num_col_dims = ctx->Attrs().Get<int>("in_num_col_dims");
    PADDLE_ENFORCE_LT(
        in_num_col_dims, in_dims.size(),
        platform::errors::InvalidArgument(
            "The attribute in_num_col_dims used to flatten Input to "
            "a 2-D tensor, is expected to be less than the number of "
            "Input's dimensions. But recieved in_num_col_dims is %d, "
            "the number of Input's dimensions is %d, Input's shape is %s.",
            in_num_col_dims, in_dims.size(), in_dims));

    auto& activation_type = ctx->Attrs().Get<std::string>("activation_type");
    if (!activation_type.empty()) {
      PADDLE_ENFORCE_EQ(activation_type, "relu",
                        platform::errors::InvalidArgument(
                            "The attribute activation_type of fc is expected "
                            "to be \"relu\", but received %s.",
                            activation_type.c_str()));
    }

    if (ctx->Attrs().Get<bool>("use_mkldnn")) {
      PADDLE_ENFORCE_EQ(
          in_dims.size() >= 2 && in_dims.size() <= 4, true,
          platform::errors::Unimplemented(
              "The Input of fc is expected to be a 2-D, 3-D or 4-D tensor when "
              "use_mkldnn is set. But recieved the number of Input's "
              "dimensions is %d, Input's shape is %s.",
              in_dims.size(), in_dims));
    }

    std::vector<int64_t> output_dims;
    FCOutputSize(in_dims, w_dims, output_dims, in_num_col_dims,
                 padding_weights);

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("Input", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type =
        OperatorWithKernel::IndicateVarDataType(ctx, "Input");
    if (ctx.Attr<bool>("use_mkldnn")) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
      using framework::proto::VarType;
      customized_type_value = (input_data_type == VarType::INT8 ||
                               input_data_type == VarType::UINT8)
                                  ? kFCMKLDNNINT8
                                  : kFCMKLDNNFP32;
    }
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class FCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor), The input tensor of fully connected operator.");
    AddInput("W", "(Tensor), The weight fc op with shape (I, O).");
    AddInput("Bias", "(Tensor, optional) Bias vector with shape (1 x O")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) The output tensor of fully connected operator. ");
    AddAttr<int>("in_num_col_dims",
                 "(int, default 1), The fc op can take tensors with more than "
                 "two dimensions as its inputs.")
        .SetDefault(1)
        .EqualGreaterThan(1);
    AddAttr<std::string>("activation_type",
                         "Activation type used in fully connected operator.")
        .SetDefault("");
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddAttr<bool>(
        "padding_weights",
        "(bool, default false) When padding weights in the fc fuse pass, "
        "the 'padding_weights' attribute is set as true.")
        .SetDefault(false);
    AddAttr<bool>(framework::kAllKernelsMustComputeRuntimeShape,
                  "Skip calling InferShape() function in the runtime.")
        .SetDefault(true);
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
    AddAttr<float>("Scale_in",
                   "(float, default 1.0f), The quantize scale of input data")
        .SetDefault(1.0f);
    AddAttr<std::vector<float>>("Scale_weights",
                                "(std::vector<float>, default {1.0f}), The "
                                "quantize scale of weights data")
        .SetDefault({1.0f});
    AddAttr<float>("Scale_out",
                   "(float, default 1.0f), The quantize scale of output data")
        .SetDefault(1.0f);
    AddAttr<bool>("force_fp32_output",
                  "(bool, default false) Force INT8 kernel output FP32, only "
                  "used in MKL-DNN INT8")
        .SetDefault(false);
    AddComment(R"DOC(
Fully Connected Operator.

The fully connected operation calculates the output based on the input, weights and bias.
The size of each dimension of the parameters checked in the infer-shape.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    fc, ops::FCOp, ops::FCOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fc, ops::FCOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FCOpKernel<paddle::platform::CPUDeviceContext, double>);
