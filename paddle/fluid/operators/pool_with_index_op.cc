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

#include "paddle/fluid/operators/pool_with_index_op.h"
#include <memory>

namespace paddle {
namespace operators {

inline int MaxPoolOutputSize(int input_size, int filter_size, int padding,
                             int stride) {
  int output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  return output_size;
}

class MaxPoolWithIndexOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::InvalidArgument(
                          "Input(X) of Pooling should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of Pooling should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Mask"), true,
                      platform::errors::InvalidArgument(
                          "Output(Mask) of Pooling should not be null."));

    auto in_x_dims = ctx->GetInputDim("X");

    std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
    std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
    std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
    bool adaptive = ctx->Attrs().Get<bool>("adaptive");

    PADDLE_ENFORCE(
        in_x_dims.size() == 4 || in_x_dims.size() == 5,
        platform::errors::InvalidArgument("Pooling intput should be 4-D or 5-D "
                                          "tensor but received %dD-Tensor",
                                          in_x_dims.size()));

    if (ctx->Attrs().Get<bool>("global_pooling")) {
      ksize.resize(static_cast<size_t>(in_x_dims.size()) - 2);
      for (size_t i = 0; i < ksize.size(); ++i) {
        paddings[i] = 0;
        ksize[i] = static_cast<int>(in_x_dims[i + 2]);
      }
    }

    PADDLE_ENFORCE_EQ(
        in_x_dims.size() - ksize.size(), 2U,
        platform::errors::InvalidArgument(
            "The input size %d minus the kernel size %d should equal to 2.",
            in_x_dims.size(), ksize.size()));
    PADDLE_ENFORCE_EQ(
        ksize.size(), strides.size(),
        platform::errors::InvalidArgument(
            "Strides size %d and pooling size %d should be the same.",
            strides.size(), ksize.size()));
    PADDLE_ENFORCE_EQ(
        ksize.size(), paddings.size(),
        platform::errors::InvalidArgument(
            "Paddings size %d and pooling size %d should be the same.",
            paddings.size(), ksize.size()));

    std::vector<int64_t> output_shape({in_x_dims[0], in_x_dims[1]});
    if (adaptive) {
      output_shape.insert(output_shape.end(), ksize.begin(), ksize.end());
    } else {
      for (size_t i = 0; i < ksize.size(); ++i) {
        output_shape.push_back(MaxPoolOutputSize(in_x_dims[i + 2], ksize[i],
                                                 paddings[i], strides[i]));
      }
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
    ctx->SetOutputDim("Mask", framework::make_ddim(output_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class MaxPoolWithIndexOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("Mask"), true,
        platform::errors::InvalidArgument("Input(Mask) must not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::InvalidArgument("Input(X) must not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasInput(framework::GradVarName("Out")), true,
                      platform::errors::InvalidArgument(
                          "Input(Out@GRAD) should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                      platform::errors::InvalidArgument(
                          "Output(X@GRAD) should not be null."));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

class MaxPool2dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "X",
        "(Tensor) The input tensor of pooling operator. "
        "The format of input tensor is NCHW, where N is batch size, C is the "
        "number of channels, H is the height of the image, "
        "and W is the width of the image.");
    AddOutput("Out",
              "(Tensor) The output tensor of pooling operator. "
              "The format of output tensor is also NCHW, "
              "where N is batch size, C is "
              "the number of channels, H is the height of the image "
              "and W is the width of the image.");
    AddOutput("Mask",
              "(Tensor) The Mask tensor of pooling operator."
              "The format of output tensor is also NCHW, "
              "where N is batch size, C is the number of channels, "
              "H is the height of the image, "
              "and W is the width of the image. "
              "It represents the index in the current feature map.");

    AddAttr<std::vector<int>>("ksize",
                              "(vector<int>) The pooling window size(height, "
                              "width) of pooling operator. "
                              "If global_pooling = true, ksize and paddings "
                              "will be ignored.");  // TODO(Chengduo): Add
                                                    // checker. (Currently,
    // TypedAttrChecker don't support vector type.)
    AddAttr<bool>(
        "global_pooling",
        "(bool, default:false) Whether to use the global pooling. "
        "If global_pooling = true, ksize and paddings will be ignored.")
        .SetDefault(false);
    AddAttr<bool>(
        "adaptive",
        "(bool, default False) When true, will perform adaptive pooling "
        "instead, "
        "output shape in H and W dimensions will be same as ksize, input data "
        "will be divided into grids specify by ksize averagely and perform "
        "pooling in each grid area to get output pooling value.")
        .SetDefault(false);
    AddAttr<std::vector<int>>("strides",
                              "(vector<int>, default {1, 1}), strides(height, "
                              "width) of pooling operator.")
        .SetDefault({1, 1});  // TODO(Chengduo): Add checker. (Currently,
    // TypedAttrChecker don't support vector type.)
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector<int>, default:{0, 0}), paddings(height, width) of pooling "
        "operator. "
        "If global_pooling = true, paddings and will be ignored.")
        .SetDefault({0, 0});  // TODO(Chengduo): Add checker. (Currently,
    // TypedAttrChecker don't support vector type.)

    AddComment(R"DOC(
MaxPool2d Operator.

The maxPooling2d with index operation calculates the output and the mask
based on the input, ksize, strides, and paddings parameters. Input(X) and
output(Out, Mask) are in NCHW format, where N is batch size, C is the
number of channels, H is the height of the feature, 
and W is the width of the feature.
Parameters(ksize, strides, paddings) are two elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out, Mask) size may be different.

Example:
  Input:
       X shape: $(N, C, H_{in}, W_{in})$
  Output:
       Out shape: $(N, C, H_{out}, W_{out})$
       Mask shape: $(N, C, H_{out}, W_{out})$
  Where
       $$
       H_{out} = \frac{(H_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1 \\
       W_{out} = \frac{(W_{in} - ksize[1] + 2 * paddings[1])}{strides[1]} + 1
       $$
  
  For adaptive = true:
       $$
       H_{out} = ksize[0]   W_{out} = ksize[1]
       $$
      

)DOC");
  }
};

class MaxPool3dWithIndexOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor) The input tensor of pooling operator. "
             "The format of input tensor is NCDHW, where N is batch size, C is "
             "the number of channels, and D, H and W are the depth, height and "
             "width of "
             "the image, respectively");
    AddOutput("Out",
              "(Tensor) The output tensor of pooling operator. "
              "The format of output tensor is also NCDHW, "
              "where N is the batch size, C is the number of channels, "
              "and D, H and W are the depth, height and "
              "width of the image, respectively.");
    AddOutput("Mask",
              "(Tensor) The Mask tensor of pooling operator. "
              "The format of output tensor is also NCDHW, "
              "where N is the batch size, C is the number of channels, and "
              "D, H and W are the depth, height and width "
              "of the image, respectively. "
              "It represents the index in the current feature map.");

    AddAttr<std::vector<int>>("ksize",
                              "(vector<int>) The pooling window size(depth, "
                              "height, width) of pooling operator. "
                              "If global_pooling = true, ksize and paddings "
                              "will be ignored.");  // TODO(Chengduo): Add
                                                    // checker. (Currently,
    // TypedAttrChecker don't support vector type.)
    AddAttr<bool>(
        "global_pooling",
        "(bool, default false) Whether to use the global pooling. "
        "If global_pooling = true, ksize and paddings will be ignored.")
        .SetDefault(false);
    AddAttr<bool>(
        "adaptive",
        "(bool, default False) When true, will perform adaptive pooling "
        "instead, "
        "output shape in H and W dimensions will be same as ksize, input data "
        "will be divided into grids specify by ksize averagely and perform "
        "pooling in each grid area to get output pooling value.")
        .SetDefault(false);
    AddAttr<std::vector<int>>("strides",
                              "(vector<int>, default {1,1,1}), strides(depth, "
                              "height, width) of pooling operator.")
        .SetDefault({1, 1, 1});  // TODO(Chengduo): Add checker. (Currently,
    // TypedAttrChecker don't support vector type.)
    AddAttr<std::vector<int>>(
        "paddings",
        "(vector, default {0,0,0}), paddings(depth, "
        "height, width) of pooling operator. "
        "If global_pooling = true, paddings and ksize will be ignored.")
        .SetDefault({0, 0, 0});  // TODO(Chengduo): Add checker. (Currently,
    // TypedAttrChecker don't support vector type.)

    AddComment(R"DOC(
MaxPool3d Operator.

The maxpooling3d with index operation calculates the output and the mask
based on the input and ksize, strides, paddings parameters.
Input(X) and output(Out, Mask) are in NCDHW format, where N is batch
size, C is the number of channels, and D, H and W are the depth, height and
width of the feature, respectively. 
Parameters(ksize, strides, paddings) are three elements.
These three elements represent depth, height and width, respectively.
The input(X) size and output(Out, Mask) size may be different.

Example:
  Input:
       X shape: $(N, C, D_{in}, H_{in}, W_{in})$
  Output:
       Out shape: $(N, C, D_{out}, H_{out}, W_{out})$
       Mask shape: $(N, C, D_{out}, H_{out}, W_{out})$
  Where
       $$
       D_{out} = \frac{(D_{in} - ksize[0] + 2 * paddings[0])}{strides[0]} + 1 \\
       H_{out} = \frac{(H_{in} - ksize[1] + 2 * paddings[1])}{strides[1]} + 1 \\
       W_{out} = \frac{(W_{in} - ksize[2] + 2 * paddings[2])}{strides[2]} + 1
       $$
  
  For adaptive = true:
       $$
       D_{out} = ksize[0]   H_{out} = ksize[1]   W_{out} = ksize[2]
       $$

)DOC");
  }
};

template <typename T>
class MaxPoolWithIndexGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Mask", this->Output("Mask"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(
    MaxPoolWithIndexOpGradNoNeedBufferVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(max_pool2d_with_index, ops::MaxPoolWithIndexOp,
                  ops::MaxPool2dWithIndexOpMaker,
                  ops::MaxPoolWithIndexGradOpMaker<paddle::framework::OpDesc>,
                  ops::MaxPoolWithIndexGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(max_pool2d_with_index_grad, ops::MaxPoolWithIndexOpGrad,
                  ops::MaxPoolWithIndexOpGradNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    max_pool2d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUDeviceContext, float, int>,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUDeviceContext, double,
                                int>);
REGISTER_OP_CPU_KERNEL(
    max_pool2d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUDeviceContext, float,
                                    int>,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUDeviceContext, double,
                                    int>);

REGISTER_OPERATOR(max_pool3d_with_index, ops::MaxPoolWithIndexOp,
                  ops::MaxPool3dWithIndexOpMaker,
                  ops::MaxPoolWithIndexGradOpMaker<paddle::framework::OpDesc>,
                  ops::MaxPoolWithIndexGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(max_pool3d_with_index_grad, ops::MaxPoolWithIndexOpGrad,
                  ops::MaxPoolWithIndexOpGradNoNeedBufferVarsInferer);

REGISTER_OP_CPU_KERNEL(
    max_pool3d_with_index,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUDeviceContext, float, int>,
    ops::MaxPoolWithIndexKernel<paddle::platform::CPUDeviceContext, double,
                                int>);
REGISTER_OP_CPU_KERNEL(
    max_pool3d_with_index_grad,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUDeviceContext, float,
                                    int>,
    ops::MaxPoolWithIndexGradKernel<paddle::platform::CPUDeviceContext, double,
                                    int>);
