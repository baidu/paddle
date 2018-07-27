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

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ReshapeOp : public framework::OperatorWithKernel {
 public:
  ReshapeOp(const std::string &type, const framework::VariableNameMap &inputs,
            const framework::VariableNameMap &outputs,
            const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ReshapeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ReshapeOp should not be null.");

    const std::vector<int> &shape = ctx->Attrs().Get<std::vector<int>>("shape");
    PADDLE_ENFORCE(!shape.empty(),
                   "The shape information must be set by Attr(shape).");

    if (ctx->HasInput("Shape") && ctx->IsRuntime()) {
      // If true, set the shape of Output(Out) according to Input(Shape) in
      // ReshapeKernel with ExecutionContext. Also check LoD in ReshapeKernel.
      ctx->ShareLoD("X", /*->*/ "Out");
      return;
    }

    auto x_dims = ctx->GetInputDim("X");
    auto out_dims = ValidateShape(shape, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    if (x_dims[0] == out_dims[0]) {
      // Only pass LoD when the first dimension of output and Input(X)
      // are the same.
      ctx->ShareLoD("X", /*->*/ "Out");
    }
  }

  static framework::DDim ValidateShape(const std::vector<int> shape,
                                       const framework::DDim &in_dims) {
    const int64_t in_size = framework::product(in_dims);
    // only one dimension can be set to -1, whose size will be automatically
    // infered.
    const int64_t unk_dim_val = -1;
    const int64_t copy_dim_val = 0;

    std::vector<int64_t> output_shape(shape.size(), 0);
    int64_t capacity = 1;
    int unk_dim_idx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == unk_dim_val) {
        PADDLE_ENFORCE(
            unk_dim_idx == -1,
            "Only one input dimension of Attr(shape) can be unknown.");
        unk_dim_idx = i;
      } else if (shape[i] == copy_dim_val) {
        PADDLE_ENFORCE(
            static_cast<int>(i) < in_dims.size(),
            "The index of dimension to copy from input shape must be less "
            "than the size of input shape.");
      } else {
        PADDLE_ENFORCE(
            shape[i] > 0,
            "Each input dimension of Attr(shape) must not be negtive except "
            "one unknown dimension.");
      }

      capacity *= (shape[i] ? shape[i] : in_dims[i]);
      output_shape[i] =
          (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
    }

    if (unk_dim_idx != -1) {
      if (in_size > 0) {
        // in_size < 0 and is un-determinate in compile time, skip the check,
        // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
        // capacity = -24, in_size = -8, output_shape[0] = 0
        // the following check will fail.
        output_shape[unk_dim_idx] = -in_size / capacity;
        PADDLE_ENFORCE_EQ(output_shape[unk_dim_idx] * capacity, -in_size,
                          "Invalid shape is given.");
      } else {
        output_shape[unk_dim_idx] = -1;
      }
    } else {
      PADDLE_ENFORCE_EQ(capacity, in_size, "Invalid shape is given.");
    }
    return framework::make_ddim(output_shape);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

class ReshapeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of reshape operator.");
    AddInput("Shape",
             "(Tensor<int32>, optional). If provided, reshape according to "
             "this given shape. That is to say it has a higher priority than "
             "the shape attribute, while the shape attribute still should be "
             "set correctly to gurantee shape inference in compile time.")
        .AsDispensable();
    AddOutput("Out", "(Tensor). The output tensor of reshape operator.");
    AddAttr<std::vector<int>>(
        "shape", "(std::vector<int>) Target shape of reshape operator.");
    AddComment(R"DOC(
Reshape Operator.

Reshape Input(X) into the shape specified by Attr(shape) or Input(Shape). The
data in Input(X) are unchanged.

Examples:

1. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [6, 8], the reshape operator will transform Input(X)
into a 2-D tensor with shape [6, 8] and leaving Input(X)'s data unchanged.

2. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [2, 3, -1, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 3, 4, 2] and leaving Input(X)'s data
unchanged. In this case, one and only dimension of Attr(shape) can be set to -1,
the value of this dimension is inferred from the total element number of
Input(X) and remaining dimensions.

3. Given a 3-D tensor Input(X) with a shape [2, 4, 6], and the target shape
specified by Attr(shape) is [-1, 0, 3, 2], the reshape operator will transform
Input(X) into a 4-D tensor with shape [2, 4, 3, 2] and leaving Input(X)'s data
unchanged. In this case, besides -1, 0 means the actual dimension value is going
to be copied from the corresponding dimension of Input(X).

Note:

1. One and only one dimension in Attr(shape) can be set -1. In this case,
the actual dimension value will be infered from the total element number of
Input(X) and remaining dimensions.

2. More than one dimensions in Attr(shape) can be set to 0, which means the real
dimension value will be copied from Input(X) at runtime. Note that the index of
0 can not exceed Rank(X). For example, Input(X) is a 3-D tensor with shape
[2, 3, 4], Attr(shape) = [2, 3, 2, 0] is an invalid input.

3. Input(Shape) has a higher priority than Attr(shape) if it is provided, while
Attr(shape) still should be set correctly to gurantee shape inference in 
compile-time.

)DOC");
  }
};

class ReshapeGradOp : public framework::OperatorWithKernel {
 public:
  ReshapeGradOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

class ReshapeKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *out = ctx.Output<framework::LoDTensor>("Out");
    auto *in = ctx.Input<framework::LoDTensor>("X");

    auto *shape_tensor = ctx.HasInput("Shape")
                             ? ctx.Input<framework::LoDTensor>("Shape")
                             : nullptr;

    framework::DDim out_dims = out->dims();

    if (shape_tensor) {
      auto *shape_data = shape_tensor->data<int>();
      framework::Tensor cpu_shape_tensor;
      if (platform::is_gpu_place(shape_tensor->place())) {
        TensorCopySync(*shape_tensor, platform::CPUPlace(), &cpu_shape_tensor);
        shape_data = cpu_shape_tensor.data<int>();
      }
      auto shape =
          std::vector<int>(shape_data, shape_data + shape_tensor->numel());
      out_dims = ReshapeOp::ValidateShape(shape, in->dims());
    }
    if (!in->lod().empty()) {
      PADDLE_ENFORCE_EQ(
          out_dims[0], in->dims()[0],
          "Reshape operator cannot reshape an input sequence batch "
          "into an output sequence batch that has a different "
          "number of time steps. Please consider using "
          "sequence_reshape op.");
    }

    bool inplace = ctx.Attr<bool>("inplace");
    out->Resize(out_dims);
    if (!inplace) {
      out->mutable_data(ctx.GetPlace(), in->type());
      framework::TensorCopySync(*in, ctx.GetPlace(), out);
      out->Resize(out_dims);
    } else {
      out->ShareDataWith(*in);
      out->Resize(out_dims);
    }
  }
};

class ReshapeGradKernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto *d_out = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *d_x = ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    d_x->mutable_data(ctx.GetPlace(), d_out->type());
    bool inplace = ctx.Attr<bool>("inplace");

    auto in_dims = d_x->dims();
    framework::TensorCopy(*d_out, ctx.GetPlace(), ctx.device_context(), d_x);
    ctx.device_context().Wait();
    d_x->Resize(in_dims);
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(reshape, ops::ReshapeOp, ops::ReshapeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(reshape_grad, ops::ReshapeGradOp);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                               ops::ReshapeKernel, int, ops::ReshapeKernel,
                               int64_t, ops::ReshapeKernel);
REGISTER_OP_CPU_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                               double, ops::ReshapeGradKernel, int,
                               ops::ReshapeGradKernel, int64_t,
                               ops::ReshapeGradKernel);

#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape, float, ops::ReshapeKernel, double,
                                ops::ReshapeKernel, int, ops::ReshapeKernel,
                                int64_t, ops::ReshapeKernel);
REGISTER_OP_CUDA_KERNEL_FUNCTOR(reshape_grad, float, ops::ReshapeGradKernel,
                                double, ops::ReshapeGradKernel, int,
                                ops::ReshapeGradKernel, int64_t,
                                ops::ReshapeGradKernel);
#endif
