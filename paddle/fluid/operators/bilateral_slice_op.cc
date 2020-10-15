/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/bilateral_slice_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using DataLayout = framework::DataLayout;

class BilateralSliceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BilateralSlice");
    OP_INOUT_CHECK(ctx->HasInput("Grid"), "Input", "Grid", "BilateralSlice");
    OP_INOUT_CHECK(ctx->HasInput("Guide"), "Input", "Guide", "BilateralSlice");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Output", "BilateralSlice");

    auto dim_x = ctx->GetInputDim("X");  // NCHW format
    PADDLE_ENFORCE_EQ(
        dim_x.size(), 4,
        platform::errors::Unimplemented(
            "Input(X) dimension must be 4, but got dimension = %d .",
            dim_x.size()));

    auto input_dims = ctx->GetInputDim("X");
    auto grid_dims = ctx->GetInputDim("Grid");
    auto guide_dims = ctx->GetInputDim("Guide");
    bool has_offset = ctx->Attrs().Get<bool>("has_offset");
    int64_t h = guide_dims[1];
    int64_t w = guide_dims[2];
    int64_t bs = grid_dims[0];
    int64_t coeffs_chans = grid_dims[1];
    int64_t input_chans = input_dims[1];

    int64_t output_chans;
    if ((!ctx->IsRuntime()) && ((coeffs_chans < 0) || (input_chans < 0))) {
      output_chans = -1;
    } else {
      if (has_offset) {
        PADDLE_ENFORCE_EQ((coeffs_chans % (input_chans + 1)), 0,
                          platform::errors::InvalidArgument(
                              "Slicing with affine offset, coefficients grid "
                              "should have n_out*(n_in+1) channels, but got %d",
                              coeffs_chans));
        output_chans = coeffs_chans / (input_chans + 1);
      } else {
        PADDLE_ENFORCE_EQ(
            (coeffs_chans % input_chans), 0,
            platform::errors::InvalidArgument(
                "Slicing without affine offset, coefficients grid "
                "should have n_out*n_in channels, but got %d .",
                coeffs_chans));
        output_chans = coeffs_chans / input_chans;
      }
    }

    std::vector<int64_t> output_dims;
    output_dims.push_back(bs);
    output_dims.push_back(output_chans);
    output_dims.push_back(h);
    output_dims.push_back(w);

    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class BilateralSliceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of bilateral_slice operator, "
             "This is a 4-D tensor with shape of [N, C, H, W]");
    AddInput("Grid",
             "This is a 5-D tensor. "
             "It should be [N, C, D, H, W].");
    AddInput("Guide",
             "This is a 3-D tensor "
             "It should be [N, H, W].");
    AddOutput("Out",
              "The output tensor of bilateral slice operator, "
              "This is a tensor in same rank with Input(X).");
    AddAttr<bool>("has_offset", "an optional bool. Defaults to False. ")
        .SetDefault(false);
    AddComment(R"DOC(
          This operator enhance input X according guide and grid
          For details of bilateral slice, please refer to paper:
          https://groups.csail.mit.edu/graphics/hdrnet/
         )DOC");
  }
};

class BilateralSliceOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BilateralSliceOpGrad");
    OP_INOUT_CHECK(ctx->HasInput("Grid"), "Input", "Grid",
                   "BilateralSliceOpGrad");
    OP_INOUT_CHECK(ctx->HasInput("Guide"), "Input", "Guide",
                   "BilateralSliceOpGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input", "Out",
                   "BilateralSliceOpGrad");

    auto dim_x = ctx->GetInputDim("X");
    auto dim_grid = ctx->GetInputDim("Grid");
    auto dim_guide = ctx->GetInputDim("Guide");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
    if (ctx->HasOutput(framework::GradVarName("Grid"))) {
      ctx->SetOutputDim(framework::GradVarName("Grid"), dim_grid);
    }
    if (ctx->HasOutput(framework::GradVarName("Guide"))) {
      ctx->SetOutputDim(framework::GradVarName("Guide"), dim_guide);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class BilateralSliceGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Grid", this->Input("Grid"));
    op->SetInput("Guide", this->Input("Guide"));

    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Grid"), this->InputGrad("Grid"));
    op->SetOutput(framework::GradVarName("Guide"), this->InputGrad("Guide"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class BilateralSliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_gpu_place(ctx.GetPlace()), true,
                      platform::errors::Unimplemented(
                          "BilateralSlice only supports GPU now."));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(bilateral_slice, ops::BilateralSliceOp,
                  ops::BilateralSliceOpMaker,
                  ops::BilateralSliceGradMaker<paddle::framework::OpDesc>,
                  ops::BilateralSliceGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(bilateral_slice_grad, ops::BilateralSliceOpGrad);
REGISTER_OP_CPU_KERNEL(bilateral_slice, ops::BilateralSliceKernel<float>,
                       ops::BilateralSliceKernel<double>);
