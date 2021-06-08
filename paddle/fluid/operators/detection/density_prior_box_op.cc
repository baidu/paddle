/*Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/detection/density_prior_box_op.h"

namespace paddle {
namespace operators {

class DensityPriorBoxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input",
                   "DensityPriorBoxOp");
    OP_INOUT_CHECK(ctx->HasInput("Image"), "Input", "Image",
                   "DensityPriorBoxOp");

    auto image_dims = ctx->GetInputDim("Image");
    auto input_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(
        image_dims.size(), 4,
        platform::errors::InvalidArgument(
            "The Input(Image) of Op(density_prior_box) should be a 4-D Tensor "
            "and data format is NCHW. But received Image's dimensions = %d, "
            "shape = [%s].",
            image_dims.size(), image_dims));
    PADDLE_ENFORCE_EQ(
        input_dims.size(), 4,
        platform::errors::InvalidArgument(
            "The Input(Input) of Op(density_prior_box) should be a 4-D Tensor "
            "and data format is NCHW. But received Input's dimensions = %d, "
            "shape = [%s].",
            input_dims.size(), input_dims));

    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_LT(
          input_dims[2], image_dims[2],
          platform::errors::InvalidArgument(
              "The input tensor Input's height"
              "of DensityPriorBoxOp should be smaller than input tensor Image's"
              "height. But received Input's height = %d, Image's height = %d",
              input_dims[2], image_dims[2]));

      PADDLE_ENFORCE_LT(
          input_dims[3], image_dims[3],
          platform::errors::InvalidArgument(
              "The input tensor Input's width"
              "of DensityPriorBoxOp should be smaller than input tensor Image's"
              "width. But received Input's width = %d, Image's width = %d",
              input_dims[3], image_dims[3]));
    }
    auto variances = ctx->Attrs().Get<std::vector<float>>("variances");

    auto fixed_sizes = ctx->Attrs().Get<std::vector<float>>("fixed_sizes");
    auto fixed_ratios = ctx->Attrs().Get<std::vector<float>>("fixed_ratios");
    auto densities = ctx->Attrs().Get<std::vector<int>>("densities");
    bool flatten = ctx->Attrs().Get<bool>("flatten_to_2d");

    PADDLE_ENFORCE_EQ(
        fixed_sizes.size(), densities.size(),
        platform::errors::InvalidArgument(
            "The length of fixed_sizes and densities must be equal. "
            "But received: fixed_sizes's length is %d, densities's length "
            "is %d",
            fixed_sizes.size(), densities.size()));
    size_t num_priors = 0;
    for (int densitie : densities) {
      num_priors += (fixed_ratios.size()) * (pow(densitie, 2));
    }
    if (!flatten) {
      std::vector<int64_t> dim_vec(4);
      dim_vec[0] = input_dims[2];
      dim_vec[1] = input_dims[3];
      dim_vec[2] = num_priors;
      dim_vec[3] = 4;
      ctx->SetOutputDim("Boxes", framework::make_ddim(dim_vec));
      ctx->SetOutputDim("Variances", framework::make_ddim(dim_vec));
    } else if (ctx->IsRuntime()) {
      int64_t dim0 = input_dims[2] * input_dims[3] * num_priors;
      ctx->SetOutputDim("Boxes", {dim0, 4});
      ctx->SetOutputDim("Variances", {dim0, 4});
    } else {
      ctx->SetOutputDim("Boxes", {-1, 4});
      ctx->SetOutputDim("Variances", {-1, 4});
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace());
  }
};

class DensityPriorBoxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor, default Tensor<float>), "
        "the input feature data of DensityPriorBoxOp, the layout is NCHW.");
    AddInput("Image",
             "(Tensor, default Tensor<float>), "
             "the input image data of DensityPriorBoxOp, the layout is NCHW.");
    AddOutput("Boxes",
              "(Tensor, default Tensor<float>), the output prior boxes of "
              "DensityPriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");
    AddOutput("Variances",
              "(Tensor, default Tensor<float>), the expanded variances of "
              "DensityPriorBoxOp. The layout is [H, W, num_priors, 4]. "
              "H is the height of input, W is the width of input, num_priors "
              "is the box count of each position.");
    AddAttr<std::vector<float>>("variances",
                                "(vector<float>) List of variances to be "
                                "encoded in density prior boxes.")
        .AddCustomChecker([](const std::vector<float>& variances) {
          PADDLE_ENFORCE_EQ(variances.size(), 4,
                            platform::errors::InvalidArgument(
                                "The length of variance must "
                                "be 4. But received: variances' length is %d.",
                                variances.size()));
          for (size_t i = 0; i < variances.size(); ++i) {
            PADDLE_ENFORCE_GT(variances[i], 0.0,
                              platform::errors::OutOfRange(
                                  "variance[%d] must be greater "
                                  "than 0. But received: variance[%d] = %f",
                                  i, i, variances[i]));
          }
        });
    AddAttr<bool>("clip", "(bool) Whether to clip out-of-boundary boxes.")
        .SetDefault(true);
    AddAttr<bool>("flatten_to_2d",
                  "(bool) Whether to flatten to 2D and "
                  "the second dim is 4.")
        .SetDefault(false);
    AddAttr<float>(
        "step_w",
        "Density prior boxes step across width, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_w) {
          PADDLE_ENFORCE_GE(step_w, 0.0,
                            platform::errors::InvalidArgument(
                                "step_w should be larger "
                                "than 0. But received: step_w = %f.",
                                step_w));
        });
    AddAttr<float>(
        "step_h",
        "Density prior boxes step across height, 0.0 for auto calculation.")
        .SetDefault(0.0)
        .AddCustomChecker([](const float& step_h) {
          PADDLE_ENFORCE_GE(step_h, 0.0,
                            platform::errors::InvalidArgument(
                                "step_h should be larger "
                                "than 0. But received: step_h = %f.",
                                step_h));
        });

    AddAttr<float>("offset",
                   "(float) "
                   "Density prior boxes center offset.")
        .SetDefault(0.5);
    AddAttr<std::vector<float>>("fixed_sizes",
                                "(vector<float>) List of fixed sizes "
                                "of generated density prior boxes.")
        .SetDefault(std::vector<float>{})
        .AddCustomChecker([](const std::vector<float>& fixed_sizes) {
          for (size_t i = 0; i < fixed_sizes.size(); ++i) {
            PADDLE_ENFORCE_GT(
                fixed_sizes[i], 0.0,
                platform::errors::OutOfRange(
                    "fixed_sizes[%d] should be "
                    "larger than 0. But received: fixed_sizes[%d] = %f",
                    i, i, fixed_sizes[i]));
          }
        });

    AddAttr<std::vector<float>>("fixed_ratios",
                                "(vector<float>) List of fixed ratios "
                                "of generated density prior boxes.")
        .SetDefault(std::vector<float>{})
        .AddCustomChecker([](const std::vector<float>& fixed_ratios) {
          for (size_t i = 0; i < fixed_ratios.size(); ++i) {
            PADDLE_ENFORCE_GT(
                fixed_ratios[i], 0.0,
                platform::errors::OutOfRange(
                    "fixed_ratios[%d] should be "
                    "larger than 0. But received: fixed_ratios[%d] = %f",
                    i, i, fixed_ratios[i]));
          }
        });

    AddAttr<std::vector<int>>("densities",
                              "(vector<float>) List of densities "
                              "of generated density prior boxes.")
        .SetDefault(std::vector<int>{})
        .AddCustomChecker([](const std::vector<int>& densities) {
          for (size_t i = 0; i < densities.size(); ++i) {
            PADDLE_ENFORCE_GT(
                densities[i], 0,
                platform::errors::OutOfRange(
                    "densities[%d] should be "
                    "larger than 0. But received: densities[%d] = %f.",
                    i, i, densities[i]));
          }
        });
    AddComment(R"DOC(
        Density Prior box operator
        Each position of the input produce N density prior boxes, N is determined by
        the count of fixed_ratios, densities, the calculation of N is as follows:
        for density in densities:
        N += size(fixed_ratios)*density^2
        )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    density_prior_box, ops::DensityPriorBoxOp, ops::DensityPriorBoxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(density_prior_box, ops::DensityPriorBoxOpKernel<float>,
                       ops::DensityPriorBoxOpKernel<double>);
