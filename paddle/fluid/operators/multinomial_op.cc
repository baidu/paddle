/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/multinomial_op.h"

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/common_infer_shape_functions.h"

namespace paddle {
namespace operators {

class MultinomialOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "A tensor contains probabilities of categories");
    AddOutput("Out", "The output tensor of multinomial op");
    AddAttr<int>("num_samples", "number of the generated samples")
        .SetDefault(1);
    AddAttr<bool>("replacement", "can a category be sampled more than once")
        .SetDefault(false);
    AddComment(R"DOC(
This OP returns a Tensor filled with the sampled categoris according to Multinomial probabilities.

      Out ~ Multinomial(X)

)DOC");
  }
};

class MultinomialOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Multinomial");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Multinomial");

    auto x_dim = ctx->GetInputDim("X");
    int64_t x_rank = x_dim.size();
    PADDLE_ENFORCE_GT(x_rank, 0,
                      platform::errors::InvalidArgument(
                          "The number of dimensions of the input probability "
                          "distribution should be > 0, but got %d.",
                          x_rank));
    PADDLE_ENFORCE_LE(x_rank, 2,
                      platform::errors::InvalidArgument(
                          "The number of dimensions of the input probability "
                          "distribution should be <= 2, but got %d.",
                          x_rank));

    std::vector<int64_t> out_dims(x_rank);
    for (int64_t i = 0; i < x_rank - 1; i++) {
      out_dims[i] = x_dim[i];
    }

    int64_t num_samples = ctx->Attrs().Get<int>("num_samples");
    PADDLE_ENFORCE_GT(
        num_samples, 0,
        platform::errors::InvalidArgument(
            "The number of samples should be > 0, but got %d.", num_samples));
    out_dims[x_rank - 1] = num_samples;

    ctx->SetOutputDim("Out", framework::make_ddim(out_dims));
  }
};

template <typename T>
class MultinomialOpKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    const int64_t num_samples = ctx.Attr<int>("num_samples");
    const bool replacement = ctx.Attr<bool>("replacement");

    auto *in_data = x->data<T>();
    auto *out_data = out->mutable_data<int64_t>(ctx.GetPlace());

    auto in_dims = x->dims();
    int64_t in_rank = in_dims.size();
    const int64_t num_categories = in_dims[in_rank - 1];
    const int64_t num_distributions = in_rank > 1 ? in_dims[in_rank - 2] : 1;

    MultinomialFunctor<T>(out_data, in_data, num_samples, replacement,
                          num_categories, num_distributions);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    multinomial, ops::MultinomialOp, ops::MultinomialOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    multinomial, ops::MultinomialOpKernel<plat::CPUDeviceContext, float>,
    ops::MultinomialOpKernel<plat::CPUDeviceContext, double>);
