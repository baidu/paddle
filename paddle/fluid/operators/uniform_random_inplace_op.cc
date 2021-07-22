/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

class UniformRandomInplaceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddComment(R"DOC(
This operator fills self tensor with random values sampled from a
uniform distribution. The random result is in a range of [min, max).
)DOC");
    AddInput("X", "The input tensor.");
    AddOutput("Out", "The output tensor of uniform random op");
    AddAttr<float>("min", "Minimum value of uniform random. [default -1.0].")
        .SetDefault(-1.0f);
    AddAttr<float>("max", "Maximun value of uniform random. [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed used for generating samples. "
                 "If seed is 0, it will use the seed of the global default "
                 "generator (which can be set by paddle.seed). "
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time. [default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_num",
                 "The number of diag elements. Note that if "
                 "diag_num is 0, it means without diag init.[default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_step", "The step between two diag element.[default 0].")
        .SetDefault(0);
    AddAttr<float>("diag_val", "The value of diag element. [default 1.0].")
        .SetDefault(1.0f);
  }
};

class UniformRandomInplaceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "UniformRandomInplaceOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "UniformRandomInplaceOp");
    PADDLE_ENFORCE_LT(
        ctx->Attrs().Get<float>("min"), ctx->Attrs().Get<float>("max"),
        platform::errors::InvalidArgument(
            "The uniform_random's min must less then max. But received min = "
            "%f great than or equal max = %f.",
            ctx->Attrs().Get<float>("min"), ctx->Attrs().Get<float>("max")));
    PADDLE_ENFORCE_GE(ctx->Attrs().Get<int>("diag_num"), 0,
                      platform::errors::InvalidArgument(
                          "The uniform_random's diag_num must greater than or "
                          "equal 0. But recevied diag_num (%d) < 0.",
                          ctx->Attrs().Get<int>("diag_num")));
    PADDLE_ENFORCE_GE(ctx->Attrs().Get<int>("diag_step"), 0,
                      platform::errors::InvalidArgument(
                          "The uniform_random's diag_step must greater than or "
                          "equal 0. But recevied diag_step (%d) < 0.",
                          ctx->Attrs().Get<int>("diag_step")));
    auto xdim = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", xdim);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

template <typename T>
class CPUUniformRandomInplaceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto out_var = ctx.OutputVar("Out");
    auto *tensor = out_var->GetMutable<framework::LoDTensor>();
    T *data = tensor->mutable_data<T>(ctx.GetPlace());
    int64_t size = tensor->numel();
    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    auto engine = paddle::framework::GetCPURandomEngine(
        static_cast<unsigned int>(ctx.Attr<int>("seed")));
    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(*engine);
    }
  }
};

class UniformRandomInplaceOpVarTypeInference
    : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

DECLARE_INPLACE_OP_INFERER(UniformRandomInplaceInToOut, {"X", "Out"});
REGISTER_OPERATOR(
    uniform_random_inplace, paddle::operators::UniformRandomInplaceOp,
    paddle::operators::UniformRandomInplaceOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::UniformRandomInplaceOpVarTypeInference,
    UniformRandomInplaceInToOut);
REGISTER_OP_CPU_KERNEL(
    uniform_random_inplace,
    paddle::operators::CPUUniformRandomInplaceKernel<float>,
    paddle::operators::CPUUniformRandomInplaceKernel<double>);
