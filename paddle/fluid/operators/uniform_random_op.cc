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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace operators {

// It seems that Eigen::Tensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T>
class CPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    framework::Tensor *tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (ctx.HasInput("Shape")) {
        const framework::Tensor *shapeTensor =
            ctx.Input<framework::Tensor>("Shape");
        const int64_t *shapeData = shapeTensor->data<int64_t>();
        std::vector<int64_t> shape(shapeData, shapeData + shapeTensor->numel());
        tensor->Resize(framework::make_ddim(shape));
      } else {
        auto shape = ctx.Attr<std::vector<int64_t>>("shape");
        tensor->Resize(framework::make_ddim(shape));
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      auto *selected_rows = out_var->GetMutable<framework::SelectedRows>();
      tensor = selected_rows->mutable_value();
      if (ctx.HasInput("Shape")) {
        const framework::Tensor *shapeTensor =
            ctx.Input<framework::Tensor>("Shape");
        const int64_t *shapeData = shapeTensor->data<int64_t>();
        std::vector<int64_t> shape(shapeData, shapeData + shapeTensor->numel());
        tensor->Resize(framework::make_ddim(shape));
        selected_rows->mutable_rows()->reserve(shape[0]);
      } else {
        auto shape = ctx.Attr<std::vector<int64_t>>("shape");
        tensor->Resize(framework::make_ddim(shape));
        selected_rows->mutable_rows()->reserve(shape[0]);
      }
    } else {
      PADDLE_THROW(
          "uniform_random_op's output only"
          "supports SelectedRows and LoDTensor");
    }
    T *data = tensor->mutable_data<T>(ctx.GetPlace());
    unsigned int seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    std::minstd_rand engine;
    if (seed == 0) {
      seed = std::random_device()();
    }
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(
        static_cast<T>(ctx.Attr<float>("min")),
        static_cast<T>(ctx.Attr<float>("max")));
    int64_t size = tensor->numel();
    for (int64_t i = 0; i < size; ++i) {
      data[i] = dist(engine);
    }
  }
};

class UniformRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UniformRandomOp should not be null.");

    PADDLE_ENFORCE(
        ctx->Attrs().Get<float>("min") < ctx->Attrs().Get<float>("max"),
        "uniform_random's min must less then max");
    if (!ctx->HasInput("Shape")) {
      auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
      std::vector<int64_t> temp;
      temp.reserve(shape.size());
      for (auto dim : shape) {
        temp.push_back(static_cast<int64_t>(dim));
      }
      ctx->SetOutputDim("Out", framework::make_ddim(temp));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }
};

class UniformRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Shape",
             "(Tensor Tensor<int>) "
             "the shape of the output tensor.")
        .AsDispensable();
    AddOutput("Out", "The output tensor of uniform random op");
    AddComment(R"DOC(
This operator initializes a tensor with random values sampled from a
uniform distribution. The random result is in set [min, max].

)DOC");
    AddAttr<std::vector<int64_t>>("shape", "The shape of the output tensor");
    AddAttr<float>("min", "Minimum value of uniform random. [default -1.0].")
        .SetDefault(-1.0f);
    AddAttr<float>("max", "Maximun value of uniform random. [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("seed",
                 "Random seed used for generating samples. "
                 "0 means use a seed generated by the system."
                 "Note that if seed is not 0, this operator will always "
                 "generate the same random numbers every time. [default 0].")
        .SetDefault(0);
    AddAttr<int>("dtype", "Output tensor data type. [default 5(FP32)].")
        .SetDefault(framework::proto::VarType::FP32);
  }
};

class UniformRandomOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto out_var_name = ctx->Output("Out").front();
    auto var_data_type = static_cast<framework::proto::VarType::Type>(
        boost::get<int>(ctx->GetAttr("dtype")));

    if (ctx->GetType(out_var_name) !=
        framework::proto::VarType::SELECTED_ROWS) {
      ctx->SetType(out_var_name, framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetDataType(out_var_name, var_data_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(uniform_random, paddle::operators::UniformRandomOp,
                  paddle::operators::UniformRandomOpMaker,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::UniformRandomOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(uniform_random,
                       paddle::operators::CPUUniformRandomKernel<float>,
                       paddle::operators::CPUUniformRandomKernel<double>);
REGISTER_OP_CPU_KERNEL(uniform_random_batch_size_like,
                       paddle::operators::CPUUniformRandomKernel<float>,
                       paddle::operators::CPUUniformRandomKernel<double>);
