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
#include "paddle/fluid/operators/uniform_random_op.h"

#include <string>

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/bfloat16.h"

namespace paddle {
namespace operators {

template <typename T>
inline void UniformRealDistribution(T *data, const int64_t &size,
                                    const float &min, const float &max,
                                    const unsigned &seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<T>";
  std::uniform_real_distribution<T> dist(static_cast<T>(min),
                                         static_cast<T>(max));
  auto engine = paddle::framework::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = dist(*engine);
  }
}

template <>
inline void UniformRealDistribution(paddle::platform::bfloat16 *data,
                                    const int64_t &size, const float &min,
                                    const float &max, const unsigned &seed) {
  VLOG(4) << "[CPU] UniformRandomKernel<bfloat16>";
  std::uniform_real_distribution<float> dist(min, max);
  auto engine = paddle::framework::GetCPURandomEngine(seed);

  for (int64_t i = 0; i < size; ++i) {
    data[i] = static_cast<paddle::platform::bfloat16>(dist(*engine));
  }
}

// It seems that Eigen::Tensor::random in GPU will SEGFAULT.
// Use std::random and thrust::random(thrust is a std library in CUDA) to
// implement uniform random.
template <typename T>
class CPUUniformRandomKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    framework::Tensor *tensor = nullptr;
    auto out_var = ctx.OutputVar("Out");
    std::vector<int64_t> new_shape;
    auto list_new_shape_tensor =
        ctx.MultiInput<framework::Tensor>("ShapeTensorList");
    if (list_new_shape_tensor.size() > 0 || ctx.HasInput("ShapeTensor")) {
      if (ctx.HasInput("ShapeTensor")) {
        auto *shape_tensor = ctx.Input<framework::Tensor>("ShapeTensor");
        new_shape = GetNewDataFromShapeTensor(shape_tensor);
      } else if (list_new_shape_tensor.size() > 0) {
        new_shape = GetNewDataFromShapeTensorList(list_new_shape_tensor);
      }
    }

    if (out_var->IsType<framework::SelectedRows>()) {
      auto *selected_rows = out_var->GetMutable<framework::SelectedRows>();
      tensor = selected_rows->mutable_value();
      auto shape = ctx.Attr<std::vector<int64_t>>("shape");
      if (!new_shape.empty()) shape = new_shape;
      tensor->Resize(framework::make_ddim(shape));
      selected_rows->mutable_rows()->reserve(shape[0]);
    } else if (out_var->IsType<framework::LoDTensor>()) {
      tensor = out_var->GetMutable<framework::LoDTensor>();
      if (!new_shape.empty()) tensor->Resize(framework::make_ddim(new_shape));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Expected type of Output(out) in uniform_random_op must be Tensor, "
          "SelectedRows. But got "
          "unsupport type: %s.",
          framework::ToTypeName(out_var->Type())));
    }
    T *data = tensor->mutable_data<T>(ctx.GetPlace());
    int64_t size = tensor->numel();

    UniformRealDistribution<T>(
        data, size, ctx.Attr<float>("min"), ctx.Attr<float>("max"),
        static_cast<unsigned int>(ctx.Attr<int>("seed")));

    unsigned int diag_num =
        static_cast<unsigned int>(ctx.Attr<int>("diag_num"));
    unsigned int diag_step =
        static_cast<unsigned int>(ctx.Attr<int>("diag_step"));
    auto diag_val = static_cast<T>(ctx.Attr<float>("diag_val"));
    if (diag_num > 0) {
      PADDLE_ENFORCE_GT(
          size, (diag_num - 1) * (diag_step + 1),
          platform::errors::InvalidArgument(
              "ShapeInvalid: the diagonal's elements is equal (num-1) "
              "* (step-1) with num %d, step %d,"
              "It should be smaller than %d, but received %d",
              diag_num, diag_step, (diag_num - 1) * (diag_step + 1), size));
      for (int64_t i = 0; i < diag_num; ++i) {
        int64_t pos = i * diag_step + i;
        data[pos] = diag_val;
      }
    }
  }
};

class UniformRandomOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "UniformRandomOp");

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

    if (ctx->HasInputs("ShapeTensorList")) {
      // top prority shape
      auto inputs_name = ctx->Inputs("ShapeTensorList");
      PADDLE_ENFORCE_GT(inputs_name.size(), 0,
                        platform::errors::InvalidArgument(
                            "Input(ShapeTensorList)'size of "
                            "Op(uniform_random) can't be zero."
                            "Please check the Attr(shape)'s size of"
                            "Op(fluid.layers.uniform_random).)"));
      auto out_dims = std::vector<int>(inputs_name.size(), -1);
      ctx->SetOutputDim("Out", framework::make_ddim(out_dims));

      return;
    }
    auto &shape = ctx->Attrs().Get<std::vector<int64_t>>("shape");
    if (ctx->HasInput("ShapeTensor") && shape.empty()) {
      auto shape_dims = ctx->GetInputDim("ShapeTensor");
      PADDLE_ENFORCE_EQ(
          shape_dims.size(), 1,
          platform::errors::InvalidArgument(
              "ShapeError: Input(ShapeTensor)' dimension size of "
              "Op(uniform_random) must be 1."
              "But received ShapeTensor's dimensions = %d, shape = [%s]",
              shape_dims.size(), shape_dims));
      int num_ele = 1;
      for (int i = 0; i < shape_dims.size(); ++i) {
        num_ele *= shape_dims[i];
      }
      auto vec_dims = std::vector<int64_t>(num_ele, -1);
      auto out_dims = framework::make_ddim(vec_dims);
      ctx->SetOutputDim("Out", out_dims);
      return;
    }

    PADDLE_ENFORCE_EQ(shape.empty(), false,
                      platform::errors::InvalidArgument(
                          "if there is no Input(ShapeTensorList) and no "
                          "Input(ShapeTensor),the "
                          "attr(shape) information must "
                          "be set by Attr(shape)."));
    std::vector<int64_t> tensor_shape;
    tensor_shape.reserve(shape.size());
    for (auto dim : shape) {
      tensor_shape.push_back(static_cast<int64_t>(dim));
    }
    ctx->SetOutputDim("Out", framework::make_ddim(tensor_shape));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        static_cast<framework::proto::VarType::Type>(ctx.Attr<int>("dtype")),
        ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    if (var_name == "ShapeTensorList" || var_name == "ShapeTensor") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class UniformRandomOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("ShapeTensor",
             "(Tensor<int64_t> or Tensor<int32_t>, optional) . If provided, "
             "uniform_random "
             "according to "
             "this given shape. It means that it has a higher priority than "
             "the shape attribute, while the shape attribute still should be "
             "set correctly to guarantee shape inference in compile time.")
        .AsDispensable();
    AddInput("ShapeTensorList",
             "(vector<Tensor<int64_t>> or vector<Tensor<int32_t>>, optional). "
             "If provided, uniform_random use this. The shape of the tensor "
             "must be [1], it has the highest priority comparing with "
             "Input(ShapeTensor) and attr(shape).")
        .AsDuplicable()
        .AsDispensable();
    AddOutput("Out", "The output tensor of uniform random op");
    AddComment(R"DOC(
This operator initializes a tensor with random values sampled from a
uniform distribution. The random result is in set [min, max).

)DOC");
    AddAttr<std::vector<int64_t>>("shape", "The shape of the output tensor")
        .SetDefault({});
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
    AddAttr<int>("diag_num",
                 "The number of diag elements. Note that if "
                 "diag_num is 0, it means without diag init.[default 0].")
        .SetDefault(0);
    AddAttr<int>("diag_step", "The step between two diag element.[default 0].")
        .SetDefault(0);
    AddAttr<float>("diag_val", "The value of diag element. [default 1.0].")
        .SetDefault(1.0f);
    AddAttr<int>("dtype", "Output tensor data type. [default 5(FP32)].")
        .SetDefault(framework::proto::VarType::FP32);
  }
};

class UniformRandomOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    auto var_data_type = static_cast<framework::proto::VarType::Type>(
        BOOST_GET_CONST(int, ctx->GetAttr("dtype")));

    if (ctx->GetOutputType("Out") != framework::proto::VarType::SELECTED_ROWS) {
      ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR);
    }
    ctx->SetOutputDataType("Out", var_data_type);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    uniform_random, paddle::operators::UniformRandomOp,
    paddle::operators::UniformRandomOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    paddle::operators::UniformRandomOpVarTypeInference);

REGISTER_OP_CPU_KERNEL(
    uniform_random, paddle::operators::CPUUniformRandomKernel<float>,
    paddle::operators::CPUUniformRandomKernel<double>,
    paddle::operators::CPUUniformRandomKernel<paddle::platform::bfloat16>);
REGISTER_OP_CPU_KERNEL(
    uniform_random_batch_size_like,
    paddle::operators::CPUUniformRandomKernel<float>,
    paddle::operators::CPUUniformRandomKernel<double>,
    paddle::operators::CPUUniformRandomKernel<paddle::platform::bfloat16>);
