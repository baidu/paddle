// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/dist_op.h"
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

template <typename T, int Rank>
struct DistFunctor<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::DefaultDevice& dev, OutType out,
                   const InType& x, const InType& y, const Array& x_bcasts,
                   const Array& y_bcasts, float p) {
    // p=0 means number of non-zero elements of (x-y)
    // p=inf means the maximum of |x-y|
    // p=-inf means the minimum of |x-y|
    // otherwise, Lp-norm = pow(sum(pow(|x-y|, p)), 1/p)
    if (p == 0) {
      out.device(dev) = (x.broadcast(x_bcasts) != y.broadcast(y_bcasts))
                            .template cast<T>()
                            .sum();
    } else if (p == INFINITY) {
      out.device(dev) =
          (x.broadcast(x_bcasts) - y.broadcast(y_bcasts)).abs().maximum();
    } else if (p == -INFINITY) {
      out.device(dev) =
          (x.broadcast(x_bcasts) - y.broadcast(y_bcasts)).abs().minimum();
    } else {
      out.device(dev) = (x.broadcast(x_bcasts) - y.broadcast(y_bcasts))
                            .abs()
                            .pow(p)
                            .sum()
                            .pow(1.0 / p);
    }
  }
};

template <typename T, int Rank>
struct DistGradFunctor<Eigen::DefaultDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;

  static void EvalX(const Eigen::DefaultDevice& dev, OutType out,
                    const OutType& in, const Array& reduce_dims,
                    const Array2& reshape_dims) {
    out.device(dev) =
        in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
  }

  static void EvalY(const Eigen::DefaultDevice& dev, OutType out,
                    const OutType& in, const Array& reduce_dims,
                    const Array2& reshape_dims) {
    out.device(dev) =
        -in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
  }

  static void EvalZ(const Eigen::DefaultDevice& dev, OutType grad,
                    const InType& out_grad, const InType& x, const InType& y,
                    const InType& out, const Array& x_bcasts,
                    const Array& y_bcasts, const Array& out_bcasts, float p) {
    auto x_minux_y = x.broadcast(x_bcasts) - y.broadcast(y_bcasts);
    auto x_minux_y_abs = x_minux_y.abs();
    auto sign = (x_minux_y > static_cast<T>(0)).template cast<T>() *
                    static_cast<T>(1.0) +
                (x_minux_y < static_cast<T>(0)).template cast<T>() *
                    static_cast<T>(-1.0);

    // 1: Lp-norm(z), z = x-y, compute dz
    if (p == 0) {
      grad.setZero();
    } else if (p == INFINITY || p == -INFINITY) {
      // p=inf or -inf, Lp-norm = |z_i|, the j-th element of dz tends to 0 if
      // j!=i, or equals to sign(z_i) * dout if j=i.
      grad.device(dev) =
          (x_minux_y_abs == out.broadcast(out_bcasts)).template cast<T>() *
          sign.eval() * out_grad.broadcast(out_bcasts);
    } else {
      // dz = pow(abs(x-y)/out, p-1) * sign(x-y) * dout
      grad.device(dev) =
          (x_minux_y_abs / out.broadcast(out_bcasts)).pow(p - 1) * sign.eval() *
          out_grad.broadcast(out_bcasts);
    }
  }
};

class DistOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Dist");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "Dist");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Dist");
    ctx->SetOutputDim("Out", {1});
  }
};

class DistOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input Tensor of Dist Op.");
    AddInput("Y", "The Right-hand-side input Tensor of Dist Op.");
    AddOutput("Out",
              "The output of Dist Op, "
              "which is the p-norm of (X - Y)");
    AddAttr<float>("p", "the norm to be computed.").SetDefault(2.0f);
    AddComment(R"DOC(
Dist Operator.
Given two tensors X and Y, compute Lp-norm of (X-Y). It is not a norm in a strict sense,
only as a measure of distance. The shapes of X and Y must be broadcastable. Where, Z = X - Y,

When p = 0, defining $0^0 = 0$, the zero-norm of Z is simply the number of non-zero elements of z.
$$
||Z||_{0} = \lim_{p \rightarrow 0} \sum_{i=1}^{m} |z_i|^p
$$

When p = inf, the inf-norm of Z is the maximum element of Z.
$$
||Z||_\infty=\max_i |z_i|
$$

When p = -inf, the negative-inf-norm of Z is the minimum element of Z.
$$
||Z||_{-\infty}=\min_i |z_i|
$$

Otherwise, the p-norm of Z follows the formula,
$$
||Z||_{p} = (\sum_{i=i}^{m} |z_i|^p)^{1/p}
$$
    )DOC");
  }
};

class DistOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    if (ctx->HasOutput(framework::GradVarName("Y"))) {
      ctx->SetOutputDim(framework::GradVarName("Y"), y_dims);
    }
  }
};

template <typename T>
class DistGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(dist, ops::DistOp, ops::DistOpMaker,
                  ops::DistGradOpMaker<paddle::framework::OpDesc>,
                  ops::DistGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(dist_grad, ops::DistOpGrad);
REGISTER_OP_CPU_KERNEL(
    dist, ops::DistKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DistGradKernel<paddle::platform::CPUDeviceContext, double>)
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    dist, ops::DistKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DistKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    dist_grad, ops::DistGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::DistGradKernel<paddle::platform::CUDADeviceContext, double>);
#endif
