/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/instance_norm_op.h"
#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

void InstanceNormOp::InferShape(framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("SavedMean"), "Output", "SavedMean",
                 "InstanceNorm");
  OP_INOUT_CHECK(ctx->HasOutput("SavedVariance"), "Output", "SavedVariance",
                 "InstanceNorm");

  const auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_NE(framework::product(x_dims), 0,
                    platform::errors::PreconditionNotMet(
                        "The Input variable X(%s) has not "
                        "been initialized. You may need to confirm "
                        "if you put exe.run(startup_program) "
                        "after optimizer.minimize function.",
                        ctx->Inputs("X").front()));
  PADDLE_ENFORCE_GE(
      x_dims.size(), 2,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "greater than or equal to 2. But received: the shape of input "
          "X = [%s], the dimension of input X =[%d]",
          x_dims, x_dims.size()));
  PADDLE_ENFORCE_LE(
      x_dims.size(), 5,
      platform::errors::InvalidArgument(
          "ShapeError: the dimension of input X must "
          "smaller than or equal to 5, But received: the shape of input "
          "X = [%s], the dimension of input X = [%d]",
          x_dims, x_dims.size()));
  auto N = x_dims[0];
  auto C = x_dims[1];
  auto NxC = N * C;

  if (ctx->HasInput("Scale")) {
    auto scale_dim = ctx->GetInputDim("Scale");

    PADDLE_ENFORCE_EQ(
        scale_dim.size(), 1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of scale must equal to 1."
            "But received: the shape of scale is [%s], the dimension "
            "of scale is [%d]",
            scale_dim, scale_dim.size()));

    bool check = !((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0));

    if (check) {
      PADDLE_ENFORCE_EQ(scale_dim[0], C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of scale must equal to [%d]"
                            "But received: the shape of scale is [%d]",
                            C, scale_dim[0]));
    }
  }
  if (ctx->HasInput("Bias")) {
    auto bias_dim = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(
        bias_dim.size(), 1UL,
        platform::errors::InvalidArgument(
            "ShapeError: the dimension of bias must equal to 1."
            "But received: the shape of bias is [%s],the dimension "
            "of bias is [%d]",
            bias_dim, bias_dim.size()));

    bool check = !((!ctx->IsRuntime()) && (framework::product(bias_dim) <= 0));
    if (check) {
      PADDLE_ENFORCE_EQ(bias_dim[0], C,
                        platform::errors::InvalidArgument(
                            "ShapeError: the shape of bias must equal to [%d]"
                            "But received: the shape of bias is [%d]",
                            C, bias_dim[0]));
    }
  }

  ctx->SetOutputDim("Y", x_dims);
  ctx->SetOutputDim("SavedMean", {NxC});
  ctx->SetOutputDim("SavedVariance", {NxC});
  ctx->ShareLoD("X", "Y");
}

framework::OpKernelType InstanceNormOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  // By default, the type of the scale, bias, mean,
  // and var tensors should both be float. (For float or float16 input tensor)
  // or double (For double input tensor).
  auto in_param_type = framework::proto::VarType::FP32;
  if (input_data_type == framework::proto::VarType::FP64) {
    in_param_type = framework::proto::VarType::FP64;
  }
  if (ctx.HasInput("Scale")) {
    PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Scale")->type(),
                      platform::errors::InvalidArgument(
                          "Scale input should be of float type"));
  }
  if (ctx.HasInput("Bias")) {
    PADDLE_ENFORCE_EQ(in_param_type, ctx.Input<Tensor>("Bias")->type(),
                      platform::errors::InvalidArgument(
                          "Bias input should be of float type"));
  }

  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void InstanceNormOpMaker::Make() {
  AddAttr<float>("epsilon", "")
      .SetDefault(1e-5)
      .AddCustomChecker([](const float &epsilon) {
        PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                          platform::errors::InvalidArgument(
                              "'epsilon' should be between 0.0 and 0.001."));
      });
  AddInput("X", "The input tensor");
  AddInput("Scale",
           "Scale is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddInput("Bias",
           "Bias is a 1-dimensional tensor of size C "
           "that is applied to the output")
      .AsDispensable();
  AddOutput("Y", "result after normalization");
  AddOutput("SavedMean",
            "Mean of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddOutput("SavedVariance",
            "Variance of the current mini batch, "
            "will apply to output when training")
      .AsIntermediate();
  AddComment(R"DOC(
Instance Normalization.

Instance Norm has been implemented as disscussed in the paper:
https://arxiv.org/pdf/1607.08022.pdf
Can be used as a normalizer function for conv2d and fully_connected operations.
The required data format for this layer is as following:
NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
}

template <typename T>
class InstanceNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int C = x_dims[1];
    const int NxC = N * C;

    const int sample_size = x->numel() / N / C;

    auto *y = ctx.Output<Tensor>("Y");
    auto *saved_mean = ctx.Output<Tensor>("SavedMean");
    auto *saved_variance = ctx.Output<Tensor>("SavedVariance");

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    auto *place = dev_ctx.eigen_device();

    Eigen::DSizes<int, 2> shape(NxC, sample_size);
// Once eigen on Windows is updated, the if branch can be removed.
#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::DSizes<int, 2> bcast(1, sample_size);
    Eigen::DSizes<int, 2> C_shape(C, 1);
    Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
    Eigen::DSizes<int, 1> rdims(1);
#else
    Eigen::IndexList<Eigen::type2index<1>, int> bcast;
    bcast.set(1, sample_size);
    Eigen::IndexList<int, Eigen::type2index<1>> C_shape;
    C_shape.set(0, C);
    Eigen::IndexList<int, Eigen::type2index<1>> NxC_shape;
    NxC_shape.set(0, NxC);
    Eigen::IndexList<Eigen::type2index<1>> rdims;
#endif

    math::SetConstant<platform::CPUDeviceContext, T> set_constant;

    saved_mean->mutable_data<T>(ctx.GetPlace());
    saved_variance->mutable_data<T>(ctx.GetPlace());
    set_constant(dev_ctx, saved_mean, static_cast<T>(0));
    set_constant(dev_ctx, saved_variance, static_cast<T>(0));

    auto saved_mean_a = framework::EigenVector<T>::Flatten(*saved_mean);
    auto saved_mean_e = saved_mean_a.reshape(NxC_shape);
    auto saved_variance_a = framework::EigenVector<T>::Flatten(*saved_variance);
    auto saved_variance_e = saved_variance_a.reshape(NxC_shape);

    auto x_e = framework::EigenVector<T>::Flatten(*x);
    auto x_arr = x_e.reshape(shape);

    saved_mean_e.device(*place) = x_arr.mean(rdims);
    auto saved_variance_arr =
        (x_arr - saved_mean_e.broadcast(bcast)).square().mean(rdims) + epsilon;

    saved_variance_e.device(*place) = saved_variance_arr.sqrt().inverse();

    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *bias = ctx.Input<Tensor>("Bias");

    Tensor scale_data;
    Tensor bias_data;
    if (!scale) {
      scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &scale_data, static_cast<T>(1));
    }

    if (!bias) {
      bias_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &bias_data, static_cast<T>(0));
    }
    auto scale_e = scale
                       ? framework::EigenVector<T>::Flatten(*scale)
                       : framework::EigenVector<T>::Flatten(
                             const_cast<const framework::Tensor &>(scale_data));
    auto scale_arr = scale_e.reshape(C_shape);
    auto bias_e = bias ? framework::EigenVector<T>::Flatten(*bias)
                       : framework::EigenVector<T>::Flatten(
                             const_cast<const framework::Tensor &>(bias_data));
    auto bias_arr = bias_e.reshape(C_shape);

    y->mutable_data<T>(ctx.GetPlace());
    auto y_e = framework::EigenVector<T>::Flatten(*y);
    auto y_arr = y_e.reshape(shape);

    // (x - mean) * inv_std * scale + bias
    Eigen::DSizes<int, 2> bcast_param(N, sample_size);
    y_arr.device(*place) = (x_arr - saved_mean_e.broadcast(bcast)) *
                               saved_variance_e.broadcast(bcast) *
                               scale_arr.broadcast(bcast_param) +
                           bias_arr.broadcast(bcast_param);
  }
};

void InstanceNormGradOp::InferShape(framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                 framework::GradVarName("Y"), "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "InstanceNormGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "InstanceNormGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                 framework::GradVarName("X"), "InstanceNormGrad");
  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[1];
  ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
  if (ctx->HasOutput(framework::GradVarName("Scale"))) {
    ctx->SetOutputDim(framework::GradVarName("Scale"), {C});
  }
  if (ctx->HasOutput(framework::GradVarName("Bias"))) {
    ctx->SetOutputDim(framework::GradVarName("Bias"), {C});
  }
}

framework::OpKernelType InstanceNormGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar(framework::GradVarName("Y"));
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

template <typename T>
class InstanceNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scale = ctx.Input<Tensor>("Scale");
    const auto *saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *saved_inv_variance = ctx.Input<Tensor>("SavedVariance");

    const auto &x_dims = x->dims();

    const int N = x_dims[0];
    const int C = x_dims[1];
    const int NxC = N * C;
    const int sample_size = x->numel() / N / C;

    auto *d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));
    d_x->mutable_data<T>(ctx.GetPlace());

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    auto *place = dev_ctx.eigen_device();

    Eigen::DSizes<int, 2> rshape(NxC, sample_size);
    Eigen::DSizes<int, 2> param_shape(N, C);
    Eigen::DSizes<int, 2> shape(NxC, sample_size);
#ifndef EIGEN_HAS_INDEX_LIST
    Eigen::DSizes<int, 1> rdims(0);
    Eigen::DSizes<int, 1> mean_rdims(1);
    Eigen::DSizes<int, 2> bcast(1, sample_size);
    Eigen::DSizes<int, 2> C_shape(C, 1);
    Eigen::DSizes<int, 2> NxC_shape(NxC, 1);
#else
    Eigen::IndexList<Eigen::type2index<0>> rdims;
    Eigen::IndexList<Eigen::type2index<1>> mean_rdims;
    Eigen::IndexList<Eigen::type2index<1>, int> bcast;
    bcast.set(1, sample_size);
    Eigen::IndexList<int, Eigen::type2index<1>> C_shape;
    C_shape.set(0, C);
    Eigen::IndexList<int, Eigen::type2index<1>> NxC_shape;
    NxC_shape.set(0, NxC);
#endif

    math::SetConstant<platform::CPUDeviceContext, T> set_constant;

    Tensor scale_data;
    if (!scale) {
      scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &scale_data, static_cast<T>(1));
    }

    auto scale_e = scale
                       ? framework::EigenVector<T>::Flatten(*scale)
                       : framework::EigenVector<T>::Flatten(
                             const_cast<const framework::Tensor &>(scale_data));
    auto mean_e = framework::EigenVector<T>::Flatten(*saved_mean);
    auto inv_var_e = framework::EigenVector<T>::Flatten(*saved_inv_variance);
    auto dy_e = framework::EigenVector<T>::Flatten(*d_y);
    auto x_e = framework::EigenVector<T>::Flatten(*x);

    auto scale_arr = scale_e.reshape(C_shape);
    auto mean_arr = mean_e.reshape(NxC_shape);
    auto inv_var_arr = inv_var_e.reshape(NxC_shape);
    auto dy_arr = dy_e.reshape(shape);
    auto x_arr = x_e.reshape(shape);

    auto tmp = (x_arr - mean_arr.eval().broadcast(bcast)) *
               inv_var_arr.eval().broadcast(bcast);

    // math: d_bias = np.sum(d_y, axis=(n,h,w))
    // math: d_scale = np.sum((X-mean) / inv_std * dy, axis=(n, h,w))
    if (d_scale && d_bias) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      d_bias->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, d_scale, static_cast<T>(0));
      set_constant(dev_ctx, d_bias, static_cast<T>(0));

      auto d_scale_e = framework::EigenVector<T>::Flatten(*d_scale);
      auto d_scale_data = d_scale_e.reshape(C_shape);
      auto d_bias_e = framework::EigenVector<T>::Flatten(*d_bias);
      auto d_bias_data = d_bias_e.reshape(C_shape);
      d_bias_data.device(*place) =
          dy_arr.sum(mean_rdims).reshape(param_shape).sum(rdims);
      d_scale_data.device(*place) =
          (tmp * dy_arr).sum(mean_rdims).reshape(param_shape).sum(rdims);
    }

    auto dy_mean =
        dy_arr.mean(mean_rdims).reshape(NxC_shape).eval().broadcast(bcast);

    Eigen::DSizes<int, 2> bcast_param(N, sample_size);
    set_constant(dev_ctx, d_x, static_cast<T>(0));
    // math: d_x = scale * inv_var * d_y - scale * inv_var * np.sum(d_y,
    // axis=(h,w))
    //             - scale * (X - mean) * inv_var.pow(3) * np.sum(d_y * (X -
    //             mean),
    //             axis=(h,w))
    auto dx_e = framework::EigenVector<T>::Flatten(*d_x);
    auto dx_arr = dx_e.reshape(shape);
    dx_arr.device(*place) = scale_arr.broadcast(bcast_param) *
                            inv_var_arr.broadcast(bcast) *
                            (dy_arr - dy_mean -
                             tmp *
                                 (dy_arr * tmp)
                                     .mean(mean_rdims)
                                     .reshape(NxC_shape)
                                     .eval()
                                     .broadcast(bcast));
  }
};

void InstanceNormDoubleGradOp::InferShape(
    framework::InferShapeContext *ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedMean"), "Input", "SavedMean",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("SavedVariance"), "Input", "SavedVariance",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("DDX"), "Input", "DDX",
                 "InstanceNormDoubleGrad");
  OP_INOUT_CHECK(ctx->HasInput("DY"), "Input", "DY", "InstanceNormDoubleGrad");

  // check output
  OP_INOUT_CHECK(ctx->HasOutput("DX"), "Output", "DX",
                 "InstanceNormDoubleGrad");

  const auto x_dims = ctx->GetInputDim("X");
  const int C = x_dims[1];
  if (ctx->HasOutput("DX")) {
    ctx->SetOutputDim("DX", x_dims);
  }
  if (ctx->HasOutput("DScale")) {
    ctx->SetOutputDim("DScale", {C});
  }
  if (ctx->HasOutput("DDY")) {
    ctx->ShareDim("X", "DDY");
  }
}

framework::OpKernelType InstanceNormDoubleGradOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  const auto *var = ctx.InputVar("DY");
  if (var == nullptr) {
    PADDLE_THROW(
        platform::errors::NotFound("cannot find gradient variable of Y"));
  }
  const Tensor *t = nullptr;
  if (var->IsType<Tensor>()) {
    t = &var->Get<Tensor>();
  } else if (var->IsType<LoDTensor>()) {
    t = &var->Get<LoDTensor>();
  }
  if (t == nullptr) {
    PADDLE_THROW(
        platform::errors::InvalidArgument("gradient variable of Y is empty"));
  }
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
}

template <typename T>
class InstanceNormDoubleGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *X = ctx.Input<Tensor>("X");
    const auto *Scale = ctx.Input<Tensor>("Scale");
    const auto *dY = ctx.Input<Tensor>("DY");
    const auto *Saved_mean = ctx.Input<Tensor>("SavedMean");
    const auto *Saved_variance = ctx.Input<Tensor>("SavedVariance");
    const auto *ddX = ctx.Input<Tensor>("DDX");
    const auto *ddScale = ctx.Input<Tensor>("DDScale");
    const auto *ddBias = ctx.Input<Tensor>("DDBias");

    auto *dX = ctx.Output<Tensor>("DX");
    auto *dScale = ctx.Output<Tensor>("DScale");
    auto *ddY = ctx.Output<Tensor>("DDY");

    auto &dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_constant;

    const auto &x_dims = X->dims();
    int N, C, H, W, D;
    ExtractNCWHD(x_dims, DataLayout::kNCHW, &N, &C, &H, &W, &D);
    const int sample_size = X->numel() / N / C;
    const int NxC = N * C;

    const T *mean_data = Saved_mean->data<T>();
    const T *inv_var_data = Saved_variance->data<T>();
    Tensor mean_tensor;
    Tensor inv_var_tensor;
    ConstEigenArrayMap<T> x_arr(X->data<T>(), sample_size, NxC);
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, NxC);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, NxC);

    Tensor mean_tile;
    mean_tile.Resize({sample_size, NxC});
    mean_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> mean_tile_data(mean_tile.mutable_data<T>(ctx.GetPlace()),
                                    sample_size, NxC);

    Tensor inv_var_tile;
    inv_var_tile.Resize({sample_size, NxC});
    inv_var_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> inv_var_tile_data(
        inv_var_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);

    mean_tile_data = mean_arr.transpose().replicate(sample_size, 1);
    inv_var_tile_data = inv_var_arr.transpose().replicate(sample_size, 1);

    Tensor Scale_data;
    if (!Scale) {
      Scale_data.mutable_data<T>({C}, ctx.GetPlace());
      set_constant(dev_ctx, &Scale_data, static_cast<T>(1));
    }
    ConstEigenVectorArrayMap<T> scale_arr(
        Scale ? Scale->data<T>() : Scale_data.data<T>(), C);

    Tensor scale_tile;
    scale_tile.Resize({sample_size, NxC});
    scale_tile.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> scale_tile_data(scale_tile.mutable_data<T>(ctx.GetPlace()),
                                     sample_size, NxC);
    scale_tile_data = scale_arr.transpose().replicate(sample_size, N);

    ConstEigenArrayMap<T> dy_arr(dY->data<T>(), sample_size, NxC);
    ConstEigenArrayMap<T> ddx_arr(ddX->data<T>(), sample_size, NxC);

    // math: dx = scale * ((x - mean) * inv_var / HxW * (np.mean(ddx,
    // axis=(h,w)) *
    //          np.sum(dy, axis=(h,w)) -
    //          np.sum(dy * ddx, axis=(h,w)) + 3 * np.mean(dy * (x - mean),
    //          axis=(h,w)) * inv_var.pow(2) *
    //          np.sum(ddx * (x - mean), axis=(h,w))) + inv_var.pow(3) / HxW *
    //          np.sum(ddx * (x - mean)) *
    //          (np.mean(dy, axis=(h,w)) - dy) + inv_var.pow(3) / HxW *
    //          np.sum(dy,
    //          axis=(h,w)) * (x - mean) *
    //          (np.mean(ddx, axis=(h,w)) - ddx)) + ddr * (dy * inv_var -
    //          inv_var *
    //          np.mean(dy, axis=(h,w)) -
    //          inv_var.pow(3) * (x - mean) * np.mean(dy * (x - mean),
    //          axis=(h,w)))

    Tensor x_sub_mean_mul_invstd;
    x_sub_mean_mul_invstd.Resize({sample_size, NxC});
    x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace());
    EigenArrayMap<T> x_sub_mean_mul_invstd_arr(
        x_sub_mean_mul_invstd.mutable_data<T>(ctx.GetPlace()), sample_size,
        NxC);
    x_sub_mean_mul_invstd_arr = (x_arr - mean_tile_data) * inv_var_tile_data;

    if (dX) {
      dX->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, dX, static_cast<T>(0));
      EigenArrayMap<T> dx_arr(dX->mutable_data<T>(ctx.GetPlace()), sample_size,
                              NxC);

      if (ddX) {
        dx_arr +=
            x_sub_mean_mul_invstd_arr * inv_var_tile_data * inv_var_tile_data /
            sample_size *
            (ddx_arr.colwise().sum() * dy_arr.colwise().sum() / sample_size -
             (dy_arr * ddx_arr).colwise().sum() +
             3. * (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() *
                 (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                 sample_size);

        dx_arr += (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                  sample_size * inv_var_tile_data * inv_var_tile_data *
                  (dy_arr.colwise().sum() / sample_size - dy_arr);

        dx_arr += (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                  sample_size * inv_var_tile_data * inv_var_tile_data *
                  (ddx_arr.colwise().sum() / sample_size - ddx_arr);

        dx_arr = scale_tile_data * dx_arr;
      }
      if (ddScale) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);

        Tensor ddscale_tile;
        ddscale_tile.Resize({sample_size, NxC});
        ddscale_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

        dx_arr += (dy_arr * inv_var_tile_data -
                   dy_arr.colwise().sum() / sample_size * inv_var_tile_data -
                   x_sub_mean_mul_invstd_arr * inv_var_tile_data *
                       (dy_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                       sample_size) *
                  ddscale_tile_data;
      }
    }
    if (dScale) {
      // math: dscale = inv_var * (dy - np.mean(dy, axis=(h,w) - (x-mean) *
      //            inv_var.pow(2) * np.mean(dy * (x-mean), axis=(h,w)))) * ddx
      dScale->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, dScale, static_cast<T>(0));
      EigenVectorArrayMap<T> dscale_arr(dScale->mutable_data<T>(ctx.GetPlace()),
                                        C);
      if (ddX) {
        Tensor first_grad;
        first_grad.Resize({sample_size, NxC});
        first_grad.mutable_data<T>(ctx.GetPlace());
        set_constant(dev_ctx, &first_grad, static_cast<T>(0));
        EigenArrayMap<T> first_grad_arr(
            first_grad.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);

        first_grad_arr +=
            inv_var_tile_data *
            (dy_arr -
             dy_arr.colwise().sum().replicate(sample_size, 1) / sample_size -
             x_sub_mean_mul_invstd_arr *
                 (dy_arr * x_sub_mean_mul_invstd_arr)
                     .colwise()
                     .sum()
                     .replicate(sample_size, 1) /
                 sample_size);
        first_grad_arr = first_grad_arr * ddx_arr;
        for (int nc = 0; nc < NxC; ++nc) {
          int c = nc % C;
          dscale_arr(c) += first_grad_arr.colwise().sum()(nc);
        }
      }
    }
    if (ddY) {
      // math: ddy = (x - mean) * inv_var * ddscale + ddbias +
      //           scale * inv_var * (ddx - (x - mean) * inv_var.pow(2) *
      //           np.mean(ddx * (x - mean), axis=(h,w)))
      ddY->mutable_data<T>(ctx.GetPlace());
      set_constant(dev_ctx, ddY, static_cast<T>(0));
      EigenArrayMap<T> ddy_arr(ddY->mutable_data<T>(ctx.GetPlace()),
                               sample_size, NxC);
      if (ddX) {
        ddy_arr += scale_tile_data * inv_var_tile_data *
                   (ddx_arr - ddx_arr.colwise().sum() / sample_size -
                    x_sub_mean_mul_invstd_arr *
                        (ddx_arr * x_sub_mean_mul_invstd_arr).colwise().sum() /
                        sample_size);
      }
      if (ddScale && ddBias) {
        ConstEigenVectorArrayMap<T> ddscale_arr(ddScale->data<T>(), C);
        Tensor ddscale_tile;
        ddscale_tile.Resize({sample_size, NxC});
        ddscale_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddscale_tile_data(
            ddscale_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddscale_tile_data = ddscale_arr.transpose().replicate(sample_size, N);

        ConstEigenVectorArrayMap<T> ddbias_arr(ddBias->data<T>(), C);
        Tensor ddbias_tile;
        ddbias_tile.Resize({sample_size, NxC});
        ddbias_tile.mutable_data<T>(ctx.GetPlace());
        EigenArrayMap<T> ddbias_tile_data(
            ddbias_tile.mutable_data<T>(ctx.GetPlace()), sample_size, NxC);
        ddbias_tile_data = ddbias_arr.transpose().replicate(sample_size, N);

        ddy_arr += x_sub_mean_mul_invstd_arr * ddscale_tile_data;
        ddy_arr += ddbias_tile_data;
      }
    }
  }
};

DECLARE_INPLACE_OP_INFERER(InstanceNormDoubleGradOpInplaceInferer,
                           {"DY", "DDY"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(instance_norm, ops::InstanceNormOp, ops::InstanceNormOpMaker,
                  ops::InstanceNormOpInferVarType,
                  ops::InstanceNormGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(instance_norm_grad, ops::InstanceNormGradOp,
                  ops::InstanceNormDoubleGradMaker<paddle::framework::OpDesc>,
                  ops::InstanceNormDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(instance_norm_grad_grad, ops::InstanceNormDoubleGradOp,
                  ops::InstanceNormDoubleGradOpInplaceInferer);

REGISTER_OP_CPU_KERNEL(
    instance_norm,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    instance_norm_grad,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::InstanceNormGradKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    instance_norm_grad_grad,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    ops::InstanceNormDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                      double>);

REGISTER_OP_VERSION(instance_norm)
    .AddCheckpoint(
        R"ROC(
      Change dispensable of attribute from False to True in instance_norm.
    )ROC",
        paddle::framework::compatible::OpVersionDesc()
            .ModifyAttr(
                "Bias",
                "The arg 'dispensable' of Input 'Bias' is changed: from "
                "'False' to 'True'.",
                true)
            .ModifyAttr(
                "Scale",
                "The arg 'dispensable' of Input 'Scale' is changed: from "
                "'False' to 'True'.",
                true));
