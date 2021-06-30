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

#include "paddle/fluid/operators/layer_norm_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

using DataLayout = framework::DataLayout;

template <typename T>
class NormDataType;

template <>
class NormDataType<platform::float16> {
 public:
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <>
class NormDataType<float> {
 public:
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <typename T>
using NormDataType = NormDataType<T>;
template <typename T>
using LayerNormParamType = typename NormDataType<T>::BatchNormParamType;

template <typename T>
class LayerNormNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = LayerNormParamType<T>;
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto epsilon = ctx.Attr<float>("epsilon");
    const auto* x = ctx.Input<Tensor>("X");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* variance = ctx.Output<Tensor>("Variance");
    const auto& x_dims = x->dims();
    std::vector<int> axes;
    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int right = static_cast<int>(matrix_dim[1]);

    // The shape of scale and bias should be equal to x.shape[begin_norm_axis:],
    // required by Ascend.
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      axes.push_back(x_dims[i]);
    }

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(1.0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", axes}});
      runner.Run(stream);
      scale = &default_scale;
    } else {
      const_cast<Tensor*>(scale)->Resize(framework::make_ddim(axes));
    }

    Tensor default_bias(x->type());
    if (!bias) {
      default_bias.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_bias}, {{"dims", axes}});
      runner.Run(stream);
      bias = &default_bias;
    } else {
      const_cast<Tensor*>(bias)->Resize(framework::make_ddim(axes));
    }

    // cast scale from LayerNormParamType to T if needed
    Tensor cast_scale(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        scale->type() == framework::proto::VarType::FP32) {
      cast_scale.Resize(scale->dims());
      cast_scale.mutable_data<T>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(x->type());
      const auto& runner_cast_scale =
          NpuOpRunner("Cast", {*scale}, {cast_scale},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_scale.Run(stream);
    } else {
      cast_scale.ShareDataWith(*scale);
    }

    // cast bias from LayerNormParamType to T if needed
    Tensor cast_bias(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        bias->type() == framework::proto::VarType::FP32) {
      cast_bias.Resize(bias->dims());
      cast_bias.mutable_data<T>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(x->type());
      const auto& runner_cast_bias =
          NpuOpRunner("Cast", {*bias}, {cast_bias},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_bias.Run(stream);
    } else {
      cast_bias.ShareDataWith(*bias);
    }

    InferNPUStorageFormatAndDims(y, x->npu_storage_layout());
    y->mutable_data<T>(ctx.GetPlace());

    // mean should be of  U type
    Tensor* tmp_mean = mean;
    Tensor cast_mean(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        (scale->type() == framework::proto::VarType::FP32 ||
         bias->type() == framework::proto::VarType::FP32)) {
      cast_mean.Resize(mean->dims());
      cast_mean.mutable_data<T>(ctx.GetPlace());
      tmp_mean = &cast_mean;
      mean->mutable_data<U>(ctx.GetPlace());
    } else {
      mean->mutable_data<T>(ctx.GetPlace());
    }

    // same for variance
    Tensor* tmp_variance = variance;
    Tensor cast_variance(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        (scale->type() == framework::proto::VarType::FP32 ||
         bias->type() == framework::proto::VarType::FP32)) {
      cast_variance.Resize(variance->dims());
      cast_variance.mutable_data<T>(ctx.GetPlace());
      tmp_variance = &cast_variance;
      variance->mutable_data<U>(ctx.GetPlace());
    } else {
      variance->mutable_data<T>(ctx.GetPlace());
    }

    const auto& runner = NpuOpRunner("LayerNorm", {*x, cast_scale, cast_bias},
                                     {*y, *tmp_mean, *tmp_variance},
                                     {{"begin_norm_axis", begin_norm_axis},
                                      {"begin_params_axis", begin_norm_axis},
                                      {"epsilon", epsilon}});
    runner.Run(stream);

    // cast back from FP16 to FP32
    if (x->type() == framework::proto::VarType::FP16 &&
        mean->type() == framework::proto::VarType::FP32) {
      auto dst_dtype = ConvertToNpuDtype(mean->type());
      const auto& runner_cast_mean =
          NpuOpRunner("Cast", {*tmp_mean}, {*mean},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_mean.Run(stream);
    }
    // same for variance
    if (x->type() == framework::proto::VarType::FP16 &&
        variance->type() == framework::proto::VarType::FP32) {
      auto dst_dtype = ConvertToNpuDtype(variance->type());
      const auto& runner_cast_variance =
          NpuOpRunner("Cast", {*tmp_variance}, {*variance},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_variance.Run(stream);
    }

    // revert shape of scale and bias
    // TODO(zhiqiu): better implementation, use tmp tensor to avoid write input
    // tensor.
    const_cast<Tensor*>(scale)->Resize(framework::make_ddim({right}));
    const_cast<Tensor*>(bias)->Resize(framework::make_ddim({right}));
  }
};

template <typename T>
class LayerNormGradNPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = LayerNormParamType<T>;
    const auto begin_norm_axis = ctx.Attr<int>("begin_norm_axis");
    const auto* x = ctx.Input<Tensor>("X");
    const auto& x_dims = x->dims();
    const auto* mean = ctx.Input<Tensor>("Mean");
    const auto* variance = ctx.Input<Tensor>("Variance");
    const auto* scale = ctx.Input<Tensor>("Scale");
    const auto* dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dscale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto matrix_dim = framework::flatten_to_2d(x_dims, begin_norm_axis);
    int right = static_cast<int>(matrix_dim[1]);

    std::vector<int> axes;
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      axes.push_back(x_dims[i]);
    }

    auto place = ctx.GetPlace();
    auto stream =
        ctx.template device_context<paddle::platform::NPUDeviceContext>()
            .stream();

    // No need to compute any gradient, jusr return
    if (!dx && !dscale && !dbias) {
      return;
    }

    // The rank of mean should be equal to x, required by Ascend.
    std::vector<int> new_shape;
    for (auto i = 0; i < begin_norm_axis; ++i) {
      new_shape.push_back(x_dims[i]);
    }
    for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
      new_shape.push_back(1);
    }

    auto mean_dims = mean->dims();
    const_cast<Tensor*>(mean)->Resize(framework::make_ddim({new_shape}));
    const_cast<Tensor*>(variance)->Resize(framework::make_ddim({new_shape}));

    Tensor default_scale(x->type());
    if (!scale) {
      default_scale.mutable_data<T>(framework::make_ddim(axes), place);
      Tensor value(x->type());
      value.mutable_data<T>({1}, place);
      FillNpuTensorWithConstant<T>(&value, static_cast<T>(1.0));
      const auto& runner =
          NpuOpRunner("FillD", {value}, {default_scale}, {{"dims", axes}});
      runner.Run(stream);
      scale = &default_scale;
    } else {
      const_cast<Tensor*>(scale)->Resize(framework::make_ddim(axes));
    }

    // cast scale from LayerNormParamType to T if needed
    Tensor cast_scale(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        scale->type() == framework::proto::VarType::FP32) {
      cast_scale.Resize(scale->dims());
      cast_scale.mutable_data<T>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(x->type());
      const auto& runner_cast_scale =
          NpuOpRunner("Cast", {*scale}, {cast_scale},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_scale.Run(stream);
    } else {
      cast_scale.ShareDataWith(*scale);
    }

    // cast mean from LayerNormParamType to T if needed
    Tensor cast_mean(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        mean->type() == framework::proto::VarType::FP32) {
      cast_mean.Resize(mean->dims());
      cast_mean.mutable_data<T>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(x->type());
      const auto& runner_cast_mean =
          NpuOpRunner("Cast", {*mean}, {cast_mean},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_mean.Run(stream);
    } else {
      cast_mean.ShareDataWith(*mean);
    }

    // cast variance from LayerNormParamType to T if needed
    Tensor cast_variance(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        variance->type() == framework::proto::VarType::FP32) {
      cast_variance.Resize(variance->dims());
      cast_variance.mutable_data<T>(ctx.GetPlace());
      auto dst_dtype = ConvertToNpuDtype(x->type());
      const auto& runner_cast_variance =
          NpuOpRunner("Cast", {*variance}, {cast_variance},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_variance.Run(stream);
    } else {
      cast_variance.ShareDataWith(*variance);
    }

    Tensor dx_(dy->type()), dscale_(dy->type()), dbias_(dy->type());
    dx = (dx == nullptr) ? &dx_ : dx;
    dscale = (dscale == nullptr) ? &dscale_ : dscale;
    dbias = (dbias == nullptr) ? &dbias_ : dbias;

    dx->Resize(x->dims());
    InferNPUStorageFormatAndDims(dx, dy->npu_storage_layout());
    dx->mutable_data<T>(ctx.GetPlace());

    dscale->Resize(framework::make_ddim(axes));

    dbias->Resize(framework::make_ddim(axes));

    // dscale should be of  U type
    Tensor* tmp_dscale = dscale;
    Tensor cast_dscale(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        (mean->type() == framework::proto::VarType::FP32 ||
         variance->type() == framework::proto::VarType::FP32)) {
      cast_dscale.Resize(dscale->dims());
      cast_dscale.mutable_data<T>(ctx.GetPlace());
      tmp_dscale = &cast_dscale;
      dscale->mutable_data<U>(ctx.GetPlace());
    } else {
      dscale->mutable_data<T>(ctx.GetPlace());
    }

    // same for dbias
    Tensor* tmp_dbias = dbias;
    Tensor cast_dbias(x->type());
    if (x->type() == framework::proto::VarType::FP16 &&
        (mean->type() == framework::proto::VarType::FP32 ||
         variance->type() == framework::proto::VarType::FP32)) {
      cast_dbias.Resize(dbias->dims());
      cast_dbias.mutable_data<T>(ctx.GetPlace());
      tmp_dbias = &cast_dbias;
      dbias->mutable_data<U>(ctx.GetPlace());
    } else {
      dbias->mutable_data<T>(ctx.GetPlace());
    }

    const auto& runner = NpuOpRunner(
        "LayerNormGrad", {*dy, *x, cast_variance, cast_mean, cast_scale},
        {*dx, *tmp_dscale, *tmp_dbias}, {});
    runner.Run(stream);

    // cast back from FP16 to FP32
    if (x->type() == framework::proto::VarType::FP16 &&
        dscale->type() == framework::proto::VarType::FP32) {
      auto dst_dtype = ConvertToNpuDtype(dscale->type());
      const auto& runner_cast_dscale =
          NpuOpRunner("Cast", {*tmp_dscale}, {*dscale},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_dscale.Run(stream);
    }
    // same for dbias
    if (x->type() == framework::proto::VarType::FP16 &&
        dbias->type() == framework::proto::VarType::FP32) {
      auto dst_dtype = ConvertToNpuDtype(dbias->type());
      const auto& runner_cast_dbias =
          NpuOpRunner("Cast", {*tmp_dbias}, {*dbias},
                      {{"dst_type", static_cast<int>(dst_dtype)}});
      runner_cast_dbias.Run(stream);
    }

    const_cast<Tensor*>(mean)->Resize(mean_dims);
    const_cast<Tensor*>(variance)->Resize(mean_dims);
    const_cast<Tensor*>(scale)->Resize(framework::make_ddim({right}));
    dscale->Resize(framework::make_ddim({right}));
    dbias->Resize(framework::make_ddim({right}));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(layer_norm, ops::LayerNormNPUKernel<float>,
                       ops::LayerNormNPUKernel<plat::float16>);
REGISTER_OP_NPU_KERNEL(layer_norm_grad, ops::LayerNormGradNPUKernel<float>,
                       ops::LayerNormGradNPUKernel<plat::float16>);
