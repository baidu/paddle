/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using Tensor = framework::Tensor;

using Array1 = Eigen::DSizes<int64_t, 1>;
using Array2 = Eigen::DSizes<int64_t, 2>;
using IndexPair = Eigen::IndexPair<int>;

template <typename DeviceContext, typename T>
static inline void TransCompute(const int rank, const Tensor& in, Tensor* out,
                                const std::vector<int>& perm,
                                const DeviceContext& dev_ctx) {
  if (rank <= 1 || rank > 5) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Weight rank of SpectralNorm should be in range [2, 5], but got %d.",
        rank));
  }

  switch (rank) {
    case 2:
      math::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, in, out, perm);
      break;
    case 3:
      math::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, in, out, perm);
      break;
    case 4:
      math::Transpose<DeviceContext, T, 4> trans4;
      trans4(dev_ctx, in, out, perm);
      break;
    case 5:
      math::Transpose<DeviceContext, T, 5> trans5;
      trans5(dev_ctx, in, out, perm);
      break;
    default:
      break;
  }
}

template <typename DeviceContext, typename T>
static inline void UpdateUandV(
    Tensor* u, Tensor* v, Tensor* weight, const int power_iters,
    const float eps, const framework::ExecutionContext& ctx) {
  if (power_iters <= 0) return;
  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  auto u_t = EigenTensor<T, 2>::From(*u);
  auto v_t = EigenTensor<T, 2>::From(*v);

  const int h = weight->dims()[0];
  const int w = weight->dims()[1];

  for (int i = 0; i < power_iters; i++) {
    // V = W^T * U / ||W^T * U||_2
    blas.MatMul(*weight, true, *u, false, T(1), v, T(0));
    auto v_t_norm =
        v_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(w));
    v_t.device(place) = v_t / (v_t_norm + v_t_norm.constant(eps));
    // U = W^T * V / ||W^T * V||_2
    blas.MatMul(*weight, false, *v, false, T(1), u, T(0));
    auto u_t_norm =
        u_t.square().sum().sqrt().eval().reshape(Array1(1)).broadcast(
            Array1(h));
    u_t.device(place) = u_t / (u_t_norm + u_t_norm.constant(eps));
  }
}

// CalcMatrixSigmaAndNormWeight will not update u and v
template <typename DeviceContext, typename T>
static inline void CalcMatrixSigmaAndNormWeight(
    Tensor* sigma, const Tensor* u, const Tensor* v,
    Tensor* weight, const framework::ExecutionContext& ctx) {
  auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
  auto blas = math::GetBlas<DeviceContext, T>(ctx);
  auto sigma_t = EigenTensor<T, 2>::From(*sigma);
  auto weight_t = EigenTensor<T, 2>::From(*weight);
  auto u_t = EigenTensor<T, 2>::From(*u);

  const int h = weight->dims()[0];
  const int w = weight->dims()[1];

  Tensor weight_v;
  weight_v.mutable_data<T>({h, 1}, ctx.GetPlace());
  blas.MatMul(*weight, false, *v, false, T(1), &weight_v, T(0));
  auto weight_v_t = EigenTensor<T, 2>::From(weight_v);
  sigma_t.device(place) = (u_t * weight_v_t)
                              .sum()
                              .eval()
                              .reshape(Array2(1, 1))
                              .broadcast(Array2(h, w));
  weight_t.device(place) = weight_t / sigma_t;
}

template <typename DeviceContext, typename T>
class SpectralNormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto weight = ctx.Input<Tensor>("Weight");
    auto u = ctx.Input<Tensor>("U");
    auto v = ctx.Input<Tensor>("V");
    auto out = ctx.Output<Tensor>("Out");
    auto u_out = ctx.Output<Tensor>("UOut");
    auto v_out = ctx.Output<Tensor>("VOut");

    int dim = ctx.Attr<int>("dim");
    int power_iters = ctx.Attr<int>("power_iters");
    float eps = ctx.Attr<float>("eps");

    const int h = u->dims()[0];
    const int w = v->dims()[0];

    Tensor weight_mat;
    auto dims = weight->dims();
    const int rank = dims.size();
    std::vector<int> real_dims;
    if (dim != 0) {
      std::vector<int> perm;
      perm.push_back(dim);
      real_dims.push_back(dims[dim]);
      for (int i = 0; i < rank; i++) {
        if (i != dim) {
          perm.push_back(i);
          real_dims.push_back(dims[i]);
        }
      }
      weight_mat.mutable_data<T>(framework::make_ddim(real_dims),
                                 ctx.GetPlace());
      TransCompute<DeviceContext, T>(rank, *weight, &weight_mat, perm, dev_ctx);
    } else {
      for (int i = 0; i < rank; i++) {
        real_dims.push_back(i);
      }
      TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
    }
    weight_mat = weight_mat.Resize({h, w});

    Tensor sigma;
    sigma.mutable_data<T>(weight_mat.dims(), ctx.GetPlace());
    TensorCopySync(*u, ctx.GetPlace(), u_out);
    TensorCopySync(*v, ctx.GetPlace(), v_out);
    UpdateUandV<DeviceContext, T>(
        &(u_out->Resize({h, 1})), &(v_out->Resize({w, 1})), &weight_mat,
        power_iters, eps, ctx);
    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(
        &sigma, &(u_out->Resize({h, 1})), &(v_out->Resize({w, 1})), &weight_mat,
        ctx);

    if (dim != 0) {
      std::vector<int> perm;
      for (int i = 0; i < rank; i++) {
        if (i < dim) {
          perm.push_back(i + 1);
        } else if (i == dim) {
          perm.push_back(0);
        } else {
          perm.push_back(i);
        }
      }
      out->mutable_data<T>(dims, ctx.GetPlace());
      TransCompute<DeviceContext, T>(
          rank, weight_mat.Resize(framework::make_ddim(real_dims)), out, perm,
          dev_ctx);
    } else {
      TensorCopySync(weight_mat.Resize(dims), ctx.GetPlace(), out);
    }
  }
};

template <typename DeviceContext, typename T>
class SpectralNormGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(ctx);
    auto weight = ctx.Input<Tensor>("Weight");
    auto u_out = ctx.Input<Tensor>("UOut");
    auto v_out = ctx.Input<Tensor>("VOut");
    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto weight_grad = ctx.Output<Tensor>(framework::GradVarName("Weight"));

    int dim = ctx.Attr<int>("dim");

    const int h = u_out->dims()[0];
    const int w = v_out->dims()[0];

    Tensor u_mat, v_mat;
    TensorCopySync(*u_out, ctx.GetPlace(), &u_mat);
    TensorCopySync(*v_out, ctx.GetPlace(), &v_mat);

    Tensor weight_mat, out_grad_mat;
    auto dims = weight->dims();
    const int rank = dims.size();
    std::vector<int> real_dims;
    if (dim != 0) {
      std::vector<int> perm;
      perm.push_back(dim);
      real_dims.push_back(dims[dim]);
      for (int i = 0; i < rank; i++) {
        if (i != dim) {
          perm.push_back(i);
          real_dims.push_back(dims[i]);
        }
      }
      weight_mat.mutable_data<T>(framework::make_ddim(real_dims),
                                 ctx.GetPlace());
      out_grad_mat.mutable_data<T>(framework::make_ddim(real_dims),
                                   ctx.GetPlace());
      TransCompute<DeviceContext, T>(rank, *weight, &weight_mat, perm, dev_ctx);
      TransCompute<DeviceContext, T>(rank, *out_grad, &out_grad_mat, perm,
                                     dev_ctx);
    } else {
      for (int i = 0; i < rank; i++) {
        real_dims.push_back(i);
      }
      TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
      TensorCopySync(*out_grad, ctx.GetPlace(), &out_grad_mat);
    }
    weight_mat = weight_mat.Resize({h, w});
    out_grad_mat = out_grad_mat.Resize({h, w});

    Tensor sigma;
    sigma.mutable_data<T>(weight_mat.dims(), ctx.GetPlace());

    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(
        &sigma, &(u_mat.Resize({h, 1})), &(v_mat.Resize({w, 1})), &weight_mat,
        ctx);

    Tensor uv;
    uv.mutable_data<T>({h, w}, ctx.GetPlace());
    blas.MatMul(u_mat.Resize({h, 1}), false, v_mat.Resize({w, 1}), false, T(1), &uv,
                T(0));

    Tensor weight_grad_mat;
    weight_grad_mat.mutable_data<T>({h, w}, ctx.GetPlace());
    auto weight_grad_mat_t = EigenTensor<T, 2>::From(weight_grad_mat);
    auto weight_mat_t = EigenTensor<T, 2>::From(weight_mat);
    auto out_grad_mat_t = EigenTensor<T, 2>::From(out_grad_mat);
    auto sigma_t = EigenTensor<T, 2>::From(sigma);
    auto uv_t = EigenTensor<T, 2>::From(uv);
    weight_mat_t.device(place) =
        weight_mat_t.sum().eval().reshape(Array2(1, 1)).broadcast(Array2(h, w));
    weight_grad_mat_t.device(place) =
        out_grad_mat_t * (out_grad_mat_t.constant(1.0) - uv_t * weight_mat_t) /
        sigma_t;

    if (dim != 0) {
      std::vector<int> perm;
      for (int i = 0; i < rank; i++) {
        if (i < dim) {
          perm.push_back(i + 1);
        } else if (i == dim) {
          perm.push_back(0);
        } else {
          perm.push_back(i);
        }
      }
      weight_grad->mutable_data<T>(dims, ctx.GetPlace());
      TransCompute<DeviceContext, T>(
          rank, weight_grad_mat.Resize(framework::make_ddim(real_dims)),
          weight_grad, perm, dev_ctx);
    } else {
      TensorCopySync(weight_grad_mat.Resize(dims), ctx.GetPlace(), weight_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
