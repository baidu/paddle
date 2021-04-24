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

#pragma once

#include <math.h>
#include <algorithm>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::Tensor;

template <typename EigenDevice, typename T, int Rank>
struct DistFunctor {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const EigenDevice& dev, OutType out, const InType& x,
                   const InType& y, const Array& x_bcasts,
                   const Array& y_bcasts, float p);
};

template <typename EigenDevice, typename T, int Rank>
struct DistGradFunctor {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void EvalX(const EigenDevice& dev, OutType out, const OutType& in,
                    const Array& reduce_dims, const Array2& reshape_dims);
  static void EvalY(const EigenDevice& dev, OutType out, const OutType& in,
                    const Array& reduce_dims, const Array2& reshape_dims);
  static void EvalZ(const EigenDevice& dev, OutType grad,
                    const InType& out_grad, const InType& x, const InType& y,
                    const InType& out, const Array& x_bcasts,
                    const Array& y_bcasts, const Array& out_bcasts, float p);
};

template <int Rank>
static void GetBraodcastDims(
    const framework::DDim& x_dims, const framework::DDim& y_dims,
    Eigen::DSizes<Eigen::DenseIndex, Rank>* x_bcast_dims,
    Eigen::DSizes<Eigen::DenseIndex, Rank>* y_bcast_dims) {
  int bcast_dims_remainder = 0;
  for (int i = 0; i < x_dims.size(); ++i) {
    if (x_dims[i] >= y_dims[i]) {
      (*x_bcast_dims)[i] = 1;
      (*y_bcast_dims)[i] = x_dims[i] / y_dims[i];
      bcast_dims_remainder += x_dims[i] % y_dims[i];
    } else {
      (*y_bcast_dims)[i] = 1;
      (*x_bcast_dims)[i] = y_dims[i] / x_dims[i];
      bcast_dims_remainder += y_dims[i] % x_dims[i];
    }
  }
  PADDLE_ENFORCE_EQ(bcast_dims_remainder, 0,
                    platform::errors::PreconditionNotMet(
                        "The input tensor of Op(dist) could not be broadcast, "
                        "X's shape is [%s], Y's shape is [%s].",
                        x_dims, y_dims));
}

static framework::DDim GetNewDims(const framework::DDim& in_dims, int rank) {
  std::vector<int64_t> new_dims_vec(rank);
  if (in_dims.size() < rank) {
    for (int i = 0; i < rank - in_dims.size(); ++i) {
      new_dims_vec[i] = 1;
    }
    for (int i = 0; i < in_dims.size(); ++i) {
      new_dims_vec[i + rank - in_dims.size()] = in_dims[i];
    }
  } else {
    new_dims_vec = vectorize(in_dims);
  }
  return framework::make_ddim(new_dims_vec);
}

template <typename DeviceContext, typename T, int Rank>
static void DistFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* y = context.Input<Tensor>("Y");
  auto* out = context.Output<Tensor>("Out");
  auto p = context.Attr<float>("p");
  out->mutable_data<T>(context.GetPlace());

  auto x_dims = context.Input<Tensor>("X")->dims();
  auto y_dims = context.Input<Tensor>("Y")->dims();

  // new dims with same size as rank, e.g. (rank=3, (4, 3) => (1, 4, 3))
  framework::DDim x_new_dims = GetNewDims(x_dims, Rank);
  framework::DDim y_new_dims = GetNewDims(y_dims, Rank);

  auto x_t = EigenTensor<T, Rank>::From(*x, x_new_dims);
  auto y_t = EigenTensor<T, Rank>::From(*y, y_new_dims);
  auto out_t = EigenTensor<T, 1>::From(*out);
  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();

  Eigen::DSizes<Eigen::DenseIndex, Rank> x_bcast_dims;
  Eigen::DSizes<Eigen::DenseIndex, Rank> y_bcast_dims;
  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  DistFunctor<std::decay_t<decltype(place)>, T, Rank>::Eval(
      place, out_t, x_t, y_t, x_bcast_dims, y_bcast_dims, p);
}

template <typename DeviceContext, typename T, int Rank>
static void DistGradFunction(const framework::ExecutionContext& context) {
  auto* x = context.Input<Tensor>("X");
  auto* y = context.Input<Tensor>("Y");
  auto* out = context.Input<Tensor>("Out");
  auto p = context.Attr<float>("p");

  auto x_grad = context.Output<Tensor>(framework::GradVarName("X"));
  auto y_grad = context.Output<Tensor>(framework::GradVarName("Y"));
  auto out_grad = context.Input<Tensor>(framework::GradVarName("Out"));

  auto x_dims = context.Input<Tensor>("X")->dims();
  auto y_dims = context.Input<Tensor>("Y")->dims();
  auto out_dims = context.Input<Tensor>("Out")->dims();

  framework::DDim x_new_dims = GetNewDims(x_dims, Rank);
  framework::DDim y_new_dims = GetNewDims(y_dims, Rank);
  framework::DDim out_new_dims = GetNewDims(out_dims, Rank);
  auto x_t = EigenTensor<T, Rank>::From(*x, x_new_dims);
  auto y_t = EigenTensor<T, Rank>::From(*y, y_new_dims);
  auto out_t = EigenTensor<T, Rank>::From(*out, out_new_dims);

  Eigen::DSizes<Eigen::DenseIndex, Rank> x_bcast_dims;
  Eigen::DSizes<Eigen::DenseIndex, Rank> y_bcast_dims;
  Eigen::DSizes<Eigen::DenseIndex, Rank> out_bcast_dims;

  GetBraodcastDims<Rank>(x_new_dims, y_new_dims, &x_bcast_dims, &y_bcast_dims);
  std::vector<int64_t> new_dims_vec(Rank);
  for (int i = 0; i < Rank; ++i) {
    new_dims_vec[i] = std::max(x_new_dims[i], y_new_dims[i]);
    out_bcast_dims[i] = new_dims_vec[i];
  }
  framework::DDim new_dims = framework::make_ddim(new_dims_vec);

  auto& place =
      *context.template device_context<DeviceContext>().eigen_device();
  auto out_grad_t = EigenTensor<T, Rank>::From(*out_grad, out_new_dims);
  framework::Tensor grad;
  grad.mutable_data<T>(new_dims, context.GetPlace());
  auto grad_t = EigenTensor<T, Rank>::From(grad);

  using Functor = DistGradFunctor<std::decay_t<decltype(place)>, T, Rank>;
  Functor::EvalZ(place, grad_t, out_grad_t, x_t, y_t, out_t, x_bcast_dims,
                 y_bcast_dims, out_bcast_dims, p);

  Eigen::DSizes<Eigen::DenseIndex, Rank * 2> x_reshape_dims;
  Eigen::DSizes<Eigen::DenseIndex, Rank * 2> y_reshape_dims;
  Eigen::DSizes<Eigen::DenseIndex, Rank> reduce_dims;
  for (int i = 0; i < x_new_dims.size(); ++i) {
    x_reshape_dims[2 * i] = x_bcast_dims[i];
    x_reshape_dims[2 * i + 1] = x_new_dims[i];
    y_reshape_dims[2 * i] = y_bcast_dims[i];
    y_reshape_dims[2 * i + 1] = y_new_dims[i];
    reduce_dims[i] = 2 * i;
  }

  // 2: if x or y is broadcasted in forward function,
  // the grad need to be sum along the broadcasted dimensions
  if (x_grad) {
    x_grad->mutable_data<T>(context.GetPlace());
    auto x_grad_t = EigenTensor<T, Rank>::From(*x_grad, x_new_dims);
    Functor::EvalX(place, x_grad_t, grad_t, reduce_dims, x_reshape_dims);
  }
  if (y_grad) {
    y_grad->mutable_data<T>(context.GetPlace());
    auto y_grad_t = EigenTensor<T, Rank>::From(*y_grad, y_new_dims);
    Functor::EvalY(place, y_grad_t, grad_t, reduce_dims, y_reshape_dims);
  }
}

template <typename DeviceContext, typename T>
class DistKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x_rank = context.Input<Tensor>("X")->dims().size();
    auto y_rank = context.Input<Tensor>("Y")->dims().size();
    auto rank = std::max(x_rank, y_rank);
    PADDLE_ENFORCE_LE(rank, 6,
                      platform::errors::Unimplemented(
                          "Op(dist) only support tensors with no more than 6 "
                          "dimensions, but X's rank is %d, Y's rank is %d.",
                          x_rank, y_rank));
    switch (rank) {
      case 1:
        DistFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        DistFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        DistFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        DistFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        DistFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        DistFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

template <typename DeviceContext, typename T>
class DistGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto x_rank = context.Input<Tensor>("X")->dims().size();
    auto y_rank = context.Input<Tensor>("Y")->dims().size();
    auto rank = std::max(x_rank, y_rank);
    PADDLE_ENFORCE_LE(rank, 6,
                      platform::errors::Unimplemented(
                          "Op(dist) only support tensors with no more than 6 "
                          "dimensions, but X's rank is %d, Y's rank is %d.",
                          x_rank, y_rank));
    switch (rank) {
      case 1:
        DistGradFunction<DeviceContext, T, 1>(context);
        break;
      case 2:
        DistGradFunction<DeviceContext, T, 2>(context);
        break;
      case 3:
        DistGradFunction<DeviceContext, T, 3>(context);
        break;
      case 4:
        DistGradFunction<DeviceContext, T, 4>(context);
        break;
      case 5:
        DistGradFunction<DeviceContext, T, 5>(context);
        break;
      case 6:
        DistGradFunction<DeviceContext, T, 6>(context);
        break;
    }
  }
};

}  // namespace operators
}  // namespace paddle
