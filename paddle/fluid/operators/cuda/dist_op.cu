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

#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

template <typename EigenDevice, typename T, int Rank>
struct DistFunctor;

template <typename EigenDevice, typename T, int Rank>
struct DistGradFunctor;

template <typename T, int Rank>
struct DistFunctor<Eigen::GpuDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(const Eigen::GpuDevice& dev, OutType out, const InType& x,
                   const InType& y, const Array& x_bcasts,
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
struct DistGradFunctor<Eigen::GpuDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;

  static void EvalX(const Eigen::GpuDevice& dev, OutType out, const OutType& in,
                    const Array& reduce_dims, const Array2& reshape_dims) {
    out.device(dev) =
        in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
  }

  static void EvalY(const Eigen::GpuDevice& dev, OutType out, const OutType& in,
                    const Array& reduce_dims, const Array2& reshape_dims) {
    out.device(dev) =
        -in.reshape(reshape_dims).sum(reduce_dims).reshape(out.dimensions());
  }

  static void EvalZ(const Eigen::GpuDevice& dev, OutType grad,
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
          sign * out_grad.broadcast(out_bcasts);
    } else {
      // dz = pow(abs(x-y)/out, p-1) * sign(x-y) * dout
      grad.device(dev) =
          (x_minux_y_abs / out.broadcast(out_bcasts)).pow(p - 1) * sign *
          out_grad.broadcast(out_bcasts);
    }
  }
};

#define INSTANTIATION(FUNCTOR, T)                  \
  template struct FUNCTOR<Eigen::GpuDevice, T, 1>; \
  template struct FUNCTOR<Eigen::GpuDevice, T, 2>; \
  template struct FUNCTOR<Eigen::GpuDevice, T, 3>; \
  template struct FUNCTOR<Eigen::GpuDevice, T, 4>; \
  template struct FUNCTOR<Eigen::GpuDevice, T, 5>; \
  template struct FUNCTOR<Eigen::GpuDevice, T, 6>
INSTANTIATION(DistFunctor, float);
INSTANTIATION(DistFunctor, double);
INSTANTIATION(DistGradFunctor, float);
INSTANTIATION(DistGradFunctor, double);
#undef INSTANTIATION

}  // namespace operators
}  // namespace paddle
