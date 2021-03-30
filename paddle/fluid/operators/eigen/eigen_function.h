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
#pragma once
#include "unsupported/Eigen/CXX11/Tensor"

namespace paddle {
namespace operators {

template <typename EigenDevice, typename T, int Rank>
struct EigenBroadcast {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(EigenDevice dev, OutType out, InType in, const Array& bcast);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenBroadcastGrad {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array2 = Eigen::DSizes<Eigen::DenseIndex, Rank * 2>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType =
      Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(EigenDevice dev, OutType out, InType in,
                   const Array& reduce_dims, const Array2& reshape_dims);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenSlice {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(EigenDevice dev, OutType out, InType in,
                   const Array& offsets, const Array& extents);
};

template <typename EigenDevice, typename T, int Rank>
struct EigenPad {
  using Array =
      std::array<std::pair<Eigen::DenseIndex, Eigen::DenseIndex>, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  static void Eval(EigenDevice dev, OutType out, InType in,
                   const Array& padding, T padding_value);
};

}  // namespace operators
}  // namespace paddle
