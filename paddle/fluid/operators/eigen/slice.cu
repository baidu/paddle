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
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

template <typename T, int Rank>
struct EigenSlice<Eigen::GpuDevice, T, Rank> {
  using Array = Eigen::DSizes<Eigen::DenseIndex, Rank>;
  using Array32Bit = Eigen::DSizes<int, Rank>;
  using InType = Eigen::TensorMap<
      Eigen::Tensor<const T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using InType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<const T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;
  using OutType = Eigen::TensorMap<
      Eigen::Tensor<T, Rank, Eigen::RowMajor, Eigen::DenseIndex>>;
  using OutType32BitIndex =
      Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor, int>,
                       Eigen::Aligned>;

  static void Eval(const Eigen::GpuDevice& dev, OutType out, const InType& in,
                   const Array& offsets, const Array& extents) {
    out.device(dev) = in.slice(offsets, extents);
  }

  static void Eval(const Eigen::GpuDevice& dev, OutType32BitIndex out,
                   const InType32BitIndex& in, const Array32Bit& offsets,
                   const Array32Bit& extents) {
    out.device(dev) = in.slice(offsets, extents);
  }
};

#define INSTANTIATION(FUNCTOR, TYPE)                  \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 1>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 2>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 3>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 4>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 5>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 6>
INSTANTIATION(EigenSlice, int);
INSTANTIATION(EigenSlice, int64_t);
INSTANTIATION(EigenSlice, float);
INSTANTIATION(EigenSlice, double);
INSTANTIATION(EigenSlice, platform::float16);
INSTANTIATION(EigenSlice, platform::bfloat16);
INSTANTIATION(EigenSlice, platform::complex<float>);
INSTANTIATION(EigenSlice, platform::complex<double>);
#undef INSTANTIATION

}  // namespace operators
}  // namespace paddle
