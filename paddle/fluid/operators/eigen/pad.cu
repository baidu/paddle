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
struct EigenPad<Eigen::GpuDevice, T, Rank> {
  using Array = std::array<std::pair<int64_t, int64_t>, Rank>;
  using Array32Bit = std::array<std::pair<int, int>, Rank>;
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
                   const Array& padding, const T value) {
    out.device(dev) = in.pad(padding, value);
  }

  static void Eval(const Eigen::GpuDevice& dev, OutType32BitIndex out,
                   const InType32BitIndex& in, const Array32Bit& padding,
                   const T value) {
    out.device(dev) = in.pad(padding, value);
  }
};

#define INSTANTIATION(FUNCTOR, TYPE)                  \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 1>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 2>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 3>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 4>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 5>; \
  template struct FUNCTOR<Eigen::GpuDevice, TYPE, 6>
INSTANTIATION(EigenPad, int);
INSTANTIATION(EigenPad, int64_t);
INSTANTIATION(EigenPad, float);
INSTANTIATION(EigenPad, double);
INSTANTIATION(EigenPad, platform::float16);
INSTANTIATION(EigenPad, platform::bfloat16);
INSTANTIATION(EigenPad, platform::complex<float>);
INSTANTIATION(EigenPad, platform::complex<double>);
#undef INSTANTIATION

}  // namespace operators
}  // namespace paddle
