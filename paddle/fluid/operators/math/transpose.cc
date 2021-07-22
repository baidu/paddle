/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/transpose.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
class TransposeFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& in, framework::Tensor* out,
                  const std::vector<int>& axis) {
    int ndims = axis.size();
    TransCompute<platform::CPUDeviceContext, T>(ndims, context, in, out, axis);
  }
};

template class TransposeFunctor<platform::CPUDeviceContext, float>;
template class TransposeFunctor<platform::CPUDeviceContext, double>;
template class TransposeFunctor<platform::CPUDeviceContext, int32_t>;
template class TransposeFunctor<platform::CPUDeviceContext, int64_t>;
template class TransposeFunctor<platform::CPUDeviceContext,
                                paddle::platform::complex64>;
template class TransposeFunctor<platform::CPUDeviceContext,
                                paddle::platform::complex128>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
