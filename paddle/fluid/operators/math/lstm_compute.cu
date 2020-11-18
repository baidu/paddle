/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/detail/lstm_gpu_kernel.h"
#include "paddle/fluid/operators/math/detail/lstm_kernel.h"
#include "paddle/fluid/operators/math/lstm_compute.h"

namespace paddle {
namespace operators {
namespace math {

template <class T>
struct LstmUnitFunctor<platform::CUDADeviceContext, T> {
  static void compute(const platform::CUDADeviceContext& context,
                      LstmMetaValue<T> value, int frame_size, int batch_size,
                      T cell_clip, const detail::ActivationType& gate_act,
                      const detail::ActivationType& cell_act,
                      const detail::ActivationType& cand_act,
                      bool old_api_version = true) {
    detail::gpu_lstm_forward<T>(context, detail::forward::lstm<T>(), value,
                                frame_size, batch_size, cell_clip, cand_act,
                                gate_act, cell_act);
  }
};

template <class T>
struct LstmUnitGradFunctor<platform::CUDADeviceContext, T> {
  static void compute(const platform::CUDADeviceContext& context,
                      LstmMetaValue<T> value, LstmMetaGrad<T> grad,
                      int frame_size, int batch_size, T cell_clip,
                      const detail::ActivationType& gate_act,
                      const detail::ActivationType& cell_act,
                      const detail::ActivationType& cand_act,
                      bool old_api_version = true) {
    detail::gpu_lstm_backward(context, detail::backward::lstm<T>(), value, grad,
                              frame_size, batch_size, cell_clip, cand_act,
                              gate_act, cell_act);
  }
};

template class LstmUnitFunctor<platform::CUDADeviceContext, float>;
template class LstmUnitFunctor<platform::CUDADeviceContext, double>;
template class LstmUnitGradFunctor<platform::CUDADeviceContext, float>;
template class LstmUnitGradFunctor<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
