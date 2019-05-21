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

#pragma once
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct ValueClip {
  HOSTDEVICE T operator()(const T& x) const {
    const T kThreshold = static_cast<T>(-64.);
    return x < kThreshold ? kThreshold : x;
  }
};

template <typename DeviceContext, typename T, bool is_test, typename Enable>
void SoftmaxFunctor<DeviceContext, T, is_test, Enable>::operator()(
    const DeviceContext& context, const int axis_dim,
    const framework::Tensor* X, framework::Tensor* Y) {
  auto in_dims = X->dims();
  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int num_classes = in_dims[kClassDim];
  const int batch_size = in_dims[kBatchDim];
  const int num_remain = num_classes / axis_dim;

  if (num_remain == 1 && platform::MayIUse(platform::avx)) {
    const T* in_data = X->data<T>();
    T* out_data = Y->data<T>();
    for (int bs = 0; bs < batch_size; ++bs) {
      T max_val = *std::max_element(in_data, in_data + num_classes);
      max_val *= -1;
      vec_add_bias<T, platform::avx>(num_classes, max_val, in_data, out_data);
      vec_exp<T>(num_classes, out_data, out_data);

      T sum = 0;
      vec_sum<T, platform::avx>(num_classes, out_data, &sum);
      sum = static_cast<T>(1) / sum;
      vec_scal<T, platform::avx>(num_classes, sum, out_data, out_data);

      in_data += num_classes;
      out_data += num_classes;
    }
    return;
  }

  auto logits = EigenMatrix<T>::From(*X);
  auto softmax = EigenMatrix<T>::From(*Y);

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);
  Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
  Eigen::DSizes<int, 2> one_axis(1, axis_dim);

  auto shifted_logits = (logits -
                         logits.maximum(along_class)
                             .eval()
                             .reshape(batch_by_one)
                             .broadcast(one_by_class))
                            .unaryExpr(ValueClip<T>());

  softmax.device(*context.eigen_device()) = shifted_logits.exp();
  softmax.device(*context.eigen_device()) = (softmax *
                                             softmax.reshape(batch_axis_remain)
                                                 .sum(along_class)
                                                 .inverse()
                                                 .eval()
                                                 .broadcast(one_axis));
}

template <class DeviceContext>
using enable_if_CPU = typename std::enable_if<
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type;

template <typename DeviceContext>
class SoftmaxFunctor<DeviceContext, float, true, enable_if_CPU<DeviceContext>> {
  void operator()(const DeviceContext& context, const int axis_dim,
                  const framework::Tensor* X, framework::Tensor* Y) {
    auto in_dims = X->dims();
    const float* in_data = X->data<float>();
    float* out_data = Y->data<float>();
    const int kBatchDim = 0;
    const int kClassDim = 1;
    // 2D data. Batch x C
    auto compute_softmax =
        jit::KernelFuncs<jit::SoftmaxTuple<float>, platform::CPUPlace>::Cache()
            .At(in_dims[kClassDim]);
    compute_softmax(in_data, out_data, in_dims[kClassDim], in_dims[kBatchDim],
                    in_dims[kClassDim] / axis_dim);
  }
};

template <typename DeviceContext, typename T>
void SoftmaxGradFunctor<DeviceContext, T>::operator()(
    const DeviceContext& context, const int axis_dim,
    const framework::Tensor* y, const framework::Tensor* y_grad,
    framework::Tensor* x_grad) {
  auto softmax = EigenMatrix<T>::From(*y);
  auto softmax_grad = EigenMatrix<T>::From(*y_grad);
  auto logits_grad = EigenMatrix<T>::From(*x_grad);

  const int kBatchDim = 0;
  const int kClassDim = 1;

  const int batch_size = softmax.dimension(kBatchDim);
  const int num_classes = softmax.dimension(kClassDim);
  const int num_remain = num_classes / axis_dim;

  Eigen::DSizes<int, 1> along_class(kClassDim);
  Eigen::DSizes<int, 2> batch_by_one(batch_size, 1);
  Eigen::DSizes<int, 2> one_by_class(1, num_classes);
  Eigen::DSizes<int, 3> batch_axis_remain(batch_size, axis_dim, num_remain);
  Eigen::DSizes<int, 2> one_axis(1, axis_dim);

  auto dot = (softmax * softmax_grad)
                 .reshape(batch_axis_remain)
                 .sum(along_class)
                 .eval()
                 .broadcast(one_axis);
  logits_grad.device(*context.eigen_device()) = (softmax_grad - dot) * softmax;
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
