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
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/hostdevice.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace operators {
namespace math {

/*
 * \brief Extracting simple operations from pooling.
 *        Both MaxPool and AvgPool need "initial", "compute" and "finalize"
 * operation.
 *        MaxPool initializes temp variable to the negative maximum to find the
 * maximum value in the pooling field.
 *        AvgPool initializes temp variable to the zero to accumulate all values
 * in pool pooling, and finally takes the average.
 *        MaxPoolGrad and AvgPoolGrad are gradient operations respectively.
 */
template <class T>
class MaxPool {
 public:
  DEVICE inline T initial() { return static_cast<T>(-FLT_MAX); }
  DEVICE inline void compute(const T& x, T* y) { *y = *y > x ? *y : x; }
  DEVICE inline void finalize(const T& pool_field, T* y) {}
};

template <class T>
class AvgPool {
 public:
  T res;
  double intermediate_res;
  DEVICE inline T initial() {
    intermediate_res = static_cast<double>(0);
    return static_cast<T>(0);
  }
  DEVICE inline void compute(const T& x, T* y) {
    intermediate_res += static_cast<double>(x);
  }
  DEVICE inline void finalize(const T& pool_field, T* y) {
    *y = static_cast<T>(intermediate_res / static_cast<double>(pool_field));
  }
};

template <class T>
class MaxPoolGrad {
 public:
  DEVICE inline void compute(const T& x, const T& y, const T& dy, T scale,
                             T* dx) {
    *dx += dy * static_cast<T>(x == y);
  }
};

template <class T>
class AvgPoolGrad {
 public:
  DEVICE inline void compute(const T& x, const T& y, const T& dy, T scale,
                             T* dx) {
    *dx += (scale * dy);
  }
};

/* used for adaptive pool to calculate start and end index of each divided grid
 */
HOSTDEVICE inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

HOSTDEVICE inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

/*
 * \brief Getting pooling results, and calculating gradient.
 *
 * In pool2d, all Tensors are in NCHW or NHWC format. Where N is batch size, C
 * is the number of channels, H and W is the height and width of feature.
 * In pool3d, all Tensors are in NCDHW or NDHWC format. Where N is batch size, C
 * is the number of channels, D, H and W is the depth, height and width of
 * feature.
 *
 * In max pooling, it is possible that the pooling region has multiple maximum
 * elements. In this case, we should compute the gradient of the first maximum
 * element.
 * This is different from average pooling. So we rewrite the max_pool_grad:
 * MaxPool2dGradFunctor, MaxPool3dGradFunctor.
 */
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename PoolProcess, typename T>
class Pool2dDirectCUDAFunctor {
 public:
  void operator()(const T* input, const std::vector<int>& input_shape,
                  const std::vector<int>& output_shape,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, T* output, gpuStream_t stream,
                  PoolProcess pool_compute);
};
#endif

template <typename DeviceContext, typename PoolProcess, typename T>
class Pool2dFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* output,
                  PoolProcess pool_compute);

  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* output, PoolProcess pool_compute);
};

template <typename DeviceContext, typename PoolProcess, typename T>
class Pool2dGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* input_grad,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* input_grad, PoolProcess pool_compute);
};

template <typename DeviceContext, class T>
class MaxPool2dGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* input_grad);
  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, framework::Tensor* input_grad);
};

template <typename DeviceContext, typename PoolProcess, typename T>
class Pool3dFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* output,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* output, PoolProcess pool_compute);
};

template <typename DeviceContext, typename PoolProcess, typename T>
class Pool3dGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool exclusive,
                  bool adaptive, framework::Tensor* input_grad,
                  PoolProcess pool_compute);
  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, bool exclusive, bool adaptive,
                  framework::Tensor* input_grad, PoolProcess pool_compute);
};

template <typename DeviceContext, class T>
class MaxPool3dGradFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  framework::Tensor* input_grad);
  // overload operator() to support argument data_format
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const framework::Tensor& output,
                  const framework::Tensor& output_grad,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string data_format, framework::Tensor* input_grad);
};

/*
 * \brief Getting max pooling results and corresponding max index, and
 * calculating gradient.
 * In up-sampling-pooling, it is necessary to know max element index.
 * In pool2d, all tensors are in NCHW format. In pool3d, all tensors are in
 * NCDHW format.
 */
template <typename DeviceContext, typename T1, typename T2>
class MaxPool2dWithIndexFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask);
};

template <typename DeviceContext, typename T1, typename T2>
class MaxPool2dWithIndexGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad);
};

template <typename DeviceContext, typename T1, typename T2>
class MaxPool3dWithIndexFunctor {
 public:
  void operator()(const DeviceContext& context, const framework::Tensor& input,
                  const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* output, framework::Tensor* mask);
};

template <typename DeviceContext, typename T1, typename T2>
class MaxPool3dWithIndexGradFunctor {
 public:
  void operator()(const DeviceContext& context,
                  const framework::Tensor& output_grad,
                  const framework::Tensor& mask, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, bool adaptive,
                  framework::Tensor* input_grad);
};

}  // namespace math
}  // namespace operators
}  // namespace paddle
