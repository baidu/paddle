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

#include <algorithm>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/pooling.h"
#if defined(__HIPCC__) || defined(__NVCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#endif

namespace paddle {
namespace operators {
template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  template <typename T2>
  HOSTDEVICE inline T2 operator()(const T2& x) const {
    return x * static_cast<T2>(n_inv);
  }

 private:
  T n_inv;
};

using Tensor = framework::Tensor;

class PoolOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class PoolOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class Pool2dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

class Pool3dOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override;
};

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings, const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const framework::DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2, paddings->size(),
                      platform::errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(), data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

template <typename T = int>
inline void UpdateKsize(std::vector<T>* ksize,
                        const framework::DDim data_dims) {
  ksize->resize(static_cast<size_t>(data_dims.size()));
  for (size_t i = 0; i < ksize->size(); ++i) {
    *(ksize->begin() + i) = static_cast<T>(data_dims[i]);
  }
}

inline int getReduceNum(const framework::Tensor& input,
                        const framework::Tensor* output,
                        const std::string data_format,
                        std::vector<int>* reduce_dim) {
  // data_format only can be NCHW
  bool channel_last = (data_format == "NHWC");
  if (channel_last) {
    return 0;
  }
  int reduce_num = 0;
  const int output_height = output->dims()[2];
  const int output_width = output->dims()[3];
  if ((output_height == 1) && (output_width == 1)) {
    reduce_dim->push_back(2);
    reduce_dim->push_back(3);
    reduce_num = input.dims()[2] * input.dims()[3];
  }
  return reduce_num;
}

template <typename DeviceContext, typename T>
class PoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::string data_format = context.Attr<std::string>("data_format");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    bool global_pooling = context.Attr<bool>("global_pooling");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }

    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
      for (int i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }
    auto& dev_ctx = context.template device_context<DeviceContext>();
    switch (ksize.size()) {
      case 2: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool2dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool2d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         true, false, out, pool_process);

        } else if (pooling_type == "avg") {
          std::vector<int> reduce_dim;
          int reduce_num = getReduceNum(*in_x, out, data_format, &reduce_dim);

          if (reduce_num > 0 &&
              adaptive) {  // for adaptive_avg_pool2d && output_size == 1
#if defined(__HIPCC__) || defined(__NVCC__)
            auto stream = dev_ctx.stream();
            TensorReduce<T, T, cub::Sum, DivideFunctor<T>>(
                *in_x, out, reduce_dim, static_cast<T>(0), cub::Sum(),
                DivideFunctor<T>(reduce_num), stream);
#else  // for cpu
            paddle::operators::math::Pool2dFunctor<
                DeviceContext, paddle::operators::math::AvgPool<T>, T>
                pool2d_forward;
            paddle::operators::math::AvgPool<T> pool_process;
            pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings,
                           data_format, exclusive, adaptive, out, pool_process);
#endif
          } else {  // avgpool_2d or  adaptive_avg_pool2d && output_size != 1
            paddle::operators::math::Pool2dFunctor<
                DeviceContext, paddle::operators::math::AvgPool<T>, T>
                pool2d_forward;
            paddle::operators::math::AvgPool<T> pool_process;
            pool2d_forward(dev_ctx, *in_x, ksize, strides, paddings,
                           data_format, exclusive, adaptive, out, pool_process);
          }
        }
      } break;
      case 3: {
        if (pooling_type == "max") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::MaxPool<T>, T>
              pool3d_forward;
          paddle::operators::math::MaxPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         true, false, out, pool_process);

        } else if (pooling_type == "avg") {
          paddle::operators::math::Pool3dFunctor<
              DeviceContext, paddle::operators::math::AvgPool<T>, T>
              pool3d_forward;
          paddle::operators::math::AvgPool<T> pool_process;
          pool3d_forward(dev_ctx, *in_x, ksize, strides, paddings, data_format,
                         exclusive, adaptive, out, pool_process);
        }
      } break;
      default: {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Pool op only supports 2D and 3D input."));
      }
    }
  }
};

template <typename DeviceContext, typename T>
class PoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* in_x = context.Input<Tensor>("X");
    const Tensor* out = context.Input<Tensor>("Out");
    const Tensor* out_grad =
        context.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = context.Output<Tensor>(framework::GradVarName("X"));

    std::string pooling_type = context.Attr<std::string>("pooling_type");
    std::vector<int> ksize = context.Attr<std::vector<int>>("ksize");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    bool exclusive = context.Attr<bool>("exclusive");
    bool adaptive = context.Attr<bool>("adaptive");
    std::string data_format = context.Attr<std::string>("data_format");
    bool global_pooling = context.Attr<bool>("global_pooling");
    std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // update paddings
    auto in_x_dims = in_x->dims();
    framework::DDim data_dims;
    if (channel_last) {
      data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    } else {
      data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
    }
    UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                  data_dims, strides, ksize);
    if (data_dims.size() * 2 == static_cast<int>(paddings.size())) {
      for (int i = 0; i < data_dims.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();
    if (in_x_grad) {
      in_x_grad->mutable_data<T>(context.GetPlace());
      paddle::operators::math::SetConstant<DeviceContext, T> set_constant;
      set_constant(dev_ctx, in_x_grad, static_cast<T>(0.0));

      switch (ksize.size()) {
        case 2: {
          if (pooling_type == "max") {
            paddle::operators::math::MaxPool2dGradFunctor<DeviceContext, T>
                pool2d_backward;
            pool2d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool2dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool2d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool2d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, exclusive, adaptive,
                            in_x_grad, pool_process);
          }
        } break;
        case 3: {
          if (pooling_type == "max") {
            paddle::operators::math::MaxPool3dGradFunctor<DeviceContext, T>
                pool3d_backward;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, in_x_grad);
          } else if (pooling_type == "avg") {
            paddle::operators::math::Pool3dGradFunctor<
                DeviceContext, paddle::operators::math::AvgPoolGrad<T>, T>
                pool3d_backward;
            paddle::operators::math::AvgPoolGrad<T> pool_process;
            pool3d_backward(dev_ctx, *in_x, *out, *out_grad, ksize, strides,
                            paddings, data_format, exclusive, adaptive,
                            in_x_grad, pool_process);
          }
        } break;
        default: {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Pool op only supports 2D and 3D input."));
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
