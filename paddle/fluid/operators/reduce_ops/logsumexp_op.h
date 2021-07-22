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

#include <algorithm>
#include <vector>
#include "paddle/fluid/operators/reduce_ops/reduce_op_function.h"

namespace paddle {
namespace operators {

#define HANDLE_DIM(NDIM, RDIM)                                            \
  if (ndim == NDIM && rdim == RDIM) {                                     \
    ReduceFunctor<DeviceContext, OutT, NDIM, RDIM, LogsumexpFunctor>(     \
        context.template device_context<DeviceContext>(), *input, output, \
        axis, keepdim);                                                   \
  }

struct LogsumexpFunctor {
  template <typename DeviceContext, typename X, typename Y, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, const Dim& dim) {
    auto x_dim = x->dimensions();
    auto t_dim = x_dim;
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      t_dim[dim[i]] = 1;
    }

    auto r_dim = x_dim;
    for (int i = 0; i < static_cast<int>(r_dim.size()); i++) {
      r_dim[i] = 1;
    }
    for (int i = 0; i < static_cast<int>(dim.size()); i++) {
      r_dim[dim[i]] = x_dim[dim[i]];
    }

    auto y_dim = y->dimensions();
    auto x_max = x->maximum(dim).eval();
    y->device(place) =
        (x_max +
         (*x - x_max.reshape(t_dim).broadcast(r_dim)).exp().sum(dim).log())
            .reshape(y_dim);
  }
};

struct LogsumexpGradFunctor {
  template <typename DeviceContext, typename X, typename Y, typename DX,
            typename DY, typename Dim>
  void operator()(const DeviceContext& place, X* x, Y* y, DX* dx, DY* dy,
                  const Dim& dim, int size) {
    dx->device(place) = dy->broadcast(dim) * (*x - y->broadcast(dim)).exp();
  }
};

template <typename DeviceContext, typename OutT>
class LogsumexpKernel : public framework::OpKernel<OutT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    output->mutable_data<OutT>(context.GetPlace());

    auto axis = context.Attr<std::vector<int>>("axis");
    auto keepdim = context.Attr<bool>("keepdim");
    auto reduce_all = context.Attr<bool>("reduce_all");

    const auto& input_dim_size = input->dims().size();
    // The dims has full dim, set the reduce_all is True
    reduce_all |= (static_cast<const int>(axis.size()) == input_dim_size);

    if (reduce_all) {
      // Flatten and reduce 1-D tensor
      auto x = EigenVector<OutT>::Flatten(*input);
      auto out = EigenScalar<OutT>::From(*output);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto reduce_dim = Eigen::array<int, 1>({{0}});
      LogsumexpFunctor()(place, &x, &out, reduce_dim);
    } else {
      int ndim = input_dim_size;
      int rdim = axis.size();
      // comments for accelerating compiling temporarily.
      // HANDLE_DIM(6, 5);
      // HANDLE_DIM(6, 4);
      // HANDLE_DIM(6, 3);
      // HANDLE_DIM(6, 2);
      // HANDLE_DIM(6, 1);
      // HANDLE_DIM(5, 4);
      // HANDLE_DIM(5, 3);
      // HANDLE_DIM(5, 2);
      // HANDLE_DIM(5, 1);
      HANDLE_DIM(4, 3);
      HANDLE_DIM(4, 2);
      HANDLE_DIM(4, 1);
      HANDLE_DIM(3, 2);
      HANDLE_DIM(3, 1);
      HANDLE_DIM(2, 1);
    }
  }
};

template <typename DeviceContext, typename T>
class LogsumexpGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Input<Tensor>("Out");
    auto* output_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = context.Output<Tensor>(framework::GradVarName("X"));
    input_grad->mutable_data<T>(context.GetPlace());

    auto axis = context.Attr<std::vector<int>>("axis");
    auto reduce_all = context.Attr<bool>("reduce_all");
    const auto input_dim_size = context.Input<Tensor>("X")->dims().size();
    reduce_all |= (static_cast<const int>(axis.size()) == input_dim_size);

    if (reduce_all) {
      auto x = EigenVector<T>::Flatten(*input);
      auto y = EigenVector<T>::Flatten(*output);
      auto dy = EigenVector<T>::Flatten(*output_grad);
      auto dx = EigenVector<T>::Flatten(*input_grad);
      auto& place =
          *context.template device_context<DeviceContext>().eigen_device();
      auto broadcast_dim =
          Eigen::array<int, 1>({{static_cast<int>(input->numel())}});
      LogsumexpGradFunctor()(place, &x, &y, &dx, &dy, broadcast_dim,
                             broadcast_dim[0]);
    } else {
      int rank = input->dims().size();
      switch (rank) {
        case 1:
          ReduceGradFunctor<DeviceContext, T, 1, LogsumexpGradFunctor>(
              context.template device_context<DeviceContext>(), *input, *output,
              *output_grad, input_grad, axis);
          break;
        case 2:
          ReduceGradFunctor<DeviceContext, T, 2, LogsumexpGradFunctor>(
              context.template device_context<DeviceContext>(), *input, *output,
              *output_grad, input_grad, axis);
          break;
        case 3:
          ReduceGradFunctor<DeviceContext, T, 3, LogsumexpGradFunctor>(
              context.template device_context<DeviceContext>(), *input, *output,
              *output_grad, input_grad, axis);
          break;
        case 4:
          ReduceGradFunctor<DeviceContext, T, 4, LogsumexpGradFunctor>(
              context.template device_context<DeviceContext>(), *input, *output,
              *output_grad, input_grad, axis);
          break;
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
