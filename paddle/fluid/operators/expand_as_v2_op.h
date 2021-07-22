/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class ExpandAsV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto target_rank = target_shape.size();
    PADDLE_ENFORCE_GE(target_rank, rank,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be greater than or equal to "
                          "the rank (%d) of the input 'x'.",
                          target_rank, rank));
    PADDLE_ENFORCE_GE(rank, 1, platform::errors::InvalidArgument(
                                   "The rank (%d) of the input 'x' for "
                                   "expand_as_v2 op must be positive.",
                                   rank));
    PADDLE_ENFORCE_LE(target_rank, MAX_RANK_SUPPORTED,
                      platform::errors::InvalidArgument(
                          "The rank (%d) of the input 'target_tensor' for "
                          "expand_as_v2 op must be less than or equal to %d.",
                          target_rank, MAX_RANK_SUPPORTED));

    switch (target_rank) {
      case 1:
        ExpandAs<1>(context);
        break;
      case 2:
        ExpandAs<2>(context);
        break;
      case 3:
        ExpandAs<3>(context);
        break;
      case 4:
        ExpandAs<4>(context);
        break;
      case 5:
        ExpandAs<5>(context);
        break;
      case 6:
        ExpandAs<6>(context);
        break;
    }
  }

 protected:
  template <int Rank>
  void ExpandAs(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<Tensor>("X");
    auto in_dims = in0->dims();
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto vec_in_dims = framework::vectorize<int>(in_dims);
    auto diff = target_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> repeat_times(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      PADDLE_ENFORCE_NE(target_shape[i], 0,
                        platform::errors::InvalidArgument(
                            "The value of target shape cannot be zero."));
      if (vec_in_dims[i] != 1) {
        PADDLE_ENFORCE_EQ(
            vec_in_dims[i], target_shape[i],
            platform::errors::InvalidArgument(
                "The value (%d) of the non-singleton dimension does not match"
                " the corresponding value (%d) in "
                "target tensor for expand_as_v2 op.",
                vec_in_dims[i], target_shape[i]));
        repeat_times[i] = 1;
      } else {
        repeat_times[i] = target_shape[i];
      }
    }
    auto* out0 = context.Output<Tensor>("Out");
    Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      bcast_dims[i] = repeat_times[i];
    }

    framework::DDim new_in_dims = framework::make_ddim(vec_in_dims);
    framework::DDim out_dims = framework::make_ddim(target_shape);

    out0->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(*in0, new_in_dims);
    out0->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, Rank>::From(*out0, out_dims);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x,
                                                                 bcast_dims);
  }
};

template <typename DeviceContext, typename T>
class ExpandAsV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    auto target_shape = context.Attr<std::vector<int>>("target_shape");
    auto x_dims = in0->dims();
    auto vec_in_dims = framework::vectorize<int>(x_dims);
    auto diff = target_shape.size() - vec_in_dims.size();
    vec_in_dims.insert(vec_in_dims.begin(), diff, 1);
    std::vector<int> repeat_times(vec_in_dims.size());
    for (size_t i = 0; i < vec_in_dims.size(); ++i) {
      repeat_times[i] = target_shape[i] / vec_in_dims[i];
    }
    std::vector<int> reshape_dims_vec;
    std::vector<int> reduce_dims_vec;
    for (size_t i = 0; i < repeat_times.size(); ++i) {
      reduce_dims_vec.push_back(reshape_dims_vec.size());
      reshape_dims_vec.push_back(repeat_times[i]);
      reshape_dims_vec.push_back(vec_in_dims[i]);
    }

    int dims = reduce_dims_vec.size();
    bool just_copy = true;
    for (size_t i = 0; i < repeat_times.size(); i++) {
      if (repeat_times[i] != 1) {
        just_copy = false;
        break;
      }
    }
    // no need reduce, just copy
    if (just_copy) {
      auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
      auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
      out0->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*in0, context.GetPlace(), context.device_context(),
                            out0);
    } else {
      PADDLE_ENFORCE_GE(dims, 1,
                        platform::errors::InvalidArgument(
                            "The rank of the input 'Out@GRAD' for "
                            "expand_as_v2_grad op must be greater than or "
                            "equal to 1, but the value received is %d.",
                            dims));
      PADDLE_ENFORCE_LE(dims, MAX_RANK_SUPPORTED,
                        platform::errors::InvalidArgument(
                            "The rank of the input 'Out@GRAD' for "
                            "expand_as_v2_grad op must be less than or equal "
                            "to %d, but the value received is %d.",
                            MAX_RANK_SUPPORTED, dims));
      switch (dims) {
        case 1:
          ExpandAsBackward<1>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 2:
          ExpandAsBackward<2>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 3:
          ExpandAsBackward<3>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 4:
          ExpandAsBackward<4>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 5:
          ExpandAsBackward<5>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 6:
          ExpandAsBackward<6>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        default:
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Only support tensor with rank being between 1 and 6. But "
              "received tensor's rank = %d.",
              dims));
      }
    }
  }

 protected:
  template <int Dims>
  void ExpandAsBackward(const framework::ExecutionContext& context,
                        const std::vector<int>& reshape_dims_vec,
                        const std::vector<int>& reduce_dims_vec) const {
    size_t reshape_size = reshape_dims_vec.size();
    size_t reduce_size = reduce_dims_vec.size();
    auto* in0 = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* out0 = context.Output<Tensor>(framework::GradVarName("X"));
    out0->mutable_data<T>(context.GetPlace());
    auto x_grad = EigenVector<T>::Flatten(*out0);
    Eigen::DSizes<Eigen::DenseIndex, Dims * 2> reshape_dims;
    for (size_t i = 0; i < reshape_size; ++i) {
      reshape_dims[i] = reshape_dims_vec[i];
    }
    Eigen::DSizes<Eigen::DenseIndex, Dims> reduce_dims;
    for (size_t i = 0; i < reduce_size; ++i) {
      reduce_dims[i] = reduce_dims_vec[i];
    }
    auto out_grad = EigenVector<T>::Flatten(*in0);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcastGrad<std::decay_t<decltype(place)>, T, Dims>::Eval(
        place, x_grad, out_grad, reduce_dims, reshape_dims);
  }
};

}  // namespace operators
}  // namespace paddle
