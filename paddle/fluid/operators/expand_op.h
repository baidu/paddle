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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"

#define MAX_RANK_SUPPORTED 6

namespace paddle {
namespace operators {
inline std::vector<int> get_expand_times(
    const framework::ExecutionContext& ctx) {
  if (ctx.HasInput("ExpandTimes")) {
    auto* expand_tensor = ctx.Input<framework::LoDTensor>("ExpandTimes");
    auto* expand_data = expand_tensor->data<int>();
    framework::Tensor cpu_expand_tensor;
    if (platform::is_gpu_place(expand_tensor->place())) {
      TensorCopySync(*expand_tensor, platform::CPUPlace(), &cpu_expand_tensor);
      expand_data = cpu_expand_tensor.data<int>();
    }
#ifdef PADDLE_WITH_ASCEND_CL
    if (platform::is_npu_place(expand_tensor->place())) {
      TensorCopySync(*expand_tensor, platform::CPUPlace(), &cpu_expand_tensor);
      expand_data = cpu_expand_tensor.data<int>();
    }
#endif
#ifdef PADDLE_WITH_XPU
    if (platform::is_xpu_place(expand_tensor->place())) {
      TensorCopySync(*expand_tensor, platform::CPUPlace(), &cpu_expand_tensor);
      expand_data = cpu_expand_tensor.data<int>();
    }
#endif
    auto vec_epxand_times =
        std::vector<int>(expand_data, expand_data + expand_tensor->numel());
    return vec_epxand_times;
  }

  auto list_expand_times_tensor =
      ctx.MultiInput<framework::Tensor>("expand_times_tensor");
  if (list_expand_times_tensor.size() > 0) {
    // get tensor from
    std::vector<int> vec_epxand_times;
    for (size_t i = 0; i < list_expand_times_tensor.size(); ++i) {
      auto tensor = list_expand_times_tensor[i];
      if (platform::is_gpu_place(tensor->place())) {
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_times.push_back(*temp.data<int32_t>());
      }
#ifdef PADDLE_WITH_XPU
      else if (platform::is_xpu_place(tensor->place())) {  // NOLINT
        framework::Tensor temp;
        TensorCopySync(*tensor, platform::CPUPlace(), &temp);
        vec_epxand_times.push_back(*temp.data<int32_t>());
      }
#endif
      else {  // NOLINT
        vec_epxand_times.push_back(*tensor->data<int32_t>());
      }
    }

    return vec_epxand_times;
  } else {
    return ctx.Attr<std::vector<int>>("expand_times");
  }
}

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, size_t D, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenTensor = framework::EigenTensor<T, D, MajorType, IndexType>;
using framework::To32BitIndex;

template <typename DeviceContext, typename T>
class ExpandKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto rank = context.Input<Tensor>("X")->dims().size();
    PADDLE_ENFORCE_GE(
        rank, 1,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'x' for Op(expand) "
            "must be greater than or equal to 1, but the value received is %d.",
            rank));
    PADDLE_ENFORCE_LE(
        rank, MAX_RANK_SUPPORTED,
        platform::errors::InvalidArgument(
            "The number of dimensions of the input 'x' for Op(expand) "
            "must be less than or equal to %d, but the value received is %d.",
            MAX_RANK_SUPPORTED, rank));
    switch (rank) {
      case 1:
        Expand<1>(context);
        break;
      case 2:
        Expand<2>(context);
        break;
      case 3:
        Expand<3>(context);
        break;
      case 4:
        Expand<4>(context);
        break;
      case 5:
        Expand<5>(context);
        break;
      case 6:
        Expand<6>(context);
        break;
    }
  }

 protected:
  template <int Rank>
  void Expand(const framework::ExecutionContext& context) const {
    auto* in0 = context.Input<Tensor>("X");

    auto in_dims = in0->dims();
    auto expand_times = get_expand_times(context);
    PADDLE_ENFORCE_EQ(
        static_cast<size_t>(in_dims.size()), expand_times.size(),
        platform::errors::InvalidArgument(
            "The number of elements (%d) of 'expand_times' for "
            "Op(expand) must be equal to the number "
            "of dimensions (%d) of the input.",
            expand_times.size(), static_cast<size_t>(in_dims.size())));
    auto* out0 = context.Output<Tensor>("Out");
    Eigen::DSizes<Eigen::DenseIndex, Rank> bcast_dims;
    for (size_t i = 0; i < expand_times.size(); ++i) {
      bcast_dims[i] = expand_times[i];
    }

    framework::DDim out_dims(in_dims);
    for (size_t i = 0; i < expand_times.size(); ++i) {
      out_dims[i] *= expand_times[i];
    }

    out0->Resize(out_dims);
    auto x = EigenTensor<T, Rank>::From(*in0);
    out0->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, Rank>::From(*out0);
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    // use 32-bit index to speed up
    bool use_32bit_index = y.size() < Eigen::NumTraits<int>::highest();
    if (use_32bit_index) {
      EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(
          place, To32BitIndex(y), To32BitIndex(x), bcast_dims);
    } else {
      EigenBroadcast<std::decay_t<decltype(place)>, T, Rank>::Eval(place, y, x,
                                                                   bcast_dims);
    }
  }
};

template <typename DeviceContext, typename T>
class ExpandGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in0 = context.Input<Tensor>("X");
    // auto& expand_times = context.Attr<std::vector<int>>("expand_times");
    auto expand_times = get_expand_times(context);
    auto x_dims = in0->dims();
    // 1. reshape_dims_vec is the broadcast parameter.
    // 2. reduce_dims_vec is the dimension parameter to compute gradients. For
    //    each dimension expanded, the gradients should be summed to original
    //    size.
    std::vector<int> reshape_dims_vec;
    std::vector<int> reduce_dims_vec;
    for (size_t i = 0; i < expand_times.size(); ++i) {
      reduce_dims_vec.push_back(reshape_dims_vec.size());
      reshape_dims_vec.push_back(expand_times[i]);
      reshape_dims_vec.push_back(x_dims[i]);
    }

    int dims = reduce_dims_vec.size();

    bool just_copy = true;
    for (size_t i = 0; i < expand_times.size(); i++) {
      if (expand_times[i] != 1) {
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
      PADDLE_ENFORCE_GE(dims, 1, platform::errors::InvalidArgument(
                                     "The number of dimensions of the input "
                                     "'Out@GRAD' for Op(expand_grad)"
                                     " must be greater than or equal to 1, but "
                                     "the value received is %d.",
                                     dims));
      PADDLE_ENFORCE_LE(dims, MAX_RANK_SUPPORTED,
                        platform::errors::InvalidArgument(
                            "The number of dimensions of the input 'Out@GRAD' "
                            "for Op(expand_grad) must be less than or equal "
                            "to %d, but the value received is %d.",
                            MAX_RANK_SUPPORTED, dims));
      switch (dims) {
        case 1:
          ExpandBackward<1>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 2:
          ExpandBackward<2>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 3:
          ExpandBackward<3>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 4:
          ExpandBackward<4>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 5:
          ExpandBackward<5>(context, reshape_dims_vec, reduce_dims_vec);
          break;
        case 6:
          ExpandBackward<6>(context, reshape_dims_vec, reduce_dims_vec);
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
  void ExpandBackward(const framework::ExecutionContext& context,
                      const std::vector<int>& reshape_dims_vec,
                      const std::vector<int>& reduce_dims_vec) const {
    size_t reshape_size = reshape_dims_vec.size();
    size_t reduce_size = reduce_dims_vec.size();
    PADDLE_ENFORCE_EQ(reshape_size, reshape_dims_vec.size(),
                      platform::errors::InvalidArgument(
                          "Inconsistent size between template Dims (%d) and "
                          "reshape dimensions (%d).",
                          reshape_size, reshape_dims_vec.size()));
    PADDLE_ENFORCE_EQ(reduce_size, reduce_dims_vec.size(),
                      platform::errors::InvalidArgument(
                          "Inconsistent size between template Dims (%d) and "
                          "reduce dimensions (%d).",
                          reduce_size, reduce_dims_vec.size()));
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
