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

#pragma once
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/math_function.h"

#define SWITCH_OUT_RANK_CASE(n)                                \
  case n: {                                                    \
    ApplyBroadcast<n>(context, in_tensors[i], out_tensors[i]); \
    break;                                                     \
  }

namespace paddle {
namespace operators {

using framework::Tensor;
using framework::DDim;
using framework::EigenTensor;

template <typename DeviceContext, typename T>
class BroadcastTensorsOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto& in_tensors = context.MultiInput<Tensor>("X");
    auto out_tensors = context.MultiOutput<Tensor>("Out");

    size_t num_ins = in_tensors.size();

    PADDLE_ENFORCE_GT(
        num_ins, 1,
        platform::errors::InvalidArgument(
            "Expected at least 2 input tensors, but only received d%.",
            in_tensors.size()));

    PADDLE_ENFORCE_EQ(
        num_ins, out_tensors.size(),
        platform::errors::InvalidArgument(
            "BroadcastTensorsOp expects equal number of inputs and outputs,"
            "but received: %d inputs v.s %d outputs",
            num_ins, out_tensors.size()));

    // Eigen has no support for dynamic ranked tensor
    // Thus we perform static expansion for each possible ranks
    for (size_t i = 0; i < num_ins; i++) {
      int out_rank = out_tensors[i]->dims().size();
      switch (out_rank) {
        SWITCH_OUT_RANK_CASE(1)
        SWITCH_OUT_RANK_CASE(2)
        SWITCH_OUT_RANK_CASE(3)
        SWITCH_OUT_RANK_CASE(4)
        SWITCH_OUT_RANK_CASE(5)
        default: {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Target tensor rank out of range"
              "Maximum supported rank for broadcast is: 5"));
        }
      }
    }
  }

  template <int OutRank>
  void ApplyBroadcast(const framework::ExecutionContext& context,
                      const Tensor* input_tensor, Tensor* output_tensor) const {
    const auto& input_dims = input_tensor->dims();
    const auto& output_dims = output_tensor->dims();

    int in_rank = input_dims.size();
    int out_rank = output_dims.size();

    // 1. Collect bcast_dims, each element of which indicates how many
    // times we need to replicate along the corresponding dimension
    // 2. Collect new_input_dims_vec. Eigen::broadcast requires same rank for
    // both input and output tensors, so we need to initialize input X with
    // expanded dims: "new_input_dims_vec"
    Eigen::DSizes<Eigen::DenseIndex, OutRank> bcast_dims;
    std::vector<int64_t> new_input_dims_vec(out_rank);
    for (int j = 0; j < out_rank; j++) {
      int out_axis = out_rank - j - 1;
      int in_axis = in_rank - j - 1;

      bcast_dims[out_axis] = output_dims[out_axis];
      new_input_dims_vec[out_axis] = 1;
      if (in_axis >= 0 && input_dims[in_axis] == output_dims[out_axis]) {
        bcast_dims[out_axis] = 1;
        new_input_dims_vec[out_axis] = input_dims[in_axis];
      }
    }
    auto new_input_dims = framework::make_ddim(new_input_dims_vec);

    // Initialize input X with new_input_dims_vec, so it's rank-aligned with the
    // output
    auto x = EigenTensor<T, OutRank>::From(*input_tensor, new_input_dims);

    output_tensor->mutable_data<T>(context.GetPlace());
    auto y = EigenTensor<T, OutRank>::From(*output_tensor, output_dims);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    EigenBroadcast<std::decay_t<decltype(place)>, T, OutRank>::Eval(place, y, x,
                                                                    bcast_dims);
  }
};

#define SWITCH_RESHAPE_DIMS(n)                                                \
  case n: {                                                                   \
    Eigen::DSizes<Eigen::DenseIndex, n> reshape_dims;                         \
    for (size_t i = 0; i < reshape_dims_vec.size(); ++i) {                    \
      reshape_dims[i] = reshape_dims_vec[i];                                  \
    }                                                                         \
    dX.device(place) =                                                        \
        dOut.reshape(reshape_dims).sum(reduce_dims).reshape(dX.dimensions()); \
    break;                                                                    \
  }

#define UPPER_SWITCH_REDUCE_DIMS(m)                       \
  case m: {                                               \
    Eigen::DSizes<Eigen::DenseIndex, m> reduce_dims;      \
    for (size_t i = 0; i < reduce_dims_vec.size(); ++i) { \
      reduce_dims[i] = reduce_dims_vec[i];                \
    }                                                     \
    switch (reshape_size) {
#define LOWER_SWITCH_REDUCE_DIMS                             \
  default: {                                                 \
    PADDLE_THROW(platform::errors::InvalidArgument(          \
        "Detected reshape size: %d out of range"             \
        "Minimum value should be larger than reduce size %d" \
        "While maximum supported is: 5",                     \
        reshape_size, reduce_size));                         \
  }                                                          \
    }                                                        \
    break;                                                   \
    }

/* ----- GradOpKernel ----- */
template <typename DeviceContext, typename T>
class BroadcastTensorsGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // Find reduce dimensions
    const auto& in_tensors =
        context.MultiInput<Tensor>(framework::GradVarName("Out"));
    auto out_tensors = context.MultiOutput<Tensor>(framework::GradVarName("X"));

    size_t num_ins = in_tensors.size();

    PADDLE_ENFORCE_GT(
        num_ins, 1,
        platform::errors::InvalidArgument(
            "Expected at least 2 input tensors, but only received d%.",
            in_tensors.size()));

    PADDLE_ENFORCE_EQ(
        num_ins, out_tensors.size(),
        platform::errors::InvalidArgument(
            "BroadcastTensorsOp expects equal number of inputs and outputs,"
            "but received: %d inputs v.s %d outputs",
            num_ins, out_tensors.size()));

    // For each In-Out tensor pair,
    // Prepare and apply broadcast dims array
    for (size_t i = 0; i < num_ins; i++) {
      const auto* input_tensor = in_tensors[i];
      auto* output_tensor = out_tensors[i];

      const auto& input_dims = input_tensor->dims();
      const auto& output_dims = output_tensor->dims();

      int in_rank = input_dims.size();
      int out_rank = output_dims.size();

      // BroadcastTensorsGrad is simply a reduce_sum along broadcasted axes
      // Here we perform the following Eigen operations:
      // dOut(Flattened) -> reshape(reshape_dims) -> reduce(reduce_dims) ->
      // reshape(dX_shape) -> dX
      // Note the last "reshape(dX_shape)" will be performed implicitly,
      // and we only need to collect reduce_dims and reshape_dims
      std::vector<int> reduce_dims_vec;
      std::vector<int> reshape_dims_vec;
      for (int j = 0; j < in_rank; j++) {
        int out_axis = out_rank - j - 1;
        int in_axis = in_rank - j - 1;

        reshape_dims_vec.push_back(input_dims[j]);
        if (out_axis < 0 || output_dims[out_axis] != input_dims[in_axis]) {
          reduce_dims_vec.push_back(in_axis);
        }
      }

      size_t reduce_size = reduce_dims_vec.size();
      size_t reshape_size = reshape_dims_vec.size();
      bool just_copy = (reduce_dims_vec.size() == 0);
      output_tensor->mutable_data<T>(context.GetPlace());
      if (just_copy) {
        // If this turns out to be a No-Op, simply perform a tensor copy
        framework::TensorCopy(*input_tensor, context.GetPlace(),
                              context.device_context(), output_tensor);
      } else {
        PADDLE_ENFORCE_GE(reduce_dims_vec.size(), 1,
                          platform::errors::InvalidArgument(
                              "The number of dimensions of the input "
                              "'Out@GRAD' for Op(broadcast_tensors)"
                              " must be greater than or equal to 1, but "
                              "the value received is %d.",
                              reduce_dims_vec.size()));
        PADDLE_ENFORCE_LE(
            reduce_dims_vec.size(), 5,
            platform::errors::InvalidArgument(
                "The number of dimensions of the input 'Out@GRAD' "
                "for Op(broadcast_tensors) must be less than or equal "
                "to 5, but the value received is %d.",
                reduce_dims_vec.size()));

        // Overall:
        // dOut(Flattened) -> reshape(reshape_dims) -> reduce(reduce_dims) ->
        // reshape(dX_shape) -> dX
        auto dX = framework::EigenVector<T>::Flatten(*output_tensor);
        auto dOut = framework::EigenVector<T>::Flatten(*input_tensor);
        auto& place =
            *context.template device_context<DeviceContext>().eigen_device();

        // Expand ReduceSize and ReshapeSize into static values
        switch (reduce_size) {
          UPPER_SWITCH_REDUCE_DIMS(1)
          SWITCH_RESHAPE_DIMS(1)
          SWITCH_RESHAPE_DIMS(2)
          SWITCH_RESHAPE_DIMS(3)
          SWITCH_RESHAPE_DIMS(4)
          SWITCH_RESHAPE_DIMS(5)
          LOWER_SWITCH_REDUCE_DIMS

          UPPER_SWITCH_REDUCE_DIMS(2)
          SWITCH_RESHAPE_DIMS(2)
          SWITCH_RESHAPE_DIMS(3)
          SWITCH_RESHAPE_DIMS(4)
          SWITCH_RESHAPE_DIMS(5)
          LOWER_SWITCH_REDUCE_DIMS

          UPPER_SWITCH_REDUCE_DIMS(3)
          SWITCH_RESHAPE_DIMS(3)
          SWITCH_RESHAPE_DIMS(4)
          SWITCH_RESHAPE_DIMS(5)
          LOWER_SWITCH_REDUCE_DIMS

          UPPER_SWITCH_REDUCE_DIMS(4)
          SWITCH_RESHAPE_DIMS(4)
          SWITCH_RESHAPE_DIMS(5)
          LOWER_SWITCH_REDUCE_DIMS

          UPPER_SWITCH_REDUCE_DIMS(5)
          SWITCH_RESHAPE_DIMS(5)
          LOWER_SWITCH_REDUCE_DIMS

          default: {
            PADDLE_THROW(platform::errors::InvalidArgument(
                "Detected reduce size: %d out of range"
                "While maximum supported is: 5",
                reduce_size));
          }
        }
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
