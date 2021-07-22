//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/assign_value_op.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_function.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/slice_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using Variable = framework::Variable;
using LoDTensorArray = framework::LoDTensorArray;
using DDim = framework::DDim;

inline std::string GetValueName(framework::proto::VarType::Type data_type) {
  std::string value_name;
  switch (data_type) {
    case framework::proto::VarType::INT32:
      value_name = "int32_values";
      break;
    case framework::proto::VarType::INT64:
      value_name = "int64_values";
      break;
    case framework::proto::VarType::FP32:
      value_name = "fp32_values";
      break;
    case framework::proto::VarType::FP64:
      value_name = "fp64_values";
      break;
    case framework::proto::VarType::BOOL:
      value_name = "bool_values";
      break;

    default:
      PADDLE_THROW(platform::errors::Unimplemented(
          "Unsupported data type(code %d) for SetValue operator, only "
          "supports bool, int32, float32 and int64.",
          data_type));
  }
  return value_name;
}

template <typename DeviceContext, typename T>
class SetValueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const int rank = ctx.Input<framework::LoDTensor>("Input")->dims().size();

    // TODO(liym27): A more elegent code to do this. C++ has to make template
    //  integer as constant, but we had better have alternative writing in the
    //  future.
    switch (rank) {
      case 1:
        SetValueCompute<1>(ctx);
        break;
      case 2:
        SetValueCompute<2>(ctx);
        break;
      case 3:
        SetValueCompute<3>(ctx);
        break;
      case 4:
        SetValueCompute<4>(ctx);
        break;
      case 5:
        SetValueCompute<5>(ctx);
        break;
      case 6:
        SetValueCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.", rank));
    }
  }

 private:
  template <size_t D>
  void SetValueCompute(const framework::ExecutionContext& ctx) const {
    auto* in = ctx.Input<framework::LoDTensor>("Input");
    auto* value_tensor = ctx.Input<framework::LoDTensor>("ValueTensor");
    auto* out = ctx.Output<framework::LoDTensor>("Out");

    auto starts_tensor_list =
        ctx.MultiInput<framework::Tensor>("StartsTensorList");
    auto ends_tensor_list = ctx.MultiInput<framework::Tensor>("EndsTensorList");
    auto steps_tensor_list =
        ctx.MultiInput<framework::Tensor>("StepsTensorList");

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    auto starts = ctx.Attr<std::vector<int64_t>>("starts");
    auto ends = ctx.Attr<std::vector<int64_t>>("ends");
    auto steps = ctx.Attr<std::vector<int64_t>>("steps");
    auto shape = ctx.Attr<std::vector<int64_t>>("shape");
    auto decrease_axes = ctx.Attr<std::vector<int64_t>>("decrease_axes");

    auto dtype = in->type();
    if (!starts_tensor_list.empty()) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }
    if (!ends_tensor_list.empty()) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }
    if (!steps_tensor_list.empty()) {
      steps = GetDataFromTensorList<int64_t>(steps_tensor_list);
    }

    auto in_dims = in->dims();
    CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends, &steps);
    auto slice_dims = GetSliceDims(in_dims, axes, starts, ends, &steps);
    auto decrease_slice_dims = GetDecreasedDims(slice_dims, decrease_axes);

    auto place = ctx.GetPlace();
    auto& eigen_place =
        *ctx.template device_context<DeviceContext>().eigen_device();

    // Here copy data from input to avoid data loss at PE and Graph level.
    // TODO(liym27): Speed up in the future version.
    // - Q: Why don't call ShareDataWith to speed up?
    // - A: Because it's not supported to ShareDataWith on OP's input and output
    // https://github.com/PaddlePaddle/Paddle/wiki/ShareDataWith-and-ShareBufferWith-are-prohibited-in-OP
    // - Q: Why don't delete Input, after all, the input and output are the same
    // Tensor at program level?
    // - A: If deleting Input, the graph will be complex, such as there will
    // be two ops points to the output in graph: op1 -> output <- set_value.
    // In this case, we have to find a way to handle the running order of
    // set_value is what we want.
    TensorCopy(*in, place, out);

    Tensor slice_tensor(dtype), pad_tensor(dtype);
    slice_tensor.mutable_data<T>(slice_dims, place);
    pad_tensor.mutable_data<T>(in_dims, place);

    auto pad_e = framework::EigenTensor<T, D>::From(pad_tensor, in_dims);
    auto out_e = framework::EigenTensor<T, D>::From(*out);
    auto slice_e = framework::EigenTensor<T, D>::From(slice_tensor, slice_dims);

    // Step 1: Set the value of out at `_index` to zero
    slice_e.device(eigen_place) = slice_e.constant(T(0));

    auto starts_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto ends_indices = Eigen::DSizes<Eigen::DenseIndex, D>();
    auto strides_indices = Eigen::DSizes<Eigen::DenseIndex, D>();

    for (size_t i = 0; i < D; ++i) {
      starts_indices[i] = 0;
      ends_indices[i] = slice_dims[i];
      strides_indices[i] = 1;
    }
    for (size_t i = 0; i < axes.size(); i++) {
      int axis_index = axes[i];
      starts_indices[axis_index] = starts[i];
      ends_indices[axis_index] = ends[i];
      strides_indices[axis_index] = steps[i];
    }

    out_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 2: Set a tensor with the same shape as out tensor. And its data at
    // '_index' is the same as value_tensor, and data out of '_index' to zero

    // - Step 2.1 Set slice tensor with value

    // NOTE(liym27): [ Why resize slice_tensor here? ]
    // A: When do broadcasting on slice_tensor and value_tensor, the shape of
    // slice_tensor should be decreased dims.
    // e.g.
    //  x[:,0] = value_tensor
    // x's shape = [3, 4], value_tensor's shape = [3]
    // We get slice_dims = [3, 1],  decrease_slice_dims = [3]
    // If do broadcasting on Tensor with shape [3, 1] and [3], the result's
    // shape is [3, 3], which cross the border;
    // If do broadcasting on Tensor with shape [3] and [3], the result's shape
    // is [3], which is right.

    slice_tensor.Resize(decrease_slice_dims);
    if (value_tensor != nullptr) {
      // ElementwiseComputeEx can do broadcasting
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_tensor, value_tensor, -1, SubFunctor<T>(), &slice_tensor);
    } else {
      Tensor value_t(dtype);
      auto value_dims = framework::make_ddim(shape);
      value_t.mutable_data<T>(value_dims, place);
      auto value_name = GetValueName(dtype);
      CopyVecotorToTensor<T>(value_name.c_str(), &value_t, ctx);
      value_t.Resize(value_dims);
      ElementwiseComputeEx<SubFunctor<T>, DeviceContext, T>(
          ctx, &slice_tensor, &value_t, -1, SubFunctor<T>(), &slice_tensor);
    }
    slice_tensor.Resize(slice_dims);

    // - Step 2.2 Pad slice tensor with 0
    pad_e.device(eigen_place) = pad_e.constant(T(0));
    pad_e.stridedSlice(starts_indices, ends_indices, strides_indices)
        .device(eigen_place) = slice_e;

    // Step 3: Set out tensor with value_tensor
    out_e.device(eigen_place) = out_e - pad_e;
  }
};

template <typename DeviceContext, typename T>
class SetValueGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar(framework::GradVarName("Out"));
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    int rank = is_tensor_array ? 1 : ctx.Input<Tensor>(framework::GradVarName("Out"))->dims().size();

    switch (rank) {
      case 1:
        SliceCompute<1>(ctx);
        break;
      case 2:
        SliceCompute<2>(ctx);
        break;
      case 3:
        SliceCompute<3>(ctx);
        break;
      case 4:
        SliceCompute<4>(ctx);
        break;
      case 5:
        SliceCompute<5>(ctx);
        break;
      case 6:
        SliceCompute<6>(ctx);
        break;
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The rank of input should be less than 7, but received %d.", rank));
    }
  }

 private:
  template <size_t D>
  void SliceCompute(const framework::ExecutionContext& ctx) const {
    const Variable* input_var = ctx.InputVar(framework::GradVarName("Out"));
    // Variable* out_var = ctx.OutputVar(framework::GradVarName("VaueTensor"));
    bool input_is_array = input_var->IsType<LoDTensorArray>();
    // bool out_is_array = out_var->IsType<LoDTensorArray>();

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int64_t> axes(axes_int.begin(), axes_int.end());
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto infer_flags = ctx.Attr<std::vector<int>>("infer_flags");

    // Step 1: Get the accurate attribute value of starts and ends
    auto starts_tensor_list = ctx.MultiInput<Tensor>("StartsTensorList");
    if (ctx.HasInput("StartsTensor")) {
      starts = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("StartsTensor"));
    } else if (starts_tensor_list.size() > 0) {
      starts = GetDataFromTensorList<int64_t>(starts_tensor_list);
    }

    auto ends_tensor_list = ctx.MultiInput<Tensor>("EndsTensorList");
    if (ctx.HasInput("EndsTensor")) {
      ends = GetDataFromTensor<int64_t>(ctx.Input<Tensor>("EndsTensor"));
    } else if (ends_tensor_list.size() > 0) {
      ends = GetDataFromTensorList<int64_t>(ends_tensor_list);
    }

    PADDLE_ENFORCE_EQ(
        starts.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of starts must be equal to the size of axes."));
    PADDLE_ENFORCE_EQ(
        ends.size(), axes.size(),
        platform::errors::InvalidArgument(
            "The size of ends must be equal to the size of axes."));

    // Step 2: Compute output
    if (input_is_array) {
      // DealTensorArray(ctx, starts, ends, out_is_array);
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Input of set_value_grad OP should be Tensor, but received TensorArray."));
      return;
    } else {
      auto in = ctx.Input<Tensor>(framework::GradVarName("Out"));
      auto out = ctx.Output<Tensor>(framework::GradVarName("ValueTensor"));

      auto in_dims = in->dims();
      auto out_dims = out->dims();
      auto slice_dims = out_dims;

      // 2.1 Infer output dims
      for (size_t i = 0; i < axes.size(); ++i) {
        // when start == -1 && end == start+1
        if (starts[i] == -1 && ends[i] == 0 && infer_flags[i] == -1) {
          auto ret =
              std::find(decrease_axis.begin(), decrease_axis.end(), axes[i]);
          if (ret != decrease_axis.end()) {
            ends[i] = in_dims[axes[i]];
          }
        }
      }

      CheckAndUpdateSliceAttrs(in_dims, axes, &starts, &ends);
      slice_dims =
          GetSliceDims<int64_t>(in_dims, axes, starts, ends, nullptr, nullptr);
      out_dims = GetDecreasedDims(slice_dims, decrease_axis);

      // 2.2 Get output
      auto offsets = Eigen::DSizes<Eigen::DenseIndex, D>();
      auto extents = Eigen::DSizes<Eigen::DenseIndex, D>();

      for (size_t i = 0; i < D; ++i) {
        offsets[i] = 0;
        extents[i] = slice_dims[i];
      }
      for (size_t i = 0; i < axes.size(); ++i) {
        offsets[axes[i]] = starts[i];
      }

      out->Resize(slice_dims);
      out->mutable_data<T>(ctx.GetPlace());

      auto in_t = framework::EigenTensor<T, D>::From(*in, in_dims);
      auto out_t = framework::EigenTensor<T, D>::From(*out, slice_dims);
      auto& eigen_place =
          *ctx.template device_context<DeviceContext>().eigen_device();

      if (in->numel() <= Eigen::NumTraits<int>::highest()) {
        // similar to tf.slice:
        // if element number less than INT_MAX, change the type of index to int
        Eigen::DSizes<int, D> offsets_32bit, extents_32bit;
        for (size_t i = 0; i < D; i++) {
          offsets_32bit[i] = offsets[i];
          extents_32bit[i] = extents[i];
        }
        EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
            eigen_place, framework::To32BitIndex(out_t),
            framework::To32BitIndex(in_t), offsets_32bit, extents_32bit);
      } else {
        EigenSlice<std::decay_t<decltype(eigen_place)>, T, D>::Eval(
            eigen_place, out_t, in_t, offsets, extents);
      }

      out->Resize(out_dims);
      // update gradient of Input
      auto input_grad = ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"));
      auto output_grad = ctx.Output<framework::LoDTensor>(framework::GradVarName("Input"));
      TensorCopy(*input_grad, ctx.GetPlace(), output_grad);
    }
  }
  
  
};

}  // namespace operators
}  // namespace paddle
