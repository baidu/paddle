/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/eigen/eigen_function.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/slice_utils.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using Variable = framework::Variable;
using LoDTensorArray = framework::LoDTensorArray;
using DDim = framework::DDim;

inline void DealTensorArray(const framework::ExecutionContext& ctx,
                            const std::vector<int64_t>& starts,
                            const std::vector<int64_t>& ends,
                            bool out_is_array) {
  auto in_array = ctx.Input<LoDTensorArray>("Input");
  // If the input is LoDTensorArray, the rank of input is 1.
  int64_t in_size = in_array->size();
  int64_t start = starts[0] < 0 ? (starts[0] + in_size) : starts[0];
  int64_t end = ends[0] < 0 ? (ends[0] + in_size) : ends[0];

  start = std::max(start, static_cast<int64_t>(0));
  end = std::max(end, static_cast<int64_t>(0));
  end = std::min(end, in_size);

  PADDLE_ENFORCE_GT(end, start,
                    platform::errors::InvalidArgument(
                        "Attr(ends) should be greater than attr(starts) in "
                        "slice op. But received end = %d, start = %d.",
                        ends[0], starts[0]));
  int64_t out_size = end - start;

  if (out_is_array) {
    auto out_array = ctx.Output<LoDTensorArray>("Out");
    out_array->resize(out_size);

    for (int i = 0; i < out_size; ++i) {
      auto* out_tensor = &out_array->at(i);
      auto in_tensor = in_array->at(i + start);
      out_tensor->set_lod(in_tensor.lod());
      if (in_tensor.memory_size() > 0) {
        TensorCopy(in_tensor, ctx.GetPlace(), out_tensor);
      } else {
        VLOG(10) << "WARNING: The input tensor 'x_tensor' holds no memory, so "
                    "nothing has been written to output array["
                 << i << "].";
      }
    }
  } else {
    auto out = ctx.Output<Tensor>("Out");
    auto in_tensor = in_array->at(start);
    TensorCopy(in_tensor, ctx.GetPlace(), out);
  }
}

template <typename DeviceContext, typename T>
class SliceKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_tensor_array = input_var->IsType<LoDTensorArray>();
    int rank = is_tensor_array ? 1 : ctx.Input<Tensor>("Input")->dims().size();

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
    const Variable* input_var = ctx.InputVar("Input");
    Variable* out_var = ctx.OutputVar("Out");
    bool input_is_array = input_var->IsType<LoDTensorArray>();
    bool out_is_array = out_var->IsType<LoDTensorArray>();

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
      DealTensorArray(ctx, starts, ends, out_is_array);
      return;
    } else {
      auto in = ctx.Input<Tensor>("Input");
      auto out = ctx.Output<Tensor>("Out");

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
    }
  }
};

template <typename DeviceContext, typename T>
class SliceGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Variable* input_var = ctx.InputVar("Input");
    bool is_array = input_var->IsType<LoDTensorArray>();
    size_t rank = is_array ? 1 : ctx.Input<Tensor>("Input")->dims().size();

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
    auto axes = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");
    std::vector<int64_t> starts(starts_int.begin(), starts_int.end());
    std::vector<int64_t> ends(ends_int.begin(), ends_int.end());

    // Get the accurate attribute value of starts and ends
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

    Variable* d_input_var = ctx.OutputVar(framework::GradVarName("Input"));
    const Variable* d_out_var = ctx.InputVar(framework::GradVarName("Out"));
    bool d_input_is_array = d_input_var->IsType<LoDTensorArray>();
    bool d_out_is_array = d_out_var->IsType<LoDTensorArray>();

    if (d_input_is_array) {
      auto* input_array = ctx.Input<LoDTensorArray>("Input");
      auto* d_in_arr =
          ctx.Output<LoDTensorArray>(framework::GradVarName("Input"));

      int64_t d_in_size = input_array->size();
      d_in_arr->resize(d_in_size);
      // If the input is LoDTensorArray, the rank of input is 1.
      // So only use the 0th element of starts.
      int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
      start = std::max(start, static_cast<int64_t>(0));
      // set zero
      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();
      auto& dev_ctx = *pool.Get(ctx.GetPlace());
      math::SetConstant<DeviceContext, T> functor;
      for (int i = 0; i < d_in_size; ++i) {
        auto dim = input_array->at(i).dims();
        d_in_arr->at(i).Resize(dim);
        d_in_arr->at(i).mutable_data<T>(ctx.GetPlace());
        functor(reinterpret_cast<const DeviceContext&>(dev_ctx),
                &d_in_arr->at(i), static_cast<T>(0));
      }

      if (d_out_is_array) {
        auto* d_out_arr =
            ctx.Input<LoDTensorArray>(framework::GradVarName("Out"));
        int d_out_size = d_out_arr->size();
        for (int i = 0; i < d_out_size; ++i) {
          TensorCopy(d_out_arr->at(i), ctx.GetPlace(),
                     &(d_in_arr->at(start + i)));
        }
      } else {
        auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
        TensorCopy(*d_out, ctx.GetPlace(), &(d_in_arr->at(start)));
      }
      return;
    }

    auto* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* d_input = ctx.Output<Tensor>(framework::GradVarName("Input"));
    d_input->mutable_data<T>(ctx.GetPlace());

    auto out_dims = d_out->dims();
    auto in_dims = d_input->dims();

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");
    auto decrease_size = decrease_axis.size();
    if (decrease_size > 0) {
      if (decrease_size == (size_t)in_dims.size()) {
        // all dims decrease
        std::vector<int> origin_out_shape(decrease_size, 1);
        out_dims = framework::make_ddim(std::vector<int>(decrease_size, 1));
      } else {
        std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
        for (size_t i = 0; i < decrease_size; ++i) {
          origin_out_shape[decrease_axis[i]] = 1;
        }

        int index = 0;
        for (size_t i = 0; i < origin_out_shape.size(); ++i) {
          if (origin_out_shape[i] == -1) {
            origin_out_shape[i] = out_dims[index];
            ++index;
          }
        }

        out_dims = framework::make_ddim(origin_out_shape);
      }
    }

    auto offsets = Eigen::array<int64_t, D>();
    auto extents = Eigen::array<int64_t, D>();
    for (size_t i = 0; i < D; ++i) {
      offsets[i] = 0;
      extents[i] = out_dims[i];
    }

    for (size_t i = 0; i < axes.size(); ++i) {
      int axis = axes[i];
      int64_t start = starts[i] < 0 ? (starts[i] + in_dims[axis]) : starts[i];
      start = std::max(start, static_cast<int64_t>(0));
      offsets[axis] = start;
    }

    Eigen::array<std::pair<int64_t, int64_t>, D> paddings;
    for (size_t i = 0; i < paddings.size(); ++i) {
      paddings[i].first = offsets[i];
      paddings[i].second = (in_dims[i] - out_dims[i]) - offsets[i];
    }
    EigenPaddingCompute(ctx, d_input, in_dims, d_out, out_dims, paddings);
  }

  template <size_t D>
  void EigenPaddingCompute(
      const framework::ExecutionContext& context, Tensor* d_input,
      const DDim& in_dims, const Tensor* d_out, const DDim& out_dims,
      const Eigen::array<std::pair<int64_t, int64_t>, D>& paddings) const {
    if (D <= 3) {
      // if dimension less than 3, cannot reduce dimension
      LaunchEigenPadding(context, d_input, in_dims, d_out, out_dims, paddings);
    } else {  // else we can reduce dimension
      // count not-zero padding number, and record the dimension
      int need_pad_num = 0, pad_dim = -1;
      for (size_t i = 0; i < D; i++) {
        if (paddings[i].first != 0 || paddings[i].second != 0) {
          need_pad_num++;
          pad_dim = i;
        }
      }

      if (need_pad_num == 0) {
        // do not need padding, pass if data address same, else copy
        if (d_input->mutable_data<T>(context.GetPlace()) == d_out->data<T>()) {
          // inplace, do not any operator, pass
        } else {
          framework::TensorCopy(
              *d_out, context.GetPlace(),
              context.template device_context<platform::DeviceContext>(),
              d_input);
        }
      } else if (need_pad_num == 1) {
        // only need padding one dimension, we can reduce dimension.
        // only the padding dimension is available for us.
        // How to reduce dimension(5 to 3 for example):
        // before(D=5):
        // in_dims:        [x1,  x2,  x3,  x4,  x5]
        // padding.first:  [0,   0,   a,   0,  0]
        // padding.second: [0,   0,   b,   0,  0]
        //                     | |
        //                     V V
        // after(D=3):
        // reshaped_in_dims:        [x1*x2,  x3,  x4*x5]
        // reshaped_padding.first:  [0,      a,     0]
        // reshaped_padding.second: [0,      b,     0]

        if (pad_dim == D - 1) {
          // only last dimension need padding,
          // reshape the dimension of tensor in 2: [preceding, padding]
          std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

          // first dimension is the accumulate of preceding dimension
          for (int i = 0; i < pad_dim; i++) {
            in_tore_shape[0] *= in_dims[i];
            out_tore_shape[0] *= out_dims[i];
          }
          // second dimension is the padding dimension
          in_tore_shape[1] = in_dims[pad_dim];
          out_tore_shape[1] = out_dims[pad_dim];

          // convert array from std::vector to DDim
          DDim reshaped_in_dims = framework::make_ddim(in_tore_shape);
          DDim reshaped_out_dims = framework::make_ddim(out_tore_shape);

          // after reshape: the first dimension do not need padding,
          // set padding[0] zero
          reshaped_padding[0].first = reshaped_padding[0].second = 0;
          // the second dimension is the previous padding dimension
          reshaped_padding[1].first = paddings[pad_dim].first;
          reshaped_padding[1].second = paddings[pad_dim].second;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        } else if (pad_dim == 0) {
          // only first dimension need padding,
          // reshape the dimension of tensor in 2: [padding, succeeding]
          // similar to (D - 1)
          std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

          // first dimension is the padding dimension
          in_tore_shape[0] = in_dims[pad_dim];
          out_tore_shape[0] = out_dims[pad_dim];
          // sencond dimension is the accumulate of succeeding dimension
          for (size_t i = pad_dim + 1; i < D; i++) {
            in_tore_shape[1] *= in_dims[i];
            out_tore_shape[1] *= out_dims[i];
          }

          // convert array from std::vector to DDim
          DDim reshaped_in_dims = framework::make_ddim(in_tore_shape);
          DDim reshaped_out_dims = framework::make_ddim(out_tore_shape);

          // after reshape:
          // the first dimension is the previous padding dimension
          reshaped_padding[0].first = paddings[pad_dim].first;
          reshaped_padding[0].second = paddings[pad_dim].second;
          // the second dimension do not need padding, set padding[1] zero
          reshaped_padding[1].first = reshaped_padding[1].second = 0;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        } else {
          // other dimension need padding
          // reshape the dimension of tensor in 3:
          // [preceding, padding, succeeding]
          std::vector<int64_t> in_tore_shape(3, 1), out_tore_shape(3, 1);
          Eigen::array<std::pair<int64_t, int64_t>, 3> reshaped_padding;

          // first dimension is the accumulate of preceding dimension
          for (int i = 0; i < pad_dim; i++) {
            in_tore_shape[0] *= in_dims[i];
            out_tore_shape[0] *= out_dims[i];
          }
          // second dimension is the padding dimension
          in_tore_shape[1] = in_dims[pad_dim];
          out_tore_shape[1] = out_dims[pad_dim];
          // third dimension is the accumulate of succeeding dimension
          for (size_t i = pad_dim + 1; i < D; i++) {
            in_tore_shape[2] *= in_dims[i];
            out_tore_shape[2] *= out_dims[i];
          }

          // convert array from std::vector to DDim
          DDim reshaped_in_dims = framework::make_ddim(in_tore_shape);
          DDim reshaped_out_dims = framework::make_ddim(out_tore_shape);

          // after reshape:
          // the first dimension do not need padding, set padding[0] zero
          reshaped_padding[0].first = reshaped_padding[2].second = 0;
          // the second dimension is the previous padding dimension
          reshaped_padding[1].first = paddings[pad_dim].first;
          reshaped_padding[1].second = paddings[pad_dim].second;
          // the third dimension do not need padding, set padding[2] zero
          reshaped_padding[2].first = reshaped_padding[2].second = 0;

          LaunchEigenPadding(context, d_input, reshaped_in_dims, d_out,
                             reshaped_out_dims, reshaped_padding);
        }
      } else {
        // need padding at many dimension, cannot reduce dimension
        LaunchEigenPadding(context, d_input, in_dims, d_out, out_dims,
                           paddings);
      }
    }
  }

  template <size_t D>
  void LaunchEigenPadding(
      const framework::ExecutionContext& context, Tensor* d_input,
      const DDim& in_dims, const Tensor* d_out, const DDim& out_dims,
      const Eigen::array<std::pair<int64_t, int64_t>, D>& paddings) const {
    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto d_in_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_input, in_dims);
    auto d_out_t =
        framework::EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
            *d_out, out_dims);

    if (d_input->numel() <= Eigen::NumTraits<int>::highest()) {
      // similar to tf.pad:
      // if element number less than INT_MAX, change the type of index to int
      Eigen::array<std::pair<int, int>, D> paddings_32bit;
      for (size_t i = 0; i < D; i++) {
        paddings_32bit[i] =
            std::make_pair(paddings[i].first, paddings[i].second);
      }
      EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
          place, framework::To32BitIndex(d_in_t),
          framework::To32BitIndex(d_out_t), paddings_32bit, static_cast<T>(0));
    } else {
      EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
          place, d_in_t, d_out_t, paddings, static_cast<T>(0));
    }
  }
};
}  // namespace operators
}  // namespace paddle
