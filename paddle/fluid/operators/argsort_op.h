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
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/transpose_op.h"

namespace paddle {
namespace operators {

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

using Tensor = framework::Tensor;

template <typename T, typename Type>
static void FullSort(int64_t input_height, int64_t input_width, int input_dim,
                     const framework::Tensor* input, T* t_out, Type* t_indices,
                     bool descending) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      for (int64_t j = 0; j < input_width; ++j) {
        col_vec.push_back(std::pair<T, Type>(e_input(j), static_cast<Type>(j)));
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (int64_t j = 0; j < input_width; ++j) {
        col_vec.push_back(
            std::pair<T, Type>(e_input(i, j), static_cast<Type>(j)));
      }
    }
    std::sort(col_vec.begin(), col_vec.end(),
              [&](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                if (descending)
                  return l.first > r.first;
                else
                  return l.first < r.first;
              });

    for (int64_t j = 0; j < input_width; ++j) {
      t_out[i * input_width + j] = col_vec[j].first;
      t_indices[i * input_width + j] = col_vec[j].second;
    }
  }
}

template <typename T, typename Type>
static void FullAssign(int64_t input_height, int64_t input_width, int input_dim,
                       const framework::Tensor* input,
                       const framework::Tensor* indices, T* t_out) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (int64_t i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = EigenVector<T>::Flatten(*input);
      auto e_indices = EigenVector<Type>::Flatten(*indices);
      for (int64_t j = 0; j < input_width; ++j) {
        t_out[i * input_width + static_cast<int64_t>(e_indices(j))] =
            e_input(j);
      }
    } else {
      auto e_input = EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices = EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      for (int64_t j = 0; j < input_width; ++j) {
        t_out[i * input_width + static_cast<int64_t>(e_indices(i, j))] =
            e_input(i, j);
      }
    }
  }
}

template <typename DeviceContext, typename T>
struct ArgsortFunctor {
  const framework::ExecutionContext& ctx;
  explicit ArgsortFunctor(const framework::ExecutionContext& ctx) : ctx(ctx) {}
  template <typename TINDEX>
  void apply() const {
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int axis = ctx.Attr<int>("axis");
    bool descending = ctx.Attr<bool>("descending");

    auto in_dims = input->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    T* out_data = output->mutable_data<T>(ctx.GetPlace());

    // Do full sort
    if (axis == -1 || axis + 1 == in_dims.size()) {
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];

      auto* ids_data = indices->mutable_data<TINDEX>(ctx.GetPlace());
      FullSort<T, TINDEX>(input_height, input_width, in_dims.size(), input,
                          out_data, ids_data, descending);
    } else {
      // If not full sort do transpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.push_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.push_back(i);
      }
      trans.push_back(axis);
      framework::DDim trans_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }

      Tensor trans_inp;
      trans_inp.mutable_data<T>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
      // Do transpose
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_ctx, *input,
                                                  &trans_inp, trans);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_dims, ctx.GetPlace());
      output->mutable_data<T>(ctx.GetPlace());

      Tensor tmp_indices;

      auto* t_ind =
          tmp_indices.mutable_data<TINDEX>(trans_dims, ctx.GetPlace());

      FullSort<T, TINDEX>(input_height, input_width, in_dims.size(), &trans_inp,
                          t_out, t_ind, descending);

      indices->mutable_data<TINDEX>(ctx.GetPlace());
      TransCompute<platform::CPUDeviceContext, TINDEX>(
          ndims, dev_ctx, tmp_indices, indices, trans);
      // transpose back
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_ctx, tmp_out,
                                                  output, trans);
    }
  }
};

template <typename DeviceContext, typename T>
class ArgsortKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int& dtype = ctx.Attr<int>("dtype");
    framework::VisitDataTypeInt(
        static_cast<framework::proto::VarType::Type>(dtype),
        ArgsortFunctor<DeviceContext, T>(ctx));
  }
};

template <typename DeviceContext, typename T>
struct ArgsortGradientFunctor {
  const framework::ExecutionContext& ctx;
  explicit ArgsortGradientFunctor(const framework::ExecutionContext& ctx)
      : ctx(ctx) {}

  template <typename TINDEX>
  void apply() {
    auto* indices = ctx.Input<Tensor>("Indices");
    auto* dX = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dO = ctx.Input<Tensor>(framework::GradVarName("Out"));
    int axis = ctx.Attr<int>("axis");

    auto in_dims = indices->dims();
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;

    dX->mutable_data<T>(ctx.GetPlace());
    auto dxt = framework::EigenVector<T>::Flatten(*dX);
    auto& place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    dxt.device(place) = dxt.constant(static_cast<T>(0));
    if (dO->numel() == 0) return;

    // Do full assign
    if (axis == -1 || axis + 1 == in_dims.size()) {
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];

      FullAssign<T, TINDEX>(input_height, input_width, in_dims.size(), dO,
                            indices, dX->data<T>());
    } else {
      // If not full assign do transpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.push_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.push_back(i);
      }
      trans.push_back(axis);
      framework::DDim trans_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }

      Tensor trans_dO;
      trans_dO.mutable_data<T>(trans_dims, ctx.GetPlace());
      Tensor trans_ind;
      trans_ind.mutable_data<TINDEX>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
      // Do transpose
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_ctx, *dO,
                                                  &trans_dO, trans);
      TransCompute<platform::CPUDeviceContext, TINDEX>(ndims, dev_ctx, *indices,
                                                       &trans_ind, trans);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_dims, ctx.GetPlace());

      FullAssign<T, TINDEX>(input_height, input_width, in_dims.size(),
                            &trans_dO, &trans_ind, t_out);

      // transpose back
      TransCompute<platform::CPUDeviceContext, T>(ndims, dev_ctx, tmp_out, dX,
                                                  trans);
    }
  }
};

template <typename DeviceContext, typename T>
class ArgsortGradientKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const int& dtype = ctx.Attr<int>("dtype");
    framework::VisitDataTypeInt(
        static_cast<framework::proto::VarType::Type>(dtype),
        ArgsortGradientFunctor<DeviceContext, T>(ctx));
  }
};

}  // namespace operators
}  // namespace paddle
