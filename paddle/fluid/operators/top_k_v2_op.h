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

/*
  The reason why we need the topk v2 is because the compatibility. We redefine
  the NaN is maximum value
  in the process of comparing. If do not add the topk v2,  will affect the
  inference result of model that traing
  by the older version paddlepaddle.
*/

#pragma once
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/transpose.h"
#include "paddle/fluid/operators/top_k_op.h"

namespace paddle {
namespace operators {

inline void GetDims(const framework::DDim& dim, int axis, int* pre, int* n,
                    int* post) {
  *pre = 1;
  *post = 1;
  *n = dim[axis];
  for (int i = 0; i < axis; ++i) {
    (*pre) *= dim[i];
  }
  for (int i = axis + 1; i < dim.size(); ++i) {
    (*post) *= dim[i];
  }
}

template <typename T, typename Type>
static void FullTopK(Type input_height, Type input_width, int input_dim,
                     const framework::Tensor* input, T* t_out, Type* t_indices,
                     const int& k, const bool& largest, const bool& sorted) {
  // when the k is small, will the partial sort
  bool partial_sort_flag = (k * 64) < input_width;

#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  // Eigen::DSizes<int, 2> flat2dims(input_height, input_width);
  for (Type i = 0; i < input_height; ++i) {
    std::vector<std::pair<T, Type>> col_vec;
    col_vec.reserve(input_width);
    if (input_dim == 1) {
      auto e_input = framework::EigenVector<T>::Flatten(*input);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(j), j));
      }
    } else {
      auto e_input = framework::EigenMatrix<T>::Reshape(*input, input_dim - 1);
      for (Type j = 0; j < input_width; ++j) {
        col_vec.emplace_back(std::pair<T, Type>(e_input(i, j), j));
      }
    }
    if (partial_sort_flag) {
      std::partial_sort(
          col_vec.begin(), col_vec.begin() + k, col_vec.end(),
          [&largest](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
            if (largest) {
              return (std::isnan(static_cast<double>(l.first)) &&
                      !std::isnan(static_cast<double>(r.first))) ||
                     (l.first > r.first);
            } else {
              return (!std::isnan(static_cast<double>(l.first)) &&
                      std::isnan(static_cast<double>(r.first))) ||
                     (l.first < r.first);
            }
          });
    } else {
      // use the nth-element to get the K-larger or K-small element
      if (largest) {
        std::nth_element(
            col_vec.begin(), col_vec.begin() + k - 1, col_vec.end(),
            [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
              return (std::isnan(static_cast<double>(l.first)) &&
                      !std::isnan(static_cast<double>(r.first))) ||
                     (l.first > r.first);
            });
        // the nth-element will get the unorder elements, sort the element
        if (sorted) {
          std::sort(col_vec.begin(), col_vec.begin() + k - 1,
                    [&largest](const std::pair<T, Type>& l,
                               const std::pair<T, Type>& r) {
                      return (std::isnan(static_cast<double>(l.first)) &&
                              !std::isnan(static_cast<double>(r.first))) ||
                             (l.first > r.first);
                    });
        }
      } else {
        std::nth_element(
            col_vec.begin(), col_vec.begin() + k - 1, col_vec.end(),
            [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
              return (!std::isnan(static_cast<double>(l.first)) &&
                      std::isnan(static_cast<double>(r.first))) ||
                     (l.first < r.first);
            });
        // the nth-element will get the unorder elements, sort the element
        if (sorted) {
          std::sort(
              col_vec.begin(), col_vec.begin() + k - 1,
              [](const std::pair<T, Type>& l, const std::pair<T, Type>& r) {
                return (!std::isnan(static_cast<double>(l.first)) &&
                        std::isnan(static_cast<double>(r.first))) ||
                       (l.first < r.first);
              });
        }
      }
    }
    for (Type j = 0; j < k; ++j) {
      t_out[i * k + j] = col_vec[j].first;
      t_indices[i * k + j] = col_vec[j].second;
    }
  }
}

template <typename T, typename Type>
static void FullTopKAssign(const Type& input_height, const Type& input_width,
                           const int& input_dim, const framework::Tensor* input,
                           const framework::Tensor* indices, T* output_data,
                           const int& k) {
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (Type i = 0; i < input_height; ++i) {
    if (input_dim == 1) {
      auto e_input = framework::EigenVector<T>::Flatten(*input);
      auto e_indices = framework::EigenVector<Type>::Flatten(*indices);
      for (Type j = 0; j < k; ++j) {
        output_data[i * input_width + e_indices(j)] = e_input(j);
      }
    } else {
      auto e_input = framework::EigenMatrix<T>::Reshape(*input, input_dim - 1);
      auto e_indices =
          framework::EigenMatrix<Type>::Reshape(*indices, input_dim - 1);
      for (Type j = 0; j < k; ++j) {
        output_data[i * input_width + e_indices(i, j)] = e_input(i, j);
      }
    }
  }
}

template <typename DeviceContext, typename T>
class TopkV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    // Get the top k elements of each row of input tensor
    auto* input = context.Input<Tensor>("X");
    auto* output = context.Output<Tensor>("Out");
    auto* indices = context.Output<Tensor>("Indices");
    const auto& in_dims = input->dims();
    int k = static_cast<int>(context.Attr<int>("k"));
    const auto& sorted = static_cast<bool>(context.Attr<bool>("sorted"));
    const auto& largest = static_cast<bool>(context.Attr<bool>("largest"));

    // axis < 0, cacluate the real axis
    int axis = static_cast<int>(context.Attr<int>("axis"));
    if (axis < 0) axis += in_dims.size();

    // if K tensor is not null, will the use K tesnor as k
    auto* k_t = context.Input<Tensor>("K");
    if (k_t) {
      k = k_t->data<int>()[0];
      framework::DDim output_dims = output->dims();
      // accroding to axis to set K value in the dim
      output_dims[axis] = k;
      output->Resize(output_dims);
      indices->Resize(output_dims);
    }

    T* output_data = output->mutable_data<T>(context.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(context.GetPlace());
    const auto& out_dims = output->dims();
    if (axis + 1 == in_dims.size()) {
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      FullTopK<T, int64_t>(input_height, input_width, in_dims.size(), input,
                           output_data, indices_data, k, largest, sorted);
    } else {
      // if the topk dims is not last dim, will tranpose and do topk
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.push_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);

      // get the trans input_dims, out_dims
      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(output->dims());
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }
      for (size_t i = 0; i < trans.size(); i++) {
        trans_out_dims[i] = out_dims[trans[i]];
      }

      Tensor trans_inp;
      trans_inp.mutable_data<T>(trans_dims, context.GetPlace());
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();

      // transpose the input value
      paddle::operators::math::TransposeFunctor<platform::CPUDeviceContext, T>
          transpose;
      paddle::operators::math::TransposeFunctor<platform::CPUDeviceContext,
                                                int64_t>
          transpose_indices;
      transpose(dev_context, *input, &trans_inp, trans);

      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];

      // Allocate the temp tensor to the save the topk indices, values
      Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_out_dims, context.GetPlace());
      Tensor tmp_indices;
      auto* t_ind =
          tmp_indices.mutable_data<int64_t>(trans_out_dims, context.GetPlace());

      // get the TopK value
      FullTopK<T, int64_t>(input_height, input_width, in_dims.size(),
                           &trans_inp, t_out, t_ind, k, largest, sorted);
      // transpose back
      transpose_indices(dev_context, tmp_indices, indices, trans);
      transpose(dev_context, tmp_out, output, trans);
    }
  }
};

template <typename DeviceContext, typename T>
class TopkV2GradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<Tensor>("Indices");
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    int axis = static_cast<int>(context.Attr<int>("axis"));

    const auto& in_dims = x->dims();
    const auto& out_dims = indices->dims();

    // axis < 0, get the real axis
    axis = (axis < 0) ? (in_dims.size() + axis) : axis;
    const size_t& k = out_dims[axis];

    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    if (axis + 1 == in_dims.size()) {
      // allocate the memory for the input_grad

      // assign the out_grad to input_grad directly
      const int64_t input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t input_width = in_dims[in_dims.size() - 1];

      // init the output grad with 0, because some input elements has no grad
      memset(x_grad_data, 0, x_grad->numel() * sizeof(T));
      // Assign the output_grad to input_grad
      FullTopKAssign(input_height, input_width, in_dims.size(), out_grad,
                     indices, x_grad_data, k);
    } else {
      // can not assign grad to input_grad, must do the transpose
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(out_dims.size() - 1);
      for (int i = axis + 1; i < out_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);
      framework::DDim trans_dims(out_dims);
      framework::DDim trans_in_dims(in_dims);
      for (size_t i = 0; i < trans.size(); i++) {
        trans_dims[i] = out_dims[trans[i]];
        trans_in_dims[i] = in_dims[trans[i]];
      }
      // transpose the out_grad, indices
      Tensor trans_dO;
      trans_dO.mutable_data<T>(trans_dims, context.GetPlace());
      Tensor trans_ind;
      trans_ind.mutable_data<int64_t>(trans_dims, context.GetPlace());
      auto& dev_context =
          context.template device_context<platform::CPUDeviceContext>();

      // Do transpose
      paddle::operators::math::TransposeFunctor<platform::CPUDeviceContext, T>
          transpose;
      paddle::operators::math::TransposeFunctor<platform::CPUDeviceContext,
                                                int64_t>
          transpose_indices;
      transpose(dev_context, *out_grad, &trans_dO, trans);
      transpose_indices(dev_context, *indices, &trans_ind, trans);
      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_in_dims, 0, trans_in_dims.size() - 1));
      const int64_t input_width = trans_in_dims[trans_in_dims.size() - 1];

      // Assign the out_grad to tranpose input_grad
      Tensor tmp_out;
      T* t_out = tmp_out.mutable_data<T>(trans_in_dims, context.GetPlace());
      memset(t_out, 0, x_grad->numel() * sizeof(T));

      FullTopKAssign<T, int64_t>(input_height, input_width, in_dims.size(),
                                 &trans_dO, &trans_ind, t_out, k);

      // Transpose back
      transpose(dev_context, tmp_out, x_grad, trans);
    }
  }
};

}  // namespace operators
}  // namespace paddle
