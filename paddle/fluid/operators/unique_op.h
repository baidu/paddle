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
#include <cmath>
#include <numeric>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/concat_and_split.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/transpose.h"

namespace paddle {
namespace operators {

template <typename InT>
struct UniqueOpFunctor {
  framework::Tensor* out_;
  framework::Tensor* index_;
  const framework::Tensor* in_;
  framework::Tensor* count_;

  UniqueOpFunctor(framework::Tensor* out, framework::Tensor* index,
                  const framework::Tensor* in,
                  framework::Tensor* count = nullptr)
      : out_(out), index_(index), in_(in), count_(count) {}

  template <typename IndexT>
  void apply() const {
    auto* in_data = in_->data<InT>();
    auto* index_data = index_->mutable_data<IndexT>(platform::CPUPlace());

    int64_t j = 0;

    // TODO(fangzeyang): Should optimize performance here.
    std::unordered_map<InT, int64_t> dict;
    std::vector<InT> uniq;

    PADDLE_ENFORCE_LT(
        in_->numel(), pow(2, 31),
        platform::errors::InvalidArgument(
            "The num of Input(X) elements should be less then INT_MAX, "
            "but received num is %d.",
            in_->numel()));

    for (auto i = 0; i < in_->numel(); i++) {
      auto it = dict.find(in_data[i]);
      if (it == dict.end()) {
        dict.emplace(std::make_pair(in_data[i], j));
        uniq.emplace_back(in_data[i]);
        index_data[i] = static_cast<IndexT>(j);
        j++;
      } else {
        index_data[i] = static_cast<IndexT>(it->second);
      }
    }

    if (count_ != nullptr) {
      // Resize the count tensor dims to allocate the memory
      count_->Resize(framework::make_ddim({static_cast<int64_t>(uniq.size())}));
      IndexT* count_data = count_->mutable_data<IndexT>(platform::CPUPlace());
      // init count_data to 0
      memset(count_data, 0, uniq.size() * sizeof(IndexT));

      const auto& index_type = index_->type();
      bool index_type_match = index_type == framework::proto::VarType::INT32 ||
                              index_type == framework::proto::VarType::INT64;
      PADDLE_ENFORCE_EQ(index_type_match, true,
                        platform::errors::InvalidArgument(
                            "Index holds the wrong type, it holds %s, "
                            "but desires to be %s or %s",
                            paddle::framework::DataTypeToString(index_type),
                            paddle::framework::DataTypeToString(
                                framework::proto::VarType::INT32),
                            paddle::framework::DataTypeToString(
                                framework::proto::VarType::INT64)));

      if (index_type == framework::proto::VarType::INT32) {
        for (auto i = 0; i < in_->numel(); ++i) {
          const IndexT& index = index_data[i];
          count_data[static_cast<int32_t>(index)] += static_cast<IndexT>(1);
        }
      } else {
        for (auto i = 0; i < in_->numel(); ++i) {
          const IndexT& index = index_data[i];
          count_data[static_cast<int64_t>(index)] += static_cast<IndexT>(1);
        }
      }
    }

    out_->Resize(framework::make_ddim({static_cast<int64_t>(uniq.size())}));
    auto out_data = out_->mutable_data<InT>(platform::CPUPlace());
    std::memcpy(out_data, uniq.data(), uniq.size() * sizeof(InT));
  }
};

static std::vector<framework::Tensor> Unbind(const framework::Tensor& in) {
  int64_t size = in.dims()[0];
  std::vector<framework::Tensor> tensors(size);
  for (int64_t i = 0; i < size; ++i) {
    tensors[i] = in.Slice(i, i + 1);
  }
  return tensors;
}

template <typename T>
static bool Equal(const framework::Tensor& a, const framework::Tensor& b) {
  if (a.numel() != b.numel()) {
    return false;
  }
  for (int64_t i = 0; i < a.numel(); ++i) {
    if (a.data<T>()[i] != b.data<T>()[i]) {
      return false;
    }
  }
  return true;
}

template <typename InT, typename IndexT>
static void UniqueFlattendTensor(const framework::ExecutionContext& context,
                                 const framework::Tensor& in,
                                 framework::Tensor* out, bool return_index,
                                 bool return_inverse, bool return_counts) {
  const InT* in_data = in.data<InT>();
  std::set<InT> unique(in_data, in_data + in.numel());
  out->Resize(framework::make_ddim({static_cast<int64_t>(unique.size())}));
  auto out_data = out->mutable_data<InT>(context.GetPlace());
  std::copy(unique.begin(), unique.end(), out_data);

  if (return_index) {
    auto* indices = context.Output<framework::Tensor>("Indices");
    indices->Resize(framework::make_ddim({out->numel()}));
    auto indices_data = indices->mutable_data<IndexT>(context.GetPlace());
    std::unordered_map<InT, IndexT> indices_map;
    indices_map.reserve(out->numel());
    for (int64_t i = 0; i < in.numel(); ++i) {
      if (indices_map.find(in_data[i]) != indices_map.end()) continue;
      indices_map[in_data[i]] = i;
    }
    for (int64_t i = 0; i < out->numel(); ++i) {
      indices_data[i] = indices_map[out_data[i]];
    }
  }

  if (return_inverse) {
    auto* inverse = context.Output<framework::Tensor>("Index");
    inverse->Resize(framework::make_ddim({in.numel()}));
    auto inverse_data = inverse->mutable_data<IndexT>(context.GetPlace());
    std::unordered_map<InT, IndexT> inverse_map;
    inverse_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      inverse_map[out_data[i]] = i;
    }
    for (int64_t i = 0; i < in.numel(); ++i) {
      inverse_data[i] = inverse_map[in_data[i]];
    }
  }

  if (return_counts) {
    auto* count = context.Output<framework::Tensor>("Counts");
    count->Resize(framework::make_ddim({out->numel()}));
    auto count_data = count->mutable_data<IndexT>(context.GetPlace());
    std::unordered_map<InT, IndexT> counts_map;
    counts_map.reserve(out->numel());
    for (int64_t i = 0; i < out->numel(); ++i) {
      counts_map[out_data[i]] = 0;
    }
    for (int64_t i = 0; i < in.numel(); i++) {
      counts_map[in_data[i]] += 1;
    }
    for (int64_t i = 0; i < out->numel(); i++) {
      count_data[i] = counts_map[out_data[i]];
    }
  }
}

template <class ForwardIt, typename InT, typename IndexT>
static ForwardIt UniqueDimImpl(const framework::ExecutionContext& context,
                               ForwardIt first, ForwardIt last,
                               const std::vector<IndexT>& sorted_indices_vec,
                               std::vector<IndexT>* inverse_vec,
                               std::vector<IndexT>* counts_vec,
                               std::vector<IndexT>* indices_vec) {
  if (first == last) {
    return last;
  }

  (*inverse_vec)[sorted_indices_vec[0]] = 0;
  (*counts_vec)[0] = 1;
  (*indices_vec)[0] = sorted_indices_vec[0];

  ForwardIt begin = first;
  ForwardIt result = first;

  while (++first != last) {
    int64_t idx_first = std::distance(begin, first);
    int64_t idx_result = std::distance(begin, result);
    if (!Equal<InT>(*result, *first)) {
      if (++result != first) {
        *result = std::move(*first);
      }
      idx_result += 1;
      (*indices_vec)[idx_result] = sorted_indices_vec[idx_first];
    }
    (*inverse_vec)[sorted_indices_vec[idx_first]] = idx_result;
    (*counts_vec)[idx_result] += 1;
  }
  return ++result;
}

template <typename DeviceContext, typename InT, typename IndexT>
static void UniqueDim(const framework::ExecutionContext& context,
                      const framework::Tensor& in, framework::Tensor* out,
                      bool return_index, bool return_inverse,
                      bool return_counts, int axis) {
  // transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
  std::vector<int> permute(in.dims().size());
  std::iota(permute.begin(), permute.end(), 0);
  permute[axis] = 0;
  permute[0] = axis;
  std::vector<int64_t> in_trans_dims_vec(framework::vectorize(in.dims()));
  in_trans_dims_vec[axis] = in.dims()[0];
  in_trans_dims_vec[0] = in.dims()[axis];
  framework::Tensor in_trans;
  framework::DDim in_trans_dims = framework::make_ddim(in_trans_dims_vec);
  in_trans.Resize(in_trans_dims);
  in_trans.mutable_data<InT>(context.GetPlace());
  auto& dev_ctx = context.template device_context<DeviceContext>();
  paddle::operators::math::TransposeFunctor<DeviceContext, InT> transpose;
  transpose(dev_ctx, in, &in_trans, permute);
  // reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  framework::DDim in_trans_flat_dims =
      framework::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // sort indices
  std::vector<IndexT> sorted_indices_vec(in_trans.dims()[0]);
  std::iota(sorted_indices_vec.begin(), sorted_indices_vec.end(), 0);
  int64_t col = in_trans.dims()[1];
  const InT* in_trans_data = in_trans.data<InT>();
  std::sort(sorted_indices_vec.begin(), sorted_indices_vec.end(),
            [&](int64_t a, int64_t b) -> bool {
              for (int64_t i = 0; i < col; ++i) {
                InT lhs = in_trans_data[i + a * col];
                InT rhs = in_trans_data[i + b * col];
                if (lhs < rhs) {
                  return true;
                } else if (lhs > rhs) {
                  return false;
                }
              }
              return false;
            });

  // sort tensor according to indices
  framework::Tensor input_sorted;
  input_sorted.Resize(in_trans_dims);
  input_sorted.mutable_data<InT>(context.GetPlace());
  InT* input_sorted_data = input_sorted.data<InT>();
  for (size_t i = 0; i < sorted_indices_vec.size(); ++i) {
    memcpy(input_sorted_data + i * col,
           in_trans_data + static_cast<int64_t>(sorted_indices_vec[i]) * col,
           col * sizeof(InT));
  }

  std::vector<framework::Tensor> input_unbind = Unbind(input_sorted);
  std::vector<IndexT> inverse_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> counts_vec(sorted_indices_vec.size(), 0);
  std::vector<IndexT> indices_vec(sorted_indices_vec.size(), 0);
  auto last = UniqueDimImpl<std::vector<framework::Tensor>::iterator, InT>(
      context, input_unbind.begin(), input_unbind.end(), sorted_indices_vec,
      &inverse_vec, &counts_vec, &indices_vec);
  input_unbind.erase(last, input_unbind.end());
  counts_vec.erase(counts_vec.begin() + input_unbind.size(), counts_vec.end());
  indices_vec.erase(indices_vec.begin() + input_unbind.size(),
                    indices_vec.end());

  math::ConcatFunctor<DeviceContext, InT> concat_functor;
  framework::Tensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = input_unbind.size();
  out_trans.Resize(framework::make_ddim(out_trans_dims_vec));
  out_trans.mutable_data<InT>(context.GetPlace());
  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(framework::make_ddim(out_trans_dims_vec));
  out->mutable_data<InT>(context.GetPlace());
  concat_functor(dev_ctx, input_unbind, 0, &out_trans);
  transpose(dev_ctx, out_trans, out, permute);

  if (return_inverse) {
    auto* inverse = context.Output<framework::Tensor>("Index");
    framework::TensorFromVector(inverse_vec, context.device_context(), inverse);
  }

  if (return_counts) {
    auto* count = context.Output<framework::Tensor>("Counts");
    framework::TensorFromVector(counts_vec, context.device_context(), count);
  }

  if (return_index) {
    auto* indices = context.Output<framework::Tensor>("Indices");
    framework::TensorFromVector(indices_vec, context.device_context(), indices);
  }
}

template <typename DeviceContext, typename InT>
struct UniqueFlattendTensorFunctor {
  const framework::ExecutionContext& ctx_;
  const framework::Tensor& in_;
  framework::Tensor* out_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueFlattendTensorFunctor(const framework::ExecutionContext& context,
                              const framework::Tensor& in,
                              framework::Tensor* out, bool return_index,
                              bool return_inverse, bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueFlattendTensor<InT, IndexT>(ctx_, in_, out_, return_index_,
                                      return_inverse_, return_counts_);
  }
};

template <typename DeviceContext, typename InT>
struct UniqueDimFunctor {
  const framework::ExecutionContext& ctx_;
  const framework::Tensor& in_;
  framework::Tensor* out_;
  const int axis_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueDimFunctor(const framework::ExecutionContext& context,
                   const framework::Tensor& in, framework::Tensor* out,
                   const int axis, bool return_index, bool return_inverse,
                   bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        axis_(axis),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueDim<DeviceContext, InT, IndexT>(
        ctx_, in_, out_, return_index_, return_inverse_, return_counts_, axis_);
  }
};

template <typename DeviceContext, typename T>
class UniqueKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    if (data_type == framework::proto::VarType::INT32) {
      PADDLE_ENFORCE_LE(
          x->numel(), INT_MAX,
          platform::errors::InvalidArgument(
              "The number of elements in Input(X) should be less than or "
              "equal to INT_MAX, but received num is %d. Please set `dtype` to "
              "int64.",
              x->numel()));
    }
    if (!context.Attr<bool>("is_sorted")) {
      auto* index = context.Output<framework::Tensor>("Index");

      framework::VisitDataType(data_type, UniqueOpFunctor<T>(out, index, x));
      return;
    }

    std::vector<int> axis_vec = context.Attr<std::vector<int>>("axis");
    bool return_index = context.Attr<bool>("return_index");
    bool return_inverse = context.Attr<bool>("return_inverse");
    bool return_counts = context.Attr<bool>("return_counts");

    if (axis_vec.empty()) {
      framework::VisitDataTypeTiny(
          data_type,
          UniqueFlattendTensorFunctor<DeviceContext, T>(
              context, *x, out, return_index, return_inverse, return_counts));
    } else {
      int axis = axis_vec[0];
      framework::VisitDataTypeTiny(
          data_type, UniqueDimFunctor<DeviceContext, T>(
                         context, *x, out, axis, return_index, return_inverse,
                         return_counts));
    }
  }
};

}  // namespace operators
}  // namespace paddle
