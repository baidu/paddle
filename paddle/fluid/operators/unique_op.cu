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
#include <thrust/adjacent_difference.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>
#include <iostream>
#include <vector>
#include "paddle/fluid/framework/tensor_util.h"  // TensorToVector()
#include "paddle/fluid/operators/unique_op.h"    // TransComute()

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

// Binary function 'less than'
template <typename InT>
struct LessThan {
  int col;
  const InT* in_trans_data;

  LessThan(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs < rhs) {
        return true;
      } else if (lhs > rhs) {
        return false;
      }
    }
    return false;
  }
};

// Binary function 'equal_to'
template <typename InT>
struct BinaryEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return false;
      }
    }
    return true;
  }
};

// Binary function 'not_equal_to'
template <typename InT>
struct BinaryNotEqual {
  int64_t col;
  const InT* in_trans_data;

  BinaryNotEqual(int64_t _col, const InT* _in_trans_data)
      : col(_col), in_trans_data(_in_trans_data) {}

  __device__ bool operator()(int64_t a, int64_t b) const {
    for (int64_t i = 0; i < col; ++i) {
      InT lhs = in_trans_data[i + a * col];
      InT rhs = in_trans_data[i + b * col];
      if (lhs != rhs) {
        return true;
      }
    }
    return false;
  }
};

// index_select() function for Tensor
template <typename InT, typename IndexT>
void IndexSelect(const framework::ExecutionContext& context,
                 const Tensor& input, const Tensor& index, Tensor* output,
                 int dim) {
  auto input_dim = input.dims();
  auto input_dim_size = input_dim.size();
  auto output_dim = output->dims();

  auto slice_size = 1;
  for (auto i = dim + 1; i < input_dim_size; i++) {
    slice_size *= input_dim[i];
  }

  auto input_width = slice_size * input_dim[dim];
  auto output_width = slice_size * output_dim[dim];

  auto outer_nums = 1;
  for (auto i = 0; i < dim; i++) {
    outer_nums *= input_dim[i];
  }

  auto index_size = index.dims()[0];

  std::vector<InT> input_vec;
  std::vector<IndexT> index_vec;
  TensorToVector(input, context.device_context(), &input_vec);
  TensorToVector(index, context.device_context(), &index_vec);
  std::vector<InT> out_vec(output->numel());

  for (int i = 0; i < index_size; i++) {
    PADDLE_ENFORCE_GE(
        index_vec[i], 0,
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim], index_vec[i]));
    PADDLE_ENFORCE_LT(
        index_vec[i], input_dim[dim],
        platform::errors::InvalidArgument(
            "Variable value (index) of OP(index_select) "
            "expected >= 0 and < %ld, but got %ld. Please check input "
            "value.",
            input_dim[dim], index_vec[i]));
  }

  for (auto i = 0; i < outer_nums; i++) {
    auto input_start_offset = i * input_width;
    auto output_start_offset = i * output_width;

    for (auto j = 0; j < index_size; j++) {
      IndexT index_value = index_vec[j];
      for (auto k = 0; k < slice_size; k++) {
        out_vec[output_start_offset + j * slice_size + k] =
            input_vec[input_start_offset + index_value * slice_size + k];
      }
    }
  }
  output->mutable_data<InT>(context.GetPlace());
  framework::TensorFromVector(out_vec, context.device_context(), output);
  output->Resize(output_dim);
}

// The core logic of computing Unique for a flattend Tensor
template <typename InT, typename IndexT, typename equal_T, typename not_equal_T>
static void UniqueFlattendCUDATensor(const framework::ExecutionContext& context,
                                     const Tensor& in, Tensor* out,
                                     bool return_index, bool return_inverse,
                                     bool return_counts, equal_T equal,
                                     not_equal_T not_equal, int64_t num_input) {
  // 0. Prepration
  Tensor in_hat;
  framework::TensorCopy(in, context.GetPlace(), &in_hat);
  auto in_data_hat = in_hat.mutable_data<InT>(context.GetPlace());

  Tensor* sorted_indices = context.Output<Tensor>("Indices");
  sorted_indices->Resize(framework::make_ddim({num_input}));
  auto sorted_indices_data =
      sorted_indices->mutable_data<IndexT>(context.GetPlace());
  thrust::sequence(thrust::device, sorted_indices_data,
                   sorted_indices_data + num_input);
  thrust::sort_by_key(thrust::device, in_data_hat, in_data_hat + num_input,
                      sorted_indices_data);

  // 1. Calculate op result: 'out'
  Tensor range;
  range.Resize(framework::make_ddim({num_input + 1}));
  auto range_data_ptr = range.mutable_data<IndexT>(context.GetPlace());
  thrust::sequence(thrust::device, range_data_ptr,
                   range_data_ptr + num_input + 1);
  framework::TensorCopy(in_hat, context.GetPlace(), out);
  int num_out;
  auto out_data = out->mutable_data<InT>(context.GetPlace());
  num_out = thrust::unique_by_key(thrust::device, out_data,
                                  out_data + num_input, range_data_ptr, equal)
                .first -
            out_data;
  out->Resize(framework::make_ddim({num_out}));

  // 3. Calculate inverse index: 'inverse'
  if (return_inverse) {
    Tensor* inverse = context.Output<Tensor>("Index");
    inverse->Resize(framework::make_ddim({num_input}));
    auto inverse_data = inverse->mutable_data<IndexT>(context.GetPlace());
    Tensor inv_loc;
    inv_loc.Resize(framework::make_ddim({num_input}));
    auto inv_loc_data_ptr = inv_loc.mutable_data<IndexT>(context.GetPlace());
    thrust::adjacent_difference(thrust::device, in_data_hat,
                                in_data_hat + num_input, inv_loc_data_ptr,
                                not_equal);
    thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
    inv_loc_data_dev[0] = 0;  // without device_ptr, segmentation fault
    thrust::inclusive_scan(thrust::device, inv_loc_data_ptr,
                           inv_loc_data_ptr + num_input, inv_loc_data_ptr);
    thrust::scatter(thrust::device, inv_loc_data_ptr,
                    inv_loc_data_ptr + num_input, sorted_indices_data,
                    inverse_data);
  }

  // 2. Calculate sorted index: 'sorted_indices'
  if (return_index) {
    Tensor indices;
    indices.Resize(framework::make_ddim({num_input}));
    auto indices_data_ptr = indices.mutable_data<IndexT>(context.GetPlace());
    thrust::copy(thrust::device, in_data_hat, in_data_hat + num_input,
                 indices_data_ptr);
    thrust::unique_by_key(thrust::device, indices_data_ptr,
                          indices_data_ptr + num_input, sorted_indices_data,
                          equal);
    sorted_indices->Resize(framework::make_ddim({num_out}));
  }

  // 4. Calculate 'counts'
  if (return_counts) {
    Tensor* counts = context.Output<Tensor>("Counts");
    counts->Resize(framework::make_ddim({num_out}));
    auto count_data = counts->mutable_data<IndexT>(context.GetPlace());
    // init 'count_data' as 0
    thrust::fill(thrust::device, count_data, count_data + num_out, 0);
    thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
    range_data_ptr_dev[num_out] = num_input;
    thrust::adjacent_difference(thrust::device, range_data_ptr + 1,
                                range_data_ptr + num_out + 1, count_data);
  }
}

// The logic of compute unique with axis required, it's a little different
// from above function
template <typename InT, typename IndexT, typename equal_T, typename not_equal_T>
static void ComputeUniqueDims(const framework::ExecutionContext& context,
                              Tensor* sorted_indices,
                              IndexT* sorted_indices_data, Tensor* out,
                              bool return_index, bool return_inverse,
                              bool return_counts, equal_T equal,
                              not_equal_T not_equal, int64_t row) {
  // 1. inverse indices: 'inverse'
  Tensor* inverse = context.Output<Tensor>("Index");
  inverse->Resize(framework::make_ddim({row}));
  auto inverse_data = inverse->mutable_data<IndexT>(context.GetPlace());
  Tensor inv_loc;
  inv_loc.Resize(framework::make_ddim({row}));
  auto inv_loc_data_ptr = inv_loc.mutable_data<IndexT>(context.GetPlace());
  thrust::adjacent_difference(thrust::device, sorted_indices_data,
                              sorted_indices_data + row, inv_loc_data_ptr,
                              not_equal);
  thrust::device_ptr<IndexT> inv_loc_data_dev(inv_loc_data_ptr);
  inv_loc_data_dev[0] = 0;
  thrust::inclusive_scan(thrust::device, inv_loc_data_ptr,
                         inv_loc_data_ptr + row, inv_loc_data_ptr);
  thrust::scatter(thrust::device, inv_loc_data_ptr, inv_loc_data_ptr + row,
                  sorted_indices_data, inverse_data);

  // 2. sorted indices
  Tensor range;
  range.Resize(framework::make_ddim({row + 1}));
  auto range_data_ptr = range.mutable_data<IndexT>(context.GetPlace());
  thrust::sequence(thrust::device, range_data_ptr, range_data_ptr + row + 1);
  int num_out;
  num_out =
      thrust::unique_by_key(thrust::device, sorted_indices_data,
                            sorted_indices_data + row, range_data_ptr, equal)
          .first -
      sorted_indices_data;
  thrust::device_ptr<IndexT> range_data_ptr_dev(range_data_ptr);
  range_data_ptr_dev[num_out] = row;
  sorted_indices->Resize(framework::make_ddim({num_out}));

  // 3. counts: 'counts'
  Tensor* counts = context.Output<Tensor>("Counts");
  counts->Resize(framework::make_ddim({num_out}));
  auto count_data = counts->mutable_data<IndexT>(context.GetPlace());
  thrust::fill(thrust::device, count_data, count_data + row, 0);
  thrust::adjacent_difference(thrust::device, range_data_ptr + 1,
                              range_data_ptr + row + 1, count_data);
}

// Calculate unique when 'axis' is set
template <typename DeviceContext, typename InT, typename IndexT>
static void UniqueDimsCUDATensor(const framework::ExecutionContext& context,
                                 const Tensor& in, Tensor* out,
                                 bool return_index, bool return_inverse,
                                 bool return_counts, int axis) {
  // 1. Transpose & reshape
  // Transpose tensor: eg. axis=1, [dim0, dim1, dim2] -> [dim1, dim0, dim2]
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
  auto& dev_ctx = context.cuda_device_context();
  paddle::operators::math::TransposeFunctor<platform::CUDADeviceContext, InT>
      transpose;
  transpose(dev_ctx,    // device
            in,         // original Tensor
            &in_trans,  // Tensor after reshape
            permute);   // index of axis

  // Reshape tensor: eg. [dim1, dim0, dim2] -> [dim1, dim0*dim2]
  framework::DDim in_trans_flat_dims =
      framework::flatten_to_2d(in_trans_dims, 1);
  in_trans.Resize(in_trans_flat_dims);

  // now 'in_trans' is 2D
  int64_t col = in_trans.dims()[1];
  int64_t row = in_trans.dims()[0];
  const InT* in_trans_data = in_trans.data<InT>();

  Tensor* sorted_indices = context.Output<Tensor>("Indices");
  sorted_indices->Resize(framework::make_ddim({row}));
  auto sorted_indices_data =
      sorted_indices->mutable_data<IndexT>(context.GetPlace());

  // 2. Calculate 'sorted_indices', 'inverse', 'counts'
  // Init index and sort
  thrust::sequence(thrust::device, sorted_indices_data,
                   sorted_indices_data + row);
  thrust::sort(thrust::device, sorted_indices_data, sorted_indices_data + row,
               LessThan<InT>(col, in_trans_data));
  ComputeUniqueDims<InT, IndexT>(
      context, sorted_indices, sorted_indices_data, out, return_index,
      return_inverse, return_counts, BinaryEqual<InT>(col, in_trans_data),
      BinaryNotEqual<InT>(col, in_trans_data), row);

  // 3. Select indices and reshape back to get 'out'
  Tensor out_trans;
  std::vector<int64_t> out_trans_dims_vec = in_trans_dims_vec;
  out_trans_dims_vec[0] = sorted_indices->numel();
  out_trans.Resize(framework::make_ddim(out_trans_dims_vec));
  out_trans.mutable_data<InT>(context.GetPlace());

  IndexSelect<InT, IndexT>(context, in_trans, *sorted_indices, &out_trans, 0);

  std::swap(out_trans_dims_vec[0], out_trans_dims_vec[axis]);
  out->Resize(framework::make_ddim(out_trans_dims_vec));
  out->mutable_data<InT>(context.GetPlace());
  std::vector<framework::Tensor> out_trans_unbind = Unbind(out_trans);
  math::ConcatFunctor<DeviceContext, InT> concat_functor;
  concat_functor(dev_ctx, out_trans_unbind, 0, &out_trans);
  transpose(dev_ctx, out_trans, out, permute);
}

// functor for processing a flattend Tensor
template <typename DeviceContext, typename InT>
struct UniqueFlattendCUDAFunctor {
  const framework::ExecutionContext& ctx_;
  const Tensor& in_;
  Tensor* out_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueFlattendCUDAFunctor(const framework::ExecutionContext& context,
                            const Tensor& in, Tensor* out, bool return_index,
                            bool return_inverse, bool return_counts)
      : ctx_(context),
        in_(in),
        out_(out),
        return_index_(return_index),
        return_inverse_(return_inverse),
        return_counts_(return_counts) {}

  template <typename IndexT>
  void apply() const {
    UniqueFlattendCUDATensor<InT, IndexT>(
        ctx_, in_, out_, return_index_, return_inverse_, return_counts_,
        thrust::equal_to<InT>(), thrust::not_equal_to<InT>(), in_.numel());
  }
};

// functor for processing a multi-dimentional Tensor
template <typename DeviceContext, typename InT>
struct UniqueDimsCUDAFunctor {
  const framework::ExecutionContext& ctx_;
  const Tensor& in_;
  Tensor* out_;
  const int axis_;
  const bool return_index_;
  const bool return_inverse_;
  const bool return_counts_;

  UniqueDimsCUDAFunctor(const framework::ExecutionContext& context,
                        const Tensor& in, Tensor* out, const int axis,
                        bool return_index, bool return_inverse,
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
    UniqueDimsCUDATensor<DeviceContext, InT, IndexT>(
        ctx_, in_, out_, return_index_, return_inverse_, return_counts_, axis_);
  }
};

// Unique_op CUDA implementation.
template <typename InT>
class UniqueKernel<platform::CUDADeviceContext, InT>
    : public framework::OpKernel<InT> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<framework::Tensor>("X");
    auto* out = context.Output<framework::Tensor>("Out");
    auto data_type = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    if (data_type == framework::proto::VarType::INT32) {
      PADDLE_ENFORCE_LE(
          x->numel() + 1, INT_MAX,
          platform::errors::InvalidArgument(
              "The number of elements in Input(X) should be less than or "
              "equal to INT_MAX, but received num is %d. Please set `dtype` to "
              "int64.",
              x->numel()));
    }

    std::vector<int> axis_vec = context.Attr<std::vector<int>>("axis");
    bool return_index = context.Attr<bool>("return_index");
    bool return_inverse = context.Attr<bool>("return_inverse");
    bool return_counts = context.Attr<bool>("return_counts");

    // if 'axis' is not required, flatten the Tensor.
    if (axis_vec.empty()) {
      framework::VisitDataTypeTiny(
          data_type,
          UniqueFlattendCUDAFunctor<platform::CUDADeviceContext, InT>(
              context, *x, out, return_index, return_inverse, return_counts));
    } else {
      // 'axis' is required.
      int axis = axis_vec[0];
      framework::VisitDataTypeTiny(
          data_type, UniqueDimsCUDAFunctor<platform::CUDADeviceContext, InT>(
                         context, *x, out, axis, return_index, return_inverse,
                         return_counts));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    unique, ops::UniqueKernel<paddle::platform::CUDADeviceContext, float>,
    ops::UniqueKernel<paddle::platform::CUDADeviceContext, double>,
    ops::UniqueKernel<paddle::platform::CUDADeviceContext, int32_t>,
    ops::UniqueKernel<paddle::platform::CUDADeviceContext, int64_t>);
