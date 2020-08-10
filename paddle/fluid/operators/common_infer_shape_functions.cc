/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/common_infer_shape_functions.h"

#include <algorithm>
#include <vector>

// This file almostly contains all the infershape functions that are used in
// operators.

namespace paddle {
namespace operators {
namespace details {
inline void GetBroadcastDimsArrays(const framework::DDim &x_dims,
                                   const framework::DDim &y_dims,
                                   int *x_dims_array, int *y_dims_array,
                                   int *out_dims_array, const int max_dim,
                                   const int axis) {
  PADDLE_ENFORCE_GE(
      axis, 0,
      platform::errors::InvalidArgument(
          "Axis should be great than or equal to 0, but received axis is %d.",
          axis));
  PADDLE_ENFORCE_LT(axis, max_dim,
                    platform::errors::InvalidArgument(
                        "Axis should be less than %d, but received axis is %d.",
                        max_dim, axis));
  if (x_dims.size() > y_dims.size()) {
    std::fill(y_dims_array, y_dims_array + axis, 1);
    if (axis + y_dims.size() < max_dim) {
      std::fill(y_dims_array + axis + y_dims.size(), y_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array + axis);
  } else {
    std::fill(x_dims_array, x_dims_array + axis, 1);
    if (axis + x_dims.size() < max_dim) {
      std::fill(x_dims_array + axis + x_dims.size(), x_dims_array + max_dim, 1);
    }
    std::copy(x_dims.Get(), x_dims.Get() + x_dims.size(), x_dims_array + axis);
    std::copy(y_dims.Get(), y_dims.Get() + y_dims.size(), y_dims_array);
  }

  for (int i = 0; i < max_dim; i++) {
    PADDLE_ENFORCE_EQ(
        x_dims_array[i] == y_dims_array[i] || x_dims_array[i] <= 1 ||
            y_dims_array[i] <= 1,
        true, platform::errors::InvalidArgument(
                  "Broadcast dimension mismatch. Operands could "
                  "not be broadcast together with the shape of X = [%s] and "
                  "the shape of Y = [%s]. Received [%d] in X is not equal to "
                  "[%d] in Y at i:%d.",
                  x_dims, y_dims, x_dims_array[i], y_dims_array[i], i));
    if ((x_dims_array[i] > 1 || y_dims_array[i] > 1) ||
        (x_dims_array[i] == 1 && y_dims_array[i] == 1)) {
      out_dims_array[i] = std::max(x_dims_array[i], y_dims_array[i]);
    } else {
      out_dims_array[i] = -1;
    }
  }
}
}  // namespace details

// shape input(0) -> output(0) without change.
void UnaryOpUnchangedInferShape(framework::InferShapeContext *ctx) {
  ctx->ShareDim(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
  ctx->ShareLoD(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
}

// shape input(0) -> output(0) without change, check if axis in range [-Rank(x),
// Rank(x)-1]
void UnaryOpUnchangedInferShapeCheckAxis(framework::InferShapeContext *ctx) {
  auto x_name = ctx->GetInputNameByIdx(0);
  auto x_dim = ctx->GetInputDim(x_name);
  auto x_rank = x_dim.size();
  auto axis = ctx->Attrs().Get<int>("axis");
  PADDLE_ENFORCE_GE(
      axis, -x_rank,
      platform::errors::InvalidArgument(
          "Attr(axis) value should be in range [-R, R-1], "
          "R is the rank of Input(X). But received axis: %d, R: %d.",
          axis, x_rank));
  PADDLE_ENFORCE_LT(
      axis, x_rank,
      platform::errors::InvalidArgument(
          "Attr(axis) value should be in range [-R, R-1], "
          "R is the rank of Input(X). But received axis: %d, R: %d.",
          axis, x_rank));
  ctx->ShareDim(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
  ctx->ShareLoD(ctx->GetInputNameByIdx(0), /*->*/ ctx->GetOutputNameByIdx(0));
}

// broadcast input(0) and input(1) -> output(0)
void BinaryOpBroadcastInferShape(framework::InferShapeContext *ctx) {
  // OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "BinaryOp");
  // OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "BinaryOp");
  // OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "ElementwiseOp");
  auto x_name = ctx->GetInputNameByIdx(0);
  auto y_name = ctx->GetInputNameByIdx(1);
  auto out_name = ctx->GetOutputNameByIdx(0);
  auto x_dims = ctx->GetInputDim(x_name);
  auto y_dims = ctx->GetInputDim(y_name);
  PADDLE_ENFORCE_EQ(
      ctx->GetInputsVarType(y_name).front(),
      framework::proto::VarType::LOD_TENSOR,
      platform::errors::InvalidArgument(
          "The input var's type should be LoDTensor, but the "
          "received is %s [%s].",
          ctx->GetInputsVarType(y_name).front(), ctx->Inputs(y_name).front()));

  if (ctx->GetInputsVarType(x_name).front() ==
      framework::proto::VarType::SELECTED_ROWS) {
    PADDLE_ENFORCE_EQ(
        y_dims.size(), 1u,
        platform::errors::InvalidArgument(
            "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
            "), Y must be scalar, the size of Y should be 1. "
            "But reveived the size of Y = %s.",
            y_dims.size()));
    PADDLE_ENFORCE_EQ(
        y_dims[0], 1,
        platform::errors::InvalidArgument(
            "For elementwise_op, if X is Sparse(VarType.SELECTED_ROWS"
            "), Y must be scalar, the first dimension of Y should be 1. "
            "But reveived the first dimension of Y = %s.",
            y_dims[0]));
  } else if (ctx->GetInputsVarType(x_name).front() !=
             framework::proto::VarType::LOD_TENSOR) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Input X's type[%s] is not supported by elementwise_op. Please set its "
        "type to LOD_TENSOR.",
        ctx->GetInputsVarType(x_name).front()));
  }

  if (x_dims == y_dims) {
    ctx->ShareDim(x_name, /*->*/ out_name);
    ctx->ShareLoD(x_name, /*->*/ out_name);
  } else {
    int max_dim = std::max(x_dims.size(), y_dims.size());
    int axis = ctx->Attrs().Get<int>("axis");
    axis = (axis == -1 ? std::abs(x_dims.size() - y_dims.size()) : axis);
    std::vector<int> x_dims_array(max_dim);
    std::vector<int> y_dims_array(max_dim);
    std::vector<int> out_dims_array(max_dim);
    details::GetBroadcastDimsArrays(x_dims, y_dims, x_dims_array.data(),
                                    y_dims_array.data(), out_dims_array.data(),
                                    max_dim, axis);
    ctx->SetOutputDim(out_name, framework::make_ddim(out_dims_array));
    // to do
    ctx->ShareLoD(x_name, /*->*/ out_name);
  }
}

}  // namespace operators
}  // namespace paddle
