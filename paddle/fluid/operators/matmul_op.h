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
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/safe_ref.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
inline framework::DDim GetXDim(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

inline framework::DDim GetYDim(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

template <typename DeviceContext, typename T>
class MatMulKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto& x =
        detail::Ref(context.Input<framework::Tensor>("X"), "Cannot find X");
    auto& y =
        detail::Ref(context.Input<framework::Tensor>("Y"), "Cannot find Y");
    auto* out = context.Output<framework::Tensor>("Out");
    out->mutable_data<T>(context.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::GetMatDim(GetXDim(x.dims()), 0,
                                     context.Attr<bool>("transpose_X"));
    auto mat_dim_b = math::GetMatDim(GetYDim(y.dims()), 0,
                                     context.Attr<bool>("transpose_Y"));
    blas.MatMul(x, mat_dim_a, y, mat_dim_b, T(1), out, T(0));
  }
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
inline framework::Tensor CombineBatchAndM(const framework::Tensor& input) {
  auto output = input;
  auto in_dims = input.dims();
  if (in_dims.size() == 3) {
    output.Resize({in_dims[0] * in_dims[1], in_dims[2]});
  }
  return output;
}

// Reshape a rank-3 tensor from P x M x N to M x (P * N).
// (Warning: This requires transposing data and writes into new memory.)
// Identity op if the tensor is not of rank 3.
template <typename DeviceContext, typename T>
inline framework::Tensor CombineBatchAndN(const DeviceContext& context,
                                          const framework::Tensor& input) {
  auto in_dims = input.dims();
  if (in_dims.size() != 3) {
    return input;
  }
  framework::Tensor output;
  output.Resize({in_dims[1], in_dims[0], in_dims[2]});
  output.mutable_data<T>(context.GetPlace());
  std::vector<int> axis = {1, 0, 2};
  math::Transpose<DeviceContext, T, 3> trans;
  trans(context, input, &output, axis);
  output.Resize({in_dims[1], in_dims[0] * in_dims[2]});

  return output;
}

inline void NormalizeTensorShape(framework::Tensor* x,
                                 const math::MatDescriptor& mat_dim_x) {
  int64_t h, w;
  h = mat_dim_x.height_;
  w = mat_dim_x.width_;
  if (mat_dim_x.trans_) {
    std::swap(w, h);
  }
  if (mat_dim_x.batch_size_) {
    x->Resize({mat_dim_x.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

inline void NormalizeXYOutTensorShape(framework::Tensor* x,
                                      framework::Tensor* y,
                                      framework::Tensor* out, bool trans_a,
                                      bool trans_b) {
  auto x_dim = GetXDim(x->dims());
  auto y_dim = GetYDim(y->dims());
  auto mat_dim_x = math::GetMatDim(x_dim, 0, trans_a);
  auto mat_dim_y = math::GetMatDim(y_dim, 0, trans_b);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({std::max(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  NormalizeTensorShape(x, mat_dim_x);
  NormalizeTensorShape(y, mat_dim_y);
}

// Using dimensional constraints on matrix multiplication, it is
// straight-forward to check the following table for when X and Y
// are both matrices.
//
// transpose_X | False    | True     | False    | True
// transpose_Y | False    | False    | True     | True
// -----------+----------+----------+----------+-----------
//        dX = | dOut Y^T | Y dOut^T | dOut Y   | Y^T dOut^T
//        dY = | X^T dOut | X dOut   | dOut^T X | dOut^T X^T
//
// When X is a vector of size K, we treat it instead as a matrix of shape
// (1, K). Similarly, when Y is a vector of size K, we treat it instead as
// a matrix of shape (K, 1).
//
// When X and Y are both 3-dimensional tensors, then the first dimension
// the batch dimension can be ignored and the exact same formulas apply
// as for two matrices.
//
// Finally, when, e.g., X is a 3-dimensional tensor but Y is a matrix, we end
// up with formulas like
//
//   dY_{ij} = \sum_{p, m} X_{pmi} dOut_{pmj}
//
// To handle this sort of scenario, we reshape X : P x M x K, dOut: P x M x N
// to X: (P * M) x K, dOut: (P * M) x N.
template <typename DeviceContext, typename T>
class MatMulGradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& context,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::GetMatDim(a.dims(), 0, trans_a);
    auto mat_dim_b = math::GetMatDim(b.dims(), 0, trans_b);
    blas.MatMul(a, mat_dim_a, b, mat_dim_b, T(1), out, T(0));
  }

  void CalcInputGrad(const framework::ExecutionContext& context,
                     const framework::Tensor& a, bool trans_a,
                     bool is_combine_m_a, const framework::Tensor& b,
                     bool trans_b, bool is_combine_m_b,
                     framework::Tensor* out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      auto& ctx = context.template device_context<DeviceContext>();
      MatMul(
          context, is_combine_m_a ? CombineBatchAndM(a)
                                  : CombineBatchAndN<DeviceContext, T>(ctx, a),
          trans_a, is_combine_m_b ? CombineBatchAndM(b)
                                  : CombineBatchAndN<DeviceContext, T>(ctx, b),
          trans_b, out);
    }
  }

  void Compute(const framework::ExecutionContext& context) const override {
    auto x = *context.Input<framework::Tensor>("X");
    auto y = *context.Input<framework::Tensor>("Y");
    auto dout =
        *context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* dx = context.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = context.Output<framework::Tensor>(framework::GradVarName("Y"));
    bool transpose_x = context.Attr<bool>("transpose_X");
    bool transpose_y = context.Attr<bool>("transpose_Y");

    NormalizeXYOutTensorShape(&x, &y, &dout, transpose_x, transpose_y);
    framework::DDim dx_dims;
    if (dx) {
      dx_dims = dx->dims();
      if (dx_dims != x.dims()) {
        dx->Resize(x.dims());
      }
    }

    framework::DDim dy_dims;
    if (dy) {
      dy_dims = dy->dims();
      if (dy_dims != y.dims()) {
        dy->Resize(y.dims());
      }
    }

    if (transpose_x && transpose_y) {
      CalcInputGrad(context, y, true, true, dout, true, false, dx);
      CalcInputGrad(context, dout, true, true, x, true, false, dy);
    } else if (transpose_x && !transpose_y) {
      CalcInputGrad(context, y, false, false, dout, true, false, dx);
      CalcInputGrad(context, x, false, false, dout, false, true, dy);
    } else if (!transpose_x && transpose_y) {
      CalcInputGrad(context, dout, false, false, y, false, true, dx);
      CalcInputGrad(context, dout, true, true, x, false, true, dy);
    } else {
      CalcInputGrad(context, dout, false, false, y, true, false, dx);
      CalcInputGrad(context, x, true, true, dout, false, true, dy);
    }

    if (dx) {
      if (dx_dims != x.dims()) {
        dx->Resize(dx_dims);
      }
    }
    if (dy) {
      if (dy_dims != y.dims()) {
        dy->Resize(dy_dims);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
