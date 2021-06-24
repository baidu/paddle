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

#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dot_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/reduce_ops/reduce_sum_op.h"

#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#endif

namespace paddle {
namespace operators {

using framework::Tensor;

struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  template <typename U>
  HOSTDEVICE inline U operator()(const U& x) const {
    return x;
  }
};

template <typename DeviceContext, typename T>
void ReduceSumForMatmulGrad(const Tensor* input, Tensor* output,
                            const std::vector<int>& reduce_dims,
                            const paddle::framework::ExecutionContext& ctx) {
#if defined(__NVCC__) || defined(__HIPCC__)
  auto stream = ctx.cuda_device_context().stream();
  TensorReduce<T, T, cub::Sum, IdentityFunctor>(*input, output, reduce_dims,
                                                static_cast<T>(0), cub::Sum(),
                                                IdentityFunctor(), stream);
#else
  ReduceKernelFunctor<DeviceContext, T, ops::SumFunctor>(
      input, output, reduce_dims, true, false, ctx)
      .template apply<T>();
#endif
}

static void GetBroadcastFromDims(const int x_ndim, const std::int64_t* x_dims,
                                 const int y_ndim, const std::int64_t* y_dims,
                                 std::int64_t* x_bd_dims,
                                 std::int64_t* y_bd_dims,
                                 std::int64_t* out_bd_dims) {
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::fill(x_bd_dims, x_bd_dims + ndim - x_ndim, 1);
  std::fill(y_bd_dims, y_bd_dims + ndim - y_ndim, 1);
  std::copy(x_dims, x_dims + x_ndim, x_bd_dims + ndim - x_ndim);
  std::copy(y_dims, y_dims + y_ndim, y_bd_dims + ndim - y_ndim);

  for (int i = 0; i < ndim; ++i) {
    PADDLE_ENFORCE_EQ(
        x_bd_dims[i] == y_bd_dims[i] || x_bd_dims[i] <= 1 || y_bd_dims[i] <= 1,
        true,
        platform::errors::InvalidArgument(
            "Input(X) and Input(Y) has error dim."
            "X_broadcast's shape[%s] must be equal to Y_broadcast's shape[%s],"
            "or X_broadcast's shape[%s] <= 1, or Y_broadcast's shape[%s] <= 1,"
            "But received X_broadcast's shape[%s] = [%s]"
            "received Y_broadcast's shape[%s] = [%s]",
            i, i, i, i, i, x_bd_dims[i], i, y_bd_dims[i]));
    if (x_bd_dims[i] == 0 || y_bd_dims[i] == 0) {
      out_bd_dims[i] = 0;
    } else {
      out_bd_dims[i] = (std::max)(x_bd_dims[i], y_bd_dims[i]);
    }
  }
}

static int64_t GetIndexMessage(const int n, const int64_t* dims,
                               const int64_t* index) {
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (dims[i] > 1) {
      sum = sum * dims[i] + index[i];
    }
  }
  return sum;
}

static void IndexIncreaseFromDims(const int ndim, const int64_t* dims,
                                  int64_t* index) {
  for (int i = ndim - 1; i >= 0; --i) {
    ++index[i];
    if (index[i] >= dims[i]) {
      index[i] -= dims[i];
    } else {
      break;
    }
  }
}

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y,
                    const std::vector<std::int64_t>& x_dims,
                    const std::vector<std::int64_t>& y_dims, Tensor* Out,
                    bool trans_x, bool trans_y,
                    const paddle::framework::ExecutionContext& ctx) {
  const int x_ndim = x_dims.size();
  const int y_ndim = y_dims.size();

  // get data ptr
  const T* x_data = X->data<T>();
  const T* y_data = Y->data<T>();

  if (x_ndim == 1 && y_ndim == 1) {
    PADDLE_ENFORCE_EQ(
        X->numel(), Y->numel(),
        platform::errors::InvalidArgument(
            "X's numbers must be equal to Y's numbers,"
            "when X/Y's dims =1. But received X has [%d] elements,"
            "received Y has [%d] elements",
            X->numel(), Y->numel()));
    VLOG(3) << "MatMul's case 1";
    Out->Resize({1});
    Out->mutable_data<T>(ctx.GetPlace());
    auto out_eigen = framework::EigenScalar<T>::From(*Out);
    auto x_eigen = framework::EigenVector<T>::Flatten(*X);
    auto y_eigen = framework::EigenVector<T>::Flatten(*Y);

    auto& dev = *ctx.template device_context<DeviceContext>().eigen_device();
    out_eigen.device(dev) = (x_eigen * y_eigen).sum();
    return;
  }

  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

  if (x_ndim == 1) {
    const int N = X->numel();
    if (trans_y) {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], N,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 1, N, y_ndim - 1, y_dims[y_ndim - 1]));
    } else {
      PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], N,
                        platform::errors::InvalidArgument(
                            "Input(Y) has error dim."
                            "Y'dims[%d] must be equal to %d"
                            "But received Y'dims[%d] is %d",
                            y_ndim - 2, N, y_ndim - 2, y_dims[y_ndim - 2]));
    }
    std::vector<std::int64_t> out_dims(y_ndim - 1);
    if (trans_y) {
      std::copy_n(y_dims.cbegin(), y_ndim - 1, out_dims.begin());
    } else {
      std::copy_n(y_dims.cbegin(), y_ndim - 2, out_dims.begin());
      out_dims.back() = y_dims.back();
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());
    if (trans_y) {
      const int M = Y->numel() / N;
      VLOG(3) << "MatMul's case 2";
      blas.GEMV(false, M, N, static_cast<T>(1), y_data, x_data,
                static_cast<T>(0), Out->data<T>());
    } else {
      const int M = y_dims[y_ndim - 1];
      const int batch_size = Y->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 3";
        blas.GEMV(true, N, M, static_cast<T>(1), y_data, x_data,
                  static_cast<T>(0), Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 4";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, static_cast<T>(1),
                         y_data, x_data, static_cast<T>(0), Out->data<T>(),
                         batch_size, M * N, 0);
      }
    }
    return;
  }

  if (y_ndim == 1) {
    const int N = Y->numel();
    if (trans_x) {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 2], N,
                        platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 2, N, x_ndim - 2, x_dims[x_ndim - 2]));
    } else {
      PADDLE_ENFORCE_EQ(x_dims[x_ndim - 1], N,
                        platform::errors::InvalidArgument(
                            "Input(X) has error dim."
                            "X'dims[%d] must be equal to %d"
                            "But received X'dims[%d] is %d",
                            x_ndim - 1, N, x_ndim - 1, x_dims[x_ndim - 1]));
    }
    std::vector<std::int64_t> out_dims(x_ndim - 1);
    if (trans_x) {
      std::copy_n(x_dims.cbegin(), x_ndim - 2, out_dims.begin());
      out_dims.back() = x_dims.back();
    } else {
      std::copy_n(x_dims.cbegin(), x_ndim - 1, out_dims.begin());
    }
    Out->Resize(framework::make_ddim(out_dims));
    Out->mutable_data<T>(ctx.GetPlace());

    if (trans_x) {
      const int M = x_dims[x_ndim - 1];
      const int batch_size = X->numel() / (M * N);
      if (batch_size == 1) {
        VLOG(3) << "MatMul's case 5";
        blas.GEMV(true, N, M, static_cast<T>(1), x_data, y_data,
                  static_cast<T>(0), Out->data<T>());
      } else {
        VLOG(3) << "MatMul's case 6";
        blas.BatchedGEMM(CblasTrans, CblasNoTrans, M, 1, N, static_cast<T>(1),
                         x_data, y_data, static_cast<T>(0), Out->data<T>(),
                         batch_size, M * N, 0);
      }
    } else {
      const int M = X->numel() / N;
      VLOG(3) << "MatMul's case 7";
      blas.GEMV(false, M, N, static_cast<T>(1), x_data, y_data,
                static_cast<T>(0), Out->data<T>());
    }
    return;
  }

  const int M = trans_x ? x_dims[x_ndim - 1] : x_dims[x_ndim - 2];
  const int K = trans_x ? x_dims[x_ndim - 2] : x_dims[x_ndim - 1];
  if (trans_y) {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 1], K,
                      platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 1, K, y_ndim - 1, y_dims[y_ndim - 1]));
  } else {
    PADDLE_ENFORCE_EQ(y_dims[y_ndim - 2], K,
                      platform::errors::InvalidArgument(
                          "Input(Y) has error dim."
                          "Y'dims[%d] must be equal to %d"
                          "But received Y'dims[%d] is %d",
                          y_ndim - 2, K, y_ndim - 2, y_dims[y_ndim - 2]));
  }
  const int N = trans_y ? y_dims[y_ndim - 2] : y_dims[y_ndim - 1];
  const int ndim = (std::max)(x_ndim, y_ndim);
  std::vector<std::int64_t> x_broadcast_dims(ndim);
  std::vector<std::int64_t> y_broadcast_dims(ndim);
  std::vector<std::int64_t> out_broadcast_dims(ndim);

  GetBroadcastFromDims(x_ndim - 2, x_dims.data(), y_ndim - 2, y_dims.data(),
                       x_broadcast_dims.data(), y_broadcast_dims.data(),
                       out_broadcast_dims.data());

  out_broadcast_dims[ndim - 2] = M;
  out_broadcast_dims[ndim - 1] = N;

  Out->Resize(framework::make_ddim(out_broadcast_dims));
  Out->mutable_data<T>(ctx.GetPlace());

  const int batch_dim = ndim - 2;
  // broadcast message
  const bool is_broadcast_dims = !std::equal(
      x_broadcast_dims.cbegin(), x_broadcast_dims.cbegin() + batch_dim,
      y_broadcast_dims.cbegin());

  const std::int64_t x_batch_size = std::accumulate(
      x_broadcast_dims.cbegin(), x_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t y_batch_size = std::accumulate(
      y_broadcast_dims.cbegin(), y_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  const std::int64_t out_batch_size = std::accumulate(
      out_broadcast_dims.cbegin(), out_broadcast_dims.cbegin() + batch_dim, 1LL,
      std::multiplies<std::int64_t>());
  if (out_batch_size == 0) return;
  if (x_batch_size == 1 && y_batch_size == 1) {
    VLOG(3) << "MatMul's case 8";
    blas.GEMM(trans_x ? CblasTrans : CblasNoTrans,
              trans_y ? CblasTrans : CblasNoTrans, M, N, K, static_cast<T>(1),
              x_data, y_data, static_cast<T>(0), Out->data<T>());
  } else if (x_batch_size == 1) {
    if (M == 1 && trans_y) {
      VLOG(3) << "MatMul's case 9";
      blas.GEMV(false, y_batch_size * N, K, static_cast<T>(1), y_data, x_data,
                static_cast<T>(0), Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 10";
      blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                       trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       static_cast<T>(1), x_data, y_data, static_cast<T>(0),
                       Out->data<T>(), out_batch_size, 0, K * N);
    }
  } else if (y_batch_size == 1) {
    if (!trans_x) {
      VLOG(3) << "MatMul's case 11";
      blas.GEMM(CblasNoTrans, trans_y ? CblasTrans : CblasNoTrans,
                x_batch_size * M, N, K, static_cast<T>(1), x_data, y_data,
                static_cast<T>(0), Out->data<T>());
    } else {
      VLOG(3) << "MatMul's case 12";
      blas.BatchedGEMM(CblasTrans, trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                       static_cast<T>(1), x_data, y_data, static_cast<T>(0),
                       Out->data<T>(), out_batch_size, M * K, 0);
    }
  } else if (!is_broadcast_dims) {
    VLOG(3) << "MatMul's case 13";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                     static_cast<T>(1), x_data, y_data, static_cast<T>(0),
                     Out->data<T>(), out_batch_size, M * K, K * N);
  } else {
    // in the case, can't use stridedgemm
    std::vector<const T*> x_ptr(out_batch_size);
    std::vector<const T*> y_ptr(out_batch_size);
    std::vector<T*> out_ptr(out_batch_size);
    std::vector<std::int64_t> index(batch_dim, 0);
    for (std::int64_t i = 0; i < out_batch_size; ++i) {
      // using the index to get offset
      const std::int64_t x_index =
          GetIndexMessage(batch_dim, x_broadcast_dims.data(), index.data());
      const std::int64_t y_index =
          GetIndexMessage(batch_dim, y_broadcast_dims.data(), index.data());

      x_ptr[i] = x_data + x_index * M * K;
      y_ptr[i] = y_data + y_index * K * N;
      out_ptr[i] = Out->data<T>() + i * M * N;
      IndexIncreaseFromDims(batch_dim, out_broadcast_dims.data(), index.data());
    }
    VLOG(3) << "MatMul's case 14";
    blas.BatchedGEMM(trans_x ? CblasTrans : CblasNoTrans,
                     trans_y ? CblasTrans : CblasNoTrans, M, N, K,
                     static_cast<T>(1), x_ptr.data(), y_ptr.data(),
                     static_cast<T>(0), out_ptr.data(), out_batch_size);
  }
}

template <typename DeviceContext, typename T>
void MatMulFunction(const Tensor* X, const Tensor* Y, Tensor* Out, bool trans_x,
                    bool trans_y,
                    const paddle::framework::ExecutionContext& ctx) {
  const std::vector<std::int64_t> x_dims = vectorize(X->dims());
  const std::vector<std::int64_t> y_dims = vectorize(Y->dims());
  MatMulFunction<DeviceContext, T>(X, Y, x_dims, y_dims, Out, trans_x, trans_y,
                                   ctx);
}

template <typename DeviceContext, typename T>
class MatMulV2Kernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto* X = ctx.Input<Tensor>("X");
    auto* Y = ctx.Input<Tensor>("Y");
    auto* Out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    MatMulFunction<DeviceContext, T>(X, Y, Out, trans_x, trans_y, ctx);
  }
};

// Reshape a rank-3 tensor from P x M x N to (P * M) x N.
// Identity op if the tensor is not of rank 3.
static framework::Tensor FoldInitDims(const framework::Tensor& input) {
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
static framework::Tensor FoldHeadAndLastDims(const DeviceContext& context,
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

/**
 * Get row matrix shape from a vector shape. If the rank of x_dim > 1, the
 * original x_dim is returned.
 */
static framework::DDim RowMatrixFromVector(const framework::DDim& x_dim) {
  if (x_dim.size() > 1) {
    return x_dim;
  }
  return framework::make_ddim({1, x_dim[0]});
}

/**
 * Get column matrix shape from a vector shape. If the ran of y_dim > 1, the
 * original y_dim is returned.
 */
static framework::DDim ColumnMatrixFromVector(const framework::DDim& y_dim) {
  if (y_dim.size() > 1) {
    return y_dim;
  }
  return framework::make_ddim({y_dim[0], 1});
}

/**
 * Reshape a tensor to 3-D or 2-D tensor by matrix descriptor.
 *
 * The shape would be [BatchSize, H, W] or [H, W].
 * If transposed, `H,W` will be swapped.
 */
static void ReshapeTensorIntoMatrixSequence(
    framework::Tensor* x, const math::MatDescriptor& descriptor) {
  int64_t h, w;
  h = descriptor.height_;
  w = descriptor.width_;
  if (descriptor.trans_) {
    std::swap(w, h);
  }
  if (descriptor.batch_size_) {
    x->Resize({descriptor.batch_size_, h, w});
  } else {
    x->Resize({h, w});
  }
}

static void ReshapeXYOutIntoMatrixSequence(framework::Tensor* x,
                                           framework::Tensor* y,
                                           framework::Tensor* out, bool trans_x,
                                           bool trans_y) {
  auto x_dim = RowMatrixFromVector(x->dims());
  auto y_dim = ColumnMatrixFromVector(y->dims());
  auto mat_dim_x = math::CreateMatrixDescriptor(x_dim, 0, trans_x);
  auto mat_dim_y = math::CreateMatrixDescriptor(y_dim, 0, trans_y);
  if (mat_dim_x.batch_size_ == 0 && mat_dim_y.batch_size_ == 0) {
    out->Resize({mat_dim_x.height_, mat_dim_y.width_});
  } else {
    out->Resize({(std::max)(mat_dim_x.batch_size_, mat_dim_y.batch_size_),
                 mat_dim_x.height_, mat_dim_y.width_});
  }

  ReshapeTensorIntoMatrixSequence(x, mat_dim_x);
  ReshapeTensorIntoMatrixSequence(y, mat_dim_y);
}

template <typename DeviceContext, typename T>
struct ConjHelper {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}
  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    dst.set_layout(src.layout());
    dst.ShareDataWith(src);
    return;
  }

  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<float>> {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}

  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<float>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<float>>(
        ctx_.GetPlace(),
        size_t(src.numel() * sizeof(paddle::platform::complex<float>)));

    platform::ForRange<DeviceContext> for_range(
        ctx_.template device_context<DeviceContext>(), src.numel());
    math::ConjFunctor<paddle::platform::complex<float>> functor(
        src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext>
struct ConjHelper<DeviceContext, paddle::platform::complex<double>> {
  explicit ConjHelper(const framework::ExecutionContext& ctx) : ctx_(ctx) {}

  HOSTDEVICE void operator()(framework::Tensor& src, framework::Tensor& dst) {
    dst.Resize(src.dims());
    auto* src_data = src.data<paddle::platform::complex<double>>();
    auto* dst_data = dst.mutable_data<paddle::platform::complex<double>>(
        ctx_.GetPlace(),
        size_t(src.numel() * sizeof(paddle::platform::complex<double>)));

    platform::ForRange<DeviceContext> for_range(
        ctx_.template device_context<DeviceContext>(), src.numel());
    math::ConjFunctor<paddle::platform::complex<double>> functor(
        src_data, src.numel(), dst_data);
    for_range(functor);
    return;
  }
  const framework::ExecutionContext& ctx_;
};

template <typename DeviceContext, typename T>
class MatMulV2GradKernel : public framework::OpKernel<T> {
 public:
  void MatMul(const framework::ExecutionContext& context,
              const framework::Tensor& a, bool trans_a,
              const framework::Tensor& b, bool trans_b,
              framework::Tensor* out) const {
    out->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<DeviceContext, T>(context);
    auto mat_dim_a = math::CreateMatrixDescriptor(a.dims(), 0, trans_a);
    auto mat_dim_b = math::CreateMatrixDescriptor(b.dims(), 0, trans_b);
    if (a.dims().size() == 3 && b.dims().size() <= 2) {
      // the transpose_X must be false, if is true, the transpose cost much time
      if (!trans_a) {
        mat_dim_a.height_ *= mat_dim_a.batch_size_;
        mat_dim_a.batch_size_ = 0;
      }
    }
    blas.MatMul(a, mat_dim_a, b, mat_dim_b, static_cast<T>(1), out,
                static_cast<T>(0));
  }

  void CalcInputGrad(const framework::ExecutionContext& context,
                     const framework::Tensor& a, bool trans_a,
                     bool is_fold_init_dims_a, const framework::Tensor& b,
                     bool trans_b, bool is_fold_init_dims_b,
                     framework::Tensor* out) const {
    if (out == nullptr) return;
    bool need_combine = (a.dims().size() == 3 || b.dims().size() == 3) &&
                        out->dims().size() == 2;
    if (!need_combine) {
      MatMul(context, a, trans_a, b, trans_b, out);
    } else {
      auto& ctx = context.template device_context<DeviceContext>();
      MatMul(context, is_fold_init_dims_a
                          ? FoldInitDims(a)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, a),
             trans_a, is_fold_init_dims_b
                          ? FoldInitDims(b)
                          : FoldHeadAndLastDims<DeviceContext, T>(ctx, b),
             trans_b, out);
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    bool transpose_x = ctx.Attr<bool>("trans_x");
    bool transpose_y = ctx.Attr<bool>("trans_y");

    auto x = *ctx.Input<framework::Tensor>("X");
    auto y = *ctx.Input<framework::Tensor>("Y");
    auto dout = *ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor y_conj(y.type());
    framework::Tensor x_conj(y.type());

    // get dims
    std::vector<std::int64_t> x_dims = vectorize(x.dims());
    std::vector<std::int64_t> y_dims = vectorize(y.dims());
    std::vector<std::int64_t> dout_dims = vectorize(dout.dims());

    int x_ndim = x_dims.size();
    int y_ndim = y_dims.size();
    int ndim = dout_dims.size();

    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    // Case1 : x's or y's dim = 1
    if (x_ndim == 1 && y_ndim == 1) {
      if (dx) dx->mutable_data<T>(ctx.GetPlace());
      if (dy) dy->mutable_data<T>(ctx.GetPlace());
      if (dout.numel() == 1) {
        DotGradFunction<DeviceContext, T>()(&x, &y, &dout, dx, dy, ctx);
        return;
      }
    }

    bool is_broadcast = true;
    if (x_ndim <= 2 || y_ndim <= 2) {
      is_broadcast = false;
    } else if (x_ndim != y_ndim) {
      is_broadcast = true;
    } else {
      is_broadcast = !std::equal(x_dims.cbegin(), x_dims.cbegin() + x_ndim - 2,
                                 y_dims.cbegin());
    }

    // Case2: no broadcast or no batch size, it aims to speed and it is same as
    // matmul in old version.
    if (!is_broadcast) {
      ReshapeXYOutIntoMatrixSequence(&x, &y, &dout, transpose_x, transpose_y);
      framework::DDim dx_dims;
      if (dx) {
        dx_dims = dx->dims();
        if (dx_dims != x.dims()) {
          dx->Resize(x.dims());
        }

        // for complex
        ConjHelper<DeviceContext, T> conj_helper(ctx);
        conj_helper(y, y_conj);
      }

      framework::DDim dy_dims;
      if (dy) {
        dy_dims = dy->dims();
        if (dy_dims != y.dims()) {
          dy->Resize(y.dims());
        }

        // for complex
        ConjHelper<DeviceContext, T> conj_helper(ctx);
        conj_helper(x, x_conj);
      }
      if (transpose_x && transpose_y) {
        CalcInputGrad(ctx, y_conj, true, true, dout, true, false, dx);
        CalcInputGrad(ctx, dout, true, true, x_conj, true, false, dy);
      } else if (transpose_x) {
        CalcInputGrad(ctx, y_conj, false, false, dout, true, false, dx);
        CalcInputGrad(ctx, x_conj, false, false, dout, false, true, dy);
      } else if (transpose_y) {
        CalcInputGrad(ctx, dout, false, false, y_conj, false, true, dx);
        CalcInputGrad(ctx, dout, true, true, x_conj, false, true, dy);
      } else {
        CalcInputGrad(ctx, dout, false, false, y_conj, true, false, dx);
        CalcInputGrad(ctx, x_conj, true, true, dout, false, true, dy);
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
    } else {
      // Case3: broadcast. It need cost much time to reduce sum for the
      // broadcast and wastes the memory.
      // So we should avoid the case in reality.
      VLOG(3) << "It need cost much time to reduce sum for the broadcast and "
                 "wastes the memory. So we should avoid the case in reality";
      Tensor dx_help, dy_help;

      ConjHelper<DeviceContext, T> conj_helper(ctx);
      conj_helper(x, x_conj);
      conj_helper(y, y_conj);
      if (transpose_x) {
        if (transpose_y) {
          // X'Y': dA = Y'G', dB = G'X'
          if (dx)
            MatMulFunction<DeviceContext, T>(&y_conj, &dout, y_dims, dout_dims,
                                             &dx_help, true, true, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&dout, &x_conj, dout_dims, x_dims,
                                             &dy_help, true, true, ctx);
        } else {
          // X'Y: dX = YG', dY = XG
          if (dx)
            MatMulFunction<DeviceContext, T>(&y_conj, &dout, y_dims, dout_dims,
                                             &dx_help, false, true, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&x_conj, &dout, x_dims, dout_dims,
                                             &dy_help, false, false, ctx);
        }
      } else {
        if (transpose_y) {
          // XY': dX = GY, dY = G'X
          if (dx)
            MatMulFunction<DeviceContext, T>(&dout, &y_conj, dout_dims, y_dims,
                                             &dx_help, false, false, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&dout, &x_conj, dout_dims, x_dims,
                                             &dy_help, true, false, ctx);
        } else {
          // XY: dX = GY', dY = X'G
          if (dx)
            MatMulFunction<DeviceContext, T>(&dout, &y_conj, dout_dims, y_dims,
                                             &dx_help, false, true, ctx);
          if (dy)
            MatMulFunction<DeviceContext, T>(&x_conj, &dout, x_dims, dout_dims,
                                             &dy_help, true, false, ctx);
        }
      }

      // get help dims
      const std::vector<std::int64_t> dx_help_dims = vectorize(dx_help.dims());
      const std::vector<std::int64_t> dy_help_dims = vectorize(dy_help.dims());

      std::vector<std::int64_t> dx_broadcast_dims(ndim);
      std::vector<std::int64_t> dy_broadcast_dims(ndim);

      std::fill(dx_broadcast_dims.data(),
                dx_broadcast_dims.data() + ndim - x_ndim, 1);
      std::fill(dy_broadcast_dims.data(),
                dy_broadcast_dims.data() + ndim - y_ndim, 1);
      std::copy(x_dims.data(), x_dims.data() + x_ndim,
                dx_broadcast_dims.data() + ndim - x_ndim);
      std::copy(y_dims.data(), y_dims.data() + y_ndim,
                dy_broadcast_dims.data() + ndim - y_ndim);

      std::vector<int> dx_reduce_dims;
      std::vector<int> dy_reduce_dims;
      for (int idx = 0; idx <= ndim - 3; idx++) {
        if (dx_help_dims[idx] != 1 && dx_broadcast_dims[idx] == 1) {
          dx_reduce_dims.push_back(idx);
        }
        if (dy_help_dims[idx] != 1 && dy_broadcast_dims[idx] == 1) {
          dy_reduce_dims.push_back(idx);
        }
      }
      // reduce sum to get grad by ReduceSum
      if (dx) {
        if (dx_reduce_dims.empty()) {
          *dx = std::move(dx_help);
        } else {
          ReduceSumForMatmulGrad<DeviceContext, T>(&dx_help, dx, dx_reduce_dims,
                                                   ctx);
        }
        dx->Resize(x.dims());
      }
      if (dy) {
        if (dy_reduce_dims.empty()) {
          *dy = std::move(dy_help);
        } else {
          ReduceSumForMatmulGrad<DeviceContext, T>(&dy_help, dy, dy_reduce_dims,
                                                   ctx);
        }
        dy->Resize(y.dims());
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
