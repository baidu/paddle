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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "thrust/device_vector.h"
#endif

namespace paddle {
namespace operators {

// Process an element in the output, used with a parallel-for
template <typename T>
struct KronElemFunctor {
  KronElemFunctor(const T* a, const T* b, T* out, const int64_t* shape_b,
                  const int64_t* stride_a, const int64_t* stride_b,
                  const int64_t* stride_out, int ndims)
      : a_(a),
        b_(b),
        out_(out),
        shape_b_(shape_b),
        stride_a_(stride_a),
        stride_b_(stride_b),
        stride_out_(stride_out),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) const {
    // it computes 1 element in the output
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_out_[i];
      index = index % stride_out_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }
    out_[idx] = a_[index_a] * b_[index_b];
  }

 private:
  const T* a_;
  const T* b_;
  T* out_;
  const int64_t* shape_b_;
  const int64_t* stride_a_;
  const int64_t* stride_b_;
  const int64_t* stride_out_;
  const int ndims_;
};

template <typename DeviceContext, typename T>
struct KronOpFunctor {
  void operator()(const DeviceContext& dev_ctx, const framework::Tensor& x,
                  const framework::Tensor& y, framework::Tensor* out) {
    int ndims = out->dims().size();
    int64_t numel = out->numel();

    const framework::DDim& dim_x = x.dims();
    const framework::DDim& dim_y = y.dims();
    const framework::DDim& dim_out = out->dims();
    const framework::DDim stride_x = framework::stride(dim_x);
    const framework::DDim stride_y = framework::stride(dim_y);
    const framework::DDim stride_out = framework::stride(dim_out);

    const int64_t *p_stride_x = nullptr, *p_stride_y = nullptr,
                  *p_stride_out = nullptr, *p_shape_y = nullptr;
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> d_stride_x(ndims);
    thrust::device_vector<int64_t> d_stride_y(ndims);
    thrust::device_vector<int64_t> d_stride_out(ndims);
    thrust::device_vector<int64_t> d_shape_y(ndims);
    thrust::copy(stride_x.Get(), stride_x.Get() + ndims, d_stride_x.begin());
    thrust::copy(stride_y.Get(), stride_y.Get() + ndims, d_stride_y.begin());
    thrust::copy(stride_out.Get(), stride_out.Get() + ndims,
                 d_stride_out.begin());
    thrust::copy(dim_y.Get(), dim_y.Get() + ndims, d_shape_y.begin());

    p_stride_x = thrust::raw_pointer_cast(d_stride_x.data());
    p_stride_y = thrust::raw_pointer_cast(d_stride_y.data());
    p_stride_out = thrust::raw_pointer_cast(d_stride_out.data());
    p_shape_y = thrust::raw_pointer_cast(d_shape_y.data());
#else
    p_stride_x = stride_x.Get();
    p_stride_y = stride_y.Get();
    p_stride_out = stride_out.Get();
    p_shape_y = dim_y.Get();
#endif

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    KronElemFunctor<T> functor(x.data<T>(), y.data<T>(), out->data<T>(),
                               p_shape_y, p_stride_x, p_stride_y, p_stride_out,
                               ndims);
    for_range(functor);
  }
};

template <typename T>
struct KronGradElemFunctor {
  KronGradElemFunctor(const T* dout, const T* A, const T* B, T* dout_a,
                      T* dout_b, const int64_t* stride_dout,
                      const int64_t* stride_a, const int64_t* stride_b,
                      const int64_t* shape_b, const int64_t numel_a,
                      const int64_t numel_b, const int ndims)
      : dout_(dout),
        A_(A),
        B_(B),
        dout_a_(dout_a),
        dout_b_(dout_b),
        stride_dout_(stride_dout),
        stride_a_(stride_a),
        stride_b_(stride_b),
        shape_b_(shape_b),
        numel_a_(numel_a),
        numel_b_(numel_b),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) {
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_dout_[i];
      index = index % stride_dout_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }

    if (dout_a_) {
      size_t index_out_a = index_a * numel_b_ + index_b;
      dout_a_[index_out_a] = dout_[idx] * B_[index_b];
    }
    if (dout_b_) {
      size_t index_out_b = index_b * numel_a_ + index_a;
      dout_b_[index_out_b] = dout_[idx] * A_[index_a];
    }
  }

 private:
  const T* dout_;
  const T* A_;
  const T* B_;
  T* dout_a_;
  T* dout_b_;
  const int64_t* stride_dout_;
  const int64_t* stride_a_;
  const int64_t* stride_b_;
  const int64_t* shape_b_;
  const int64_t numel_a_;
  const int64_t numel_b_;
  const int ndims_;
};

template <typename T>
struct KronGradElemFunctor<platform::complex<T>> {
  KronGradElemFunctor(const platform::complex<T>* dout,
                      const platform::complex<T>* A,
                      const platform::complex<T>* B,
                      platform::complex<T>* dout_a,
                      platform::complex<T>* dout_b, const int64_t* stride_dout,
                      const int64_t* stride_a, const int64_t* stride_b,
                      const int64_t* shape_b, const int64_t numel_a,
                      const int64_t numel_b, const int ndims)
      : dout_(dout),
        A_(A),
        B_(B),
        dout_a_(dout_a),
        dout_b_(dout_b),
        stride_dout_(stride_dout),
        stride_a_(stride_a),
        stride_b_(stride_b),
        shape_b_(shape_b),
        numel_a_(numel_a),
        numel_b_(numel_b),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) {
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_dout_[i];
      index = index % stride_dout_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }

    if (dout_a_) {
      size_t index_out_a = index_a * numel_b_ + index_b;
      dout_a_[index_out_a] =
          dout_[idx] *
          platform::complex<T>(B_[index_b].real, -B_[index_b].imag);
    }
    if (dout_b_) {
      size_t index_out_b = index_b * numel_a_ + index_a;
      dout_b_[index_out_b] =
          dout_[idx] *
          platform::complex<T>(A_[index_a].real, -A_[index_a].imag);
    }
  }

 private:
  const platform::complex<T>* dout_;
  const platform::complex<T>* A_;
  const platform::complex<T>* B_;
  platform::complex<T>* dout_a_;
  platform::complex<T>* dout_b_;
  const int64_t* stride_dout_;
  const int64_t* stride_a_;
  const int64_t* stride_b_;
  const int64_t* shape_b_;
  const int64_t numel_a_;
  const int64_t numel_b_;
  const int ndims_;
};

struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor() {}

  template <typename U>
  HOSTDEVICE inline U operator()(const U& x) const {
    return x;
  }
};

template <typename DeviceContext, typename T>
struct KronGradOpFunctor {
  void operator()(const DeviceContext& dev_ctx, const framework::Tensor& dout,
                  const framework::Tensor& x, const framework::Tensor& y,
                  framework::Tensor* dx, framework::Tensor* dy) {
    int ndims = dout.dims().size();
    int64_t numel = dout.numel();
    int64_t numel_x = x.numel();
    int64_t numel_y = y.numel();

    const framework::DDim& dim_x = x.dims();
    const framework::DDim& dim_y = y.dims();
    const framework::DDim& dim_dout = dout.dims();

    const framework::DDim stride_x = framework::stride(dim_x);
    const framework::DDim stride_y = framework::stride(dim_y);
    const framework::DDim stride_dout = framework::stride(dim_dout);

    const int64_t* p_stride_x = nullptr;
    const int64_t* p_stride_y = nullptr;
    const int64_t* p_stride_dout = nullptr;
    const int64_t* p_shape_y = nullptr;
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> d_stride_x(ndims);
    thrust::device_vector<int64_t> d_stride_y(ndims);
    thrust::device_vector<int64_t> d_stride_dout(ndims);
    thrust::device_vector<int64_t> d_shape_y(ndims);
    thrust::copy(stride_x.Get(), stride_x.Get() + ndims, d_stride_x.begin());
    thrust::copy(stride_y.Get(), stride_y.Get() + ndims, d_stride_y.begin());
    thrust::copy(stride_dout.Get(), stride_dout.Get() + ndims,
                 d_stride_dout.begin());
    thrust::copy(dim_y.Get(), dim_y.Get() + ndims, d_shape_y.begin());

    p_stride_x = thrust::raw_pointer_cast(d_stride_x.data());
    p_stride_y = thrust::raw_pointer_cast(d_stride_y.data());
    p_stride_dout = thrust::raw_pointer_cast(d_stride_dout.data());
    p_shape_y = thrust::raw_pointer_cast(d_shape_y.data());
#else
    p_stride_x = stride_x.Get();
    p_stride_y = stride_y.Get();
    p_stride_dout = stride_dout.Get();
    p_shape_y = dim_y.Get();
#endif
    // dout_x: dout * kron(ones(X), Y) re-aranged in shape (numel_x, numel_y)
    // dout_y: dout * kron(X, ones(Y)) re-aranged in shaoe (numel_y, numel_x)
    framework::Tensor dout_x;
    T* p_dout_x = nullptr;
    if (dx) {
      dout_x.mutable_data<T>({numel_x, numel_y}, dev_ctx.GetPlace());
      p_dout_x = dout_x.data<T>();
    }
    framework::Tensor dout_y;
    T* p_dout_y = nullptr;
    if (dy) {
      dout_y.mutable_data<T>({numel_y, numel_x}, dev_ctx.GetPlace());
      p_dout_y = dout_y.data<T>();
    }

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    KronGradElemFunctor<T> func(dout.data<T>(), x.data<T>(), y.data<T>(),
                                p_dout_x, p_dout_y, p_stride_dout, p_stride_x,
                                p_stride_y, p_shape_y, numel_x, numel_y, ndims);
    for_range(func);

// reduce_sum along aixs 1
#if defined(__NVCC__) || defined(__HIPCC__)
    auto stream = dev_ctx.stream();  // it is a cuda device_context
    if (dx) {
      TensorReduce<T, T, cub::Sum, IdentityFunctor>(
          dout_x, dx, {1}, static_cast<T>(0), cub::Sum(), IdentityFunctor(),
          stream);
    }
    if (dy) {
      TensorReduce<T, T, cub::Sum, IdentityFunctor>(
          dout_y, dy, {1}, static_cast<T>(0), cub::Sum(), IdentityFunctor(),
          stream);
    }
#else
    auto* place = dev_ctx.eigen_device();
    Eigen::array<int, 1> reduce_dim = {1};
    if (dx) {
      auto eigen_dout_x = framework::EigenMatrix<T>::Reshape(dout_x, 1);
      auto eigen_vec_dx = framework::EigenVector<T>::Flatten(*dx);
      eigen_vec_dx.device(*place) = eigen_dout_x.sum(reduce_dim);
    }
    if (dy) {
      auto eigen_dout_y = framework::EigenMatrix<T>::Reshape(dout_y, 1);
      auto eigen_vec_dy = framework::EigenVector<T>::Flatten(*dy);
      eigen_vec_dy.device(*place) = eigen_dout_y.sum(reduce_dim);
    }
#endif
  }
};

inline framework::Tensor UnsqueezeTo(const framework::Tensor& src, int ndims) {
  const framework::DDim& shape = src.dims();
  int rank = shape.size();
  framework::Tensor res;
  res.ShareDataWith(src);
  PADDLE_ENFORCE_LE(
      rank, ndims,
      platform::errors::InvalidArgument(
          "The input Tensor's rank should be less than or equal to ndims"
          "Received input Tensor's rank = %d, ndims = %d",
          rank, ndims));
  if (rank < ndims) {
    std::vector<int64_t> new_dim(ndims, 1);
    for (int i = ndims - rank; i < ndims; i++) {
      new_dim[i] = shape[i - ndims + rank];
    }
    res.Resize(framework::make_ddim(new_dim));
  }
  return res;
}

template <typename DeviceContext, typename T>
class KronKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");

    auto* out = ctx.Output<framework::Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    int ndims = out->dims().size();
    framework::Tensor xx = UnsqueezeTo(*x, ndims);
    framework::Tensor yy = UnsqueezeTo(*y, ndims);

    KronOpFunctor<DeviceContext, T> func;
    func(dev_ctx, xx, yy, out);
  }
};

template <typename DeviceContext, typename T>
class KronGradKernel : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* dout = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<framework::Tensor>(framework::GradVarName("Y"));
    if (dx) {
      dx->mutable_data<T>(ctx.GetPlace());
    }
    if (dy) {
      dy->mutable_data<T>(ctx.GetPlace());
    }

    int ndims = dout->dims().size();
    framework::Tensor xx = UnsqueezeTo(*x, ndims);
    framework::Tensor yy = UnsqueezeTo(*y, ndims);

    framework::Tensor* pdxx = nullptr;
    framework::Tensor* pdyy = nullptr;
    framework::Tensor dxx;
    framework::Tensor dyy;
    if (dx) {
      dxx = UnsqueezeTo(*dx, ndims);
      pdxx = &dxx;
    }

    if (dy) {
      dyy = UnsqueezeTo(*dy, ndims);
      pdyy = &dyy;
    }

    KronGradOpFunctor<DeviceContext, T> func;
    func(dev_ctx, *dout, xx, yy, pdxx, pdyy);
  }
};

}  // namespace operators
}  // namespace paddle
