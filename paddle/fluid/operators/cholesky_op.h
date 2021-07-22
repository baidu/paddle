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

#include <numeric>
#include <vector>
#include "Eigen/Cholesky"
#include "Eigen/Core"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
class CholeskyCPUKernel : public framework::OpKernel<T> {
 public:
  // different with EigenMatrix in framework/eigen.h
  using EigenMatrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using InputMatrixMap = Eigen::Map<const EigenMatrix>;
  using OutputMatrixMap = Eigen::Map<EigenMatrix>;
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");

    bool upper = context.Attr<bool>("upper");
    auto& dims = x->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    auto m = dims[dims.size() - 1];

    const auto* x_data = x->data<T>();
    auto* out_data = out->mutable_data<T>(context.GetPlace());
    // Cholesky decomposition for each matrix, maybe can use multi threads
    for (int i = 0; i < batch_count; i++) {
      auto input = InputMatrixMap(x_data + i * m * m, m, m);
      auto output = OutputMatrixMap(out_data + i * m * m, m, m);
      if (upper) {
        Eigen::LLT<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
            Eigen::UpLoType::Upper>
            llt_decomposition(input);
        PADDLE_ENFORCE_EQ(llt_decomposition.info(), Eigen::Success,
                          platform::errors::InvalidArgument(
                              "Cholesky decomposition was not successful. The "
                              "%d-th input matrice "
                              "might not be not be positive definite.",
                              i));
        output = llt_decomposition.matrixU();
      } else {
        Eigen::LLT<
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
            Eigen::UpLoType::Lower>
            llt_decomposition(input);
        PADDLE_ENFORCE_EQ(llt_decomposition.info(), Eigen::Success,
                          platform::errors::InvalidArgument(
                              "Cholesky decomposition was not successful. The "
                              "%d-th input matrice "
                              "might not be not be positive definite.",
                              i));
        output = llt_decomposition.matrixL();
      }
    }
  }
};

/*! Use these functors to implement tril, triu, diagonal and other operators */
template <typename T>
struct EyeFunctor {
  EyeFunctor(const int m, const int n, T* output)
      : m_(m), n_(n), output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int global_row = index / n_;
    const int col = index - global_row * n_;
    const int batch = global_row / m_;
    const int row = global_row - batch * m_;
    output_[index] = col == row ? static_cast<T>(1) : static_cast<T>(0);
  }

  const int m_, n_;
  T* output_;
};

template <typename T>
struct MatrixBandPartFunctor {
  /*! Set output as input value outside a central band and 0 inside that band.
   * That is: output[i, j, ..., m, n] = in_band(m, n) * input[i, j, ..., m, n]
   * where: in_band(m, n) = (num_lower < 0 || (m-n) <= num_lower)) && (num_upper
   * < 0 || (n-m) <= num_upper)
   */
  MatrixBandPartFunctor(const int m, const int n, const int num_lower_diags,
                        const int num_upper_diags, const T* input, T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = static_cast<T>(0);
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const T* input_;
  T* output_;
};

template <typename T>
struct MatrixSetDiagFunctor {
  /*! Overwrite specified diagonals of output by the values in diagonal.
   * diagonals can be a central band specified by num_diags and
   * upper_diag_index, where upper_diag_index=0 refers to the main diagonal,
   * positive value means superdiagonal and negative value means subdiagonal.
   * When it is a band, `diag` has a shape [i, j, ..., num_diags, max_diag_len]
   * and the num_diags diagonals has a up to down layout. Otherwise it has a
   * shape [i, j, ..., max_diag_len].
   */
  MatrixSetDiagFunctor(const int m, const int n, const int num_diags,
                       const int max_diag_len, const int upper_diag_index,
                       const T* diag, T* output)
      : m_(m),
        n_(n),
        num_diags_(num_diags),
        max_diag_len_(max_diag_len),
        upper_diag_index_(upper_diag_index),
        diag_(diag),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int batch_and_diag_index = index / max_diag_len_;
    const int index_in_the_diagonal =
        index - batch_and_diag_index * max_diag_len_;
    const int batch = batch_and_diag_index / num_diags_;
    const int diag_index_in_input = batch_and_diag_index - batch * num_diags_;
    // diag_index=0 refers to the main diagonal
    const int diag_index = upper_diag_index_ - diag_index_in_input;
    // shift down for subdiagonal if diag_index < 0
    const int y_index =
        index_in_the_diagonal + (0 > -diag_index ? 0 : -diag_index);
    // shift right for superdiagonal if diag_index > 0
    const int x_index =
        index_in_the_diagonal + (0 > diag_index ? 0 : diag_index);

    // Upper-bound checks for diagonals shorter than max_diag_len.
    // y_index and x_index are nonnegative by construction.
    if (y_index < m_ && x_index < n_) {
      const int out_index = batch * m_ * n_ + y_index * n_ + x_index;
      output_[out_index] = diag_[index];
    }
  }

  const int m_, n_, num_diags_, max_diag_len_, upper_diag_index_;
  const T* diag_;
  T* output_;
};

template <typename T>
struct MatrixDiagPartFunctor {
  /*! Similar to MatrixSetDiagFunctor but return the diagonals. diag_index=0
   * refers to the main diagonal, positive value means superdiagonal and
   * negative value means subdiagonal */
  MatrixDiagPartFunctor(const int m, const int n, const int num_diags,
                        const int max_diag_len, const int upper_diag_index,
                        const T padding, const T* input, T* output)
      : m_(m),
        n_(n),
        num_diags_(num_diags),
        max_diag_len_(max_diag_len),
        upper_diag_index_(upper_diag_index),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int batch_and_mapped_diag_index = index / max_diag_len_;
    const int index_in_the_diagonal =
        index - batch_and_mapped_diag_index * max_diag_len_;
    const int batch = batch_and_mapped_diag_index / num_diags_;
    const int mapped_diag_index =
        batch_and_mapped_diag_index - batch * num_diags_;
    // diag_index=0 refers to the main diagonal
    const int diag_index = upper_diag_index_ - mapped_diag_index;
    // shift down for subdiagonal if diag_index < 0
    const int y_index =
        index_in_the_diagonal + (0 > -diag_index ? 0 : -diag_index);
    // shift right for superdiagonal if diag_index > 0
    const int x_index =
        index_in_the_diagonal + (0 > diag_index ? 0 : diag_index);
    if (y_index < m_ && x_index < n_) {
      output_[index] = input_[batch * m_ * n_ + y_index * m_ + x_index];
    } else {
      output_[index] = padding_;
    }
  }

  const int m_, n_, num_diags_, max_diag_len_, upper_diag_index_;
  const T padding_;
  const T* input_;
  T* output_;
};

template <typename T>
struct MatrixBandPartScaleEndFunctor {
  /*! Compared with MatrixBandPartFunctor, it scale up values at the end of
   * band. It can be used to fuse the following operations, which actually
   * output triangular with diagonal scaled up:
   * 1. dig = matrix_diag_part(middle)
   * 2. middle = matrix_set_diag(middle, diag * scalar)
   * 3. middle = matrix_band_part(middle, -1, 0)
   */
  MatrixBandPartScaleEndFunctor(const int m, const int n,
                                const int num_lower_diags,
                                const int num_upper_diags, const T scale,
                                const T* input, T* output)
      : m_(m),
        n_(n),
        num_lower_diags_(num_lower_diags),
        num_upper_diags_(num_upper_diags),
        scale_(scale),
        input_(input),
        output_(output) {}

  HOSTDEVICE void operator()(size_t index) const {
    const int col = index % n_;
    const int row = (index / n_) % m_;
    const int band_start = (num_lower_diags_ < 0 ? 0 : row - num_lower_diags_);
    const int band_end =
        (num_upper_diags_ < 0 ? n_ : row + num_upper_diags_ + 1);
    if (col < band_start || col >= band_end) {
      output_[index] = 0;
    } else if (col == band_end - 1) {
      output_[index] = scale_ * input_[index];
    } else {
      output_[index] = input_[index];
    }
  }

  const int m_, n_, num_lower_diags_, num_upper_diags_;
  const T scale_;
  const T* input_;
  T* output_;
};

template <typename T>
struct AddtoScaleFunctor {
  AddtoScaleFunctor(const T scale, const T* input, T* output)
      : scale_(scale), input_(input), output_(output) {}
  HOSTDEVICE void operator()(size_t index) const {
    output_[index] += input_[index];
    output_[index] *= scale_;
  }
  const T scale_;
  const T* input_;
  T* output_;
};

template <typename DeviceContext, typename T>
class CholeskyGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* out = context.Input<Tensor>("Out");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    auto* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());

    bool upper = context.Attr<bool>("upper");
    auto& dims = out->dims();
    int batch_count = 1;
    for (int i = 0; i < dims.size() - 2; i++) {
      batch_count *= dims[i];
    }
    auto m = dims[dims.size() - 1];
    int tensor_size = batch_count * m * m;

    auto& dev_ctx = context.template device_context<DeviceContext>();

    std::vector<int> axis(dims.size() - 2);
    std::iota(axis.begin(), axis.end(), 0);
    axis.insert(axis.end(), {dims.size() - 1, dims.size() - 2});
    Tensor l, l_grad;
    paddle::operators::math::TransposeFunctor<DeviceContext, T> transpose;
    if (upper) {
      l.mutable_data<T>(dims, context.GetPlace());
      l_grad.mutable_data<T>(dims, context.GetPlace());
      transpose(dev_ctx, *out, &l, axis);
      transpose(dev_ctx, *out_grad, &l_grad, axis);
    } else {
      l = *out;
      l_grad = *out_grad;
    }
    auto* l_data = l.data<T>();

    /*！ refer to Iain Murray (2016); arXiv 1602.07527 */
    /*! phi = matmul(L.transpose(-1, -2), grad) */
    Tensor middle;
    auto* middle_data = middle.mutable_data<T>(dims, context.GetPlace());
    auto trans_desc = math::CreateMatrixDescriptor(dims, 0, true);
    auto no_trans_desc = math::CreateMatrixDescriptor(dims, 0, false);
    auto blas = math::GetBlas<DeviceContext, T>(context);
    blas.MatMul(l, trans_desc, l_grad, no_trans_desc, T(1), &middle, T(0));

    /*! phi.tril_().diagonal(0, -2, -1).mul_(0.5) */
    platform::ForRange<DeviceContext> for_range(dev_ctx, tensor_size);
    MatrixBandPartScaleEndFunctor<T> matrix_band_part_scale_end_functor(
        m, m, /* num_lower_diags */ m, /* num_upper_diags */ 0,
        /* scale */ 0.5, middle_data, middle_data);
    for_range(matrix_band_part_scale_end_functor);

    // Compute inverse by solving the triangular linear system AX = B, where B
    // is the identity matrix. The matrix X would be overwritten on B
    Tensor identity;
    auto* identity_data = identity.mutable_data<T>(dims, context.GetPlace());
    EyeFunctor<T> eye_functor(m, m, identity_data);
    for_range(eye_functor);
    // TODO(guosheng): use trsmBatched for GPU
    for (int i = 0; i < batch_count; i++) {
      blas.TRSM(/*side*/ CblasLeft, /*uplo*/ CblasLower,
                /*trans*/ CblasNoTrans, /*diag*/ CblasNonUnit, /*m*/ m, /*n*/ m,
                /*alpha*/ T(1), l_data + i * m * m, /*lda*/ m,
                identity_data + i * m * m, /*ldb*/ m);
    }
    Tensor& l_inverse = identity;

    /*! x_grad = matmul(matmul(L_inverse.transpose(-1, -2), phi), L_inverse) */
    Tensor middle1;
    middle1.mutable_data<T>(dims, context.GetPlace());
    blas.MatMul(l_inverse, trans_desc, middle, no_trans_desc, T(1), &middle1,
                T(0));
    blas.MatMul(middle1, no_trans_desc, l_inverse, no_trans_desc, T(1), x_grad,
                T(0));

    /*! x_grad.add(x_grad.transpose(-1, -2)).mul_(0.5) */
    Tensor x_grad_trans;
    auto* x_grad_trans_data =
        x_grad_trans.mutable_data<T>(dims, context.GetPlace());
    transpose(dev_ctx, *x_grad, &x_grad_trans, axis);
    AddtoScaleFunctor<T> addto_scale_functor(0.5, x_grad_trans_data,
                                             x_grad_data);
    for_range(addto_scale_functor);
  }
};

}  // namespace operators
}  // namespace paddle
