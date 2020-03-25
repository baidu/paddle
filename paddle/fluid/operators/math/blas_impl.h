//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#pragma once
#include <cmath>
#include <limits>
#include <vector>
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CBlas;

template <>
struct CBlas<int8_t> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_THROW("Blas GEMM don't support int8_t");
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_THROW("Blas VCOPY don't support int8_t");
  }
};

#ifdef PADDLE_WITH_MKLML
template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    platform::dynload::cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static float *GEMM_ALLOC(ARGS... args) {
    return platform::dynload::cblas_sgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    platform::dynload::cblas_sgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    platform::dynload::cblas_sgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    platform::dynload::cblas_sgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_sgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    platform::dynload::cblas_saxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    platform::dynload::cblas_scopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    platform::dynload::cblas_sgemv(args...);
  }

  template <typename... ARGS>
  static float DOT(ARGS... args) {
    return platform::dynload::cblas_sdot(args...);
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    platform::dynload::cblas_sscal(args...);
  }

  template <typename... ARGS>
  static float ASUM(ARGS... args) {
    return platform::dynload::cblas_sasum(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    platform::dynload::cblas_sgemm_batch(args...);
  }

  template <typename... ARGS>
  static void VADD(ARGS... args) {
    platform::dynload::vsAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    platform::dynload::vsSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    platform::dynload::vsMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    platform::dynload::vsDiv(args...);
  }

  template <typename... ARGS>
  static void VEXP(ARGS... args) {
    platform::dynload::vsExp(args...);
  }

  template <typename... ARGS>
  static void VSQUARE(ARGS... args) {
    platform::dynload::vsSqr(args...);
  }

  template <typename... ARGS>
  static void VPOW(ARGS... args) {
    platform::dynload::vsPowx(args...);
  }

  template <typename... ARGS>
  static void VINV(ARGS... args) {
    platform::dynload::vsInv(args...);
  }

  template <typename... ARGS>
  static void VMERF(ARGS... args) {
    platform::dynload::vmsErf(args...);
  }
#if !defined(_WIN32)
  template <typename... ARGS>
  static void CSRMM(ARGS... args) {
    platform::dynload::mkl_scsrmm(args...);
  }
#endif
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    platform::dynload::cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static double *GEMM_ALLOC(ARGS... args) {
    return platform::dynload::cblas_dgemm_alloc(args...);
  }

  template <typename... ARGS>
  static void GEMM_PACK(ARGS... args) {
    platform::dynload::cblas_dgemm_pack(args...);
  }

  template <typename... ARGS>
  static void GEMM_COMPUTE(ARGS... args) {
    platform::dynload::cblas_dgemm_compute(args...);
  }

  template <typename... ARGS>
  static void GEMM_FREE(ARGS... args) {
    platform::dynload::cblas_dgemm_free(args...);
  }

#ifdef PADDLE_WITH_LIBXSMM
  template <typename... ARGS>
  static void SMM_GEMM(ARGS... args) {
    libxsmm_dgemm(args...);
  }
#endif

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    platform::dynload::cblas_daxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    platform::dynload::cblas_dcopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    platform::dynload::cblas_dgemv(args...);
  }

  template <typename... ARGS>
  static double DOT(ARGS... args) {
    return platform::dynload::cblas_ddot(args...);
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    platform::dynload::cblas_dscal(args...);
  }

  template <typename... ARGS>
  static double ASUM(ARGS... args) {
    return platform::dynload::cblas_dasum(args...);
  }

  template <typename... ARGS>
  static void GEMM_BATCH(ARGS... args) {
    platform::dynload::cblas_dgemm_batch(args...);
  }

  template <typename... ARGS>
  static void VADD(ARGS... args) {
    platform::dynload::vdAdd(args...);
  }

  template <typename... ARGS>
  static void VSUB(ARGS... args) {
    platform::dynload::vdSub(args...);
  }

  template <typename... ARGS>
  static void VMUL(ARGS... args) {
    platform::dynload::vdMul(args...);
  }

  template <typename... ARGS>
  static void VDIV(ARGS... args) {
    platform::dynload::vdDiv(args...);
  }

  template <typename... ARGS>
  static void VEXP(ARGS... args) {
    platform::dynload::vdExp(args...);
  }

  template <typename... ARGS>
  static void VSQUARE(ARGS... args) {
    platform::dynload::vdSqr(args...);
  }

  template <typename... ARGS>
  static void VPOW(ARGS... args) {
    platform::dynload::vdPowx(args...);
  }

  template <typename... ARGS>
  static void VINV(ARGS... args) {
    platform::dynload::vdInv(args...);
  }

  template <typename... ARGS>
  static void VMERF(ARGS... args) {
    platform::dynload::vmdErf(args...);
  }
#if !defined(_WIN32)
  template <typename... ARGS>
  static void CSRMM(ARGS... args) {
    platform::dynload::mkl_dcsrmm(args...);
  }
#endif
};

#else

template <>
struct CBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_sgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_saxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_scopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_sgemv(args...);
  }
};

template <>
struct CBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    cblas_dgemm(args...);
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    cblas_daxpy(args...);
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    cblas_dcopy(args...);
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    cblas_dgemv(args...);
  }
};
#endif

template <>
struct CBlas<platform::float16> {
  static void GEMM(...) { PADDLE_THROW("float16 GEMM not supported on CPU"); }
  static void SMM_GEMM(...) {
    PADDLE_THROW("float16 SMM_GEMM not supported on CPU");
  }
  static void VMUL(...) { PADDLE_THROW("float16 VMUL not supported on CPU"); }
  static void VEXP(...) { PADDLE_THROW("float16 VEXP not supported on CPU"); }
  static void VSQUARE(...) {
    PADDLE_THROW("float16 VSQUARE not supported on CPU");
  }
  static void VPOW(...) { PADDLE_THROW("float16 VPOW not supported on CPU"); }
  static void DOT(...) { PADDLE_THROW("float16 DOT not supported on CPU"); };
  static void SCAL(...) { PADDLE_THROW("float16 SCAL not supported on CPU"); };
  static void ASUM(...) { PADDLE_THROW("float16 ASUM not supported on CPU"); };
#ifdef PADDLE_WITH_MKLML
  static void GEMM_BATCH(...) {
    PADDLE_THROW("float16 GEMM_BATCH not supported on CPU");
  }
#endif
};

#ifdef PADDLE_WITH_MKLML
template <>
template <typename T>
T *Blas<platform::CPUDeviceContext>::GEMM_ALLOC(const CBLAS_IDENTIFIER id,
                                                const int M, const int N,
                                                const int K) const {
  return CBlas<T>::GEMM_ALLOC(id, M, N, K);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM_PACK(const CBLAS_IDENTIFIER id,
                                                 const CBLAS_TRANSPOSE trans,
                                                 int M, int N, int K,
                                                 const T alpha, const T *src,
                                                 const int ld, T *dst) const {
  CBlas<T>::GEMM_PACK(CblasRowMajor, id, trans, M, N, K, alpha, src, ld, dst);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM_COMPUTE(
    int transA, int transB, int M, int N, int K, const T *A, const int lda,
    const T *B, const int ldb, T beta, T *C, const int ldc) const {
  CBlas<T>::GEMM_COMPUTE(CblasRowMajor, transA, transB, M, N, K, A, lda, B, ldb,
                         beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM_FREE(T *data) const {
  CBlas<T>::GEMM_FREE(data);
}
#endif

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                            CBLAS_TRANSPOSE transB, int M,
                                            int N, int K, T alpha, const T *A,
                                            const T *B, T beta, T *C) const {
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  CBlas<T>::GEMM(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(bool transA, bool transB, int M,
                                            int N, int K, T alpha, const T *A,
                                            int lda, const T *B, int ldb,
                                            T beta, T *C, int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor, transA == false ? CblasNoTrans : CblasTrans,
                 transB == false ? CblasNoTrans : CblasTrans, M, N, K, alpha, A,
                 lda, B, ldb, beta, C, ldc);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                            CBLAS_TRANSPOSE transB, int M,
                                            int N, int K, T alpha, const T *A,
                                            int lda, const T *B, int ldb,
                                            T beta, T *C, int ldc) const {
  CBlas<T>::GEMM(CblasRowMajor, transA, transB, M, N, K, alpha, A, lda, B, ldb,
                 beta, C, ldc);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const framework::Tensor &mat_a, bool trans_a,
                                 const framework::Tensor &mat_b, bool trans_b,
                                 T alpha, framework::Tensor *mat_out,
                                 T beta) const {
  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out->dims();
  PADDLE_ENFORCE(dim_a.size() == 2 && dim_b.size() == 2 && dim_out.size() == 2,
                 "The input and output of matmul be matrix");
  PADDLE_ENFORCE(
      mat_a.place() == mat_b.place() && mat_a.place() == mat_out->place(),
      "The places of matrices must be same");

  int M = dim_out[0];
  int N = dim_out[1];
  int K = !trans_a ? dim_a[1] : dim_a[0];

  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !trans_b ? CblasNoTrans : CblasTrans;

  this->GEMM(transA, transB, M, N, K, alpha, mat_a.data<T>(), mat_b.data<T>(),
             beta, mat_out->data<T>());
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::AXPY(int n, T alpha, const T *x,
                                            T *y) const {
  CBlas<T>::AXPY(n, alpha, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VCOPY(int n, const T *x, T *y) const {
  CBlas<T>::VCOPY(n, x, 1, y, 1);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VADD(int n, const T *x, const T *y,
                                            T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VADD(n, x, y, z);
#else
  if (x == z) {
    this->template AXPY<T>(n, 1., y, z);
  } else {
    this->template VCOPY<T>(n, y, z);
    this->template AXPY<T>(n, 1., x, z);
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VSUB(int n, const T *x, const T *y,
                                            T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VSUB(n, x, y, z);
#else
  // try to find if openblas support vsub
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VMUL(int n, const T *x, const T *y,
                                            T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VMUL(n, x, y, z);
#else
  // try to find if openblas support vmul
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] * y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VDIV(int n, const T *x, const T *y,
                                            T *z) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VDIV(n, x, y, z);
#else
  // try to find if openblas support vdiv
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] / y[i];
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VEXP(int n, const T *x, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VEXP(n, x, y);
#else
  // try to find if openblas support vexp
  for (int i = 0; i < n; ++i) {
    y[i] = std::exp(x[i]);
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VSQUARE(int n, const T *x, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VSQUARE(n, x, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = x[i] * x[i];
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VPOW(int n, const T *x, T a,
                                            T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VPOW(n, x, a, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(x[i], a);
  }
#endif
}

template <>
template <typename T>
T Blas<platform::CPUDeviceContext>::DOT(int n, const T *x, const T *y) const {
#ifdef PADDLE_WITH_MKLML
  return CBlas<T>::DOT(n, x, 1, y, 1);
#else
  // try to find if openblas support cblas_dot
  T sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += x[i] * y[i];
  }
  return sum;
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::SCAL(int n, const T a, T *x) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::SCAL(n, a, x, 1);
#else
  // try to find if openblas support cblas_scal
  for (int i = 0; i < n; ++i) {
    x[i] = a * x[i];
  }
#endif
}

template <>
template <typename T>
T Blas<platform::CPUDeviceContext>::ASUM(int n, T *x, int inc) const {
  auto sum = static_cast<T>(0.0);
#ifdef PADDLE_WITH_MKLML
  sum = CBlas<T>::ASUM(n, x, inc);
#else
  // TODO(jczaja): check if openblas does provide cblas_sasum/cblas_dasum
  for (int c = 0; c < n; ++c) {
    sum += x[c];
  }
#endif
  return sum;
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::GEMV(bool trans_a, int M, int N, T alpha,
                                            const T *A, const T *B, T beta,
                                            T *C) const {
  CBLAS_TRANSPOSE transA = !trans_a ? CblasNoTrans : CblasTrans;
  CBlas<T>::GEMV(CblasRowMajor, transA, M, N, alpha, A, N, B, 1, beta, C, 1);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB) const {
#ifdef PADDLE_WITH_MKLML
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);
  for (int k = 0; k < batchCount; ++k) {
    a_array[k] = &A[k * strideA];
    b_array[k] = &B[k * strideB];
    c_array[k] = &C[k * M * N];
  }

  CBlas<T>::GEMM_BATCH(CblasRowMajor, &transA, &transB, &M, &N, &K, &alpha,
                       a_array.data(), &lda, b_array.data(), &ldb, &beta,
                       c_array.data(), &ldc, 1 /* group_count */, &batchCount);
#else
  for (int k = 0; k < batchCount; ++k) {
    auto *Ak = &A[k * strideA];
    auto *Bk = &B[k * strideB];
    auto *Ck = &C[k * M * N];
    this->template GEMM<T>(transA, transB, M, N, K, alpha, Ak, Bk, beta, Ck);
  }
#endif
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA)
template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::BatchedGEMMWithHead(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int W1, int H1, int W2,
    int H2, T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB, int64_t head_number,
    bool split_b_vertical) const {
  int lda = (transA == CblasNoTrans) ? W1 : H1;
  int ldb = (transB == CblasNoTrans) ? W2 : H2;
  auto a_array = std::vector<const T *>(batchCount);
  auto b_array = std::vector<const T *>(batchCount);
  auto c_array = std::vector<T *>(batchCount);

  if (split_b_vertical) {
    int ldc = W2;
    int sub_width = W2 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W2 / head_number)
                                : i * (W2 / head_number) * H2;
      int sub_matC_offset = i * W2 / head_number;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor, &transA, &transB, &H1, &sub_width,
                           &H2, &alpha, a_array.data(), &lda, b_array.data(),
                           &ldb, &beta, c_array.data(), &ldc,
                           1 /* group_count */, &batchCount);
    }

  } else {
    PADDLE_ENFORCE_EQ(W1, H2);
    int ldc = W2 * head_number;
    int sub_width = W1 / head_number;

    for (int i = 0; i < head_number; i++) {
      int sub_matA_offset = (transA == CblasNoTrans)
                                ? i * (W1 / head_number)
                                : i * (W1 / head_number) * H1;
      int sub_matB_offset = (transB == CblasNoTrans)
                                ? i * (W1 / head_number) * W2
                                : i * (W1 / head_number);
      int sub_matC_offset = i * W2;
      for (int k = 0; k < batchCount; ++k) {
        a_array[k] = &A[k * strideA] + sub_matA_offset;
        b_array[k] = &B[k * strideB] + sub_matB_offset;
        c_array[k] = &C[k * H1 * head_number * W2] + sub_matC_offset;
      }

      CBlas<T>::GEMM_BATCH(CblasRowMajor, &transA, &transB, &H1, &W2,
                           &sub_width, &alpha, a_array.data(), &lda,
                           b_array.data(), &ldb, &beta, c_array.data(), &ldc,
                           1 /* group_count */, &batchCount);
    }
  }
}
#endif

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const int M, const int N, const int K,
                                 const T *A, const T *B, T *C) const {
  this->template GEMM<T>(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                         static_cast<T>(1), A, K, B, N, static_cast<T>(0), C,
                         N);
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::MatMul(const int M, const int N,
                                              const int K, const T *A,
                                              const T *B, T *C) const {
#ifdef PADDLE_WITH_LIBXSMM
  // Refer to https://github.com/hfp/libxsmm/blob/master/README.md
  // But the threshold is custom constexpr int LIBXSMM_THRESHOLD = 20 * 20 * 20;

  // Since the matrix is very small,
  // so the unit of calculation is already very fast,
  // and the if( M*N*K < LIBXSMM_THRESHOLD) would be overhead,
  // use xsmm directly.
  // Note: SMM use ColMajor
  const char transa = 'N';
  const char transb = 'N';
  const T alpha = static_cast<T>(1);
  const T beta = static_cast<T>(0);
  CBlas<T>::SMM_GEMM(&transa, &transb, &N, &M, &K, &alpha, B, &N, A, &K, &beta,
                     C, &N);
  return;
#endif

  CBlas<T>::GEMM(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                 static_cast<T>(1), A, K, B, N, static_cast<T>(0), C, N);
}

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMul(const framework::Tensor &mat_a,
                                 const MatDescriptor &dim_a,
                                 const framework::Tensor &mat_b,
                                 const MatDescriptor &dim_b, T alpha,
                                 framework::Tensor *mat_out, T beta) const {
  PADDLE_ENFORCE_EQ(dim_a.width_, dim_b.height_);
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;
  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    this->template GEMM<T>(transA, transB, dim_a.height_, dim_b.width_,
                           dim_a.width_, alpha, mat_a.data<T>(),
                           mat_b.data<T>(), beta, mat_out->data<T>());
  } else {
    PADDLE_ENFORCE(dim_a.batch_size_ == dim_b.batch_size_ ||
                       dim_a.batch_size_ == 0 || dim_b.batch_size_ == 0,
                   "dim_a.batch_size should be equal to dim_b.batch_size, or "
                   "one of dim_a.batch_size and dim_b.batch_size should be 0. "
                   "But got dim_a.batch_size = %d, dim_b.batch_size = %d.",
                   dim_a.batch_size_, dim_b.batch_size_);
    this->template BatchedGEMM<T>(
        transA, transB, dim_a.height_, dim_b.width_, dim_a.width_, alpha,
        mat_a.data<T>(), mat_b.data<T>(), beta, mat_out->data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_, dim_b.stride_);
  }
}

#if defined(PADDLE_WITH_MKLML) && !defined(PADDLE_WITH_CUDA)
/*
 * Multiple two matrixes with multiple heads
 *
 * A new parameter, i.e head_number is added compared to normal MatMul.
 * The head_number describes the number of heads a matrix is vertically
 * split.
 *
 * When user calls this API, the multiplication of two big matrixes is split
 * into multiplication of several (head_number_) small matrixes. e.g. if Mat A
 * is [3, 24] and Mat B is [24, 4], when multiple A and B with head_number as
 * 4, Mat A will be split as 4 matrix of [3, 6] and Mat B will be
 * (horizontally) split as 4 matrix of [6, 4]. The result of final matrix
 * will be 4 matrix of [3, 4], i.e. [3, 16].
 * Another example is A is [3, 8], B is [2, 16], head_number is 4. In this
 * case, A will be split as [3, 2], B will be (vertically) split as
 * [2, 4]. The final result will be 4 matrix of 4 matrix of [3,4], i.e. [3, 16]
 */
template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::MatMulWithHead(const framework::Tensor &mat_a,
                                         const MatDescriptor &dim_a,
                                         const framework::Tensor &mat_b,
                                         const MatDescriptor &dim_b, T alpha,
                                         int head_number,
                                         framework::Tensor *mat_out, T beta,
                                         bool mat_b_split_vertical) const {
  PADDLE_ENFORCE_EQ(dim_a.width_ % head_number, 0);
  PADDLE_ENFORCE_GE(head_number, 1);
  PADDLE_ENFORCE_LE(head_number, dim_a.width_);
  CBLAS_TRANSPOSE transA = !dim_a.trans_ ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = !dim_b.trans_ ? CblasNoTrans : CblasTrans;

  if (mat_b_split_vertical) {
    PADDLE_ENFORCE_EQ(dim_b.height_, dim_a.width_ / head_number);
    PADDLE_ENFORCE_EQ(dim_b.width_ % head_number, 0);
  }

  if (dim_a.batch_size_ == 0 && dim_b.batch_size_ == 0) {
    int lda = !dim_a.trans_ ? dim_a.width_ : dim_a.height_;
    int ldb = !dim_b.trans_ ? dim_b.width_ : dim_b.height_;
    int sub_matA_offset;
    int sub_matB_offset;
    int sub_matC_offset;
    int sub_mat_M = dim_a.height_;
    int sub_mat_N;
    int sub_mat_K;
    int ldc;

    for (int i = 0; i < head_number; i++) {
      sub_matA_offset = dim_a.trans_
                            ? i * (dim_a.width_ / head_number) * dim_a.height_
                            : i * (dim_a.width_ / head_number);
      if (mat_b_split_vertical) {
        sub_matB_offset = dim_b.trans_
                              ? i * (dim_b.width_ / head_number) * dim_b.height_
                              : i * (dim_b.width_ / head_number);
        sub_matC_offset = i * dim_b.width_ / head_number;

        sub_mat_N = dim_b.width_ / head_number;
        sub_mat_K = dim_b.height_;

        ldc = dim_b.width_;
      } else {
        sub_matB_offset =
            dim_b.trans_ ? i * (dim_b.height_ / head_number)
                         : i * (dim_b.height_ / head_number) * dim_b.width_;
        sub_matC_offset = i * dim_b.width_;

        sub_mat_N = dim_b.width_;
        sub_mat_K = dim_a.width_ / head_number;

        ldc = head_number * dim_b.width_;
      }

      this->template GEMM<T>(transA, transB, sub_mat_M, sub_mat_N, sub_mat_K,
                             alpha, mat_a.data<T>() + sub_matA_offset, lda,
                             mat_b.data<T>() + sub_matB_offset, ldb, beta,
                             mat_out->data<T>() + sub_matC_offset, ldc);
    }
  } else {
    PADDLE_ENFORCE_EQ((dim_a.batch_size_ == dim_b.batch_size_ ||
                       dim_a.batch_size_ == 0 || dim_b.batch_size_ == 0),
                      true);

    this->template BatchedGEMMWithHead<T>(
        transA, transB, dim_a.width_, dim_a.height_, dim_b.width_,
        dim_b.height_, alpha, mat_a.data<T>(), mat_b.data<T>(), beta,
        mat_out->data<T>(),
        dim_a.batch_size_ == 0 ? dim_b.batch_size_ : dim_a.batch_size_,
        dim_a.stride_, dim_b.stride_, head_number, mat_b_split_vertical);
  }
}
#endif

template <typename DeviceContext>
template <typename T>
void Blas<DeviceContext>::VINV(int n, const T *a, T *y) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VINV(n, a, y);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = 1.0 / a[i];
  }
#endif
}

template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::VMERF(int n, const T *a, T *y,
                                             int64_t mode) const {
#ifdef PADDLE_WITH_MKLML
  CBlas<T>::VMERF(n, a, y, mode);
#else
  for (int i = 0; i < n; ++i) {
    y[i] = std::erf(a[i]);
  }
#endif
}

#ifdef PADDLE_WITH_MKLML
template <>
template <typename T>
void Blas<platform::CPUDeviceContext>::CSRMM(
    const char *transa, const int *m, const int *n, const int *k,
    const T *alpha, const char *matdescra, const T *val, const int *indx,
    const int *pntrb, const int *pntre, const T *b, const int *ldb,
    const T *beta, T *c, const int *ldc) const {
  CBlas<T>::CSRMM(transa, m, n, k, alpha, matdescra, val, indx, pntrb, pntre, b,
                  ldb, beta, c, ldc);
}
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
