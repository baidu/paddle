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

#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/cublas.h"

#include "paddle/fluid/platform/gpu_info.h"

DECLARE_bool(enable_cublas_tensor_op_math);

namespace paddle {
namespace operators {
namespace math {

template <typename T>
struct CUBlas;

template <>
struct CUBlas<float> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasSgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasSaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasSscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasScopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasSgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasSgemmStridedBatched(args...));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "SgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      cublasOperation_t transa, cublasOperation_t transb, int m,
                      int n, int k, const float *alpha, const void *A,
                      cudaDataType_t Atype, int lda, const void *B,
                      cudaDataType_t Btype, int ldb, const float *beta, void *C,
                      cudaDataType_t Ctype, int ldc) {
// Because the gcc 4.8 doesn't expand template parameter pack that
// appears in a lambda-expression, I can not use template parameter pack
// here.
#if CUDA_VERSION >= 8000
    VLOG(5) << "use_tensor_op_math: "
            << (dev_ctx->tensor_core_available() ? "True" : "False");
    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasSgemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc));
    });
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasSgemmEx is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasStrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasSgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasSgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasSmatinvBatched(args...));
  }
};

template <>
struct CUBlas<double> {
  template <typename... ARGS>
  static void GEMM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDgemm(args...));
  }

  template <typename... ARGS>
  static void AXPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDaxpy(args...));
  }

  template <typename... ARGS>
  static void SCAL(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDscal(args...));
  }

  template <typename... ARGS>
  static void VCOPY(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDcopy(args...));
  }

  template <typename... ARGS>
  static void GEMV(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDgemv(args...));
  }

  template <typename... ARGS>
  static void GEMM_STRIDED_BATCH(ARGS... args) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasDgemmStridedBatched(args...));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "DgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  template <typename... ARGS>
  static void GEMM_EX(ARGS... args) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Currently there are not cublasDgemmEx."));
  }

  template <typename... ARGS>
  static void TRSM(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasDtrsm(args...));
  }

  template <typename... ARGS>
  static void GETRF_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasDgetrfBatched(args...));
  }

  template <typename... ARGS>
  static void GETRI_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasDgetriBatched(args...));
  }

  template <typename... ARGS>
  static void MATINV_BATCH(ARGS... args) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasDmatinvBatched(args...));
  }
};

template <>
struct CUBlas<platform::float16> {
  using float16 = platform::float16;

  static void GEMM(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float16 *alpha, const float16 *A, int lda,
                   const float16 *B, int ldb, const float16 *beta, float16 *C,
                   int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(
        platform::dynload::cublasHgemm(handle, transa, transb, m, n, k,
                                       reinterpret_cast<const __half *>(alpha),
                                       reinterpret_cast<const __half *>(A), lda,
                                       reinterpret_cast<const __half *>(B), ldb,
                                       reinterpret_cast<const __half *>(beta),
                                       reinterpret_cast<__half *>(C), ldc));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const float16 *alpha, const float16 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const float16 *B,                // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const float16 *beta, float16 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const __half *>(alpha),
        reinterpret_cast<const __half *>(A), lda, strideA,
        reinterpret_cast<const __half *>(B), ldb, strideB,
        reinterpret_cast<const __half *>(beta), reinterpret_cast<__half *>(C),
        ldc, strideC, batchCount));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "HgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      cublasOperation_t transa, cublasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      cudaDataType_t Atype, int lda, const void *B,
                      cudaDataType_t Btype, int ldb, const void *beta, void *C,
                      cudaDataType_t Ctype, int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }
};

template <>
struct CUBlas<platform::complex64> {
  using complex64 = platform::complex64;

  static void GEMV(cublasHandle_t handle, cublasOperation_t transa, int m,
                   int n, const complex64 *alpha, const complex64 *A, int lda,
                   const complex64 *B, int ldb, const complex64 *beta,
                   complex64 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasCgemv(
        handle, transa, m, n, reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A), lda,
        reinterpret_cast<const cuFloatComplex *>(B), ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C), ldc));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const complex64 *alpha, const complex64 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const complex64 *B,              // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const complex64 *beta, complex64 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasCgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A), lda, strideA,
        reinterpret_cast<const cuFloatComplex *>(B), ldb, strideB,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C), ldc, strideC, batchCount));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const complex64 *alpha, const complex64 *A, int lda,
                   const complex64 *B, int ldb, const complex64 *beta,
                   complex64 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasCgemm(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const cuFloatComplex *>(alpha),
        reinterpret_cast<const cuFloatComplex *>(A), lda,
        reinterpret_cast<const cuFloatComplex *>(B), ldb,
        reinterpret_cast<const cuFloatComplex *>(beta),
        reinterpret_cast<cuFloatComplex *>(C), ldc));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      cublasOperation_t transa, cublasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      cudaDataType_t Atype, int lda, const void *B,
                      cudaDataType_t Btype, int ldb, const void *beta, void *C,
                      cudaDataType_t Ctype, int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }
};

template <>
struct CUBlas<platform::complex128> {
  using complex128 = platform::complex128;

  static void GEMV(cublasHandle_t handle, cublasOperation_t transa, int m,
                   int n, const complex128 *alpha, const complex128 *A, int lda,
                   const complex128 *B, int ldb, const complex128 *beta,
                   complex128 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasZgemv(
        handle, transa, m, n, reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A), lda,
        reinterpret_cast<const cuDoubleComplex *>(B), ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C), ldc));
  }

  static void GEMM_STRIDED_BATCH(cublasHandle_t handle,
                                 cublasOperation_t transa,
                                 cublasOperation_t transb, int m, int n, int k,
                                 const complex128 *alpha, const complex128 *A,
                                 int lda, long long int strideA,  // NOLINT
                                 const complex128 *B,             // NOLINT
                                 int ldb, long long int strideB,  // NOLINT
                                 const complex128 *beta, complex128 *C, int ldc,
                                 long long int strideC,  // NOLINT
                                 int batchCount) {
#if CUDA_VERSION >= 8000
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasZgemmStridedBatched(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A), lda, strideA,
        reinterpret_cast<const cuDoubleComplex *>(B), ldb, strideB,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C), ldc, strideC, batchCount));
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "CgemmStridedBatched is not supported on cuda <= 7.5"));
#endif
  }

  static void GEMM(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const complex128 *alpha, const complex128 *A, int lda,
                   const complex128 *B, int ldb, const complex128 *beta,
                   complex128 *C, int ldc) {
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasZgemm(
        handle, transa, transb, m, n, k,
        reinterpret_cast<const cuDoubleComplex *>(alpha),
        reinterpret_cast<const cuDoubleComplex *>(A), lda,
        reinterpret_cast<const cuDoubleComplex *>(B), ldb,
        reinterpret_cast<const cuDoubleComplex *>(beta),
        reinterpret_cast<cuDoubleComplex *>(C), ldc));
  }

  // NOTES: GEMM_EX can use Tensor Core to accelerate matrix multiply.
  // https://docs.nvidia.com/cuda/cublas/index.html#cublassetmathmode
  template <typename... ARGS>
  static void GEMM_EX(platform::CUDADeviceContext *dev_ctx,
                      cublasOperation_t transa, cublasOperation_t transb, int m,
                      int n, int k, const void *alpha, const void *A,
                      cudaDataType_t Atype, int lda, const void *B,
                      cudaDataType_t Btype, int ldb, const void *beta, void *C,
                      cudaDataType_t Ctype, int ldc,
                      cudaDataType_t computeType) {
#if CUDA_VERSION >= 8000
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
#if CUDA_VERSION >= 9000
    bool use_tensor_op_math = dev_ctx->tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");
#endif  // CUDA_VERSION >= 9000

    dev_ctx->TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasGemmEx(
          handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
          beta, C, Ctype, ldc, computeType, algo));
    });
#else
    PADDLE_THROW(platform::errors::Unimplemented(
        "cublasGemmEx is not supported on cuda <= 7.5"));
#endif
  }
};

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(CBLAS_TRANSPOSE transA,
                                             CBLAS_TRANSPOSE transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             const T *B, T beta, T *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx, cuTransB, cuTransA, N, M, K, &alpha, B,
                       CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta, C,
                       CUDA_R_32F, N);
  } else {
#endif  // CUDA_VERSION >= 8000
    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A,
                      lda, &beta, C, N);
    });

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 *A,
    const platform::float16 *B, platform::float16 beta,
    platform::float16 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas fp16 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  float h_alpha = static_cast<float>(alpha);
  float h_beta = static_cast<float>(beta);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::float16>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &h_alpha, B, CUDA_R_16F, ldb, A,
      CUDA_R_16F, lda, &h_beta, C, CUDA_R_16F, N, CUDA_R_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<platform::float16>::GEMM(handle, cuTransB, cuTransA, N, M, K,
                                    &h_alpha, h_B, ldb, h_A, lda, &h_beta, h_C,
                                    N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::complex64 alpha, const platform::complex64 *A,
    const platform::complex64 *B, platform::complex64 beta,
    platform::complex64 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas complex64 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<float> c_alpha =
      thrust::complex<float>(alpha.real, alpha.imag);
  thrust::complex<float> c_beta = thrust::complex<float>(beta.real, beta.imag);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::complex64>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &c_alpha, B, CUDA_C_32F, ldb, A,
      CUDA_C_32F, lda, &c_beta, C, CUDA_C_32F, N, CUDA_C_32F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<platform::complex64>::GEMM(handle, cuTransB, cuTransA, N, M, K,
                                      &c_alpha, h_B, ldb, h_A, lda, &c_beta,
                                      h_C, N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::complex128 alpha, const platform::complex128 *A,
    const platform::complex128 *B, platform::complex128 beta,
    platform::complex128 *C) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;

  // TODO(kexinzhao): add processing code for compute capability < 53 case
  PADDLE_ENFORCE_GE(
      context_.GetComputeCapability(), 53,
      platform::errors::InvalidArgument(
          "cublas complex128 gemm requires GPU compute capability >= 53,"
          "but received %d",
          context_.GetComputeCapability()));

  thrust::complex<double> c_alpha =
      thrust::complex<double>(alpha.real, alpha.imag);
  thrust::complex<double> c_beta =
      thrust::complex<double>(beta.real, beta.imag);

#if CUDA_VERSION >= 8000
  // cublasHgemm does true FP16 computation which is slow for non-Volta
  // GPUs. So use cublasGemmEx instead which does pesudo FP16 computation:
  // input/output in fp16, computation in fp32, which can also be accelerated
  // using tensor cores in volta GPUs.
  auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
  CUBlas<platform::complex128>::GEMM_EX(
      &cuda_ctx, cuTransB, cuTransA, N, M, K, &c_alpha, B, CUDA_C_64F, ldb, A,
      CUDA_C_64F, lda, &c_beta, C, CUDA_C_64F, N, CUDA_C_64F);
#else
  // CUDA 7.5 does not support cublasGemmEx, hence we fall back to use hgemm

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<platform::complex128>::GEMM(handle, cuTransB, cuTransA, N, M, K,
                                       &c_alpha, h_B, ldb, h_A, lda, &c_beta,
                                       h_C, N);
  });
#endif  // CUDA_VERSION >= 8000
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMM(bool transA, bool transB, int M,
                                             int N, int K, T alpha, const T *A,
                                             int lda, const T *B, int ldb,
                                             T beta, T *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

#if CUDA_VERSION >= 8000
  if (FLAGS_enable_cublas_tensor_op_math && std::is_same<T, float>::value) {
    auto &cuda_ctx = const_cast<platform::CUDADeviceContext &>(context_);
    CUBlas<T>::GEMM_EX(&cuda_ctx, cuTransB, cuTransA, N, M, K, &alpha, B,
                       CUDA_R_32F, ldb, A, CUDA_R_32F, lda, &beta, C,
                       CUDA_R_32F, ldc);
  } else {
#endif  // CUDA_VERSION >= 8000

    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha, B, ldb, A,
                      lda, &beta, C, ldc);
    });

#if CUDA_VERSION >= 8000
  }
#endif  // CUDA_VERSION >= 8000
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMM(
    bool transA, bool transB, int M, int N, int K, platform::float16 alpha,
    const platform::float16 *A, int lda, const platform::float16 *B, int ldb,
    platform::float16 beta, platform::float16 *C, int ldc) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<platform::float16>::GEMM(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                    B, ldb, A, lda, &beta, C, ldc);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::AXPY(int n, T alpha, const T *x,
                                             T *y) const {
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::AXPY(handle, n, &alpha, x, 1, y, 1);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::SCAL(int n, const T alpha, T *x) const {
  context_.CublasCall(
      [&](cublasHandle_t handle) { CUBlas<T>::SCAL(handle, n, &alpha, x, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::VCOPY(int n, const T *x, T *y) const {
  context_.CublasCall(
      [&](cublasHandle_t handle) { CUBlas<T>::VCOPY(handle, n, x, 1, y, 1); });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::GEMV(bool trans_a, int M, int N,
                                             T alpha, const T *A, const T *B,
                                             T beta, T *C) const {
  cublasOperation_t cuTransA = !trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GEMV(handle, cuTransA, N, M, &alpha, A, N, B, 1, &beta, C, 1);
  });
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::GEMV(
    bool trans_a, int M, int N, platform::float16 alpha,
    const platform::float16 *A, const platform::float16 *B,
    platform::float16 beta, platform::float16 *C) const {
  // Because cublas doesn't support half gemv, we use cublasHgemm to achieve it.
  if (trans_a) {
    this->template GEMM<platform::float16>(CblasNoTrans, CblasNoTrans, 1, N, M,
                                           alpha, B, A, beta, C);
  } else {
    this->template GEMM<platform::float16>(CblasNoTrans, CblasNoTrans, M, 1, N,
                                           alpha, A, B, beta, C);
  }
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T *A, const T *B, T beta, T *C, int batchCount,
    int64_t strideA, int64_t strideB) const {
  // Note that cublas follows fortran order, so the order is different from
  // the cblas convention.
  int lda = (transA == CblasNoTrans) ? K : M;
  int ldb = (transB == CblasNoTrans) ? N : K;
  int ldc = N;
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (transB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  const int64_t strideC = M * N;

#if CUDA_VERSION >= 9010
  if ((FLAGS_enable_cublas_tensor_op_math && (std::is_same<T, float>::value)) ||
      std::is_same<T, paddle::platform::float16>::value) {
    cublasGemmAlgo_t algo = CUBLAS_GEMM_DFALT;
    bool use_tensor_op_math = context_.tensor_core_available();
    if (use_tensor_op_math) {
      algo = CUBLAS_GEMM_DFALT_TENSOR_OP;
    }
    VLOG(5) << "use_tensor_op_math: "
            << (use_tensor_op_math ? "True" : "False");

    auto fp = std::is_same<T, float>::value ? CUDA_R_32F : CUDA_R_16F;
    context_.TensorCoreCublasCallIfAvailable([&](cublasHandle_t handle) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cublasGemmStridedBatchedEx(
          handle, cuTransB, cuTransA, N, M, K, &alpha, B, fp, ldb, strideB, A,
          fp, lda, strideA, &beta, C, fp, ldc, strideC, batchCount, fp, algo));
    });
  } else {
#endif  // CUDA_VERSION >= 9010

    context_.CublasCall([&](cublasHandle_t handle) {
      CUBlas<T>::GEMM_STRIDED_BATCH(handle, cuTransB, cuTransA, N, M, K, &alpha,
                                    B, ldb, strideB, A, lda, strideA, &beta, C,
                                    ldc, strideC, batchCount);
    });

#if CUDA_VERSION >= 9010
  }
#endif  // CUDA_VERSION >= 9010
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    T alpha, const T **A, const T **B, T beta, T **C, int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<T>(transA, transB, M, N, K, alpha, A[k], B[k], beta,
                           C[k]);
  }
}

template <>
template <>
inline void Blas<platform::CUDADeviceContext>::BatchedGEMM(
    CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB, int M, int N, int K,
    platform::float16 alpha, const platform::float16 **A,
    const platform::float16 **B, platform::float16 beta, platform::float16 **C,
    int batchCount) const {
  for (int k = 0; k < batchCount; ++k) {
    this->template GEMM<platform::float16>(transA, transB, M, N, K, alpha, A[k],
                                           B[k], beta, C[k]);
  }
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::TRSM(CBLAS_SIDE side, CBLAS_UPLO uplo,
                                             CBLAS_TRANSPOSE transA,
                                             CBLAS_DIAG diag, int M, int N,
                                             T alpha, const T *A, int lda, T *B,
                                             int ldb) const {
  // solve row major `op ( A ) X = α B` by taking it as `X' op ( A' )  =  α B'`
  // where ' stands for transpose
  cublasSideMode_t cuSide =
      (side == CblasLeft) ? CUBLAS_SIDE_RIGHT : CUBLAS_SIDE_LEFT;
  cublasFillMode_t cuUplo =
      (uplo == CblasLower) ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  // use CUBLAS_OP_C (conjugate transpose) for complex
  cublasOperation_t cuTransA =
      (transA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuDiag =
      (diag == CblasUnit) ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::TRSM(handle, cuSide, cuUplo, cuTransA, cuDiag, N, M, &alpha, A,
                    lda, B, ldb);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGETRF(int n, T **a, int *ipiv,
                                                     int *info,
                                                     int batch_size) const {
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRF_BATCH(handle, n, a, n, ipiv, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedGETRI(int n, const T **a,
                                                     const int *ipiv, T **a_inv,
                                                     int *info,
                                                     int batch_size) const {
  PADDLE_ENFORCE_NE(
      a_inv, a,
      platform::errors::InvalidArgument(
          "cuBLAS fuction 'cublas<S/D>getrfBatched' cannot be executed "
          "in-place. The memory space of output matrix (address: %p) cannot "
          "overlap memory space of input matrix (address: %p).",
          a_inv, a));
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::GETRI_BATCH(handle, n, a, n, ipiv, a_inv, n, info, batch_size);
  });
}

template <>
template <typename T>
void Blas<platform::CUDADeviceContext>::BatchedMatInv(int n, const T **a,
                                                      T **a_inv, int *info,
                                                      int batch_size) const {
  context_.CublasCall([&](cublasHandle_t handle) {
    CUBlas<T>::MATINV_BATCH(handle, n, a, n, a_inv, n, info, batch_size);
  });
}

}  // namespace math
}  // namespace operators
}  // namespace paddle
