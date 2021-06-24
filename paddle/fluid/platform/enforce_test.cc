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

#include <list>

#include "gtest/gtest.h"
#include "paddle/fluid/platform/enforce.h"

TEST(ENFORCE, OK) {
  PADDLE_ENFORCE(true, paddle::platform::errors::Unavailable(
                           "PADDLE_ENFORCE is ok %d now %f.", 123, 0.345));
  size_t val = 1;
  const size_t limit = 10;
  PADDLE_ENFORCE(val < limit, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE tests failed."));
}

TEST(ENFORCE, FAILED) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE(false, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE won't work %d at all.", 123));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("PADDLE_ENFORCE won't work 123 at all.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);

  caught_exception = false;
  try {
    PADDLE_ENFORCE(false, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE won't work at all."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("PADDLE_ENFORCE won't work at all.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);

  caught_exception = false;
  try {
    PADDLE_ENFORCE(false, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE won't work at all."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    EXPECT_NE(std::string(error.what()).find(" at "), 0UL);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE, NO_ARG_OK) {
  int a = 2;
  int b = 2;
  PADDLE_ENFORCE_EQ(a, b, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_EQ tests failed."));
  // test enforce with extra message.
  PADDLE_ENFORCE_EQ(a, b, paddle::platform::errors::Unavailable(
                              "Some %s wrong in PADDLE_ENFORCE_EQ.", "info"));
}

TEST(ENFORCE_EQ, NO_EXTRA_MSG_FAIL) {
  int a = 2;
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3, paddle::platform::errors::InvalidArgument(
                                    "The result is not equal correct result."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected a == 1 + 3, but received a:2 != 1 "
                            "+ 3:4.") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_EQ, EXTRA_MSG_FAIL) {
  int a = 2;
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, 1 + 3, paddle::platform::errors::InvalidArgument(
                                    "The result is not equal correct result."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(
        ex_msg.find("Expected a == 1 + 3, but received a:2 != 1 + 3:4.") !=
        std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_NE, OK) {
  PADDLE_ENFORCE_NE(1, 2, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_NE tests failed."));
  PADDLE_ENFORCE_NE(1.0, 2UL, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE_NE tests failed."));
}
TEST(ENFORCE_NE, FAIL) {
  bool caught_exception = false;

  try {
    // 2UL here to check data type compatible
    PADDLE_ENFORCE_NE(1.0, 1UL,
                      paddle::platform::errors::Unavailable(
                          "Expected 1.0 != 1UL, but received 1.0:1 == 1UL:1."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected 1.0 != 1UL, but "
                            "received 1.0:1 == 1UL:1.") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_GT, OK) {
  PADDLE_ENFORCE_GT(2, 1, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_GT tests failed."));
}
TEST(ENFORCE_GT, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GT(1, 2, paddle::platform::errors::InvalidArgument(
                                "Expected 1 > 2, but received 1:1 <= 2:2."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected 1 > 2, but received 1:1 <= 2:2.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_GE, OK) {
  PADDLE_ENFORCE_GE(2, 2, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_GE tests failed."));
  PADDLE_ENFORCE_GE(3, 2, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_GE tests failed."));
  PADDLE_ENFORCE_GE(3.21, 2.0, paddle::platform::errors::Unavailable(
                                   "PADDLE_ENFORCE_GE tests failed."));
}
TEST(ENFORCE_GE, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GE(1, 2, paddle::platform::errors::InvalidArgument(
                                "Expected 1 >= 2, but received 1:1 < 2:2."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected 1 >= 2, but received 1:1 < 2:2.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_LE, OK) {
  PADDLE_ENFORCE_LE(1, 1, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_LE tests failed."));
  PADDLE_ENFORCE_LE(1UL, 1UL, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE_LE tests failed."));
  PADDLE_ENFORCE_LE(2, 3, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_LE tests failed."));
  PADDLE_ENFORCE_LE(2UL, 3UL, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE_LE tests failed."));
  PADDLE_ENFORCE_LE(2.0, 3.2, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE_LE tests failed."));
}
TEST(ENFORCE_LE, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_GT(1, 2, paddle::platform::errors::InvalidArgument(
                                "Expected 1 > 2, but received 1:1 <= 2:2."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected 1 > 2, but received 1:1 <= 2:2.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_LT, OK) {
  PADDLE_ENFORCE_LT(3, 10, paddle::platform::errors::Unavailable(
                               "PADDLE_ENFORCE_LT tests failed."));
  PADDLE_ENFORCE_LT(2UL, 3UL, paddle::platform::errors::Unavailable(
                                  "PADDLE_ENFORCE_LT tests failed."));
  PADDLE_ENFORCE_LT(2, 3, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_LT tests failed."));
}
TEST(ENFORCE_LT, FAIL) {
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_LT(
        1UL, 0.12,
        paddle::platform::errors::InvalidArgument(
            "Expected 1UL < 0.12, but received 1UL:1 >= 0.12:0.12."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("Expected 1UL < 0.12, but "
                            "received 1UL:1 >= 0.12:0.12.") !=
                std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

TEST(ENFORCE_NOT_NULL, OK) {
  int* a = new int;
  PADDLE_ENFORCE_NOT_NULL(a, paddle::platform::errors::Unavailable(
                                 "PADDLE_ENFORCE_NOT_NULL tests failed."));
  delete a;
}
TEST(ENFORCE_NOT_NULL, FAIL) {
  bool caught_exception = false;
  try {
    int* a = nullptr;
    PADDLE_ENFORCE_NOT_NULL(
        a, paddle::platform::errors::Unavailable("The a should not be null."));
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("The a should not be null.") != std::string::npos);
  }
  EXPECT_TRUE(caught_exception);
}

struct Dims {
  size_t dims_[4];

  bool operator==(const Dims& o) const {
    for (size_t i = 0; i < 4; ++i) {
      if (dims_[i] != o.dims_[i]) return false;
    }
    return true;
  }
};

std::ostream& operator<<(std::ostream& os, const Dims& d) {
  for (size_t i = 0; i < 4; ++i) {
    if (i == 0) {
      os << "[";
    }
    os << d.dims_[i];
    if (i == 4 - 1) {
      os << "]";
    } else {
      os << ", ";
    }
  }
  return os;
}

TEST(ENFORCE_USER_DEFINED_CLASS, EQ) {
  Dims a{{1, 2, 3, 4}}, b{{1, 2, 3, 4}};
  PADDLE_ENFORCE_EQ(a, b, paddle::platform::errors::Unavailable(
                              "PADDLE_ENFORCE_EQ tests failed."));
}

TEST(ENFORCE_USER_DEFINED_CLASS, NE) {
  Dims a{{1, 2, 3, 4}}, b{{5, 6, 7, 8}};
  bool caught_exception = false;
  try {
    PADDLE_ENFORCE_EQ(a, b, paddle::platform::errors::Unavailable(
                                "PADDLE_ENFORCE_EQ tests failed."));
  } catch (paddle::platform::EnforceNotMet&) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
}

TEST(EOF_EXCEPTION, THROW_EOF) {
  bool caught_eof = false;
  try {
    PADDLE_THROW_EOF();
  } catch (paddle::platform::EOFException& error) {
    caught_eof = true;
    std::string ex_msg = error.what();
    EXPECT_TRUE(ex_msg.find("There is no next data.") != std::string::npos);
  }
  EXPECT_TRUE(caught_eof);
}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
template <typename T>
bool CheckCudaStatusSuccess(T value, const std::string& msg = "success") {
  PADDLE_ENFORCE_CUDA_SUCCESS(value);
  return true;
}

template <typename T>
bool CheckCudaStatusFailure(T value, const std::string& msg) {
  try {
    PADDLE_ENFORCE_CUDA_SUCCESS(value);
    return false;
  } catch (paddle::platform::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    std::cout << ex_msg << std::endl;
    return ex_msg.find(msg) != std::string::npos;
  }
}
#ifdef PADDLE_WITH_HIP
TEST(enforce, hip_success) {
  EXPECT_TRUE(CheckCudaStatusSuccess(hipSuccess));
  EXPECT_TRUE(CheckCudaStatusFailure(hipErrorInvalidValue, "Hip error"));
  EXPECT_TRUE(CheckCudaStatusFailure(hipErrorOutOfMemory, "Hip error"));

  EXPECT_TRUE(CheckCudaStatusSuccess(HIPRAND_STATUS_SUCCESS));
  EXPECT_TRUE(
      CheckCudaStatusFailure(HIPRAND_STATUS_VERSION_MISMATCH, "Hiprand error"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(HIPRAND_STATUS_NOT_INITIALIZED, "Hiprand error"));

  EXPECT_TRUE(CheckCudaStatusSuccess(miopenStatusSuccess));
  EXPECT_TRUE(
      CheckCudaStatusFailure(miopenStatusNotInitialized, "Miopen error"));
  EXPECT_TRUE(CheckCudaStatusFailure(miopenStatusAllocFailed, "Miopen error"));

  EXPECT_TRUE(CheckCudaStatusSuccess(rocblas_status_success));
  EXPECT_TRUE(
      CheckCudaStatusFailure(rocblas_status_invalid_handle, "Rocblas error"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(rocblas_status_invalid_value, "Rocblas error"));
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
  EXPECT_TRUE(CheckCudaStatusSuccess(ncclSuccess));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclUnhandledCudaError, "Rccl error"));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclSystemError, "Rccl error"));
#endif
}
#else
TEST(enforce, cuda_success) {
  EXPECT_TRUE(CheckCudaStatusSuccess(cudaSuccess));
  EXPECT_TRUE(CheckCudaStatusFailure(cudaErrorInvalidValue, "CUDA error"));

  EXPECT_TRUE(CheckCudaStatusFailure(cudaErrorMemoryAllocation, "CUDA error"));

  EXPECT_TRUE(CheckCudaStatusFailure(
      cudaErrorInsufficientDriver,
      "This indicates that the installed NVIDIA CUDA driver is older than the "
      "CUDA runtime library. This is not a supported configuration.Users "
      "should install an updated NVIDIA display driver to allow the "
      "application to run"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      cudaErrorContextIsDestroyed,
      "This error indicates that the context current to the calling thread has "
      "been destroyed using cuCtxDestroy, or is a primary context which has "
      "not yet been initialized"));

  EXPECT_TRUE(CheckCudaStatusSuccess(CURAND_STATUS_SUCCESS));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CURAND_STATUS_VERSION_MISMATCH, "CURAND error"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CURAND_STATUS_NOT_INITIALIZED, "CURAND error"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CURAND_STATUS_ARCH_MISMATCH,
      "Architecture mismatch, GPU does not support requested feature"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CURAND_STATUS_LENGTH_NOT_MULTIPLE,
                             "Length requested is not a multple of dimension"));

  EXPECT_TRUE(CheckCudaStatusSuccess(CUDNN_STATUS_SUCCESS));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CUDNN_STATUS_NOT_INITIALIZED, "CUDNN error"));
  EXPECT_TRUE(CheckCudaStatusFailure(CUDNN_STATUS_ALLOC_FAILED, "CUDNN error"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUDNN_STATUS_BAD_PARAM,
      "An incorrect value or parameter was passed to the function. To correct, "
      "ensure that all the parameters being passed have valid values"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUDNN_STATUS_LICENSE_ERROR,
      "The functionality requested requires some license and an error was "
      "detected when trying to check the current licensing. This error can "
      "happen if the license is not present or is expired or if the "
      "environment variable NVIDIA_LICENSE_FILE is not set properly"));

  EXPECT_TRUE(CheckCudaStatusSuccess(CUBLAS_STATUS_SUCCESS));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CUBLAS_STATUS_NOT_INITIALIZED, "CUBLAS error"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CUBLAS_STATUS_INVALID_VALUE, "CUBLAS error"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUBLAS_STATUS_EXECUTION_FAILED,
      "The GPU program failed to execute. This is often caused by a launch "
      "failure of the kernel on the GPU, which can be caused by multiple "
      "reasons.  To correct: check that the hardware, an appropriate version "
      "of the driver, and the cuBLAS library are correctly installed"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUBLAS_STATUS_MAPPING_ERROR,
      "An access to GPU memory space failed, which is usually caused by a "
      "failure to bind a texture. To correct: prior to the function call, "
      "unbind any previously bound textures"));

  EXPECT_TRUE(CheckCudaStatusSuccess(CUSOLVER_STATUS_SUCCESS));
  EXPECT_TRUE(CheckCudaStatusFailure(CUSOLVER_STATUS_NOT_INITIALIZED,
                                     "CUSOLVER error"));
  EXPECT_TRUE(
      CheckCudaStatusFailure(CUSOLVER_STATUS_ALLOC_FAILED, "CUSOLVER error"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUSOLVER_STATUS_INTERNAL_ERROR,
      "An internal cuSolver operation failed. This error is usually caused by "
      "a cudaMemcpyAsync() failure.To correct: check that the hardware, an "
      "appropriate version of the driver, and the cuSolver library are "
      "correctly installed. Also, check that the memory passed as a parameter "
      "to the routine is not being deallocated prior to the routine’s "
      "completion"));
  EXPECT_TRUE(CheckCudaStatusFailure(
      CUSOLVER_STATUS_INVALID_VALUE,
      "An unsupported value or parameter was passed to the function (a "
      "negative vector size, for example).To correct: ensure that all the "
      "parameters being passed have valid values"));

#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
  EXPECT_TRUE(CheckCudaStatusSuccess(ncclSuccess));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclUnhandledCudaError, "NCCL error"));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclSystemError, "NCCL error"));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclInternalError,
                                     "An internal check failed. This is either "
                                     "a bug in NCCL or due to memory "
                                     "corruption"));
  EXPECT_TRUE(CheckCudaStatusFailure(ncclInvalidUsage,
                                     "The call to NCCL is incorrect. This is "
                                     "usually reflecting a programming error"));
#endif
}
#endif
#endif

struct CannotToStringType {
  explicit CannotToStringType(int num) : num_(num) {}

  bool operator==(const CannotToStringType& other) const {
    return num_ == other.num_;
  }

  bool operator!=(const CannotToStringType& other) const {
    return num_ != other.num_;
  }

 private:
  int num_;
};

TEST(enforce, cannot_to_string_type) {
  static_assert(
      !paddle::platform::details::CanToString<CannotToStringType>::kValue,
      "CannotToStringType must not be converted to string");
  static_assert(paddle::platform::details::CanToString<int>::kValue,
                "int can be converted to string");
  CannotToStringType obj1(3), obj2(4), obj3(3);

  PADDLE_ENFORCE_NE(obj1, obj2, paddle::platform::errors::InvalidArgument(
                                    "Object 1 is not equal to Object 2"));
  PADDLE_ENFORCE_EQ(obj1, obj3, paddle::platform::errors::InvalidArgument(
                                    "Object 1 is equal to Object 3"));

  std::string msg = "Compare obj1 with obj2";
  try {
    PADDLE_ENFORCE_EQ(obj1, obj2,
                      paddle::platform::errors::InvalidArgument(msg));
  } catch (paddle::platform::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    LOG(INFO) << ex_msg;
    EXPECT_TRUE(ex_msg.find(msg) != std::string::npos);
    EXPECT_TRUE(
        ex_msg.find("Expected obj1 == obj2, but received obj1 != obj2") !=
        std::string::npos);
  }

  msg = "Compare x with y";
  try {
    int x = 3, y = 2;
    PADDLE_ENFORCE_EQ(x, y, paddle::platform::errors::InvalidArgument(msg));
  } catch (paddle::platform::EnforceNotMet& error) {
    std::string ex_msg = error.what();
    LOG(INFO) << ex_msg;
    EXPECT_TRUE(ex_msg.find(msg) != std::string::npos);
    EXPECT_TRUE(ex_msg.find("Expected x == y, but received x:3 != y:2") !=
                std::string::npos);
  }

  std::set<int> set;
  PADDLE_ENFORCE_EQ(set.begin(), set.end(),
                    paddle::platform::errors::InvalidArgument(
                        "The begin and end of set is not equal."));
  set.insert(3);
  PADDLE_ENFORCE_NE(set.begin(), set.end(),
                    paddle::platform::errors::InvalidArgument(
                        "The begin and end of set is equal."));

  std::list<float> list;
  PADDLE_ENFORCE_EQ(list.begin(), list.end(),
                    paddle::platform::errors::InvalidArgument(
                        "The begin and end of list is not equal."));
  list.push_back(4);
  PADDLE_ENFORCE_NE(list.begin(), list.end(),
                    paddle::platform::errors::InvalidArgument(
                        "The begin and end of list is equal."));
}

TEST(GET_DATA_SAFELY_MACRO, SUCCESS) {
  int* a = new int(10);
  GET_DATA_SAFELY(a, "Input", "X", "dummy");
}

TEST(GET_DATA_SAFELY_MACRO, FAIL) {
  bool caught_exception = false;
  try {
    int* a = nullptr;
    GET_DATA_SAFELY(a, "Input", "X", "dummy");
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
}

TEST(OP_INOUT_CHECK_MACRO, SUCCESS) {
  OP_INOUT_CHECK(true, "Input", "X", "dummy");
}

TEST(OP_INOUT_CHECK_MACRO, FAIL) {
  bool caught_exception = false;
  try {
    OP_INOUT_CHECK(false, "Input", "X", "dummy");
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
}

TEST(BOOST_GET_SAFELY, SUCCESS) {
  paddle::framework::Attribute attr;
  attr = true;
  bool rlt = BOOST_GET(bool, attr);
  EXPECT_EQ(rlt, true);
}

TEST(BOOST_GET_SAFELY, FAIL) {
  paddle::framework::Attribute attr;
  attr = true;
  bool caught_exception = false;
  try {
    BOOST_GET(int, attr);
  } catch (paddle::platform::EnforceNotMet& error) {
    caught_exception = true;
  }
  EXPECT_TRUE(caught_exception);
}
