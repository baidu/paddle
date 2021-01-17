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
#include "paddle/fluid/platform/float16.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <bitset>
#include <iostream>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/enforce.h"

#include "hip/hip_runtime.h"

#define ARITHMETIC_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const half* in1, const half* in2, half* out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define COMPOUND_KERNEL(op_type, sign) \
  __global__ void op_type(half* in1, const half* in2) { in1[0] sign in2[0]; }

#define COMPARISON_KERNEL(op_type, sign)                                 \
  __global__ void op_type(const half* in1, const half* in2, bool* out) { \
    out[0] = in1[0] sign in2[0];                                         \
  }

#define ARITHMETIC_KERNEL_LAUNCH(op_type)                                     \
  void Test##op_type(float v_in1, float v_in2, float v_out) {                 \
    LOG(INFO) << "Test " << #op_type << " on GPU!";                           \
    half *in1, *in2, *out;                                                    \
    half *d_in1, *d_in2, *d_out;                                              \
    int size = sizeof(half);                                                  \
    hipMalloc(reinterpret_cast<void**>(&d_in1), size);                        \
    hipMalloc(reinterpret_cast<void**>(&d_in2), size);                        \
    hipMalloc(reinterpret_cast<void**>(&d_out), size);                        \
    in1 = reinterpret_cast<half*>(malloc(size));                              \
    in2 = reinterpret_cast<half*>(malloc(size));                              \
    out = reinterpret_cast<half*>(malloc(size));                              \
    in1[0] = half(float16(v_in1));                                            \
    in2[0] = half(float16(v_in2));                                            \
    hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);                       \
    hipMemcpy(d_in2, in2, size, hipMemcpyHostToDevice);                       \
    hipLaunchKernelGGL(op_type, dim3(1), dim3(1), 0, 0, d_in1, d_in2, d_out); \
    hipMemcpy(out, d_out, size, hipMemcpyDeviceToHost);                       \
    EXPECT_EQ(static_cast<float>(float16(out[0])), v_out);                    \
    free(in1);                                                                \
    free(in2);                                                                \
    free(out);                                                                \
    hipFree(d_in1);                                                           \
    hipFree(d_in2);                                                           \
    hipFree(d_out);                                                           \
  }

#define COMPOUND_KERNEL_LAUNCH(op_type)                                \
  void Test##op_type(float v_in1, float v_in2, float v_out) {          \
    LOG(INFO) << "Test " << #op_type << " on GPU!";                    \
    half *in1, *in2;                                                   \
    half *d_in1, *d_in2;                                               \
    int size = sizeof(half);                                           \
    hipMalloc(reinterpret_cast<void**>(&d_in1), size);                 \
    hipMalloc(reinterpret_cast<void**>(&d_in2), size);                 \
    in1 = reinterpret_cast<half*>(malloc(size));                       \
    in2 = reinterpret_cast<half*>(malloc(size));                       \
    in1[0] = half(float16(v_in1));                                     \
    in2[0] = half(float16(v_in2));                                     \
    hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);                \
    hipMemcpy(d_in2, in2, size, hipMemcpyHostToDevice);                \
    hipLaunchKernelGGL(op_type, dim3(1), dim3(1), 0, 0, d_in1, d_in2); \
    hipMemcpy(in1, d_in1, size, hipMemcpyDeviceToHost);                \
    EXPECT_EQ(static_cast<float>(float16(in1[0])), v_out);             \
    free(in1);                                                         \
    free(in2);                                                         \
    hipFree(d_in1);                                                    \
    hipFree(d_in2);                                                    \
  }

#define COMPARISON_KERNEL_LAUNCH(op_type)                                     \
  void Test##op_type(float v_in1, float v_in2, bool v_out) {                  \
    LOG(INFO) << "Test " << #op_type << " on GPU!";                           \
    half *in1, *in2;                                                          \
    half *d_in1, *d_in2;                                                      \
    bool *out, *d_out;                                                        \
    int size = sizeof(half);                                                  \
    hipMalloc(reinterpret_cast<void**>(&d_in1), size);                        \
    hipMalloc(reinterpret_cast<void**>(&d_in2), size);                        \
    hipMalloc(reinterpret_cast<void**>(&d_out), 1);                           \
    in1 = reinterpret_cast<half*>(malloc(size));                              \
    in2 = reinterpret_cast<half*>(malloc(size));                              \
    out = reinterpret_cast<bool*>(malloc(1));                                 \
    in1[0] = half(float16(v_in1));                                            \
    in2[0] = half(float16(v_in2));                                            \
    hipMemcpy(d_in1, in1, size, hipMemcpyHostToDevice);                       \
    hipMemcpy(d_in2, in2, size, hipMemcpyHostToDevice);                       \
    hipLaunchKernelGGL(op_type, dim3(1), dim3(1), 0, 0, d_in1, d_in2, d_out); \
    hipMemcpy(out, d_out, 1, hipMemcpyDeviceToHost);                          \
    EXPECT_EQ(out[0], v_out);                                                 \
    free(in1);                                                                \
    free(in2);                                                                \
    free(out);                                                                \
    hipFree(d_in1);                                                           \
    hipFree(d_in2);                                                           \
    hipFree(d_out);                                                           \
  }

#ifdef PADDLE_CUDA_FP16
namespace paddle {
namespace platform {

#if CUDA_VERSION < 9000
ARITHMETIC_KERNEL(Add, +)
ARITHMETIC_KERNEL(Sub, -)
ARITHMETIC_KERNEL(Mul, *)
ARITHMETIC_KERNEL(Div, /)

ARITHMETIC_KERNEL_LAUNCH(Add)
ARITHMETIC_KERNEL_LAUNCH(Sub)
ARITHMETIC_KERNEL_LAUNCH(Mul)
ARITHMETIC_KERNEL_LAUNCH(Div)

// Negative sign kernel
__global__ void Neg(half* in) { in[0] = -in[0]; }

void TestNeg(float v_in, float v_out) {
  LOG(INFO) << "Test Neg on GPU!";
  half *in, *d_in;
  int size = sizeof(half);
  hipMalloc(reinterpret_cast<void**>(&d_in), size);
  in = reinterpret_cast<half*>(malloc(size));
  in[0] = half(float16(v_in));
  hipMemcpy(d_in, in, size, hipMemcpyHostToDevice);
  hipLaunchKernelGGL(Neg, dim3(1), dim3(1), 0, 0, d_in);
  hipMemcpy(in, d_in, size, hipMemcpyDeviceToHost);
  EXPECT_EQ(static_cast<float>(float16(in[0])), v_out);
  free(in);
  hipFree(d_in);
}

COMPOUND_KERNEL(AddAssign, +=)
COMPOUND_KERNEL(SubAssign, -=)
COMPOUND_KERNEL(MulAssign, *=)
COMPOUND_KERNEL(DivAssign, /=)

COMPOUND_KERNEL_LAUNCH(AddAssign)
COMPOUND_KERNEL_LAUNCH(SubAssign)
COMPOUND_KERNEL_LAUNCH(MulAssign)
COMPOUND_KERNEL_LAUNCH(DivAssign)

COMPARISON_KERNEL(Equal, ==)
COMPARISON_KERNEL(NotEqual, !=)
COMPARISON_KERNEL(Less, <)
COMPARISON_KERNEL(LessEqual, <=)
COMPARISON_KERNEL(Greater, >)
COMPARISON_KERNEL(GreaterEqual, >=)

COMPARISON_KERNEL_LAUNCH(Equal)
COMPARISON_KERNEL_LAUNCH(NotEqual)
COMPARISON_KERNEL_LAUNCH(Less)
COMPARISON_KERNEL_LAUNCH(LessEqual)
COMPARISON_KERNEL_LAUNCH(Greater)
COMPARISON_KERNEL_LAUNCH(GreaterEqual)

TEST(float16, arithmetic_on_gpu) {
  TestAdd(1, 2, 3);
  TestSub(2, 1, 1);
  TestMul(2, 3, 6);
  TestDiv(6, 2, 3);
  TestNeg(1, -1);
}

TEST(float16, compound_on_gpu) {
  TestAddAssign(1, 2, 3);
  TestSubAssign(2, 1, 1);
  TestMulAssign(2, 3, 6);
  TestDivAssign(6, 2, 3);
}

TEST(float16, comparision_on_gpu) {
  TestEqual(1, 1, true);
  TestEqual(1, 2, false);
  TestNotEqual(2, 3, true);
  TestNotEqual(2, 2, false);
  TestLess(3, 4, true);
  TestLess(3, 3, false);
  TestLessEqual(3, 3, true);
  TestLessEqual(3, 2, false);
  TestGreater(4, 3, true);
  TestGreater(4, 4, false);
  TestGreaterEqual(4, 4, true);
  TestGreaterEqual(4, 5, false);
}
#endif  // CUDA_VERSION

TEST(float16, conversion_on_gpu) {
  // Explicit conversion to and from cuda half
  EXPECT_EQ(float16(half(float16(1.0f))).x, 0x3c00);
  EXPECT_EQ(float16(half(float16(0.5f))).x, 0x3800);
  EXPECT_EQ(float16(half(float16(0.33333f))).x, 0x3555);
  EXPECT_EQ(float16(half(float16(0.0f))).x, 0x0000);
  EXPECT_EQ(float16(half(float16(-0.0f))).x, 0x8000);
  EXPECT_EQ(float16(half(float16(65504.0f))).x, 0x7bff);
  EXPECT_EQ(float16(half(float16(65536.0f))).x, 0x7c00);

  // Assignment operator
  float16 v_assign;
  v_assign = half(float16(1.0f));
  EXPECT_EQ(v_assign.x, 0x3c00);
}

TEST(float16, lod_tensor_on_gpu) {
  framework::LoDTensor src_tensor;
  framework::LoDTensor gpu_tensor;
  framework::LoDTensor dst_tensor;

  float16* src_ptr = src_tensor.mutable_data<float16>(
      framework::make_ddim({2, 2}), CPUPlace());

  float16 arr[4] = {float16(1.0f), float16(0.5f), float16(0.33333f),
                    float16(0.0f)};
  memcpy(src_ptr, arr, 4 * sizeof(float16));

  // CPU LoDTensor to GPU LoDTensor
  CUDAPlace gpu_place(0);
  CUDADeviceContext gpu_ctx(gpu_place);
  framework::TensorCopy(src_tensor, gpu_place, gpu_ctx, &gpu_tensor);

  // GPU LoDTensor to CPU LoDTensor
  framework::TensorCopy(gpu_tensor, CPUPlace(), gpu_ctx, &dst_tensor);

  // Sync before comparing LoDTensors
  gpu_ctx.Wait();
  const float16* dst_ptr = dst_tensor.data<float16>();
  ASSERT_NE(src_ptr, dst_ptr);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(src_ptr[i].x, dst_ptr[i].x);
  }
}

template <typename T>
struct Functor {
  bool operator()(const T& val) {
    return std::type_index(typeid(T)) ==
           std::type_index(typeid(platform::float16));
  }
};

TEST(float16, typeid) {
  // the framework heavily used typeid hash
  Functor<float16> functor;
  float16 a = float16(.0f);
  Functor<int> functor2;
  int b(0);

  // compile time assert
  PADDLE_ENFORCE_EQ(
      functor(a), true,
      platform::errors::Unavailable("The float16 support in GPU failed."));
  PADDLE_ENFORCE_EQ(
      functor2(b), false,
      platform::errors::Unavailable("The float16 support in GPU failed."));
}

// GPU test
TEST(float16, isinf) {
  float16 a;
  a.x = 0x7c00;
  float16 b = float16(INFINITY);
  // underflow to 0
  float16 native_a(5e-40f);
  EXPECT_EQ(std::isinf(a), true);
  EXPECT_EQ(std::isinf(b), true);
#ifndef _WIN32
  // overflow to inf
  float16 native_b(5e40f);
  EXPECT_EQ(std::isinf(native_b), true);
#endif
  EXPECT_EQ(native_a, float16(0));
}

TEST(float16, isnan) {
  float16 a;
  a.x = 0x7fff;
  float16 b = float16(NAN);
  float16 c = float16(5e40);
  // inf * +-0 will get a nan
  float16 d = c * float16(0);
  EXPECT_EQ(std::isnan(a), true);
  EXPECT_EQ(std::isnan(b), true);
  EXPECT_EQ(std::isnan(d), true);
}

TEST(float16, cast) {
  float16 a;
  a.x = 0x0070;
  auto b = a;
  {
    // change semantic, keep the same value
    float16 c = reinterpret_cast<float16&>(reinterpret_cast<unsigned&>(b));
    EXPECT_EQ(b, c);
  }

  {
    // use uint32 low 16 bit store float16
    uint32_t c = reinterpret_cast<uint32_t&>(b);
    float16 d;
    d.x = c;
    EXPECT_EQ(b, d);
  }
}

}  // namespace platform
}  // namespace paddle
#endif  // PADDLE_CUDA_FP16
