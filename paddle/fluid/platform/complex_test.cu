// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/complex.h"

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <thrust/complex.h>
#include <bitset>
#include <iostream>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/eigen_ext.h"
#include "paddle/fluid/platform/enforce.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
namespace paddle {
namespace platform {

TEST(complex, conversion_on_gpu) {
  // *********** complex<float> *************
  // thrust<float> from and to complex<float>
  complex<float> a(1.0f, 2.0f);
  EXPECT_EQ(complex<float>(thrust::complex<float>(a)).real, 1.0);
  EXPECT_EQ(complex<float>(thrust::complex<float>(a)).imag, 2.0);

  complex<double> a1(1.0, 2.0);
  EXPECT_EQ(complex<double>(thrust::complex<double>(a1)).real, 1.0);
  EXPECT_EQ(complex<double>(thrust::complex<double>(a1)).imag, 2.0);

#if defined(PADDLE_WITH_HIP)
  EXPECT_EQ(hipFloatComplex(a).real(), 1.0);
  EXPECT_EQ(hipFloatComplex(a).imag(), 2.0);
  EXPECT_EQ(hipDoubleComplex(a).real(), 1.0);
  EXPECT_EQ(hipDoubleComplex(a).imag(), 2.0);

  EXPECT_EQ(hipFloatComplex(a1).real(), 1.0);
  EXPECT_EQ(hipFloatComplex(a1).imag(), 2.0);
  EXPECT_EQ(hipDoubleComplex(a1).real(), 1.0);
  EXPECT_EQ(hipDoubleComplex(a1).imag(), 2.0);
#else
  EXPECT_EQ(cuCrealf(cuFloatComplex(a)), 1.0);
  EXPECT_EQ(cuCimagf(cuFloatComplex(a)), 2.0);
  EXPECT_EQ(cuCreal(cuDoubleComplex(a)), 1.0);
  EXPECT_EQ(cuCimag(cuDoubleComplex(a)), 2.0);

  EXPECT_EQ(cuCrealf(cuFloatComplex(a1)), 1.0);
  EXPECT_EQ(cuCimagf(cuFloatComplex(a1)), 2.0);
  EXPECT_EQ(cuCreal(cuDoubleComplex(a1)), 1.0);
  EXPECT_EQ(cuCimag(cuDoubleComplex(a1)), 2.0);
#endif

  EXPECT_EQ(complex<float>().real, 0.0f);
  EXPECT_EQ(complex<float>().imag, 0.0f);

  EXPECT_EQ(complex<float>(1.0f, 1.0f).real, 1.0f);
  EXPECT_EQ(complex<float>(1.0f, 1.0f).imag, 1.0f);
  EXPECT_EQ(complex<float>(0.0f, 1.0f).real, 0.0f);
  EXPECT_EQ(complex<float>(0.0f, 1.0f).imag, 1.0f);

  EXPECT_EQ(complex<float>(1.0f).real, 1.0f);
  EXPECT_EQ(complex<float>(1.0f).imag, 0.0f);

  // int to complex<float>
  EXPECT_EQ(complex<float>(1).real, 1.0f);
  EXPECT_EQ(complex<float>(0).real, 0.0f);
  EXPECT_EQ(complex<float>(2).real, 2.0f);
  EXPECT_EQ(complex<float>(-2).real, -2.0f);

  // bool to complex
  EXPECT_EQ(complex<float>(true).real, 1.0f);
  EXPECT_EQ(complex<float>(true).imag, 0.0f);

  // complex<double> to complex<float>
  EXPECT_EQ(complex<float>(complex<double>(1.0, 2.0)).real, 1.0f);
  EXPECT_EQ(complex<float>(complex<double>(1.0, 2.0)).imag, 2.0f);

  // std::complex<float> to complex<float>
  EXPECT_EQ(complex<float>(std::complex<float>(1.0f, 2.0f)).real, 1.0f);
  EXPECT_EQ(complex<float>(std::complex<float>(1.0f, 2.0f)).imag, 2.0f);
  EXPECT_EQ(complex<float>(std::complex<double>(1.0, 2.0)).real, 1.0f);
  EXPECT_EQ(complex<float>(std::complex<double>(1.0, 2.0)).imag, 2.0f);

  // Assignment operator
  complex<float> c = 1.0f;
  EXPECT_EQ(c.real, 1.0f);
  EXPECT_EQ(c.imag, 0.0f);
  c = complex<float>(2.0, 2.0);
  EXPECT_EQ(c.real, 2.0f);
  EXPECT_EQ(c.imag, 2.0f);

  // Conversion operator
  EXPECT_EQ(static_cast<float>(complex<float>(0.5f)), 0.5f);
  EXPECT_NEAR(static_cast<double>(complex<float>(0.33333)), 0.33333, 0.01);
  EXPECT_EQ(static_cast<int>(complex<float>(-1)), -1);
  EXPECT_EQ(static_cast<bool>(complex<float>(true)), true);

  // *********** complex<double> *************
  // double to complex<double>
  EXPECT_EQ(complex<double>().real, 0.0);
  EXPECT_EQ(complex<double>().imag, 0.0);

  EXPECT_EQ(complex<double>(1.0, 1.0).real, 1.0);
  EXPECT_EQ(complex<double>(1.0, 1.0).imag, 1.0);
  EXPECT_EQ(complex<double>(0.0, 1.0).real, 0.0);
  EXPECT_EQ(complex<double>(0.0, 1.0).imag, 1.0);

  EXPECT_EQ(complex<double>(1.0).real, 1.0);
  EXPECT_EQ(complex<double>(1.0).imag, 0.0);

  // int to complex<double>
  EXPECT_EQ(complex<double>(1).real, 1.0);
  EXPECT_EQ(complex<double>(0).real, 0.0);
  EXPECT_EQ(complex<double>(2).real, 2.0);
  EXPECT_EQ(complex<double>(-2).real, -2.0);

  // bool to complex
  EXPECT_EQ(complex<double>(true).real, 1.0);
  EXPECT_EQ(complex<double>(true).imag, 0.0);

  // complex<float> to complex<double>
  EXPECT_EQ(complex<double>(complex<float>(1.0f, 2.0f)).real, 1.0);
  EXPECT_EQ(complex<double>(complex<float>(1.0f, 2.0f)).imag, 2.0);

  // std::complex<float> to complex<double>
  EXPECT_EQ(complex<double>(std::complex<double>(1.0, 2.0)).real, 1.0);
  EXPECT_EQ(complex<double>(std::complex<double>(1.0, 2.0)).imag, 2.0);
  EXPECT_EQ(complex<double>(std::complex<double>(1.0, 2.0)).real, 1.0);
  EXPECT_EQ(complex<double>(std::complex<double>(1.0, 2.0)).imag, 2.0);

  // Assignment operator
  complex<double> c1 = 1.0;
  EXPECT_EQ(c1.real, 1.0);
  EXPECT_EQ(c1.imag, 0.0);
  c1 = complex<double>(2.0, 2.0);
  EXPECT_EQ(c1.real, 2.0);
  EXPECT_EQ(c1.imag, 2.0);

  // Conversion operator
  EXPECT_EQ(static_cast<double>(complex<double>(0.5)), 0.5);
  EXPECT_NEAR(static_cast<double>(complex<double>(0.33333)), 0.33333, 0.01);
  EXPECT_EQ(static_cast<int>(complex<double>(-1)), -1);
  EXPECT_EQ(static_cast<bool>(complex<double>(true)), true);
}

TEST(bfloat16, comparison_cpu) {
  // *********** complex<float> *************
  EXPECT_TRUE(complex<float>(1.0f) == complex<float>(1.0f));
  EXPECT_TRUE(complex<float>(1.0f, 2.0f) == complex<float>(1.0f, 2.0f));
  EXPECT_FALSE(complex<float>(-1.0f) == complex<float>(-0.5f));
  EXPECT_TRUE(complex<float>(1.0f) != complex<float>(0.5f));
  EXPECT_FALSE(complex<float>(-1.0f) != complex<float>(-1.0f));
  EXPECT_TRUE(complex<float>(1.0f) < complex<float>(2.0f));
  EXPECT_FALSE(complex<float>(-1.0f) < complex<float>(-1.0f));
  EXPECT_TRUE(complex<float>(1.0f) <= complex<float>(1.0f));
  EXPECT_TRUE(complex<float>(2.0f) > complex<float>(1.0f));
  EXPECT_FALSE(complex<float>(-2.0f) > complex<float>(-2.0f));
  EXPECT_TRUE(complex<float>(2.0f) >= complex<float>(2.0f));

  // *********** complex<double> *************
  EXPECT_TRUE(complex<double>(1.0) == complex<double>(1.0));
  EXPECT_TRUE(complex<double>(1.0, 2.0) == complex<double>(1.0, 2.0));
  EXPECT_FALSE(complex<double>(-1.0) == complex<double>(-0.5f));
  EXPECT_TRUE(complex<double>(1.0) != complex<double>(0.5f));
  EXPECT_FALSE(complex<double>(-1.0) != complex<double>(-1.0));
  EXPECT_TRUE(complex<double>(1.0) < complex<double>(2.0));
  EXPECT_FALSE(complex<double>(-1.0) < complex<double>(-1.0));
  EXPECT_TRUE(complex<double>(1.0) <= complex<double>(1.0));
  EXPECT_TRUE(complex<double>(2.0) > complex<double>(1.0));
  EXPECT_FALSE(complex<double>(-2.0) > complex<double>(-2.0));
  EXPECT_TRUE(complex<double>(2.0) >= complex<double>(2.0));
}

TEST(complex, arithmetic_cpu) {
  // *********** complex<float> *************
  complex<float> a = complex<float>(1, 1) + complex<float>(1, 1);
  EXPECT_NEAR(a.real, 2, 0.001);
  EXPECT_NEAR(a.imag, 2, 0.001);

  complex<float> b = complex<float>(-5, -5) + complex<float>(5, 5);
  EXPECT_EQ(b.real, 0);
  EXPECT_EQ(b.imag, 0);

  complex<float> c =
      complex<float>(0.33333f, 0.33333f) + complex<float>(0.66667f, 0.66667f);
  EXPECT_NEAR(c.real, 1.0f, 0.01);
  EXPECT_NEAR(c.imag, 1.0f, 0.01);

  complex<float> d = complex<float>(3) - complex<float>(5);
  EXPECT_EQ(d.real, -2);
  EXPECT_EQ(d.imag, 0);

  complex<float> e =
      complex<float>(0.66667f, 0.66667f) - complex<float>(0.33333f, 0.33333f);
  EXPECT_NEAR(e.real, 0.33334f, 0.01);
  EXPECT_NEAR(e.imag, 0.33334f, 0.01);

  complex<float> f = complex<float>(0.33f, 0.33f) * complex<float>(0.2f, 0.2f);
  EXPECT_NEAR(f.real, 0.0f, 0.01);
  EXPECT_NEAR(f.imag, 0.132f, 0.01);

  complex<float> g = complex<float>(0.33f, 0.33f) / complex<float>(0.2f, 0.2f);
  EXPECT_NEAR(g.real, 1.65f, 0.01);
  EXPECT_NEAR(g.imag, 0.0f, 0.01);

  complex<float> h = -complex<float>(0.33f, 0.33f);
  EXPECT_NEAR(h.real, -0.33f, 0.01);
  EXPECT_NEAR(h.imag, -0.33f, 0.01);
  h = -complex<float>(-0.33f, -0.33f);
  EXPECT_NEAR(h.real, 0.33f, 0.01);
  EXPECT_NEAR(h.imag, 0.33f, 0.01);

  complex<float> i = complex<float>(1.0, 1.0);
  i += complex<float>(2.0, 2.0);
  EXPECT_NEAR(i.real, 3.0f, 0.01);
  EXPECT_NEAR(i.imag, 3.0f, 0.01);
  i -= complex<float>(1.0, 1.0);
  EXPECT_NEAR(i.real, 2.0f, 0.01);
  EXPECT_NEAR(i.imag, 2.0f, 0.01);
  i *= complex<float>(3, 2);
  EXPECT_NEAR(i.real, 2.0f, 0.01);
  EXPECT_NEAR(i.imag, 10.0f, 0.01);
  i /= complex<float>(3, 2);
  EXPECT_NEAR(i.real, 2.0f, 0.01);
  EXPECT_NEAR(i.imag, 2.0f, 0.01);

  // *********** complex<double> *************
  complex<double> a1 = complex<double>(1, 1) + complex<double>(1, 1);
  EXPECT_NEAR(a1.real, 2, 0.001);
  EXPECT_NEAR(a1.imag, 2, 0.001);

  complex<double> b1 = complex<double>(-5, -5) + complex<double>(5, 5);
  EXPECT_EQ(b1.real, 0);
  EXPECT_EQ(b1.imag, 0);

  complex<double> c1 =
      complex<double>(0.33333f, 0.33333f) + complex<double>(0.66667f, 0.66667f);
  EXPECT_NEAR(c1.real, 1.0f, 0.01);
  EXPECT_NEAR(c1.imag, 1.0f, 0.01);

  complex<double> d1 = complex<double>(3) - complex<double>(5);
  EXPECT_EQ(d1.real, -2);
  EXPECT_EQ(d1.imag, 0);

  complex<double> e1 =
      complex<double>(0.66667f, 0.66667f) - complex<double>(0.33333f, 0.33333f);
  EXPECT_NEAR(e1.real, 0.33334f, 0.01);
  EXPECT_NEAR(e1.imag, 0.33334f, 0.01);

  complex<double> f1 =
      complex<double>(0.33f, 0.33f) * complex<double>(0.2f, 0.2f);
  EXPECT_NEAR(f1.real, 0.0f, 0.01);
  EXPECT_NEAR(f1.imag, 0.132f, 0.01);

  complex<double> g1 =
      complex<double>(0.33f, 0.33f) / complex<double>(0.2f, 0.2f);
  EXPECT_NEAR(g1.real, 1.65f, 0.01);
  EXPECT_NEAR(g1.imag, 0.0f, 0.01);

  complex<double> h1 = -complex<double>(0.33f, 0.33f);
  EXPECT_NEAR(h1.real, -0.33f, 0.01);
  EXPECT_NEAR(h1.imag, -0.33f, 0.01);
  h1 = -complex<double>(-0.33f, -0.33f);
  EXPECT_NEAR(h1.real, 0.33f, 0.01);
  EXPECT_NEAR(h1.imag, 0.33f, 0.01);

  complex<double> i1 = complex<double>(1.0, 1.0);
  i1 += complex<double>(2.0, 2.0);
  EXPECT_NEAR(i1.real, 3.0f, 0.01);
  EXPECT_NEAR(i1.imag, 3.0f, 0.01);
  i1 -= complex<double>(1.0, 1.0);
  EXPECT_NEAR(i1.real, 2.0f, 0.01);
  EXPECT_NEAR(i1.imag, 2.0f, 0.01);
  i1 *= complex<double>(3, 2);
  EXPECT_NEAR(i1.real, 2.0f, 0.01);
  EXPECT_NEAR(i1.imag, 10.0f, 0.01);
  i1 /= complex<double>(3, 2);
  EXPECT_NEAR(i1.real, 2.0f, 0.01);
  EXPECT_NEAR(i1.imag, 2.0f, 0.01);
}

TEST(complex, print) {
  complex<float> a(1.0f);
  std::cout << a << std::endl;

  complex<double> b(1.0);
  std::cout << b << std::endl;
}

TEST(complex, isinf) {
  // *********** complex<float> *************
  complex<float> a;
  a.real = float(INFINITY);
  EXPECT_EQ(std::isinf(a), true);
  a.imag = float(INFINITY);
  EXPECT_EQ(std::isinf(a), true);

  complex<float> b = float(INFINITY);
  EXPECT_EQ(std::isinf(b), true);

  complex<float> c(float(INFINITY), 0);
  EXPECT_EQ(std::isinf(c), true);

  // *********** complex<double> *************
  complex<double> a1;
  a1.real = double(INFINITY);
  EXPECT_EQ(std::isinf(a1), true);
  a1.imag = double(INFINITY);
  EXPECT_EQ(std::isinf(a1), true);

  complex<double> b1 = double(INFINITY);
  EXPECT_EQ(std::isinf(b1), true);

  complex<double> c1(double(INFINITY), 0);
  EXPECT_EQ(std::isinf(c1), true);
}

TEST(complex, isnan) {
  // *********** complex<float> *************
  complex<float> a;
  a.real = float(NAN);
  EXPECT_EQ(std::isnan(a), true);
  a.imag = float(NAN);
  EXPECT_EQ(std::isnan(a), true);

  complex<float> b = float(NAN);
  EXPECT_EQ(std::isnan(b), true);

  complex<float> c(float(NAN), 0);
  EXPECT_EQ(std::isnan(c), true);

  // *********** complex<double> *************
  complex<double> a1;
  a1.real = double(NAN);
  EXPECT_EQ(std::isnan(a1), true);
  a1.imag = double(NAN);
  EXPECT_EQ(std::isnan(a1), true);

  complex<double> b1 = double(NAN);
  EXPECT_EQ(std::isnan(b1), true);

  complex<double> c1(double(NAN), 0);
  EXPECT_EQ(std::isnan(c1), true);
}

}  // namespace platform
}  // namespace paddle
#endif