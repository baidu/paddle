/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.1 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.1

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cstdint>
#include "paddle/fluid/platform/hostdevice.h"

#define INT_BITS 32

namespace paddle {
namespace platform {

template <typename T, int Size>
struct alignas(sizeof(T) * Size) CudaAlignedVector {
  T val[Size];
};

struct FastDivMod {
  // 1st value represents the result of input number divides by recorded divisor
  // 2nd value represents the result of input number modulo by recorded divisor
  using DivModT = CudaAlignedVector<uint32_t, 2>;

  FastDivMod() {}
  HOSTDEVICE FastDivMod(uint32_t d) : divisor(d) {
    static_assert(sizeof(unsigned int) == 4,
                  "Only Support 32-bit unsigned int.");

    for (shift_val = 0; shift_val < INT_BITS; ++shift_val) {
      auto shift_limit = 1 << shift_val;
      if (shift_limit >= divisor) break;
    }
    uint64_t long_one = 1;
    uint64_t temp_div =
        ((long_one << INT_BITS) * ((long_one << shift_val) - divisor)) /
            divisor +
        1;
    multiplier = temp_div;
  }

  __device__ __forceinline__ uint32_t Div(uint32_t n) const {
    uint32_t t = __umulhi(n, multiplier);
    return (t + n) >> shift_val;
  }

  __device__ __forceinline__ DivModT Divmod(uint32_t n) const {
    uint32_t q = Div(n);
    DivModT result = {q, n - q * divisor};
    return result;
  }

  int32_t divisor;
  int32_t shift_val;
  uint32_t multiplier;
};

/*
* Only the address of input data is the multiplier of 1,2,4, vectorized load
* with corresponding multiplier-value is possible. Moreover, the maximum length
* of vectorized load is 128 bits once. Hence, valid length of vectorized load
* shall be determined under both former constraints.
*/
template <typename T>
int GetVectorizedSize(const T *pointer) {
  constexpr int max_load_bits = 128;
  int valid_vec_size = max_load_bits / CHAR_BIT / sizeof(T);
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec8 =
      std::alignment_of<CudaAlignedVector<T, 8>>::value;  // NOLINT
  constexpr int vec4 =
      std::alignment_of<CudaAlignedVector<T, 4>>::value;  // NOLINT
  constexpr int vec2 =
      std::alignment_of<CudaAlignedVector<T, 2>>::value;  // NOLINT
  if (address % vec8 == 0) {
    /*
    * Currently, decide to deal with no more than 4 data once while adopting
    * vectorization load/store, if performance test shows that dealing with
    * 8 data once in vectorization load/store does get optimized, return code
    * below can be changed into " return std::min(8, valid_vec_size); " .
    */
    return std::min(4, valid_vec_size);
  } else if (address % vec4 == 0) {
    return std::min(4, valid_vec_size);
  } else if (address % vec2 == 0) {
    return std::min(2, valid_vec_size);
  } else {
    return 1;
  }
}

}  // namespace platform
}  // namespace paddle
