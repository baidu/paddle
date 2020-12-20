/*
 * Copyright (c) 2017, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MANAGED_CUH
#define MANAGED_CUH 

#include <new>

struct managed {
  static void *operator new(size_t n) {
    void *ptr = 0;
    cudaError_t result = cudaMallocManaged(&ptr, n);
    if (cudaSuccess != result || 0 == ptr) throw std::bad_alloc();
    return ptr;
  }

  static void operator delete(void *ptr) noexcept { cudaFree(ptr); }
};

#endif  // MANAGED_CUH
