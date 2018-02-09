/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include <cuda_runtime.h>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/framework/mixed_vector.h"
#include "paddle/platform/gpu_info.h"

template <typename T>
using vec = paddle::framework::Vector<T>;

TEST(mixed_vector, CPU_VECTOR) {
  vec<int> tmp;
  for (int i = 0; i < 10; ++i) {
    tmp.push_back(i);
  }
  ASSERT_EQ(tmp.size(), 10);
  vec<int> tmp2;
  tmp2 = tmp;
  ASSERT_EQ(tmp2.size(), 10);
  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp2[i], i);
    ASSERT_EQ(tmp2[i], tmp[i]);
  }
  int cnt = 0;
  for (auto& t : tmp2) {
    ASSERT_EQ(t, cnt);
    ++cnt;
  }
}

static __global__ void multiply_10(int* ptr) {
  for (int i = 0; i < 10; ++i) {
    ptr[i] *= 10;
  }
}

cudaStream_t GetCUDAStream(paddle::platform::CUDAPlace place) {
  return reinterpret_cast<const paddle::platform::CUDADeviceContext*>(
             paddle::platform::DeviceContextPool::Instance().Get(place))
      ->stream();
}

TEST(mixed_vector, GPU_VECTOR) {
  vec<int> tmp;
  for (int i = 0; i < 10; ++i) {
    tmp.push_back(i);
  }
  ASSERT_EQ(tmp.size(), 10);
  paddle::platform::CUDAPlace gpu(0);

  multiply_10<<<1, 1, 0, GetCUDAStream(gpu)>>>(tmp.MutableData(gpu));

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp[i], i * 10);
  }
}

TEST(mixed_vector, MultiGPU) {
  if (paddle::platform::GetCUDADeviceCount() < 2) {
    LOG(WARNING) << "Skip mixed_vector.MultiGPU since there are not multiple "
                    "GPUs in your machine.";
    return;
  }

  vec<int> tmp;
  for (int i = 0; i < 10; ++i) {
    tmp.push_back(i);
  }
  ASSERT_EQ(tmp.size(), 10);
  paddle::platform::CUDAPlace gpu0(0);
  multiply_10<<<1, 1, 0, GetCUDAStream(gpu0)>>>(tmp.MutableData(gpu0));
  paddle::platform::CUDAPlace gpu1(1);
  multiply_10<<<1, 1, 0, GetCUDAStream(gpu1)>>>(tmp.MutableData(gpu1));

  for (int i = 0; i < 10; ++i) {
    ASSERT_EQ(tmp[i], i * 100);
  }
}
