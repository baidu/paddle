/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/device_code.h"
#include <utility>
#include "gtest/gtest.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/init.h"

constexpr auto saxpy_code = R"(
extern "C" __global__
void saxpy_kernel(float a, float *x, float* y, float* z, size_t n) {
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n;
       tid += blockDim.x * gridDim.x) {
    z[tid] = a * x[tid] + y[tid];
  }
}
)";

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
TEST(DeviceCode, cuda) {
  if (!paddle::platform::dynload::HasNVRTC() ||
      !paddle::platform::dynload::HasCUDADriver()) {
    return;
  }

  paddle::framework::InitDevices({0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::CUDADeviceCode code(place, "saxpy_kernel", saxpy_code);

  paddle::framework::Tensor cpu_x;
  paddle::framework::Tensor cpu_y;
  paddle::framework::Tensor cpu_z;

  float scale = 2;
  auto dims = paddle::framework::make_ddim(
      {static_cast<int64_t>(256), static_cast<int64_t>(1024)});
  cpu_x.mutable_data<float>(dims, paddle::platform::CPUPlace());
  cpu_y.mutable_data<float>(dims, paddle::platform::CPUPlace());

  size_t n = cpu_x.numel();
  for (size_t i = 0; i < n; ++i) {
    cpu_x.data<float>()[i] = static_cast<float>(i);
  }
  for (size_t i = 0; i < n; ++i) {
    cpu_y.data<float>()[i] = static_cast<float>(0.5);
  }

  paddle::framework::Tensor x;
  paddle::framework::Tensor y;
  paddle::framework::Tensor z;

  float* x_data = x.mutable_data<float>(dims, place);
  float* y_data = y.mutable_data<float>(dims, place);
  float* z_data = z.mutable_data<float>(dims, place);

  TensorCopySync(cpu_x, place, &x);
  TensorCopySync(cpu_y, place, &y);

  EXPECT_EQ(code.Compile(), true);

  std::vector<void*> args = {&scale, &x_data, &y_data, &z_data, &n};
  code.SetNumThreads(1024);
  code.SetWorkloadPerThread(1);
  code.Launch(n, &args);

  auto* dev_ctx = paddle::platform::DeviceContextPool::Instance().Get(place);
  dev_ctx->Wait();

  TensorCopySync(z, paddle::platform::CPUPlace(), &cpu_z);
  for (size_t i = 0; i < n; i++) {
    EXPECT_EQ(cpu_z.data<float>()[i], static_cast<float>(i) * scale + 0.5);
  }
}

TEST(DeviceCodePool, cuda) {
  if (!paddle::platform::dynload::HasNVRTC() ||
      !paddle::platform::dynload::HasCUDADriver()) {
    return;
  }

  paddle::framework::InitDevices({0});
  paddle::platform::CUDAPlace place = paddle::platform::CUDAPlace(0);
  paddle::platform::DeviceCodePool& pool =
      paddle::platform::DeviceCodePool::Init({place});
  size_t num_device_codes_before = pool.size(place);
  EXPECT_EQ(num_device_codes_before, 0UL);

  std::unique_ptr<paddle::platform::DeviceCode> code(
      new paddle::platform::CUDADeviceCode(place, "saxpy_kernel", saxpy_code));
  LOG(INFO) << "origin ptr: " << code.get();
  pool.Set(std::move(code));
  size_t num_device_codes_after = pool.size(place);
  EXPECT_EQ(num_device_codes_after, 1UL);

  paddle::platform::DeviceCode* code_get = pool.Get(place, "saxpy_kernel");
  LOG(INFO) << "get ptr: " << code_get;
}
#endif
