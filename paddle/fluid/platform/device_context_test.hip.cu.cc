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
#include "paddle/fluid/platform/device_context.h"

#include <vector>

#include "glog/logging.h"
#include "gtest/gtest.h"

TEST(Device, Init) {
  using paddle::platform::DeviceContext;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    delete device_context;
  }
}

TEST(Device, CUDADeviceContext) {
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::CUDAPlace;

  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; i++) {
    CUDADeviceContext* device_context = new CUDADeviceContext(CUDAPlace(i));
    Eigen::GpuDevice* gpu_device = device_context->eigen_device();
    ASSERT_NE(nullptr, gpu_device);
    miopenHandle_t cudnn_handle = device_context->cudnn_handle();
    ASSERT_NE(nullptr, cudnn_handle);
    delete device_context;
  }
}

TEST(Device, DeviceContextPool) {
  using paddle::platform::DeviceContextPool;
  using paddle::platform::CUDADeviceContext;
  using paddle::platform::Place;
  using paddle::platform::CPUPlace;
  using paddle::platform::CUDAPlace;

  DeviceContextPool& pool = DeviceContextPool::Instance();
  auto cpu_dev_ctx1 = pool.Get(CPUPlace());
  auto cpu_dev_ctx2 = pool.Get(CPUPlace());
  ASSERT_EQ(cpu_dev_ctx2, cpu_dev_ctx1);

  std::vector<Place> gpu_places;
  int count = paddle::platform::GetCUDADeviceCount();
  for (int i = 0; i < count; ++i) {
    auto dev_ctx = pool.Get(CUDAPlace(i));
    ASSERT_NE(dev_ctx, nullptr);
  }
}
