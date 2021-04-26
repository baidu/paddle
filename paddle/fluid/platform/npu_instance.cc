/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/npu_instance.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace platform {

AclInstance::~AclInstance() {}

AclInstance &AclInstance::Instance() {
  static AclInstance instance;
  return instance;
}

AclInstance::AclInstance() {
  PADDLE_ENFORCE_NPU_SUCCESS(aclInit(nullptr));
  VLOG(4) << "Call aclrtSetDevice ";
  // NOTE(zhiqiu): why set devices here?
  // Because ACL creates a default context which contains 2 streams
  // when calling aclrtSetDeviceId, so usually we do not need to
  // create contexts explicitly. And, for each device, aclrtSetDeviceId
  // need to call parily with aclrtResetDeviceId to destory the default
  // context. Here, we use this singleton and static instance to manage
  // the devices to make sure they will be resetted before program exit.
  devices_ = platform::GetSelectedNPUDevices();
  for (auto it = devices_.rbegin(); it != devices_.rend(); ++it) {
    SetNPUDeviceId(*it);
    VLOG(4) << "Call aclrtSetDevice " << *it;
  }
}

void AclInstance::Finalize() {
  // NOTE(zhiqiu): DO NOT perform finalize in destructor
  // to avoid problems caused by destructor order of static
  // object.
  for (size_t i = 0; i < devices_.size(); ++i) {
    auto status = aclrtResetDevice(devices_[i]);
    VLOG(4) << "Call aclrtResetDevice " << devices_[i]
            << " status = " << status;
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(
        platform::NPUPlace(devices_[i]));
    auto npu_stream =
        static_cast<platform::NPUDeviceContext *>(dev_ctx)->NPUstream();
    npu_stream->Destroy();
    VLOG(4) << "Call destropy NPU stream";
  }
  auto status = aclFinalize();
  VLOG(4) << "Call aclFinalize, status = " << status;
}

}  // namespace platform
}  // namespace paddle
