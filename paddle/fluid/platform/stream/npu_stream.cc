/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/stream/npu_stream.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/npu_info.h"

namespace paddle {
namespace platform {
namespace stream {

static bool is_callback_exec_ = false;

bool NPUStream::Init(const Place& place) {
  PADDLE_ENFORCE_EQ(is_npu_place(place), true,
                    platform::errors::InvalidArgument(
                        "NPU stream must be created using npu place."));
  place_ = place;
  NPUDeviceGuard guard(BOOST_GET_CONST(NPUPlace, place_).device);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateStream(&stream_));
  callback_manager_.reset(new StreamCallbackManager<aclrtStream>(stream_));
  is_callback_exec_ = false;
  (void)pthread_create(&callback_thread_id_, nullptr, *ProcessCallback,
                       &is_callback_exec_);
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSubscribeReport(
      static_cast<uint64_t>(callback_thread_id_), stream_));
  VLOG(3) << "NPUStream Init stream: " << stream_;
  return true;
}

void NPUStream::Destroy() {
  NPUDeviceGuard guard(BOOST_GET_CONST(NPUPlace, place_).device);
  Wait();
  WaitCallback();
  is_callback_exec_ = true;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtUnSubscribeReport(
      static_cast<uint64_t>(callback_thread_id_), stream_));
  if (stream_) {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyStream(stream_));
  }
  stream_ = nullptr;
}

void NPUStream::Wait() const {
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSynchronizeStream(stream_));
}

}  // namespace stream
}  // namespace platform
}  // namespace paddle
