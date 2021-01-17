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

#pragma once

#include <cstdint>
#include <memory>
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream_callback_manager.h"

namespace paddle {
namespace platform {
namespace stream {

#ifdef PADDLE_WITH_HIP

enum class Priority : uint8_t {
  kNull = 0x0,
  kHigh = 0x1,
  kNormal = 0x2,
};

class CUDAStream final {
 public:
  CUDAStream() = default;
  explicit CUDAStream(const Place& place,
                      const Priority& priority = Priority::kNormal) {
    Init(place, priority);
  }
  virtual ~CUDAStream() { Destroy(); }

  bool Init(const Place& place, const Priority& priority = Priority::kNormal);

  template <typename Callback>
  void AddCallback(Callback&& callback) const {
    callback_manager_->AddCallback(callback);
  }

  template <typename Callback>
  void RecordEvent(hipEvent_t ev, Callback callback) const {
    callback();
    PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(ev, stream_));
  }

  void RecordEvent(hipEvent_t ev) const {
    PADDLE_ENFORCE_CUDA_SUCCESS(hipEventRecord(ev, stream_));
  }

  void WaitEvent(hipEvent_t ev) const {
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamWaitEvent(stream_, ev, 0));
  }

  void Wait() const;
  void WaitCallback() const { callback_manager_->Wait(); }

  const hipStream_t& raw_stream() const { return stream_; }
  void Destroy();

 private:
  Place place_;
  hipStream_t stream_{nullptr};
  Priority priority_{Priority::kNormal};
  std::unique_ptr<StreamCallbackManager> callback_manager_;

  DISABLE_COPY_AND_ASSIGN(CUDAStream);
};

#endif

}  // namespace stream
}  // namespace platform
}  // namespace paddle
