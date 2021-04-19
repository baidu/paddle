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

#pragma once

#include <cstdint>
#include <memory>

#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/stream_callback_manager.h"

namespace paddle {
namespace platform {
namespace stream {

#ifdef PADDLE_WITH_ASCEND_CL

class NPUStream final {
 public:
  NPUStream() = default;
  explicit NPUStream(const Place& place) { Init(place); }
  virtual ~NPUStream() { Destroy(); }

  bool Init(const Place& place);

  template <typename Callback>
  void AddCallback(Callback&& callback) {
    is_callback_exec_ = false;
    std::thread td(ProcessCallback, &is_callback_exec_);
    std::ostringstream oss;
    oss << td.get_id();
    callback_thread_id_ = std::stoull(oss.str());
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtSubscribeReport(
        static_cast<uint64_t>(callback_thread_id_), stream_));

    callback_manager_->AddCallback(callback);
  }

  static void ProcessCallback(void* arg) {
    // aclrtContext context = nullptr;
    // PADDLE_ENFORCE_NPU_SUCCESS(aclrtCreateContext(&context, place_.device));
    while (true) {
      // timeout value is 100ms
      (void)aclrtProcessReport(1000);
      if (*(static_cast<bool*>(arg)) == true) {
        return;
      }
    }
    // PADDLE_ENFORCE_NPU_SUCCESS(aclrtDestroyContext(context));
  }

  // bool GetCallbackExecuteFlag() const {
  //   return is_callback_exec_;
  // }

  void SetCallbackExecuteFlag(bool callback_flag) {
    is_callback_exec_ = callback_flag;
  }

  uint64_t GetCallbackThreadId() const { return callback_thread_id_; }

  template <typename Callback>
  void RecordEvent(aclrtEvent ev, Callback callback) const {
    callback();
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(ev, stream_));
  }

  void RecordEvent(aclrtEvent ev) const {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtRecordEvent(ev, stream_));
  }

  void WaitEvent(aclrtEvent ev) const {
    PADDLE_ENFORCE_NPU_SUCCESS(aclrtStreamWaitEvent(stream_, ev));
  }

  void Wait() const;
  void WaitCallback() const { callback_manager_->Wait(); }

  aclrtStream raw_stream() const { return stream_; }
  void Destroy();

 private:
  Place place_;
  aclrtStream stream_{nullptr};
  std::unique_ptr<StreamCallbackManager<aclrtStream>> callback_manager_;
  bool is_callback_exec_;
  uint64_t callback_thread_id_;

  DISABLE_COPY_AND_ASSIGN(NPUStream);
};

#endif

}  // namespace stream
}  // namespace platform
}  // namespace paddle
