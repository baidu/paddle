//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>
#include <vector>

#include "paddle/fluid/framework/details/op_handle_base.h"
#include "paddle/fluid/framework/details/stream_executor_impl.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/stream/paddle_stream.h"

namespace paddle {
namespace framework {
namespace details {

struct FetchOpHandle : public OpHandleBase {
 public:
  FetchOpHandle(ir::Node *node, FeedFetchList *data, size_t offset,
                std::vector<Scope *> *local_scopes,
                std::vector<Scope *> *local_exec_scopes, StreamExecutor *exec);

  ~FetchOpHandle();

  void RecordWaitEventOnCtx(platform::DeviceContext *waited_ctx) override;

  void WaitAndMergeCPUTensors() const;

  std::string Name() const override;

  bool IsMultiDeviceTransfer() override;

 protected:
  void RunImpl() override;

  std::vector<Scope *> GetLocalScopes() override { return *local_scopes_; }

  void WaitInputVarGenerated(const platform::Place &place) override;

 private:
  FeedFetchList *data_;
  size_t offset_;
  std::vector<Scope *> *local_scopes_;
  std::vector<Scope *> *local_exec_scopes_;
  std::vector<LoDTensor> tensors_;
  // const paddle::platform::stream::BaseStream* d2h_;
  StreamExecutor *exec_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
