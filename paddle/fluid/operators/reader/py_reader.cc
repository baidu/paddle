// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/py_reader.h"
#include <memory>

#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace operators {
namespace reader {

PyReader::PyReader(
    const std::shared_ptr<LoDTensorBlockingQueue>& queue,
    const std::vector<framework::DDim>& dims,
    const std::vector<framework::proto::VarType::Type>& var_types,
    const std::vector<bool>& need_check_feed)
    : framework::FileReader(dims, var_types, need_check_feed) {
  PADDLE_ENFORCE_NOT_NULL(queue,
                          platform::errors::PreconditionNotMet(
                              "LoDTensorBlockingQueue must not be null."));
  queue_ = queue;
}

void PyReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  bool success;
  VLOG(0) << "PyReader: Befoer ReadNext BlockingQueue Size: " << queue_->Size();
  platform::Timer timer;
  timer.Start();
  queue_->Pop(out, &success);
  // *out = queue_->Pop(&success);
  VLOG(0) << "PyReader: out data ptr: " << reinterpret_cast<uintptr_t>(out);
  if (!success) out->clear();
  timer.Pause();
  VLOG(0) << "PyReader: ReadNext: copy time: " << timer.ElapsedSec() << " s";
}

PyReader::~PyReader() { queue_->Close(); }

void PyReader::Shutdown() { queue_->Close(); }

void PyReader::Start() { queue_->ReOpen(); }

SharedPyReader::SharedPyReader(
    const std::shared_ptr<SharedLoDTensorBlockingQueue>& queue,
    const std::vector<framework::DDim>& dims,
    const std::vector<framework::proto::VarType::Type>& var_types,
    const std::vector<bool>& need_check_feed)
    : framework::FileReader(dims, var_types, need_check_feed) {
  PADDLE_ENFORCE_NOT_NULL(queue,
                          platform::errors::PreconditionNotMet(
                              "LoDTensorBlockingQueue must not be null."));
  queue_ = queue;
}

std::shared_ptr<std::vector<framework::LoDTensor>>
SharedPyReader::ReadNextShared() {
  VLOG(0) << "SharedPyReader: Befoer ReadNext BlockingQueue Size: "
          << queue_->Size();
  platform::Timer timer;
  timer.Start();
  auto out = queue_->Pop();
  timer.Pause();
  VLOG(0) << "SharedPyReader: out data ptr: "
          << reinterpret_cast<uintptr_t>(&out);
  VLOG(0) << "SharedPyReader: ReadNext: copy time: " << timer.ElapsedSec()
          << " s";
  return out;
}

SharedPyReader::~SharedPyReader() { queue_->Close(); }

void SharedPyReader::Shutdown() { queue_->Close(); }

void SharedPyReader::Start() { queue_->ReOpen(); }

MultiPyReader::MultiPyReader(
    const std::shared_ptr<MultiLoDTensorBlockingQueue>& queue,
    const std::vector<framework::DDim>& dims,
    const std::vector<framework::proto::VarType::Type>& var_types,
    const std::vector<bool>& need_check_feed)
    : framework::FileReader(dims, var_types, need_check_feed) {
  PADDLE_ENFORCE_NOT_NULL(queue,
                          platform::errors::PreconditionNotMet(
                              "LoDTensorBlockingQueue must not be null."));
  queue_ = queue;
}

void MultiPyReader::ReadNext(std::vector<framework::LoDTensor>* out) {
  bool success;
  VLOG(0) << "PyReader: Befoer ReadNext BlockingQueue Size: " << queue_->Size();
  platform::Timer timer;
  timer.Start();
  queue_->Pop(out, &success);
  // *out = queue_->Pop(&success);
  VLOG(0) << "PyReader: out data ptr: " << reinterpret_cast<uintptr_t>(out);
  if (!success) out->clear();
  timer.Pause();
  VLOG(0) << "PyReader: ReadNext: copy time: " << timer.ElapsedSec() << " s";
}

MultiPyReader::~MultiPyReader() { queue_->Close(); }

void MultiPyReader::Shutdown() { queue_->Close(); }

void MultiPyReader::Start() { queue_->ReOpen(); }

}  // namespace reader
}  // namespace operators
}  // namespace paddle
