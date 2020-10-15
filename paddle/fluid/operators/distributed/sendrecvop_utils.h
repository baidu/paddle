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

#pragma once
#include <iostream>
#include <memory>
#include <string>
#include <typeindex>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace framework {
class Tensor;
class Variable;
}  // namespace framework
namespace memory {
namespace allocation {
class Allocation;
}  // namespace allocation
}  // namespace memory
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

using VarMsg = sendrecv::VariableMessage;

class TensorPayload final {
 public:
  explicit TensorPayload(const framework::Tensor& tensor);
  explicit TensorPayload(std::shared_ptr<memory::Allocation> allocation);

  TensorPayload(const TensorPayload& o) = default;
  TensorPayload& operator=(const TensorPayload& o) = default;

  void* ptr() const;
  size_t memory_size() const;

 private:
  std::shared_ptr<memory::Allocation> allocation_;
  size_t offset_;
  size_t memory_size_;
};

inline void SerializeDestroyCallback(void* payload) {
  if (payload != nullptr) {
    auto* shared_payload = reinterpret_cast<TensorPayload*>(payload);
    delete shared_payload;
  }
}

TensorPayload GetTensorPayload(framework::Variable* var,
                               const platform::DeviceContext& ctx,
                               VarMsg* request);

TensorPayload GetSelectedRowsPayload(framework::Variable* var,
                                     const platform::DeviceContext& ctx,
                                     VarMsg* request);

inline framework::proto::VarType::Type ToVarType(
    sendrecv::VariableMessage::Type type) {
  switch (type) {
    case sendrecv::VariableMessage::FP32:
      return framework::proto::VarType::FP32;  // NOLINT
    case sendrecv::VariableMessage::FP64:
      return framework::proto::VarType::FP64;  // NOLINT
    case sendrecv::VariableMessage::INT32:
      return framework::proto::VarType::INT32;  // NOLINT
    case sendrecv::VariableMessage::INT64:
      return framework::proto::VarType::INT64;  // NOLINT
    case sendrecv::VariableMessage::BOOL:
      return framework::proto::VarType::BOOL;  // NOLINT
    default:
      PADDLE_THROW(
          platform::errors::InvalidArgument("Not support type id: %d.", type));
  }
}

template <template <typename> class T, typename Elem>
std::string VectorElemName(const T<Elem>& arg) {
  return typeid(Elem).name();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
