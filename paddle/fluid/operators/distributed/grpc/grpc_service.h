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

#include <grpc++/impl/codegen/async_stream.h>
#include <grpc++/impl/codegen/async_unary_call.h>
#include <grpc++/impl/codegen/proto_utils.h>
#include <grpc++/impl/codegen/rpc_method.h>
#include <grpc++/impl/codegen/service_type.h>
#include <grpc++/impl/codegen/status.h>
#include <grpc++/impl/codegen/stub_options.h>
#include <grpc++/impl/codegen/sync_stream.h>
#include <grpc++/support/byte_buffer.h>
#include "paddle/fluid/operators/distributed/grpc/grpc_variable_response.h"
#include "paddle/fluid/platform/profiler.h"

// NOTE: This method was originally created by tensorflow
//       (https://github.com/tensorflow/tensorflow/) we borrow this
//       method and did some modifications so that we can parse gRPC
//       requests without too much copying of the tensor data.

namespace grpc {
class CompletionQueue;
class Channel;
class RpcService;
class ServerCompletionQueue;
class ServerContext;

// Support parsing/unparsing of tensorflow::VariableResponse.
// Wire-format is identical to RecvVariableResponse.
template <>
class SerializationTraits<
    paddle::operators::distributed::GRPCVariableResponse> {
 public:
  static Status Serialize(
      const paddle::operators::distributed::GRPCVariableResponse& msg,
      grpc_byte_buffer** bp, bool* own_buffer) {
    PADDLE_THROW(paddle::platform::errors::Unimplemented(
        "SerializationTraits::Serialize not implemented!"));
    return Status();
  }
  static Status Deserialize(
      grpc_byte_buffer* buffer,
      paddle::operators::distributed::GRPCVariableResponse* msg,
      int max_message_size = INT_MAX) {
    if (buffer == nullptr) {
      return Status(StatusCode::INTERNAL, "No payload");
    }

    Status result = g_core_codegen_interface->ok();
    if (result.ok()) {
      paddle::operators::distributed::GrpcByteSource source(buffer);
      int ret = msg->Parse(&source);
      if (ret != 0) {
        result = Status(StatusCode::INTERNAL, "VariableResponse parse error");
      }
    }
    g_core_codegen_interface->grpc_byte_buffer_destroy(buffer);
    return result;
  }
};
}  // namespace grpc

namespace paddle {
namespace operators {
namespace distributed {

enum class GrpcMethod {
  kSendVariable,
  kGetVariable,
  kPrefetchVariable,
  kCheckpointNotify,
  kGetVariableNoBarrier,
  kGetMonomerVariable,
  kGetMonomerBarrier,
  kRequestNotify,
  kRequestSendAndRecv,
  // when you add new handler, change kGrpcNumMethods at the same time!
};

static const int kGrpcNumMethods =
    static_cast<int>(GrpcMethod::kRequestSendAndRecv) + 1;

inline const char* GrpcMethodName(GrpcMethod id) {
  switch (id) {
    case GrpcMethod::kSendVariable:
      return "/sendrecv.SendRecvService/SendVariable";
    case GrpcMethod::kGetVariable:
      return "/sendrecv.SendRecvService/GetVariable";
    case GrpcMethod::kGetVariableNoBarrier:
      return "/sendrecv.SendRecvService/GetVariableNoBarrier";
    case GrpcMethod::kGetMonomerVariable:
      return "/sendrecv.SendRecvService/GetMonomerVariable";
    case GrpcMethod::kGetMonomerBarrier:
      return "/sendrecv.SendRecvService/GetMonomerBarrier";
    case GrpcMethod::kPrefetchVariable:
      return "/sendrecv.SendRecvService/PrefetchVariable";
    case GrpcMethod::kCheckpointNotify:
      return "/sendrecv.SendRecvService/CheckpointNotify";
    case GrpcMethod::kRequestNotify:
      return "/sendrecv.SendRecvService/DistributeNotify";
    case GrpcMethod::kRequestSendAndRecv:
      return "/sendrecv.SendRecvService/SendAndRecvVariable";
  }

  // Shouldn't be reached.
  PADDLE_THROW(platform::errors::InvalidArgument(
      "Invalid id: not found valid method name"));
  return nullptr;
}

class GrpcService final {
 public:
  class AsyncService : public ::grpc::Service {
   public:
    AsyncService() {
      for (int i = 0; i < kGrpcNumMethods; ++i) {
        AddMethod(new ::grpc::internal::RpcServiceMethod(
            GrpcMethodName(static_cast<GrpcMethod>(i)),
            ::grpc::internal::RpcMethod::NORMAL_RPC, nullptr));
        ::grpc::Service::MarkMethodAsync(i);
      }
    }
    virtual ~AsyncService() {}

    // Make RequestAsyncUnary public for grpc_call.h
    using ::grpc::Service::RequestAsyncUnary;
  };
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
