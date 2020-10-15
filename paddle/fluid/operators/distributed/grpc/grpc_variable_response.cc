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

#include <stdint.h>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/io/coded_stream.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_bytebuffer_stream.h"
#include "paddle/fluid/operators/distributed/grpc/grpc_variable_response.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler.h"

namespace google {
namespace protobuf {
namespace io {
class ZeroCopyInputStream;
}  // namespace io
}  // namespace protobuf
}  // namespace google
namespace grpc {
class ByteBuffer;
}  // namespace grpc

namespace paddle {
namespace operators {
namespace distributed {

enum WireType {
  WIRETYPE_VARINT = 0,
  WIRETYPE_LENGTH_DELIMITED = 2,
};

inline int GetTagFieldNumber(uint32_t tag) { return tag >> 3; }

inline WireType GetTagWireType(uint32_t tag) {
  return static_cast<WireType>(tag & 0x7);
}

bool ReadVarintSizeAsInt(::google::protobuf::io::CodedInputStream* input,
                         int* result) {
  uint64_t v;
  if (input->ReadVarint64(&v) && v <= static_cast<uint64_t>(INT_MAX)) {
    *result = static_cast<int>(v);
    return true;
  } else {
    return false;
  }
}

int GRPCVariableResponse::Parse(const ::grpc::ByteBuffer& byte_buffer) {
  GrpcByteBufferSource source;
  source.Init(byte_buffer);
  GrpcByteBufferSourceWrapper r(&source);

  return Parse(&r);
}

bool ParseLodData(::google::protobuf::io::CodedInputStream* input,
                  std::vector<int64_t>* lod) {
  while (true) {
    auto p = input->ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);

    if (!p.second) {
      return (tag == 0);
    }

    switch (tag) {
      case sendrecv::VariableMessage_LodData::kLodDataFieldNumber: {
        uint64_t v;
        if (wt == WIRETYPE_VARINT) {
          if (!input->ReadVarint64(&v)) {
            return false;
          }
          lod->push_back(v);
          break;
        }

        if (wt == WIRETYPE_LENGTH_DELIMITED) {
          int num_bytes = 0;
          if (!input->ReadVarintSizeAsInt(&num_bytes)) {
            return tag;
          }
          int start_pos = input->CurrentPosition();
          while (input->CurrentPosition() - start_pos < num_bytes) {
            uint64_t v;
            if (!input->ReadVarint64(&v)) {
              return tag;
            }
            lod->push_back(v);
          }
          break;
        }

        return false;
      }
      default: { return false; }
    }
  }

  return true;
}

int GRPCVariableResponse::Parse(Source* source) {
  ::google::protobuf::io::ZeroCopyInputStream* input_stream =
      source->contents();
  ::google::protobuf::io::CodedInputStream input(input_stream);
  input.SetTotalBytesLimit(INT_MAX, INT_MAX);

  while (true) {
    auto p = input.ReadTagWithCutoff(127);
    int tag = GetTagFieldNumber(p.first);
    WireType wt = GetTagWireType(p.first);
    if (!p.second) {
      if (tag != 0) {
        return -1;
      }
      return 0;
    }

    switch (tag) {
      case sendrecv::VariableMessage::kVarnameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return tag;
        }

        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return tag;
        }

        meta_.set_varname(temp);
        break;
      }
      case sendrecv::VariableMessage::kTypeFieldNumber: {
        uint32_t v;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return tag;
        }

        meta_.set_type(static_cast<::sendrecv::VarType>(v));
        break;
      }
      case sendrecv::VariableMessage::kDataTypeFieldNumber: {
        uint32_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint32(&v)) {
          return tag;
        }

        meta_.set_data_type(static_cast<::sendrecv::VariableMessage_Type>(v));
        break;
      }
      case sendrecv::VariableMessage::kDimsFieldNumber: {
        // not packed
        if (wt == WIRETYPE_VARINT) {
          uint64_t v;
          if (!input.ReadVarint64(&v)) {
            return tag;
          }
          meta_.add_dims(v);
          break;
        }

        // packed
        if (wt == WIRETYPE_LENGTH_DELIMITED) {
          int num_bytes = 0;
          if (!input.ReadVarintSizeAsInt(&num_bytes)) {
            return tag;
          }
          int start_pos = input.CurrentPosition();
          while (input.CurrentPosition() - start_pos < num_bytes) {
            uint64_t v;
            if (!input.ReadVarint64(&v)) {
              return tag;
            }
            meta_.add_dims(v);
          }
          break;
        }
        return tag;
      }
      case sendrecv::VariableMessage::kLodLevelFieldNumber: {
        uint64_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }
        meta_.set_lod_level(static_cast<int64_t>(v));
        break;
      }
      case sendrecv::VariableMessage::kLodFieldNumber: {
        int length = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &length)) {
          return tag;
        }

        std::pair<::google::protobuf::io::CodedInputStream::Limit, int> p =
            input.IncrementRecursionDepthAndPushLimit(length);

        std::vector<int64_t> lod_data;
        if (p.second < 0 || !ParseLodData(&input, &lod_data)) {
          return tag;
        }

        if (!input.DecrementRecursionDepthAndPopLimit(p.first)) {
          return tag;
        }

        if (lod_data.size() == 0) {
          break;
        }

        auto lod = meta_.add_lod();
        for (uint32_t i = 0; i < lod_data.size(); i++) {
          lod->add_lod_data(lod_data[i]);
        }
        break;
      }
      case sendrecv::VariableMessage::kSlrHeightFieldNumber: {
        uint64_t v = 0;
        if ((wt != WIRETYPE_VARINT) || !input.ReadVarint64(&v)) {
          return tag;
        }
        meta_.set_slr_height(static_cast<int64_t>(v));
        break;
      }
      case sendrecv::VariableMessage::kSerializedFieldNumber: {
        int num_bytes = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &num_bytes)) {
          return tag;
        }

        if (!ProcSerializedField(tag, &input, num_bytes)) {
          return tag;
        }

        break;
      }
      case sendrecv::VariableMessage::kRowsFieldNumber: {
        PADDLE_ENFORCE((meta_.type() == sendrecv::SELECTED_ROWS ||
                        meta_.type() == sendrecv::LOD_TENSOR) &&
                           meta_.varname() != "",
                       platform::errors::PreconditionNotMet(
                           "meta info should be got first!"));

        int num_bytes = 0;
        if (wt != WIRETYPE_LENGTH_DELIMITED ||
            !ReadVarintSizeAsInt(&input, &num_bytes)) {
          return tag;
        }

        if (!CopySelectRowsData(&input, *dev_ctx_, num_bytes)) {
          return tag;
        }
        break;
      }
      case sendrecv::VariableMessage::kOutVarnameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return tag;
        }

        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return tag;
        }

        meta_.set_out_varname(temp);
        break;
      }
      case sendrecv::VariableMessage::kProfileFieldNumber: {
        uint64_t profiling = 0;
        if (!input.ReadVarint64(&profiling)) {
          return tag;
        }
        meta_.set_profile(profiling);
        int64_t listener_id = platform::ListenerId();
        if (listener_id <= 0) {
          break;
        }
        if (profiling == platform::kEnableProfiler &&
            !platform::IsProfileEnabled()) {
          platform::EnableProfiler(platform::ProfilerState::kCPU);
        } else if (profiling == platform::kDisableProfiler &&
                   platform::IsProfileEnabled()) {
          platform::DisableProfiler(
              platform::EventSortingKey::kDefault,
              string::Sprintf("%s_%lld", FLAGS_rpc_server_profile_path,
                              listener_id));
        }
        break;
      }
      case sendrecv::VariableMessage::kTrainerIdFieldNumber: {
        uint64_t trainer_id = 0;
        if (!input.ReadVarint64(&trainer_id)) {
          return tag;
        }
        meta_.set_trainer_id(trainer_id);
        break;
      }
      case sendrecv::VariableMessage::kTableNameFieldNumber: {
        uint32_t length;
        if ((wt != WIRETYPE_LENGTH_DELIMITED) || !input.ReadVarint32(&length)) {
          return tag;
        }

        std::string temp;
        if (!input.ReadString(&temp, length)) {
          return tag;
        }

        meta_.set_table_name(temp);
        break;
      }
      default: {
        // Unknown tag, return unknown error.
        return -1;
      }
    }
  }

  return 0;
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
