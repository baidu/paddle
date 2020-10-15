// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
//

#include "paddle/fluid/operators/distributed/brpc/brpc_variable_response.h"
#include "paddle/fluid/operators/distributed/send_recv.pb.h"

namespace paddle {
namespace operators {
namespace distributed {

namespace pb = ::google::protobuf;
using vr = ::sendrecv::VariableMessage;

int BRPCVariableResponse::Parse(Source* source) {
  pb::io::ZeroCopyInputStream* input_stream = source->contents();
  pb::io::CodedInputStream input(input_stream);
  input.SetTotalBytesLimit(INT_MAX, INT_MAX);

  while (1) {
    unsigned int tag = 0;
    if (!input.ReadLittleEndian32(&tag)) {
      break;
    }

    uint64_t num_bytes = 0;
    if (!input.ReadLittleEndian64(&num_bytes)) {
      break;
    }

    int field = static_cast<int>(tag);
    int ret = field == 0 ? -1 : field;
    switch (field) {
      case vr::kSerializedFieldNumber: {
        if (!ProcSerializedField(field, &input, num_bytes)) {
          return ret;
        }
        break;
      }
      case vr::kRowsFieldNumber: {
        PADDLE_ENFORCE((meta_.type() == sendrecv::SELECTED_ROWS ||
                        meta_.type() == sendrecv::LOD_TENSOR) &&
                           meta_.varname() != "",
                       platform::errors::PreconditionNotMet(
                           "meta info should be got first!"));

        if (!CopySelectRowsData(&input, *dev_ctx_, num_bytes)) {
          return ret;
        }
        break;
      }
      default: {
        PADDLE_THROW(platform::errors::Unavailable(
            "not surpported %u fieldnumber", field));
        return ret;
      }
    }
  }

  return 0;
}
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
