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

#include "paddle/fluid/operators/distributed/variable_response.h"
#include <vector>
#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"

DEFINE_string(rpc_server_profile_path, "./profile_ps",
              "the profile log file path");

namespace paddle {
namespace operators {
namespace distributed {

bool VariableResponse::ReadRaw(::google::protobuf::io::CodedInputStream* input,
                               const platform::DeviceContext& dev_ctx,
                               platform::Place place, void* dest,
                               int64_t size) {
  const void* data = NULL;
  int size_to_write = 0;
  int64_t length = size;
  int total_written = 0;

  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto& gpu_dev_ctx =
        static_cast<const platform::CUDADeviceContext&>(dev_ctx);
    platform::CPUPlace cpu;

    char* p = reinterpret_cast<char*>(dest);
    while (total_written < length) {
      if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
        return false;
      }
      // NOTE: if raw buffer is large and have two neighbor fields of raw
      // buffers GetDirectBufferPointer can get all of them, use length to
      // truncate it.
      if (total_written + size_to_write > length) {
        size_to_write = length - total_written;
      }
      // This log is useful to see how long a internal block size is of rpc.
      VLOG(7) << "copy " << size_to_write << " data to CUDAPlace";
      memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place),
                   reinterpret_cast<void*>(p), cpu, data, size_to_write,
                   gpu_dev_ctx.stream());
      p += size_to_write;
      total_written += size_to_write;

      input->Skip(size_to_write);
    }
    gpu_dev_ctx.Wait();
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "Unexpected branch, please compile with PADDLE_WITH_CUDA"));
#endif
    return true;
  } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    auto& xpu_dev_ctx = static_cast<const platform::XPUDeviceContext&>(dev_ctx);
    platform::CPUPlace cpu;
    char* p = reinterpret_cast<char*>(dest);
    while (total_written < length) {
      if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
        return false;
      }

      if (total_written + size_to_write > length) {
        size_to_write = length - total_written;
      }

      memory::Copy(BOOST_GET_CONST(platform::XPUPlace, place),
                   reinterpret_cast<void*>(p), cpu, data, size_to_write);
      p += size_to_write;
      total_written += size_to_write;
      input->Skip(size_to_write);
    }
    xpu_dev_ctx.Wait();
#else
    PADDLE_ENFORCE_NOT_NULL(
        nullptr,
        platform::errors::Unimplemented(
            "Not supported XPU, please compile with option WITH_XPU=ON."));
#endif
    return true;
  }

  char* p = reinterpret_cast<char*>(dest);
  while (total_written < length) {
    if (!input->GetDirectBufferPointer(&data, &size_to_write)) {
      return false;
    }
    // NOTE: if raw buffer is large and have two neighbor fields of raw buffers
    // GetDirectBufferPointer can get all of them, use length to truncate it.
    if (total_written + size_to_write > length) {
      size_to_write = length - total_written;
    }
    // TODO(gongwb): can we avoid copy?
    platform::CPUPlace cpu;
    // This log is useful to see how long a internal block size is of rpc.
    VLOG(7) << "copy " << size_to_write << " data to CPUPlace";
    memory::Copy(cpu, reinterpret_cast<void*>(p), cpu, data, size_to_write);

    p += size_to_write;
    total_written += size_to_write;

    input->Skip(size_to_write);
  }

  return true;
}

bool VariableResponse::CopyLodTensorData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, const framework::DDim& dims,
    int length) {
  auto server_var = GetVar();
  if (!server_var) {
    LOG(ERROR) << "recved var should not on current server: "
               << meta_.varname();
    return false;
  }
  auto* tensor = GetVar()->GetMutable<framework::LoDTensor>();
  tensor->Resize(dims);
  framework::LoD lod;
  for (int i = 0; i < meta_.lod_level(); ++i) {
    framework::Vector<size_t> v;
    for (int j = 0; j < meta_.lod(i).lod_data_size(); ++j) {
      v.push_back(meta_.lod(i).lod_data(j));
    }
    lod.push_back(v);
  }
  tensor->set_lod(lod);

  void* tensor_data =
      tensor->mutable_data(ctx.GetPlace(), ToVarType(meta_.data_type()));

  VLOG(6) << "Tensor.memory_size = " << tensor->memory_size()
          << ", Buffer Size = " << length << ", dims:" << dims
          << ", numel:" << tensor->numel();
  PADDLE_ENFORCE_GE(
      tensor->memory_size(), static_cast<unsigned int>(length),
      platform::errors::InvalidArgument(
          "The memory size of tensor: %s should greater than length: %s",
          tensor->memory_size(), length));
  return ReadRaw(input, ctx, tensor->place(), tensor_data, length);
}

inline framework::DDim GetDims(
    const ::google::protobuf::RepeatedField<::google::protobuf::int64>& dims) {
  std::vector<int> vecdims;
  for (auto& d : dims) {
    vecdims.push_back(d);
  }
  return framework::make_ddim(vecdims);
}

bool VariableResponse::CopySelectRowsTensorData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, const framework::DDim& dims,
    int length) {
  auto* slr = GetVar()->GetMutable<framework::SelectedRows>();
  slr->set_height(meta_.slr_height());
  auto* tensor = slr->mutable_value();
  tensor->Resize(dims);
  PADDLE_ENFORCE_EQ(
      static_cast<size_t>(tensor->numel()),
      length / framework::SizeOfType(paddle::operators::distributed::ToVarType(
                   meta_.data_type())),
      platform::errors::InvalidArgument(
          "length: %s should equal to memory size of tensor: %s", length,
          tensor->numel() *
              framework::SizeOfType(paddle::operators::distributed::ToVarType(
                  meta_.data_type()))));
  void* tensor_data = tensor->mutable_data(
      ctx.GetPlace(),
      paddle::operators::distributed::ToVarType(meta_.data_type()));

  if (!ReadRaw(input, ctx, tensor->place(), tensor_data, length)) {
    return false;
  }

  return true;
}

bool VariableResponse::CopySelectRowsData(
    ::google::protobuf::io::CodedInputStream* input,
    const platform::DeviceContext& ctx, int length) {
  auto* slr = GetVar()->GetMutable<framework::SelectedRows>();
  slr->mutable_rows()->clear();
  slr->mutable_rows()->resize(length / sizeof(int64_t));  // int64
  int64_t* rows_data = slr->mutable_rows()->data();

  // copy rows CPU data, GPU data will be copied lazily.
  platform::CPUPlace cpu;
  if (!ReadRaw(input, ctx, cpu, rows_data, length)) {
    return false;
  }

  return true;
}

bool VariableResponse::ProcSerializedField(
    int tag, ::google::protobuf::io::CodedInputStream* input,
    int64_t num_bytes) {
  PADDLE_ENFORCE(
      (meta_.type() == sendrecv::SELECTED_ROWS ||
       meta_.type() == sendrecv::LOD_TENSOR ||
       meta_.type() == sendrecv::NCCL_ID) &&
          meta_.varname() != "",
      platform::errors::PreconditionNotMet("meta info should be got first!"));

  if (meta_.type() == sendrecv::NCCL_ID) {
#ifdef PADDLE_WITH_CUDA
    auto* var = scope_->FindVar(meta_.varname());
    if (var != nullptr) {
      ncclUniqueId* id = var->GetMutable<ncclUniqueId>();
      if (!ReadRaw(input, *dev_ctx_, platform::CPUPlace(), id->internal,
                   num_bytes)) {
        return false;
      }
    }
    return true;
#else
    PADDLE_THROW(
        platform::errors::PreconditionNotMet("Please compiled with CUDA!"));
    return false;
#endif
  }

  VLOG(7) << "ProcSerializedField:" << meta_.varname()
          << ", type:" << meta_.type() << std::endl;
  framework::DDim dims = GetDims(meta_.dims());
  if (meta_.type() == sendrecv::LOD_TENSOR) {
    PADDLE_ENFORCE_GE(
        meta_.lod_size(), 0,
        platform::errors::PreconditionNotMet("lod info should be got first!"));
    if (!CopyLodTensorData(input, *dev_ctx_, dims, num_bytes)) {
      return false;
    }

    return true;
  }

  if (meta_.type() == sendrecv::SELECTED_ROWS) {
    if (!CopySelectRowsTensorData(input, *dev_ctx_, dims, num_bytes)) {
      return false;
    }
    return true;
  }

  PADDLE_THROW(platform::errors::InvalidArgument(
      "The type: %s of var: %s is not supported", meta_.type(),
      meta_.varname()));

  return false;
}

};  // namespace distributed
};  // namespace operators
};  // namespace paddle
