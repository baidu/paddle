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

#ifdef PADDLE_WITH_NCCL
#include <nccl.h>
#endif
#include <sys/time.h>
#include <limits>
#include <memory>
#include <thread>  // NOLINT

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_rdma_pool.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_sendrecvop_utils.h"
#include "paddle/fluid/operators/distributed/brpc/brpc_variable_response.h"
#include "paddle/fluid/operators/distributed/distributed_pb.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {
namespace distributed {

class IOBufWriter {
 public:
  static void Append(const std::string& varname, butil::IOBuf* iobuf, int k,
                     const char* v, int64_t vlen) {
    if (vlen >= std::numeric_limits<int>::max() || vlen < 0) {
      PADDDLE_THROW(platform::errors::Unavailable(
          "Variable lenght is invalid. Variable name is %s, length is %d.",
          varname, vlen));
    }

    iobuf->append(reinterpret_cast<char*>(&k), 4);
    iobuf->append(reinterpret_cast<char*>(&vlen), 8);
    iobuf->append(v, vlen);
  }

  static void AppendTCPZeroCopy(butil::IOBuf* iobuf, int k, const char* v,
                                int64_t vlen, bool in_cuda_pinned,
                                void (*destroy)(void*), void* user_data) {
    VLOG(7) << "AppendTCPZeroCopy "
            << " k:" << k
            << " data:" << static_cast<void*>(const_cast<char*>(v))
            << " data_size:" << vlen << " in_cuda_pinned:" << in_cuda_pinned;

    iobuf->append(reinterpret_cast<char*>(&k), 4);
    iobuf->append(reinterpret_cast<char*>(&vlen), 8);

    // FIXME(gongwb): use append_zerocopy
    /*
    if (in_cuda_pinned) {
      iobuf->append_zerocopy(v, vlen, IOBufWriter::FreeMemory);
    } else {
      iobuf->append_zerocopy(v, vlen, nullptr);
    }
    */
    iobuf->append(v, vlen);
    destroy(user_data);
  }

#ifdef PADDLE_WITH_BRPC_RDMA
  static void AppendRdmaZeroCopy(const std::string varname, butil::IOBuf* iobuf,
                                 int k, const char* v, int64_t vlen,
                                 bool in_cuda_pinned, void (*destroy)(void*),
                                 void* user_data) {
    VLOG(7) << "AppendRdmaZeroCopy varname:" << varname << " k:" << k
            << " data:" << static_cast<void*>(const_cast<char*>(v))
            << " data_size:" << vlen << " in_cuda_pinned:" << in_cuda_pinned;

    iobuf->append(reinterpret_cast<char*>(&k), 4);
    iobuf->append(reinterpret_cast<char*>(&vlen), 8);

    RdmaMemPool::Instance().Register(
        varname, static_cast<void*>(const_cast<char*>(v)), vlen);

    // FIXME(gongwb): use append_zerocopy
    // iobuf->append_zerocopy(v, vlen, nullptr);
    iobuf->append(v, vlen);
    destroy(user_data);
    return;
  }
#endif

  static void AppendZeroCopy(const std::string varname, butil::IOBuf* iobuf,
                             int k, const char* v, int64_t vlen,
                             bool in_cuda_pinned, void (*destroy)(void*),
                             void* user_data) {
    if (vlen >= std::numeric_limits<int>::max() || vlen < 0) {
      PADDDLE_THROW(platform::errors::Unavailable(
          "Variable lenght is invalid. Variable name is %s, length is %d.",
          varname, vlen));
    }

#ifdef PADDLE_WITH_BRPC_RDMA
    IOBufWriter::AppendRdmaZeroCopy(varname, iobuf, k, v, vlen, in_cuda_pinned,
                                    destroy, user_data);
#else
    IOBufWriter::AppendTCPZeroCopy(iobuf, k, v, vlen, in_cuda_pinned, destroy,
                                   user_data);
#endif
  }
};

void SerializeToIOBuf(const std::string& name, framework::Variable* var,
                      const platform::DeviceContext& ctx, VarMsg* request,
                      butil::IOBuf* iobuf, const std::string& out_varname,
                      bool var_is_not_stable, int trainer_id,
                      const std::string& table_name) {
  std::unique_ptr<TensorPayload> payload;

  request->set_varname(name);
  request->set_trainer_id(trainer_id);
  // Note: normally the profiler is enabled in 1 trainer, hence only
  // 1 trainer returns true for ShouldSendProfileState(). It tells PS
  // servers the trainer's profiling state so that PS can follow the
  // trainer.
  if (platform::ShouldSendProfileState()) {
    if (platform::IsProfileEnabled()) {
      request->set_profile(platform::kEnableProfiler);
    } else {
      request->set_profile(platform::kDisableProfiler);
    }
  }
  if (!out_varname.empty()) {
    request->set_out_varname(out_varname);
  }
  if (!table_name.empty()) {
    request->set_table_name(table_name);
  }
  if (var->IsType<framework::LoDTensor>()) {
    request->set_type(::sendrecv::LOD_TENSOR);
    payload.reset(new TensorPayload(GetTensorPayload(var, ctx, request)));
  } else if (var->IsType<framework::SelectedRows>()) {
    request->set_type(::sendrecv::SELECTED_ROWS);
    payload.reset(new TensorPayload(GetSelectedRowsPayload(var, ctx, request)));
#ifdef PADDLE_WITH_NCCL
  } else if (var->IsType<ncclUniqueId>()) {
    request->set_type(::sendrecv::NCCL_ID);
    const ncclUniqueId& uid = var->Get<ncclUniqueId>();
    // TODO(gongwb): use append_zero to avoid data copy.
    IOBufWriter::Append(name, iobuf,
                        sendrecv::VariableMessage::kSerializedFieldNumber,
                        uid.internal, NCCL_UNIQUE_ID_BYTES);
    return;
#endif
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Serialize does not support type: %s", typeid(var->Type()).name()));
  }

  PADDLE_ENFORCE_NOT_NULL(
      payload,
      platform::errors::InvalidArgument(
          "Not support type: %s, need to be LOD_TENSOR or SELECTED_ROWS.",
          var->Type()));

  // FIXME(gongwb): it seems that can use zero copy.
  if (var_is_not_stable) {
    IOBufWriter::Append(
        name, iobuf, ::sendrecv::VariableMessage::kSerializedFieldNumber,
        static_cast<const char*>(payload->ptr()), payload->memory_size());
  } else {
    if (platform::is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
      IOBufWriter::AppendZeroCopy(
          name, iobuf, ::sendrecv::VariableMessage::kSerializedFieldNumber,
          static_cast<const char*>(payload->ptr()), payload->memory_size(),
          true, SerializeDestroyCallback, static_cast<void*>(payload.get()));
      payload.release();
#endif
    } else {
      IOBufWriter::AppendZeroCopy(
          name, iobuf, ::sendrecv::VariableMessage::kSerializedFieldNumber,
          static_cast<const char*>(payload->ptr()), payload->memory_size(),
          false, SerializeDestroyCallback, static_cast<void*>(payload.get()));
      payload.release();
    }
  }

  if (var->IsType<framework::SelectedRows>()) {
    auto* slr = var->GetMutable<framework::SelectedRows>();
    PADDLE_ENFORCE_EQ(VectorElemName(slr->rows()), typeid(int64_t).name(),
                      platform::errors::InvalidArgument(
                          "Got wrong type: %s, expect type: int64_t",
                          VectorElemName(slr->rows())));
    size_t rows_memory_size = slr->rows().size() * sizeof(int64_t);

    IOBufWriter::Append(name, iobuf,
                        ::sendrecv::VariableMessage::kRowsFieldNumber,
                        reinterpret_cast<const char*>(slr->rows().data()),
                        static_cast<int64_t>(rows_memory_size));
  }
}

void DeserializeFromIOBuf(const ::sendrecv::VariableMessage& meta,
                          const butil::IOBuf& iobuf,
                          const platform::DeviceContext& ctx,
                          const framework::Scope* scope,
                          framework::Variable** var, int* trainer_id) {
  operators::distributed::BRPCVariableResponse resp(scope, &ctx);
  PADDLE_ENFORCE_EQ(
      resp.Parse(iobuf, meta), 0,
      platform::errors::InvalidArgument("parse iobuf to tensor error!"));
  *var = resp.GetVar();
  *trainer_id = resp.GetTrainerId();
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
