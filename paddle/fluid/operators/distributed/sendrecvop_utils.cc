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
#include <memory>

#include "paddle/fluid/operators/distributed/sendrecvop_utils.h"

namespace paddle {
namespace framework {
class Variable;
}  // namespace framework
namespace memory {
namespace allocation {
class Allocation;
}  // namespace allocation
}  // namespace memory
}  // namespace paddle

DEFINE_bool(rpc_disable_reuse_port, false, "Disable SO_REUSEPORT or not.");
DEFINE_int32(rpc_retry_bind_port, 3,
             "Retry to bind the address if address is already used.");

namespace paddle {
namespace operators {
namespace distributed {

using VarMsg = sendrecv::VariableMessage;

static TensorPayload GetCommunicationAllocationFromTensor(
    const platform::DeviceContext& ctx, const framework::Tensor& tensor) {
  if (is_gpu_place(ctx.GetPlace())) {
#ifdef PADDLE_WITH_CUDA
    PADDLE_ENFORCE_EQ(
        is_gpu_place(tensor.place()), true,
        platform::errors::PreconditionNotMet("Please run in gpu place."));
    auto& gpu_dev_ctx =
        reinterpret_cast<const platform::CUDADeviceContext&>(ctx);
    auto copy_size = tensor.numel() * framework::SizeOfType(tensor.type());
    platform::CUDAPinnedPlace cuda_pinned;
    auto result = memory::AllocShared(cuda_pinned, copy_size);

    memory::Copy(cuda_pinned, result->ptr(),
                 BOOST_GET_CONST(platform::CUDAPlace, tensor.place()),
                 tensor.data<void>(), copy_size, gpu_dev_ctx.stream());
    ctx.Wait();
    return TensorPayload(result);
#else
    PADDLE_THROW(
        platform::errors::Unavailable("This situation should not be happened"));
#endif
  } else {
    return TensorPayload(tensor);
  }
}
TensorPayload GetTensorPayload(framework::Variable* var,
                               const platform::DeviceContext& ctx,
                               VarMsg* request) {
  auto tensor = var->Get<framework::LoDTensor>();
  // FIXME(wuyi): data types in send_recv.proto is copied from
  // framework.proto
  request->set_data_type(static_cast<VarMsg::Type>(tensor.type()));
  for (auto& dim : framework::vectorize(tensor.dims())) {
    request->add_dims(dim);
  }
  const framework::LoD lod = tensor.lod();
  if (lod.size() > 0) {
    request->set_lod_level(lod.size());
    for (auto& each : lod) {
      VarMsg::LodData* lod_inner = request->add_lod();
      for (auto& d : each) {
        lod_inner->add_lod_data(d);
      }
    }
  }
  return GetCommunicationAllocationFromTensor(ctx, tensor);
}

TensorPayload GetSelectedRowsPayload(framework::Variable* var,
                                     const platform::DeviceContext& ctx,
                                     VarMsg* request) {
  auto* slr = var->GetMutable<framework::SelectedRows>();
  request->set_data_type(static_cast<VarMsg::Type>(slr->value().type()));
  request->set_lod_level(0);
  request->set_slr_height(slr->height());

  for (auto& dim : framework::vectorize(slr->value().dims())) {
    request->add_dims(dim);
  }

  auto* tensor = slr->mutable_value();
  return GetCommunicationAllocationFromTensor(ctx, *tensor);
}

TensorPayload::TensorPayload(std::shared_ptr<memory::Allocation> allocation)
    : allocation_(allocation), offset_(0), memory_size_(allocation->size()) {}
TensorPayload::TensorPayload(const framework::Tensor& tensor)
    : allocation_(tensor.Holder()),
      offset_(tensor.offset()),
      memory_size_(tensor.numel() * framework::SizeOfType(tensor.type())) {}
void* TensorPayload::ptr() const {
  return reinterpret_cast<void*>(
      reinterpret_cast<uintptr_t>(allocation_->ptr()) + offset_);
}
size_t TensorPayload::memory_size() const { return memory_size_; }
}  // namespace distributed
}  // namespace operators
}  // namespace paddle
