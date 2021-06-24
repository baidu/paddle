/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/barrier_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class BarrierOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype = platform::ToNCCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<void>();
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    auto stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    ncclRedOp_t nccl_red_type = ncclSum;
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
#ifdef PADDLE_WITH_RCCL
    PADDLE_ENFORCE_CUDA_SUCCESS(hipStreamSynchronize(stream));
#else
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(stream));
#endif
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with NCCL."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(barrier, ops::BarrierOpCUDAKernel<int>);
