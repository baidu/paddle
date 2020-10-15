/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_allgather_op.h"

#include <memory>

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CAllGatherOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    ncclDataType_t dtype = platform::ToNCCLDataType(in->type());

    int nranks = ctx.Attr<int>("nranks");
    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    PADDLE_ENFORCE_EQ(
        nranks, comm->nranks(),
        platform::errors::InvalidArgument("nranks: %s should equal to %s",
                                          nranks, comm->nranks()));

    framework::DDim out_dims = in->dims();
    out_dims[0] *= nranks;
    out->mutable_data<T>(out_dims, place);

    int64_t send_numel = in->numel();
    const T* send_buff = in->data<T>();
    T* recv_buff = out->data<T>();

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllGather(
        send_buff, recv_buff, send_numel, static_cast<ncclDataType_t>(dtype),
        comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(c_allgather, ops::CAllGatherOpCUDAKernel<float>,
                        ops::CAllGatherOpCUDAKernel<double>,
                        ops::CAllGatherOpCUDAKernel<int>,
                        ops::CAllGatherOpCUDAKernel<int64_t>,
                        ops::CAllGatherOpCUDAKernel<plat::float16>);
