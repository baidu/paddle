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

#include "paddle/fluid/operators/collective/send_op_v2.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class SendOpV2CUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
#if NCCL_VERSION_CODE >= 2703
    auto x = ctx.Input<framework::LoDTensor>("X");
    int numel = x->numel();
    ncclDataType_t dtype = platform::ToNCCLDataType(x->type());

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    int peer = ctx.Attr<int>("peer");
    PADDLE_ENFORCE_LT(
        peer, comm->nranks(),
        platform::errors::InvalidArgument("The value of peer (%d) you set must "
                                          "be less than comm->nranks (%d).",
                                          peer, comm->nranks()));

    // Send number of elements to the receiver, as the receiver may have
    // no information of the Tensor size.
    int* numel_ptr = nullptr;
    PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&numel_ptr, sizeof(int)));
    PADDLE_ENFORCE_CUDA_SUCCESS(
        cudaMemcpy(numel_ptr, &numel, sizeof(int), cudaMemcpyHostToDevice));

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupStart());
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
        numel_ptr, 1, ncclInt, peer, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclSend(
        x->data<T>(), numel, dtype, peer, comm->comm(), stream));
    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclGroupEnd());
    VLOG(3) << "rank " << comm->rank() << " send "
            << framework::product(x->dims()) << " to " << peer;
#else
    PADDLE_THROW(
        platform::errors::Unavailable("NCCL version >= 2.7.3 is needed."));
#endif
#else
    PADDLE_THROW(
        platform::errors::Unavailable("PaddlePaddle should compile with GPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(send_v2, ops::SendOpV2CUDAKernel<float>,
                        ops::SendOpV2CUDAKernel<double>,
                        ops::SendOpV2CUDAKernel<int>,
                        ops::SendOpV2CUDAKernel<int64_t>,
                        ops::SendOpV2CUDAKernel<plat::float16>);
