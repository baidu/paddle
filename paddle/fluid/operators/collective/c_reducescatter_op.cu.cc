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

#include "paddle/fluid/operators/collective/c_reducescatter_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CReduceScatterOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int rid = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);
    int nranks = comm->nranks();

    auto out_dims = in->dims();
    PADDLE_ENFORCE_EQ(out_dims[0] % nranks, 0,
                      platform::errors::InvalidArgument(
                          "The input tensor X's "
                          "dim[0] (%d) should be divisible by nranks(%d)",
                          out_dims[0], nranks));
    out_dims[0] = out_dims[0] / nranks;
    out->mutable_data<T>(out_dims, place);

    int64_t recv_numel = in->numel() / nranks;
    const T* send_buff = in->data<T>();
    T* recv_buff = out->data<T>();
    int dtype = platform::ToNCCLDataType(in->type());

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclReduceScatter(
        send_buff, recv_buff, recv_numel, static_cast<ncclDataType_t>(dtype),
        ncclSum, comm->comm(), stream));
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

REGISTER_OP_CUDA_KERNEL(c_reducescatter, ops::CReduceScatterOpCUDAKernel<float>,
                        ops::CReduceScatterOpCUDAKernel<double>,
                        ops::CReduceScatterOpCUDAKernel<int>,
                        ops::CReduceScatterOpCUDAKernel<int64_t>,
                        ops::CReduceScatterOpCUDAKernel<plat::float16>);
