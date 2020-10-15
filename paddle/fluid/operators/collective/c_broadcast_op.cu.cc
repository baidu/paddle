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

#include "paddle/fluid/operators/collective/c_broadcast_op.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto x = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
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

    int root = ctx.Attr<int>("root");
    if (root == comm->rank()) {
      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclBcast(
          reinterpret_cast<void*>(const_cast<T*>(x->data<T>())), numel, dtype,
          root, comm->comm(), stream));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. sent "
              << x->numel();

      if (out != x) {
        framework::TensorCopy(
            *static_cast<const framework::Tensor*>(x), place,
            *platform::DeviceContextPool::Instance().Get(place),
            static_cast<framework::Tensor*>(out));
      }
    } else {
      PADDLE_ENFORCE_CUDA_SUCCESS(
          platform::dynload::ncclBcast(out->mutable_data<T>(place), numel,
                                       dtype, root, comm->comm(), stream));
      VLOG(3) << "rank " << comm->rank() << " invoke Bcast. recieved "
              << framework::product(out->dims());
    }

    out->Resize(x->dims());
    out->set_lod(x->lod());
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

REGISTER_OP_CUDA_KERNEL(c_broadcast, ops::CBroadcastOpCUDAKernel<float>,
                        ops::CBroadcastOpCUDAKernel<double>,
                        ops::CBroadcastOpCUDAKernel<int>,
                        ops::CBroadcastOpCUDAKernel<int64_t>,
                        ops::CBroadcastOpCUDAKernel<plat::float16>);
