/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CBroadcastOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto x = ctx.Input<framework::LoDTensor>("X");
    void *ptr = reinterpret_cast<void*>(const_cast<T*>(x->data<T>()));
    int numel = x->numel();
    HcclDataType dtype = platform::ToHCCLDataType(x->type());

    auto out = ctx.Output<framework::LoDTensor>("Out");

    int ring_id = ctx.Attr<int>("ring_id");
    auto place = ctx.GetPlace();
    auto comm = paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    int root = ctx.Attr<int>("root");
    std::string group = std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);

    VLOG(3) << "begin hccl broadcast, parameter is: "<< "root " << root
      << ", group is " << group << ", comm: " << comm->comm() << ", stream: " << stream;

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclBroadcast(ptr, numel,
                                  dtype, (uint32_t)root, comm->comm(), stream));

    VLOG(3) << "rank " << comm->rank() << " invoke Bcast. recieved "
            << framework::product(out->dims());

    dev_ctx->Wait();

    if (out != x) {
      framework::TensorCopy(
          *static_cast<const framework::Tensor*>(x), place,
          *platform::DeviceContextPool::Instance().Get(place),
          static_cast<framework::Tensor*>(out));
    }
    dev_ctx->Wait();

    out->Resize(x->dims());
    out->set_lod(x->lod());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(c_broadcast,
                        ops::CBroadcastOpASCENDKernel<int>,
                        ops::CBroadcastOpASCENDKernel<int8_t>,
                        ops::CBroadcastOpASCENDKernel<float>,
                        ops::CBroadcastOpASCENDKernel<plat::float16>);
