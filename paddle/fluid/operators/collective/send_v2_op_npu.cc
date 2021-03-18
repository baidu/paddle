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

#include "paddle/fluid/operators/collective/send_v2_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CSendOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto x = ctx.Input<framework::LoDTensor>("X");
    int numel = x->numel();
    hcclDataType_t dtype = platform::ToHCCLDataType(x->type());

    auto place = ctx.GetPlace();
    int ring_id = ctx.Attr<int>("ring_id");
    auto comm = platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    std::string tag =
        std::to_string(ring_id) + "_" + std::to_string(comm->NextTagId());
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    int destRank = ctx.Attr<int>("peer");
    int srTag = ctx.Attr<int>("srTag");

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::hcom_send(
        tag.c_str(), reinterpret_cast<void*>(const_cast<T*>(x->data<T>())),
        (u64)numel, dtype, destRank, srTag, group.c_str(), stream));

    VLOG(3) << "Dest rank:" << destRank << " Invoke hcom send. Sent "
            << x->numel();

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

REGISTER_OP_NPU_KERNEL(send_v2, ops::CSendOpASCENDKernel<int>,
                       ops::CSendOpASCENDKernel<int8_t>,
                       ops::CSendOpASCENDKernel<float>,
                       ops::CSendOpASCENDKernel<plat::float16>);
