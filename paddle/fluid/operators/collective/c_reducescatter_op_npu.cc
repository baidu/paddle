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

#include "paddle/fluid/operators/collective/c_reducescatter_op.h"

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

template <typename T>
class CReduceScatterOpAscendKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    int ring_id = ctx.Attr<int>("ring_id");
    std::string group = std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto place = ctx.GetPlace();
    auto comm = platform::HCCLCommContext::Instance().Get(ring_id, place);
    int nranks = comm->nranks();

    auto out_dims = in->dims();
    PADDLE_ENFORCE_EQ(out_dims[0] % nranks, 0,
                      platform::errors::InvalidArgument(
                          "The input tensor X's "
                          "dim[0] (%d) should be divisible by nranks(%d)",
                          out_dims[0], nranks));

    out_dims[0] = out_dims[0] / nranks;
    out->mutable_data<T>(out_dims, place);

    uint64_t recv_numel = in->numel() / nranks;

    void* inputPtr = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));
    void* outputPtr = reinterpret_cast<void*>(out->data<T>());
    HcclDataType dtype = platform::ToHCCLDataType(in->type());

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }
    VLOG(3) << "begin hccl reduce scatter, parameter is: "
      << "recv_numel: " << recv_numel
      << "dtype: " << dtype
      << "hccl_red_type: " << HCCL_REDUCE_SUM
      << ", group is: " << group;

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclReduceScatter(
        inputPtr, outputPtr, recv_numel, dtype, HCCL_REDUCE_SUM, comm->comm(), (void*)stream));
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

REGISTER_OP_NPU_KERNEL(c_reducescatter,
                        ops::CReduceScatterOpAscendKernel<int8_t>,
                        ops::CReduceScatterOpAscendKernel<int>,
                        ops::CReduceScatterOpAscendKernel<float>,
                        ops::CReduceScatterOpAscendKernel<plat::float16>);
