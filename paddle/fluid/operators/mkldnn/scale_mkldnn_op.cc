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

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

template <typename T>
class ScaleMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    bool is_inplaced = x->IsSharedBufferWith(*out);

    platform::ActivationMKLDNNHandler<T> handler(
        mkldnn::algorithm::eltwise_linear, ctx, dev_ctx, ctx.GetPlace(), x,
        ctx.InputName("X"), is_inplaced);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);
    auto activation_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    activation_p->execute(astream, {{MKLDNN_ARG_FROM, *src_memory_p},
                                    {MKLDNN_ARG_TO, *dst_memory_p}});
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(scale, MKLDNN, paddle::platform::CPUPlace,
                   ops::ScaleMKLDNNKernel<float>,
                   ops::ScaleMKLDNNKernel<paddle::platform::bfloat16>);
