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

using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using paddle::platform::CPUDeviceContext;
using paddle::platform::CreateKey;
using paddle::platform::MKLDNNGetDataType;
using paddle::platform::MKLDNNMemDesc;
using platform::to_void_cast;


template <typename T>
class ReduceMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void RunKernel(const framework::ExecutionContext& ctx, dnnl::algorithm reduction_type) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* input = ctx.Input<LoDTensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    int out_dtype = ctx.Attr<int>("out_dtype");
    int in_dtype = ctx.Attr<int>("in_dtype");

    auto x_dims = input->dims();
    auto x_rank = x_dims.size();

    auto dims = ctx.Attr<std::vector<int>>("dim"); // dims to reduce
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    //bool keep_dim = ctx.Attr<bool>("keep_dim");
    // Change data formats



    platform::ReductionMKLDNNHandler<T> handler(
        reduction_type, 0.0f, 0.0f, dev_ctx, onednn_engine,
        ctx.GetPlace(), input, output,
        ctx.InputName("X"), dims);

    auto src_memory_p = handler.AcquireSrcMemory(input);
    auto dst_memory_p = handler.AcquireDstMemory(output);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto reduction_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    reduction_p->execute(astream, reduction_args);
    astream.wait();

    output->set_layout(framework::DataLayout::kMKLDNN);
    output->set_format(
        platform::GetMKLDNNFormat(dst_memory_p->get_desc().reshape(
            paddle::framework::vectorize<int64_t>(output->dims()))));
  }


};

}  // namespace operators
}  // namespace paddle


