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
class ExpandMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_vec_dims = framework::vectorize(x->dims());
    auto out_vec_dims = framework::vectorize(out->dims());

    dnnl::memory::format_tag x_format_tag = x->format();
    if (x_vec_dims.size() != out_vec_dims.size()) {
      x_format_tag =
          GetExtendedFormatTag(x_vec_dims, out_vec_dims.size(), x_format_tag);
    }

    out->set_format(x_format_tag);

    platform::BroadcastDataMKLDNNHandler<T> handler(
        dnnl::algorithm::binary_add, dev_ctx, onednn_engine, ctx.GetPlace(),
        out, x, 0.0f, 1.0f, ctx.InputName("X"), x_vec_dims);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);
    auto binary_p = handler.AcquireForwardPrimitive();

    const std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC_0, *dst_memory_p},
        {DNNL_ARG_SRC_1, *src_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    binary_p->execute(astream, args);
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(*dst_memory_p));
  }

 private:
  dnnl::memory::format_tag GetExtendedFormatTag(
      std::vector<int64_t>& dims, int new_size,
      mkldnn::memory::format_tag format_tag) const {
    mkldnn::memory::desc md(dims, platform::MKLDNNGetDataType<T>(), format_tag);
    std::vector<int64_t> new_dims(new_size, 1);
    std::copy(dims.begin(), dims.end(),
              new_dims.begin() + new_size - dims.size());

    dims = std::move(new_dims);
    return platform::GetMKLDNNFormat(md.reshape(dims));
  }
};

template <typename T>
class ExpandGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto dx_vec_dims = framework::vectorize(dx->dims());
    auto dout_vec_dims = framework::vectorize(dout->dims());

    if (dx_vec_dims.size() != dout_vec_dims.size()) {
      dx_vec_dims.insert(dx_vec_dims.begin(),
                         dout_vec_dims.size() - dx_vec_dims.size(), 1);
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (dout_vec_dims == dx_vec_dims) {
      mkldnn::memory::data_type dout_type =
          framework::ToMKLDNNDataType(dout->type());
      std::string key = platform::CreateKey(
          dev_ctx, dout_vec_dims, dout->format(), dout->format(), dout_type);
      platform::ReorderMKLDNNHandler reorder_handler(
          dout_vec_dims, dout->type(), dout_type, dev_ctx, onednn_engine, key);

      auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
          dout->format(), platform::to_void_cast(dout->data<T>()));

      auto reorder_dst_memory_p =
          reorder_handler.AcquireDstMemory(dx, dout->format(), ctx.GetPlace());

      auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                      reorder_dst_memory_p);

      reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
      astream.wait();

      dx->set_layout(framework::DataLayout::kMKLDNN);
      dx->set_format(
          platform::GetMKLDNNFormat(reorder_dst_memory_p->get_desc()));
    } else {
      platform::ReductionMKLDNNHandler<T> handler(
          dnnl::algorithm::reduction_sum, 0.0f, 0.0f, dev_ctx, onednn_engine,
          ctx.GetPlace(), dout, dx, ctx.InputName("X"), dx_vec_dims);

      auto src_memory_p = handler.AcquireSrcMemory(dout);
      auto dst_memory_p = handler.AcquireDstMemory(dx);

      std::unordered_map<int, dnnl::memory> reduction_args = {
          {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

      auto reduction_p = handler.AcquireForwardPrimitive();

      reduction_p->execute(astream, reduction_args);
      astream.wait();
      dx->set_layout(framework::DataLayout::kMKLDNN);
      dx->set_format(platform::GetMKLDNNFormat(dst_memory_p->get_desc().reshape(
          paddle::framework::vectorize<int64_t>(dx->dims()))));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(expand_v2, MKLDNN, paddle::platform::CPUPlace,
                   ops::ExpandMKLDNNKernel<float>,
                   ops::ExpandMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(expand_v2_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ExpandGradMKLDNNKernel<float>,
                   ops::ExpandGradMKLDNNKernel<paddle::platform::bfloat16>);
