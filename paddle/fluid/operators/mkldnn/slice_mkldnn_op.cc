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
class SliceMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("Input");
    auto* out = ctx.Output<Tensor>("Out");

    auto x_vec_dims = framework::vectorize(x->dims());
    auto out_vec_dims = framework::vectorize(out->dims());

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");

    std::vector<int64_t> axes(ctx.Attr<std::vector<int>>("axes").begin(),
                              ctx.Attr<std::vector<int>>("axes").end());
    std::vector<int64_t> starts(ctx.Attr<std::vector<int>>("starts").begin(),
                                ctx.Attr<std::vector<int>>("starts").end());
    std::vector<int64_t> ends(ctx.Attr<std::vector<int>>("ends").begin(),
                              ctx.Attr<std::vector<int>>("ends").end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    std::vector<int64_t> offsets(x_vec_dims.size(), 0);
    std::vector<int64_t> slice_dims(x_vec_dims);

    for (size_t i = 0; i < axes.size(); ++i) {
      starts[i] = starts[i] < 0 ? x_vec_dims[axes[i]] + starts[i] : starts[i];
      ends[i] = ends[i] < 0 ? x_vec_dims[axes[i]] + ends[i]
                            : std::min(ends[i], x_vec_dims[axes[i]]);
      offsets[axes[i]] = starts[i];
      slice_dims[axes[i]] = ends[i] - starts[i];
    }

    mkldnn::memory::data_type x_type = framework::ToMKLDNNDataType(x->type());
    auto key = platform::CreateKey(dev_ctx, x_vec_dims, axes, starts, ends,
                                   x->format(), x_type);

    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, x->type(), x_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->format(), platform::to_void_cast(x->data<T>()));
    auto slice_mem_p = reorder_handler.AcquireSubmemory(slice_dims, offsets,
                                                        reorder_src_memory_p);
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, slice_dims, 0, x->format(), ctx.GetPlace());

    auto reorder_p =
        reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p);
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(
        reorder_dst_memory_p->get_desc().reshape(out_vec_dims)));
  }
};

template <typename T>
class SliceGradMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("Input"));

    auto dx_vec_dims = framework::vectorize(dx->dims());
    auto dout_vec_dims = framework::vectorize(dout->dims());

    auto axes_int = ctx.Attr<std::vector<int>>("axes");
    auto starts_int = ctx.Attr<std::vector<int>>("starts");
    auto ends_int = ctx.Attr<std::vector<int>>("ends");

    std::vector<int64_t> axes(ctx.Attr<std::vector<int>>("axes").begin(),
                              ctx.Attr<std::vector<int>>("axes").end());
    std::vector<int64_t> starts(ctx.Attr<std::vector<int>>("starts").begin(),
                                ctx.Attr<std::vector<int>>("starts").end());
    std::vector<int64_t> ends(ctx.Attr<std::vector<int>>("ends").begin(),
                              ctx.Attr<std::vector<int>>("ends").end());

    auto decrease_axis = ctx.Attr<std::vector<int>>("decrease_axis");

    std::vector<int64_t> offsets(dx_vec_dims.size(), 0);
    std::vector<int64_t> slice_dims(dx_vec_dims);

    for (size_t i = 0; i < axes.size(); ++i) {
      starts[i] = starts[i] < 0 ? dx_vec_dims[axes[i]] + starts[i] : starts[i];
      ends[i] = ends[i] < 0 ? dx_vec_dims[axes[i]] + ends[i]
                            : std::min(ends[i], dx_vec_dims[axes[i]]);
      offsets[axes[i]] = starts[i];
      slice_dims[axes[i]] = ends[i] - starts[i];
    }

    mkldnn::memory::data_type dout_type =
        framework::ToMKLDNNDataType(dout->type());
    mkldnn::memory::desc md(dout_vec_dims, platform::MKLDNNGetDataType<T>(),
                            dout->format());
    mkldnn::memory::format_tag reorder_format_tag =
        platform::GetMKLDNNFormat(md.reshape(slice_dims));

    auto key = platform::CreateKey(dev_ctx, dout_vec_dims, axes, starts, ends,
                                   reorder_format_tag, dout_type);

    platform::ReorderMKLDNNHandler reorder_handler(
        slice_dims, dout->type(), dout_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        reorder_format_tag, platform::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        dx, dx_vec_dims, 0, reorder_format_tag, ctx.GetPlace());
    memset(dx->data<T>(), 0, reorder_dst_memory_p->get_desc().get_size());

    auto slice_mem_p = reorder_handler.AcquireSubmemory(slice_dims, offsets,
                                                        reorder_dst_memory_p);

    auto reorder_p =
        reorder_handler.AcquireReorder(slice_mem_p, reorder_src_memory_p);
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *slice_mem_p);
    astream.wait();

    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(reorder_format_tag);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(slice, MKLDNN, paddle::platform::CPUPlace,
                   ops::SliceMKLDNNKernel<float>,
                   ops::SliceMKLDNNKernel<paddle::platform::bfloat16>);

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(slice_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::SliceGradMKLDNNKernel<float>,
                   ops::SliceGradMKLDNNKernel<paddle::platform::bfloat16>);