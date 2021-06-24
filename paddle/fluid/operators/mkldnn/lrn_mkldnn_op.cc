/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

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
using paddle::platform::MKLDNNDeviceContext;

template <typename T>
class LRNMKLDNNHandler : public platform::MKLDNNHandlerT<T, mkldnn::lrn_forward,
                                                         mkldnn::lrn_backward> {
 public:
  LRNMKLDNNHandler(const framework::ExecutionContext& ctx,
                   const MKLDNNDeviceContext& dev_ctx,
                   const mkldnn::engine mkldnn_engine,
                   platform::Place cpu_place, const Tensor* input,
                   const std::string& unique_name)

      : platform::MKLDNNHandlerT<T, mkldnn::lrn_forward, mkldnn::lrn_backward>(
            dev_ctx, mkldnn_engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(input->dims()),
                                unique_name)) {
    if (!this->isCached()) {
      const int n = ctx.Attr<int>("n");
      // MKL-DNN implements LRN in a caffe way:
      // http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
      // Where sum of squares is divided by size of normalization window
      // this is not the case for PaddlePaddle LRN.
      // Hence we need to compensate for this diffrence by
      // multipliing alpha by size of window(n)
      const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
      const float beta = ctx.Attr<float>("beta");
      const float k = ctx.Attr<float>("k");
      bool is_test = ctx.Attr<bool>("is_test");

      auto dims = framework::vectorize(input->dims());

      auto src_md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(),
                                         input->format());

      this->AcquireForwardPrimitiveDescriptor(
          is_test ? mkldnn::prop_kind::forward_inference
                  : mkldnn::prop_kind::forward_training,
          mkldnn::algorithm::lrn_across_channels, src_md, n, alpha, beta, k);
    }
  }

  LRNMKLDNNHandler(const framework::ExecutionContext& ctx,
                   const MKLDNNDeviceContext& dev_ctx,
                   platform::Place cpu_place, const Tensor* in_x,
                   const Tensor* out_grad, Tensor* in_x_grad,
                   const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::lrn_forward, mkldnn::lrn_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(in_x->dims()),
                                unique_name)) {
    if (!this->isBwdCached()) {
      PADDLE_ENFORCE_EQ(
          ctx.Attr<bool>("is_test"), false,
          platform::errors::PreconditionNotMet(
              "is_test attribute should be set to False in training phase."));

      const int n = ctx.Attr<int>("n");
      const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
      const float beta = ctx.Attr<float>("beta");
      const float k = ctx.Attr<float>("k");

      auto dims = framework::vectorize<int64_t>(in_x->dims());

      auto src_md = mkldnn::memory::desc(dims, platform::MKLDNNGetDataType<T>(),
                                         in_x->format());
      auto diff_md = mkldnn::memory::desc(
          dims, platform::MKLDNNGetDataType<T>(), out_grad->format());

      this->AcquireForwardPrimitiveDescriptor(
          mkldnn::prop_kind::forward_training,
          mkldnn::algorithm::lrn_across_channels, src_md, n, alpha, beta, k);

      this->AcquireBackwardPrimitiveDescriptor(
          mkldnn::algorithm::lrn_across_channels, src_md, diff_md, n, alpha,
          beta, k);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireWorkspaceMemory(Tensor* workspace) {
    T* ptr = workspace->mutable_data<T>(
        this->place_, this->fwd_pd_->workspace_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->workspace_desc(),
                                            ptr, "@wrk_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireBackwardWorkspaceMemory(
      const Tensor* workspace) {
    const T* workspace_data = workspace->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->workspace_desc(),
        platform::to_void_cast<T>(workspace_data), "@bwd-wrk_mem_p");
  }
};

template <typename T>
class LRNMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE_EQ(
        is_float_type, true,
        platform::errors::PreconditionNotMet("DNNL LRN must use float data."));
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL LRN must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<Tensor>("X");
    auto out = ctx.Output<Tensor>("Out");
    auto mid = ctx.Output<Tensor>("MidOut");

    LRNMKLDNNHandler<T> handler(ctx, dev_ctx, mkldnn_engine, ctx.GetPlace(), x,
                                ctx.OutputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(out);

    auto lrn_p = handler.AcquireForwardPrimitive();

    auto workspace_memory = handler.AcquireWorkspaceMemory(mid);
    mid->set_layout(framework::DataLayout::kMKLDNN);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (!workspace_memory->get_desc().is_zero()) {
      mid->set_format(platform::GetMKLDNNFormat(*workspace_memory));
      lrn_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DST, *dst_memory},
                               {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      lrn_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DST, *dst_memory}});
    }
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class LRNMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE_EQ(is_float_type, true,
                      platform::errors::PreconditionNotMet(
                          "DNNL LRN GradOpKernel must use float data."));
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL LRNGrad must use CPUPlace"));

    auto in_x = ctx.Input<Tensor>("X");
    auto mid = ctx.Input<Tensor>("MidOut");

    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

    LRNMKLDNNHandler<T> handler(ctx, dev_ctx, ctx.GetPlace(), in_x, out_grad,
                                in_x_grad, ctx.InputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(in_x);
    auto workspace = handler.AcquireBackwardWorkspaceMemory(mid);
    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    auto lrn_bwd = handler.AcquireBackwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    lrn_bwd->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                               {MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                               {MKLDNN_ARG_WORKSPACE, *workspace}});
    astream.wait();

    in_x_grad->set_layout(framework::DataLayout::kMKLDNN);
    in_x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(lrn, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(lrn_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNGradOpKernel<float>);
