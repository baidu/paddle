/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;
using paddle::platform::MKLDNNMemDesc;

using dnnl::memory;  // Note: paddle has also "memory" namespace
using dnnl::primitive;
using dnnl::prop_kind;
using dnnl::softmax_backward;
using dnnl::softmax_forward;
using dnnl::stream;
using platform::to_void_cast;

template <typename T>
class SoftmaxMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                      mkldnn::softmax_backward> {
 public:
  SoftmaxMKLDNNHandler(const MKLDNNDeviceContext& dev_ctx,
                       const mkldnn::engine mkldnn_engine,
                       platform::Place cpu_place, const Tensor* input,
                       Tensor* output, const int axis,
                       const std::string uniq_name, bool is_inplaced)
      : platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                 mkldnn::softmax_backward>(
            dev_ctx, mkldnn_engine, cpu_place,
            // Softmax may be inplace then uniq_name is no longer unique
            is_inplaced ? platform::CreateKey(
                              dev_ctx, framework::vectorize(input->dims()),
                              axis, uniq_name)
                        : platform::CreateKey(
                              dev_ctx, framework::vectorize(input->dims()),
                              uniq_name)) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(
          input->dims(), output->dims(),
          platform::errors::InvalidArgument(
              "The shape of input and output tensor must be identical."));

      auto softmax_tz = framework::vectorize(input->dims());
      auto md = memory::desc(softmax_tz, platform::MKLDNNGetDataType<T>(),
                             input->format());

      this->AcquireForwardPrimitiveDescriptor(prop_kind::forward_scoring, md,
                                              axis);
    }
  }

  SoftmaxMKLDNNHandler(const framework::ExecutionContext& ctx,
                       const MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const Tensor* out,
                       const Tensor* out_grad, Tensor* in_x_grad,
                       const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::softmax_forward,
                                 mkldnn::softmax_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(out->dims()),
                                unique_name)) {
    if (!this->isBwdCached()) {
      PADDLE_ENFORCE_EQ(
          out_grad->dims(), in_x_grad->dims(),
          platform::errors::InvalidArgument("The shape of softmax_grad's input "
                                            "and output must be identical."));

      auto dims = out_grad->dims();  // input and output share the same shape
      const int axis = CanonicalAxis(ctx.Attr<int>("axis"), dims.size());
      auto softmax_tz = framework::vectorize<int64_t>(dims);

      auto data_softmax_md = MKLDNNMemDesc(
          softmax_tz, platform::MKLDNNGetDataType<T>(), out->format());
      auto diff_softmax_md = MKLDNNMemDesc(
          softmax_tz, platform::MKLDNNGetDataType<T>(), out_grad->format());

      this->AcquireForwardPrimitiveDescriptor(prop_kind::forward_scoring,
                                              data_softmax_md, axis);
      this->AcquireBackwardPrimitiveDescriptor(diff_softmax_md, data_softmax_md,
                                               axis);
    }
  }
};

template <typename T>
class SoftmaxMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");
    bool is_inplaced = input->IsSharedBufferWith(*output);

    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), input->dims().size());

    SoftmaxMKLDNNHandler<T> handler(dev_ctx, mkldnn_engine, ctx.GetPlace(),
                                    input, output, axis, ctx.OutputName("Out"),
                                    is_inplaced);

    auto softmax_src_memory_p = handler.AcquireSrcMemory(input);
    // For Inplace src and and dst are the same memory object
    auto softmax_dst_memory_p =
        is_inplaced ? softmax_src_memory_p : handler.AcquireDstMemory(output);

    auto softmax_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    softmax_p->execute(astream, {{DNNL_ARG_SRC, *softmax_src_memory_p},
                                 {DNNL_ARG_DST, *softmax_dst_memory_p}});
    astream.wait();

    const bool is_test = ctx.Attr<bool>("is_test");
    if (!is_test) {
      T* output_data = output->mutable_data<T>(ctx.GetPlace());
      std::for_each(output_data, &output_data[output->numel()], [](T& val) {
        val = std::max(val, static_cast<T>(exp(-64)));
      });
    }

    output->set_layout(framework::DataLayout::kMKLDNN);
    // Softmax output format is the same as input one
    output->set_format(input->format());
  }
};

template <typename T>
class SoftmaxMKLDNNGradKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL SoftmaxGrad must use CPUPlace"));
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const Tensor* output = ctx.Input<Tensor>("Out");
    auto* out_grad = ctx.template Input<Tensor>(framework::GradVarName("Out"));
    auto* in_x_grad = ctx.template Output<Tensor>(framework::GradVarName("X"));

    SoftmaxMKLDNNHandler<T> handler(ctx, dev_ctx, ctx.GetPlace(), output,
                                    out_grad, in_x_grad, ctx.InputName("Out"));

    auto dst_memory_p = handler.AcquireDstMemory(output);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(in_x_grad);

    auto softmax_bwd_p = handler.AcquireBackwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    softmax_bwd_p->execute(astream,
                           {{MKLDNN_ARG_DST, *dst_memory_p},
                            {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                            {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
    astream.wait();

    in_x_grad->set_layout(framework::DataLayout::kMKLDNN);
    in_x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory_p));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(softmax, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNKernel<float>,
                   ops::SoftmaxMKLDNNKernel<paddle::platform::bfloat16>);
REGISTER_OP_KERNEL(softmax_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::SoftmaxMKLDNNGradKernel<float>);
