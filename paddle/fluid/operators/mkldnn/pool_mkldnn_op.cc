/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Pool must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    platform::PoolingMKLDNNHandler<T> handler(ctx, dev_ctx, mkldnn_engine,
                                              ctx.GetPlace(), input, output,
                                              ctx.OutputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(input);
    auto dst_memory = handler.AcquireDstMemory(output);

    auto pool_p = handler.AcquireForwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    if ((ctx.Attr<bool>("is_test") == false) &&
        (ctx.Attr<std::string>("pooling_type") == "max")) {
      // Training
      auto workspace_memory = handler.AcquireWorkspaceMemory();
      pool_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                                {MKLDNN_ARG_DST, *dst_memory},
                                {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      // Inference
      pool_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                                {MKLDNN_ARG_DST, *dst_memory}});
    }
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class PoolMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL PoolGrad must use CPUPlace"));
    const Tensor* in_x = ctx.Input<Tensor>("X");
    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(
        in_x->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for Input tensor"));
    PADDLE_ENFORCE_NE(
        in_x->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for Input tensor"));

    PADDLE_ENFORCE_EQ(out_grad->layout(), DataLayout::kMKLDNN,
                      platform::errors::InvalidArgument(
                          "Wrong layout set for Input output_grad tensor"));
    PADDLE_ENFORCE_NE(out_grad->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Wrong format set for Input output_grad tensor"));

    PADDLE_ENFORCE_EQ(
        ctx.Attr<bool>("is_test"), false,
        platform::errors::InvalidArgument(
            "is_test attribute should be set to False in training phase."));

    std::string pooling_type = ctx.Attr<std::string>("pooling_type");

    std::vector<int> ksize_temp = ctx.Attr<std::vector<int>>("ksize");
    std::vector<int64_t> ksize(begin(ksize_temp), end(ksize_temp));

    std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
    std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

    std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

    bool global_pooling = ctx.Attr<bool>("global_pooling");
    std::string padding_algorithm = ctx.Attr<std::string>("padding_algorithm");

    auto in_x_dims = in_x->dims();
    framework::DDim data_dims =
        framework::slice_ddim(in_x_dims, 2, in_x_dims.size());

    if (global_pooling) {
      UpdateKsize(&ksize, data_dims);
    }

    UpdatePadding(&paddings, global_pooling, 0, padding_algorithm, data_dims,
                  strides, ksize);

    platform::PoolingMKLDNNHandler<T>::ComputeAdaptivePoolParameters(
        ctx, paddle::framework::vectorize(in_x->dims()), ksize, strides);

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    std::vector<mkldnn::primitive> pipeline;

    auto diff_src_tz = paddle::framework::vectorize<int64_t>(in_x_grad->dims());
    auto diff_dst_tz = paddle::framework::vectorize<int64_t>(out_grad->dims());

    // Get an unique name from "argument" name of "Out" variable
    // This name will be used as key when referring info from device context
    const std::string key = platform::CreateKey(
        diff_src_tz, pooling_type, ksize, strides, paddings,
        memory::data_type::f32, in_x->format(), ctx.InputName("Out"));

    platform::PoolingMKLDNNHandler<T> handler(
        diff_dst_tz, diff_src_tz, ksize, strides, paddings, pooling_type,
        ctx.Attr<bool>("ceil_mode"), in_x->format(), out_grad->format(),
        paddle::framework::ToMKLDNNDataType(out_grad->type()), dev_ctx,
        ctx.GetPlace(), ctx.InputName("Out"), ctx.Attr<bool>("exclusive"));

    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    auto pool_bwd_p = handler.AcquireBackwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    if (pooling_type == "max") {
      // Max - pooling needs Workspace
      auto workspace_memory = handler.AcquireWorkspaceMemory();
      pool_bwd_p->execute(astream, {{MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                                    {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      // Average Pooling
      pool_bwd_p->execute(astream, {{MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory}});
    }
    astream.wait();

    in_x_grad->set_layout(DataLayout::kMKLDNN);
    in_x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(pool2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNOpKernel<float>,
                   ops::PoolMKLDNNOpKernel<int8_t>,
                   ops::PoolMKLDNNOpKernel<uint8_t>);

REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNGradOpKernel<float>);
