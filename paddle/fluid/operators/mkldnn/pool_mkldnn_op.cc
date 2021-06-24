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
class PoolingMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                      mkldnn::pooling_backward> {
 public:
  PoolingMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const Tensor* input,
                       Tensor* output, const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                 mkldnn::pooling_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(input->dims()),
                                framework::ToMKLDNNDataType(input->type()),
                                unique_name)) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(input->layout(), DataLayout::kMKLDNN,
                        platform::errors::InvalidArgument(
                            "Wrong layout set for Input tensor."));
      PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Wrong format set for Input tensor."));

      const std::string pooling_type = ctx.Attr<std::string>("pooling_type");

      std::vector<int> ksize_temp = ctx.Attr<std::vector<int>>("ksize");
      std::vector<int64_t> ksize(begin(ksize_temp), end(ksize_temp));

      std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
      std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

      std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

      const bool global_pooling = ctx.Attr<bool>("global_pooling");
      const std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      // Only 2D pooling is supported now
      PADDLE_ENFORCE_EQ(
          ksize.size(), 2,
          platform::errors::InvalidArgument(
              "The ksize must be 2D, i.e. 2D pooling, but received %dD.",
              ksize.size()));
      PADDLE_ENFORCE_EQ(
          pooling_type == "max" || pooling_type == "avg", true,
          platform::errors::InvalidArgument(
              "The pooling_type must be 'max' or 'avg', but received %s.",
              pooling_type));
      PADDLE_ENFORCE_EQ(
          input->dims().size(), 4,
          platform::errors::InvalidArgument(
              "Input dim must be with 4, i.e. NCHW, but received %d.",
              input->dims().size()));

      const auto input_dims = input->dims();
      framework::DDim data_dims =
          framework::slice_ddim(input_dims, 2, input_dims.size());

      if (global_pooling) {
        operators::UpdateKsize(&ksize, data_dims);
      }

      operators::UpdatePadding(&paddings, global_pooling, 0, padding_algorithm,
                               data_dims, strides, ksize);

      const auto src_tz = paddle::framework::vectorize(input->dims());
      const auto dst_tz = paddle::framework::vectorize(output->dims());

      const auto is_test = ctx.Attr<bool>("is_test");

      const auto dt = framework::ToMKLDNNDataType(input->type());

      const auto exclude_padding = ctx.Attr<bool>("exclusive");

      const auto src_md = mkldnn::memory::desc(src_tz, dt, input->format());
      /* create memory descriptor for pooling without specified format
       * ('any') which lets a primitive (pooling in this case) choose
       * the memory format preferred for best performance
       */

      const auto dst_md =
          platform::MKLDNNMemDesc(dst_tz, dt, MKLDNNMemoryFormat::any);

      auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);

      const bool ceil_mode = ctx.Attr<bool>("ceil_mode");

      if (ceil_mode) {
        CorrectOutputSize(src_tz, dst_tz, ksize, paddings, strides,
                          mkldnn_paddings[1]);
      }

      ComputeAdaptivePoolParameters(ctx, src_tz, &ksize, &strides);

      this->AcquireForwardPrimitiveDescriptor(
          is_test ? mkldnn::prop_kind::forward_inference
                  : mkldnn::prop_kind::forward_training,
          pooling_type == "max"
              ? mkldnn::algorithm::pooling_max
              : (exclude_padding
                     ? mkldnn::algorithm::pooling_avg_exclude_padding
                     : mkldnn::algorithm::pooling_avg_include_padding),
          src_md, dst_md, strides, ksize, mkldnn_paddings[0],
          mkldnn_paddings[1]);
    }
  }

  PoolingMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       platform::Place cpu_place, const Tensor* in_x,
                       const Tensor* out_grad, Tensor* in_x_grad,
                       const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::pooling_forward,
                                 mkldnn::pooling_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(in_x->dims()),
                                framework::ToMKLDNNDataType(in_x->type()),
                                unique_name)) {
    if (!this->isBwdCached()) {
      PADDLE_ENFORCE_EQ(in_x->layout(), DataLayout::kMKLDNN,
                        platform::errors::InvalidArgument(
                            "Wrong layout set for Input tensor"));
      PADDLE_ENFORCE_NE(in_x->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Wrong format set for Input tensor"));

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
      std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      auto in_x_dims = in_x->dims();
      framework::DDim data_dims =
          framework::slice_ddim(in_x_dims, 2, in_x_dims.size());

      if (global_pooling) {
        operators::UpdateKsize(&ksize, data_dims);
      }

      operators::UpdatePadding(&paddings, global_pooling, 0, padding_algorithm,
                               data_dims, strides, ksize);

      auto src_tz = paddle::framework::vectorize<int64_t>(in_x->dims());
      auto diff_src_tz =
          paddle::framework::vectorize<int64_t>(in_x_grad->dims());
      auto diff_dst_tz =
          paddle::framework::vectorize<int64_t>(out_grad->dims());

      const auto dt = framework::ToMKLDNNDataType(in_x->type());
      auto src_md = mkldnn::memory::desc(src_tz, dt, in_x->format());
      auto dst_md =
          mkldnn::memory::desc(diff_dst_tz, dt, MKLDNNMemoryFormat::any);
      auto diff_dst_md = mkldnn::memory::desc(
          diff_dst_tz, platform::MKLDNNGetDataType<T>(), out_grad->format());
      auto diff_src_md =
          mkldnn::memory::desc(diff_src_tz, platform::MKLDNNGetDataType<T>(),
                               MKLDNNMemoryFormat::any);

      auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);
      const bool ceil_mode = ctx.Attr<bool>("ceil_mode");

      if (ceil_mode) {
        CorrectOutputSize(src_tz, diff_dst_tz, ksize, paddings, strides,
                          mkldnn_paddings[1]);
      }
      ComputeAdaptivePoolParameters(ctx, diff_src_tz, &ksize, &strides);

      const auto exclude_padding = ctx.Attr<bool>("exclusive");

      this->AcquireForwardPrimitiveDescriptor(
          mkldnn::prop_kind::forward_training,
          pooling_type == "max"
              ? mkldnn::algorithm::pooling_max
              : (exclude_padding
                     ? mkldnn::algorithm::pooling_avg_exclude_padding
                     : mkldnn::algorithm::pooling_avg_include_padding),
          src_md, dst_md, strides, ksize, mkldnn_paddings[0],
          mkldnn_paddings[1]);

      this->AcquireBackwardPrimitiveDescriptor(
          pooling_type == "max"
              ? mkldnn::algorithm::pooling_max
              : (exclude_padding
                     ? mkldnn::algorithm::pooling_avg_exclude_padding
                     : mkldnn::algorithm::pooling_avg_include_padding),
          diff_src_md, diff_dst_md, strides, ksize, mkldnn_paddings[0],
          mkldnn_paddings[1]);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireWorkspaceMemory(void) {
    mkldnn::memory::desc workspace_md = this->fwd_pd_->workspace_desc();
    // Pooling PD has to be passed to Grad op that
    // may be executed by diffrent thread, hence
    // for that one we use key that does not contain TID
    auto local_key = this->key_common_ + "@workspace";
    auto mem_p = std::static_pointer_cast<mkldnn::memory>(
        this->dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);
      mem_p = std::static_pointer_cast<mkldnn::memory>(
          this->dev_ctx_.GetBlob(local_key));
      if (mem_p == nullptr) {
        mem_p = std::make_shared<mkldnn::memory>(workspace_md, this->engine_);
        this->dev_ctx_.SetBlob(local_key, mem_p);
      }
    }
    return mem_p;
  }

  static void ComputeAdaptivePoolParameters(
      const paddle::framework::ExecutionContext& ctx,
      const std::vector<int64_t>& src_tz, std::vector<int64_t>* ksize,
      std::vector<int64_t>* strides) {
    if (ctx.Attr<bool>("adaptive")) {
      // https://github.com/oneapi-src/oneDNN/tree/bkocot/adaptive-pooling/rfcs/20200818-adaptive-pooling
      auto IH = static_cast<double>(src_tz[src_tz.size() - 2]);
      auto IW = static_cast<double>(src_tz[src_tz.size() - 1]);
      auto OH = static_cast<double>(ksize->at(0));
      auto OW = static_cast<double>(ksize->at(1));

      strides->at(0) =
          static_cast<int64_t>(floor((IH * 2.0) / OH) - floor(IH / OH));
      strides->at(1) =
          static_cast<int64_t>(floor((IW * 2.0) / OW) - floor(IW / OW));
      ksize->at(0) =
          static_cast<int64_t>(ceil((IH * 2.0) / OH) - floor(IH / OH));
      ksize->at(1) =
          static_cast<int64_t>(ceil((IW * 2.0) / OW) - floor(IW / OW));
    }
  }

 private:
  static inline int ComputeCeiledOutput(int input_size, int kernel_size,
                                        int padding, int stride) {
    return (input_size - kernel_size + 2 * padding) / stride + 1;
  }

  static inline void CorrectOutputSize(
      const std::vector<int64_t>& src_tz, const std::vector<int64_t>& dst_tz,
      const std::vector<int64_t>& kernel_size,
      const std::vector<int64_t>& paddings, const std::vector<int64_t>& strides,
      std::vector<int64_t>& right_bot_padding) {  // NOLINT
    for (size_t i = 0; i < right_bot_padding.size(); i++) {
      int desired_size = ComputeCeiledOutput(src_tz[i + 2], kernel_size[i],
                                             paddings[i], strides[i]);
      if (desired_size != dst_tz[i + 2]) {
        right_bot_padding[i] += strides[i] - 1;
      }
    }
  }
};

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Pool must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    PoolingMKLDNNHandler<T> handler(ctx, dev_ctx, ctx.GetPlace(), input, output,
                                    ctx.OutputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(input);
    auto dst_memory = handler.AcquireDstMemory(output);

    auto pool_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
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

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    PoolingMKLDNNHandler<T> handler(ctx, dev_ctx, ctx.GetPlace(), in_x,
                                    out_grad, in_x_grad, ctx.InputName("Out"));

    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    auto pool_bwd_p = handler.AcquireBackwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (ctx.Attr<std::string>("pooling_type") == "max") {
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
                   ops::PoolMKLDNNOpKernel<uint8_t>,
                   ops::PoolMKLDNNOpKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNGradOpKernel<float>);
