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

#include "mkldnn.hpp"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/quantize_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using platform::to_void_cast;
using Tensor = framework::Tensor;
using framework::DataLayout;
using mkldnn::stream;
using platform::GetMKLDNNFormat;

template <typename T>
class QuantOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto scale_data = ctx.Attr<float>("Scale");
    auto scale_shift = ctx.Attr<float>("Shift");
    bool with_shift = scale_shift != 0.0f;
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE_NE(
        scale_data, 0.0f,
        platform::errors::InvalidArgument("Quantization scale cannot be 0.0"));
    PADDLE_ENFORCE_GE(scale_shift, 0,
                      platform::errors::Unimplemented(
                          "Quantization shift must be nonnegative."));
    PADDLE_ENFORCE_LE(
        scale_shift, 255,
        platform::errors::Unimplemented(
            "Quantization shift must be less than or equal to 255."));

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& engine = dev_ctx.GetEngine();

    std::vector<primitive> pipeline;
    auto src_tz = paddle::framework::vectorize<int64_t>(input->dims());
    auto dst_tz = paddle::framework::vectorize<int64_t>(output->dims());

    const T* input_data = input->data<T>();

    bool is_negative_input = ctx.Attr<bool>("is_negative_input");
    bool bfloat16 = ctx.Attr<bool>("bfloat16");

    std::string key = platform::CreateKey(
        platform::ThreadIDasStr(), src_tz, scale_data, scale_shift,
        is_negative_input, ctx.OutputName("Output"));
    const std::string key_prim = key + "@r";
    const std::string key_src_mem = key + "@s";
    const std::string key_dst_mem = key + "@d";

    std::shared_ptr<mkldnn::memory> src_memory;
    std::shared_ptr<mkldnn::memory> dst_memory;
    std::shared_ptr<reorder> reorder_p;
    reorder_p = std::static_pointer_cast<reorder>(dev_ctx.GetBlob(key_prim));

    if (reorder_p == nullptr) {
      std::string out_layout = ctx.Attr<std::string>("output_format");
      MKLDNNMemoryFormat out_format =
          platform::data_format_to_memory_format(out_layout);
      mkldnn::primitive_attr attri;
      int mask = 0;
      attri.set_output_scales(mask, {scale_data});

      if (with_shift) {
        mkldnn::post_ops post_operations;
        post_operations.append_sum();
        attri.set_post_ops(post_operations);
        uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
        // memset casts scale_shift to unsigned char (uint8_t) internally
        std::memset(output_data, scale_shift, output->numel());
      }

      auto src_md = platform::MKLDNNMemDesc({src_tz}, memory::data_type::f32,
                                            input->format());
      src_memory = std::make_shared<mkldnn::memory>(
          src_md, engine, to_void_cast<T>(input_data));

      std::shared_ptr<mkldnn::memory::desc> dst_md;
      if (bfloat16) {
        platform::SetDstMemoryQuantized<paddle::platform::bfloat16>(
            ctx, output, dst_tz, engine, dst_md, dst_memory, out_format);
      } else if (is_negative_input && !with_shift) {
        platform::SetDstMemoryQuantized<int8_t>(ctx, output, dst_tz, engine,
                                                dst_md, dst_memory, out_format);
      } else {
        platform::SetDstMemoryQuantized<uint8_t>(
            ctx, output, dst_tz, engine, dst_md, dst_memory, out_format);
      }
      auto reorder_pd = std::shared_ptr<reorder::primitive_desc>(
          new reorder::primitive_desc(*src_memory, *dst_memory, attri));
      reorder_p = std::shared_ptr<reorder>(new reorder(*reorder_pd));

      dev_ctx.SetBlob(key_prim, reorder_p);
      dev_ctx.SetBlob(key_src_mem, src_memory);
      dev_ctx.SetBlob(key_dst_mem, dst_memory);
    } else {
      src_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_src_mem));
      src_memory->set_data_handle(to_void_cast<T>(input_data));

      dst_memory = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(key_dst_mem));
      auto place = ctx.GetPlace();

      if (bfloat16) {
        dst_memory->set_data_handle(
            output->mutable_data<paddle::platform::bfloat16>(place));
      } else if (with_shift || !is_negative_input) {
        uint8_t* output_data = output->mutable_data<uint8_t>(ctx.GetPlace());
        if (with_shift) std::memset(output_data, scale_shift, output->numel());
        dst_memory->set_data_handle(output_data);
      } else {
        dst_memory->set_data_handle(
            output->mutable_data<int8_t>(ctx.GetPlace()));
      }
    }

    mkldnn::stream astream(engine);
    reorder_p->execute(astream, *src_memory, *dst_memory);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(quantize, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::QuantOpKernel<float>);
