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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/operators/conv_op.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace platform {
class MKLDNNDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::GetMKLDNNFormat;
using platform::to_void_cast;

inline MKLDNNMemoryFormat GetWeightsFormat(const MKLDNNMemoryFormat format,
                                           const int groups,
                                           const bool is_conv3d) {
  if (is_conv3d) {
    return (groups == 1) ? format : MKLDNNMemoryFormat::goidhw;
  } else {
    return (groups == 1) ? format : MKLDNNMemoryFormat::goihw;
  }
}

static mkldnn::memory::data_type GetDstType(bool is_int8, bool is_bfloat16,
                                            bool force_fp32_output,
                                            std::string fuse_activation,
                                            bool fuse_residual_conn,
                                            const Tensor* residual_param) {
  auto dst_dt = mkldnn::memory::data_type::f32;
  if (is_int8) {
    dst_dt = (fuse_activation == "relu" || fuse_activation == "relu6")
                 ? mkldnn::memory::data_type::u8
                 : mkldnn::memory::data_type::s8;
    if (force_fp32_output) {
      dst_dt = mkldnn::memory::data_type::f32;
    }
    if (fuse_residual_conn && residual_param) {
      auto residual_dt = framework::ToMKLDNNDataType(residual_param->type());
      if (dst_dt != residual_dt) dst_dt = residual_dt;
    }
  } else {
    if (!force_fp32_output && is_bfloat16) {
      dst_dt = mkldnn::memory::data_type::bf16;
      if (fuse_residual_conn && residual_param) {
        dst_dt = framework::ToMKLDNNDataType(residual_param->type());
      }
    }
  }
  return dst_dt;
}

template <typename T, typename K, typename T_out>
class ConvMKLDNNHandlerT
    : public platform::MKLDNNHandlerT<T, mkldnn::convolution_forward,
                                      mkldnn::convolution_backward_data,
                                      mkldnn::convolution_backward_weights> {
 public:
  ConvMKLDNNHandlerT(const paddle::framework::ExecutionContext& ctx,
                     const platform::MKLDNNDeviceContext& dev_ctx,
                     const mkldnn::engine mkldnn_engine,
                     platform::Place cpu_place, const Tensor* input,
                     const Tensor* filter, const Tensor* bias, Tensor* output,
                     const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::convolution_forward,
                                 mkldnn::convolution_backward_data,
                                 mkldnn::convolution_backward_weights>(
            dev_ctx, mkldnn_engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(input->dims()),
                                unique_name)) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(
          input->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The input tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, input->layout()));
      PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Wrong format set for Input tensor"));

      PADDLE_ENFORCE_EQ(
          filter->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The Filter tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, filter->layout()));
      PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Wrong format set for Filter tensor"));

      PADDLE_ENFORCE_GE(
          input->dims().size(), 4,
          platform::errors::InvalidArgument(
              "Input must be with 4 or 5 dimensions, i.e. NCHW or "
              "NCDHW, but got dimension = %d .",
              input->dims().size()));
      PADDLE_ENFORCE_LE(
          input->dims().size(), 5,
          platform::errors::InvalidArgument(
              "Input must be with 4 or 5 dimensions, i.e. NCHW or "
              "NCDHW, but got dimension = %d .",
              input->dims().size()));

      PADDLE_ENFORCE_GE(
          filter->dims().size(), 4,
          platform::errors::InvalidArgument(
              "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
              "OIDHW, but got dimension = %d .",
              filter->dims().size()));
      PADDLE_ENFORCE_LE(
          filter->dims().size(), 5,
          platform::errors::InvalidArgument(
              "Filter must be with 4 or 5 dimensions, i.e. OIHW or "
              "OIDHW, but got dimension = %d .",
              filter->dims().size()));

      if (bias) {
        PADDLE_ENFORCE_EQ(
            bias->layout(), DataLayout::kMKLDNN,
            platform::errors::InvalidArgument(
                "The Bias tensor's layout should be %d, but got %d.",
                DataLayout::kMKLDNN, bias->layout()));
        PADDLE_ENFORCE_NE(bias->format(), MKLDNNMemoryFormat::undef,
                          platform::errors::InvalidArgument(
                              "Got wrong format for Bias tensor."));

        PADDLE_ENFORCE_EQ(bias->dims().size(), 1,
                          platform::errors::InvalidArgument(
                              "Bias must only have 1 dimension, "
                              "i.e. X, but got dimension = %d .",
                              bias->dims().size()));
      }

      const std::string fuse_activation =
          ctx.Attr<std::string>("fuse_activation");
      const float fuse_alpha = ctx.Attr<float>("fuse_alpha");
      const float fuse_beta = ctx.Attr<float>("fuse_beta");
      const bool fuse_residual_conn =
          ctx.Attr<bool>("fuse_residual_connection");
      const int groups = ctx.Attr<int>("groups");
      const std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      const auto input_dims = input->dims();
      const auto data_dims =
          framework::slice_ddim(input_dims, 2, input_dims.size());
      const auto filter_dims = filter->dims();
      const auto filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());

      const auto ksize = framework::vectorize(filter_data_dims);
      const bool is_test = ctx.Attr<bool>("is_test");

      auto strides_temp = ctx.Attr<std::vector<int>>("strides");
      std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

      auto paddings_temp = ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

      auto dilations_temp = ctx.Attr<std::vector<int>>("dilations");
      std::vector<int64_t> dilations(begin(dilations_temp),
                                     end(dilations_temp));

      UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                               data_dims, strides, ksize);

      std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                     [](int64_t i) { return i - 1; });

      const auto src_tz = paddle::framework::vectorize(input->dims());

      auto weights_tz = paddle::framework::vectorize(filter->dims());
      platform::GetGroupConvWeightsTz(weights_tz, groups);

      const auto dst_tz = paddle::framework::vectorize(output->dims());

      const mkldnn::memory::dims stride_dims = strides;
      const auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);
      const mkldnn::memory::dims dilations_dims = dilations;

      /* create memory descriptor for convolution without specified format
       * ('any') which lets a primitive (convolution in this case) choose
       * the memory format preferred for best performance
       */
      auto chosen_memory_format = MKLDNNMemoryFormat::any;

      auto data_type = mkldnn::memory::data_type::f32;
      if (ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16" ||
          std::is_same<T_out, platform::bfloat16>::value)
        data_type = mkldnn::memory::data_type::bf16;

      const auto src_md =
          platform::MKLDNNMemDesc(src_tz, data_type, chosen_memory_format);
      const auto weights_md = platform::MKLDNNMemDesc(weights_tz, data_type,
                                                      MKLDNNMemoryFormat::any);
      const auto dst_md = platform::MKLDNNMemDesc(
          dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);

      const auto fwd_prop_kind = is_test ? mkldnn::prop_kind::forward_inference
                                         : mkldnn::prop_kind::forward_training;

      const mkldnn::primitive_attr conv_attr = CreatePostOps(
          fuse_activation, fuse_alpha, fuse_beta, fuse_residual_conn);

      if (bias) {
        auto bias_tz = framework::vectorize(bias->dims());
        auto bias_md =
            platform::MKLDNNMemDesc(bias_tz, data_type, MKLDNNMemoryFormat::x);

        this->AcquireForwardPrimitiveDescriptor(
            conv_attr, fwd_prop_kind, dnnl::algorithm::convolution_direct,
            src_md, weights_md, bias_md, dst_md, stride_dims, dilations_dims,
            mkldnn_paddings[0], mkldnn_paddings[1]);
      } else {
        this->AcquireForwardPrimitiveDescriptor(
            conv_attr, fwd_prop_kind, dnnl::algorithm::convolution_direct,
            src_md, weights_md, dst_md, stride_dims, dilations_dims,
            mkldnn_paddings[0], mkldnn_paddings[1]);
      }
    }
  }

  ConvMKLDNNHandlerT(const framework::ExecutionContext& ctx,
                     const platform::MKLDNNDeviceContext& dev_ctx,
                     platform::Place cpu_place, const Tensor* in,
                     const Tensor* filter, const Tensor* bias,
                     const Tensor* out_grad, Tensor* filter_grad,
                     Tensor* in_x_grad, const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::convolution_forward,
                                 mkldnn::convolution_backward_data,
                                 mkldnn::convolution_backward_weights>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(in->dims()),
                                unique_name)) {
    if (!this->isBwdCached()) {
      PADDLE_ENFORCE_EQ(
          in->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The input tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, in->layout()));
      PADDLE_ENFORCE_NE(in->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong format for Input tensor."));

      PADDLE_ENFORCE_EQ(
          filter->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The filter tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, filter->layout()));
      PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong format for Filter tensor."));

      PADDLE_ENFORCE_EQ(
          out_grad->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The output_grad tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, out_grad->layout()));
      PADDLE_ENFORCE_NE(out_grad->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Wrong format set for output_grad tensor"));

      PADDLE_ENFORCE_EQ(
          ctx.Attr<bool>("is_test"), false,
          platform::errors::InvalidArgument(
              "is_test attribute should be set to False in training phase."));

      std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
      std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

      std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

      std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
      std::vector<int64_t> dilations(begin(dilations_temp),
                                     end(dilations_temp));

      std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      int groups = ctx.Attr<int>("groups");

      auto input_dims = in->dims();
      auto data_dims = framework::slice_ddim(input_dims, 2, input_dims.size());
      auto filter_dims = filter->dims();
      auto filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());

      auto ksize = framework::vectorize(filter_data_dims);

      UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                               data_dims, strides, ksize);

      auto src_tz = framework::vectorize(in->dims());
      auto weights_tz = framework::vectorize(filter->dims());

      int g = std::max(groups, 1);
      platform::GetGroupConvWeightsTz(weights_tz, g);
      auto dst_tz = paddle::framework::vectorize(out_grad->dims());

      /* create memory descriptor for conv backward without specified format
       * ('any') which lets a primitive (conv backward in this case) choose
       * the memory format preferred for best performance
       */
      const auto chosen_memory_format = MKLDNNMemoryFormat::any;
      const auto weights_format = MKLDNNMemoryFormat::any;

      auto src_md = platform::MKLDNNMemDesc(
          src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
      const auto dst_md = platform::MKLDNNMemDesc(
          dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);
      auto diff_src_md = platform::MKLDNNMemDesc(
          src_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);
      auto weights_md = platform::MKLDNNMemDesc(
          weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
      auto diff_weights_md = platform::MKLDNNMemDesc(
          weights_tz, platform::MKLDNNGetDataType<T>(), weights_format);
      auto diff_dst_md = platform::MKLDNNMemDesc(
          dst_tz, platform::MKLDNNGetDataType<T>(), chosen_memory_format);

      auto mkldnn_paddings = platform::ToMkldnnPadding(paddings);
      std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                     [](int64_t i) { return i - 1; });
      const mkldnn::memory::dims dilations_dims = dilations;

      const mkldnn::memory::dims stride_dims = strides;
      // Recreating FWD PD. For training there are no post ops in convolution
      mkldnn::primitive_attr conv_attr;
      if (bias) {
        auto bias_tz = framework::vectorize(bias->dims());
        auto bias_md = platform::MKLDNNMemDesc(
            bias_tz, mkldnn::memory::data_type::f32, MKLDNNMemoryFormat::x);

        this->AcquireForwardPrimitiveDescriptor(
            conv_attr, mkldnn::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, src_md, weights_md, bias_md,
            dst_md, stride_dims, dilations_dims, mkldnn_paddings[0],
            mkldnn_paddings[1]);
      } else {
        this->AcquireForwardPrimitiveDescriptor(
            conv_attr, mkldnn::prop_kind::forward_training,
            dnnl::algorithm::convolution_direct, src_md, weights_md, dst_md,
            stride_dims, dilations_dims, mkldnn_paddings[0],
            mkldnn_paddings[1]);
      }

      this->AcquireBackwardPrimitiveDescriptor(
          mkldnn::algorithm::convolution_direct, diff_src_md, weights_md,
          diff_dst_md, strides, dilations_dims, mkldnn_paddings[0],
          mkldnn_paddings[1]);

      this->AcquireBackwardWeightsPrimitiveDescriptor(
          mkldnn::algorithm::convolution_direct, src_md, diff_weights_md,
          diff_dst_md, strides, dilations_dims, mkldnn_paddings[0],
          mkldnn_paddings[1]);
    }
  }

  mkldnn::primitive_attr CreatePostOps(
      std::string fuse_activation, float fuse_alpha, float fuse_beta,
      bool fuse_residual_conn, const std::vector<float> output_shift_scale = {},
      float sum_scale = 1.0f) {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    if (output_shift_scale.size() > 0) {
      int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
      conv_attr.set_output_scales(mask, output_shift_scale);
    }

    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_swish,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "hard_swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(
          scale, mkldnn::algorithm::eltwise_hardswish, fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<mkldnn::memory>
  AcquireWeightsMemoryWithReorderFromDataPrimitive(
      const framework::Tensor* filter, const int groups, const bool is_conv3d) {
    const K* filter_data = filter->data<K>();
    auto weights_tz = framework::vectorize(filter->dims());
    platform::GetGroupConvWeightsTz(weights_tz, groups);

    auto user_src_md = platform::MKLDNNMemDesc(
        weights_tz, platform::MKLDNNGetDataType<K>(),
        GetWeightsFormat(filter->format(), groups, is_conv3d));

    return this->AcquireMemoryWithReorder(
        user_src_md, this->bwd_pd_->weights_desc(),
        to_void_cast<K>(filter_data), "@weights_mem_d_p", false);
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryWithReorder(
      const framework::Tensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(
        input, "@src_mem_p_user", "@src_mem_p_target", "@src_mem_p",
        this->fwd_pd_->src_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireSrcMemoryWithReorderFromWeightsPrimitive(
      const framework::Tensor* input) {
    return this->AcquireMemoryWithReorderPrimitive(
        input, "@src_mem_w_p_user", "@src_mem_w_p_target", "@src_mem_w_p",
        this->bwd_w_pd_->src_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
      const framework::Tensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad, "@diff_dst_mem_w_p_user", "@diff_dst_mem_w_p_target",
        "@diff_dst_mem_w_p", this->bwd_w_pd_->diff_dst_desc());
  }

  std::shared_ptr<mkldnn::memory>
  AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
      const framework::Tensor* out_grad) {
    return this->AcquireMemoryWithReorderPrimitive(
        out_grad, "@diff_dst_mem_p_user", "@diff_dst_mem_p_target",
        "@diff_dst_mem_p", this->bwd_pd_->diff_dst_desc());
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryWithReorderPrimitive(
      const framework::Tensor* in_mem, const char* key_mem_user,
      const char* key_mem_target, const char* key_mem,
      const mkldnn::memory::desc& mem_md) {
    const T* in_mem_data = in_mem->data<T>();
    const std::string user_key_suffix{key_mem_user};
    auto user_mem_p = this->AcquireMemory(user_key_suffix);

    if (!user_mem_p) {
      auto user_mem_md = platform::MKLDNNMemDesc(
          framework::vectorize(in_mem->dims()),
          platform::MKLDNNGetDataType<T>(), in_mem->format());
      return this->AcquireMemoryWithReorder(
          user_mem_md, mem_md, to_void_cast<T>(in_mem_data), key_mem);
    } else {
      const std::string target_key_suffix{key_mem_target};
      const auto target_mem_p = this->AcquireMemory(target_key_suffix);
      user_mem_p->set_data_handle(to_void_cast<T>(in_mem_data));
      if (user_mem_p != target_mem_p) {
        this->AcquireReorder(user_mem_p, target_mem_p, key_mem);
      }
      return target_mem_p;
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryWithReorder(
      const framework::Tensor* filter, const int groups, const bool is_conv3d,
      const bool is_test) {
    // This is workaround to make execution faster, delete
    // if statement after including md inside Tensor
    auto weights_mem_p = this->AcquireMemory("@weights_mem_p_target");
    if (is_test && weights_mem_p) {
      return weights_mem_p;
    } else {
      const K* filter_data = filter->data<K>();
      auto weights_tz = framework::vectorize(filter->dims());
      platform::GetGroupConvWeightsTz(weights_tz, groups);

      auto user_src_md = platform::MKLDNNMemDesc(
          weights_tz, platform::MKLDNNGetDataType<K>(),
          GetWeightsFormat(filter->format(), groups, is_conv3d));

      return this->AcquireMemoryWithReorder(
          user_src_md, this->fwd_pd_->weights_desc(),
          to_void_cast<K>(filter_data), "@weights_mem_p", is_test);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryWithReorder(
      const framework::Tensor* bias, const bool is_test) {
    auto bias_mem_p = this->AcquireMemory("@bias_mem_p_target");
    if (is_test && bias_mem_p) {
      return bias_mem_p;
    } else {
      const K* bias_data = bias->data<K>();
      auto user_bias_md = platform::MKLDNNMemDesc(
          framework::vectorize(bias->dims()), platform::MKLDNNGetDataType<K>(),
          MKLDNNMemoryFormat::x);

      return this->AcquireMemoryWithReorder(
          user_bias_md, this->fwd_pd_->bias_desc(), to_void_cast<K>(bias_data),
          "@bias_mem_p", is_test);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireResidualMemory(
      const framework::Tensor* residual_param) {
    void* residual_data =
        residual_param->type() == framework::DataTypeTrait<T_out>::DataType()
            ? to_void_cast<T_out>(residual_param->data<T_out>())
            : to_void_cast<T>(residual_param->data<T>());
    auto residual_mem_p = this->AcquireMemory("@user_residual_data_mem_p");
    if (residual_mem_p) {
      residual_mem_p->set_data_handle(residual_data);
      return residual_mem_p;
    } else {
      auto user_residual_md = platform::MKLDNNMemDesc(
          framework::vectorize(residual_param->dims()),
          framework::ToMKLDNNDataType(residual_param->type()),
          residual_param->format());

      return this->AcquireMemoryFromPrimitive(user_residual_md, residual_data,
                                              "@user_residual_data_mem_p");
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryWithResidual(
      framework::Tensor* output, const framework::Tensor* residual_param) {
    std::shared_ptr<dnnl::memory> dst_memory_p;
    if (residual_param->format() !=
        platform::GetMKLDNNFormat(this->fwd_pd_->dst_desc())) {
      auto residual_memory_p = this->AcquireResidualMemory(residual_param);
      dst_memory_p = this->template AcquireDstMemory<T_out>(output);
      this->AcquireReorder(residual_memory_p, dst_memory_p, "@residual_dst");
    } else {
      // Changing ShareDataWith to TensorCopy results in performance drop
      // on ResNet architectures
      // (https://github.com/PaddlePaddle/Paddle/issues/22964)
      output->ShareDataWith(*residual_param);
      dst_memory_p = this->template AcquireDstMemory<T_out>(output);
    }
    return dst_memory_p;
  }
};

template <typename T, typename K>
class ConvMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Conv must use CPUPlace"));
    bool is_INT8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    bool is_BFLOAT16 = ctx.Attr<std::string>("mkldnn_data_type") == "bfloat16";
    auto residual_param = ctx.Input<Tensor>("ResidualData");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    std::string fuse_activation = ctx.Attr<std::string>("fuse_activation");
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    auto dst_dt =
        GetDstType(is_INT8, is_BFLOAT16, force_fp32_output, fuse_activation,
                   fuse_residual_conn, residual_param);
    if (!is_INT8) {
      if (dst_dt == mkldnn::memory::data_type::f32) {
        ComputeFP32<float>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::bf16) {
        ComputeFP32<platform::bfloat16>(ctx);
      }
    } else {
      if (dst_dt == mkldnn::memory::data_type::f32) {
        ComputeINT8<float>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::u8) {
        ComputeINT8<uint8_t>(ctx);
      } else if (dst_dt == mkldnn::memory::data_type::s8) {
        ComputeINT8<int8_t>(ctx);
      }
    }
  }

  template <typename T_out>
  void ComputeFP32(const paddle::framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const bool is_test = ctx.Attr<bool>("is_test");
    const bool is_conv3d = ctx.Attr<std::vector<int>>("strides").size() == 3U;
    const bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");

    const auto* input = ctx.Input<Tensor>("Input");
    const auto* filter = ctx.Input<Tensor>("Filter");
    const auto* bias =
        ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    auto* output = ctx.Output<Tensor>("Output");

    ConvMKLDNNHandlerT<T, K, T_out> handler(
        ctx, dev_ctx, mkldnn_engine, ctx.GetPlace(), input, filter, bias,
        output, ctx.InputName("Input") + ctx.InputName("Filter"));

    auto src_memory_p = handler.AcquireSrcMemoryWithReorder(input);

    auto weights_memory_p = handler.AcquireWeightsMemoryWithReorder(
        filter, ctx.Attr<int>("groups"), is_conv3d, is_test);

    std::shared_ptr<dnnl::memory> dst_memory_p;
    if (fuse_residual_conn) {
      auto* residual_param = ctx.Input<Tensor>("ResidualData");
      dst_memory_p =
          handler.AcquireDstMemoryWithResidual(output, residual_param);
    } else {
      dst_memory_p = handler.template AcquireDstMemory<T_out>(output);
    }

    auto conv_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, dnnl::memory> args = {
        {MKLDNN_ARG_SRC, *src_memory_p},
        {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
        {MKLDNN_ARG_DST, *dst_memory_p}};

    if (bias) {
      auto bias_memory_p = handler.AcquireBiasMemoryWithReorder(bias, is_test);
      args.insert({MKLDNN_ARG_BIAS, *bias_memory_p});
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    conv_p->execute(astream, args);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }

  template <typename T_out>
  void ComputeINT8(const paddle::framework::ExecutionContext& ctx) const {
    const bool is_test = ctx.Attr<bool>("is_test");

    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* input = ctx.Input<Tensor>("Input");
    auto* output = ctx.Output<Tensor>("Output");

    PADDLE_ENFORCE_EQ(input->layout(), DataLayout::kMKLDNN,
                      platform::errors::InvalidArgument(
                          "The input tensor's layout should be %d, but got %d.",
                          DataLayout::kMKLDNN, input->layout()));
    PADDLE_ENFORCE_NE(input->format(), MKLDNNMemoryFormat::undef,
                      platform::errors::InvalidArgument(
                          "Got wrong format for Input tensor."));

    PADDLE_ENFORCE_GE(input->dims().size(), 4,
                      platform::errors::InvalidArgument(
                          "Input must be with 4 or 5 dimensions, i.e. NCHW or "
                          "NCDHW, but got dimension = %d .",
                          input->dims().size()));
    PADDLE_ENFORCE_LE(input->dims().size(), 5,
                      platform::errors::InvalidArgument(
                          "Input must be with 4 or 5 dimensions, i.e. NCHW or "
                          "NCDHW, but got dimension = %d .",
                          input->dims().size()));

    std::string fuse_activation = ctx.Attr<std::string>("fuse_activation");
    bool fuse_residual_conn = ctx.Attr<bool>("fuse_residual_connection");
    bool unsigned_output =
        (fuse_activation == "relu" || fuse_activation == "relu6");

    const T* input_data = input->data<T>();

    auto src_tz = paddle::framework::vectorize(input->dims());

    mkldnn::memory::data_type src_dt =
        paddle::framework::ToMKLDNNDataType(input->type());

    std::string key =
        platform::CreateKey(dev_ctx, src_tz, src_dt,
                            ctx.InputName("Input") + ctx.InputName("Filter"));

    const std::string key_conv_pd = key + "@conv_pd";
    bool need_s8_to_u8 = false;
    std::shared_ptr<mkldnn::convolution_forward> conv_p;
    std::shared_ptr<mkldnn::memory> src_memory_p;
    std::shared_ptr<mkldnn::memory> user_src_memory_p;
    std::shared_ptr<mkldnn::memory> dst_memory_p;
    std::vector<primitive> pipeline;
    std::shared_ptr<mkldnn::convolution_forward::primitive_desc> conv_pd;
    std::shared_ptr<platform::ConvMKLDNNHandler> handler;

    // This is workaround for hacky implementation
    // of conv int8 mkl-dnn. Once conv fp32 and conv int8
    // are merged/unified, this will disappear
    auto key_tid = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    auto prim_key = key_tid + "@conv_p";
    auto dst_key = key_tid + "@dst_mem_p";
    auto src_key = key_tid + "@src_mem_p";
    auto weights_key = key_tid + "@weights_mem_p";
    auto bias_key = key_tid + "@bias_mem_p";
    auto user_src_key = key_tid + "@user_src_mem_p";
    auto user_residual_key = key_tid + "@user_residual_data_mem_p";
    auto src_reorder_key = key_tid + "@src_mem_preorder_p";
    auto residual_reorder_key = key_tid + "@residual_data_mem_preorder_p";

    conv_p = std::static_pointer_cast<mkldnn::convolution_forward>(
        dev_ctx.GetBlob(prim_key));

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (conv_p == nullptr || !is_test) {
      float fuse_alpha = ctx.Attr<float>("fuse_alpha");
      float fuse_beta = ctx.Attr<float>("fuse_beta");
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

      auto* filter = ctx.Input<Tensor>("Filter");

      PADDLE_ENFORCE_EQ(
          filter->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument(
              "The filter tensor's layout should be %d, but got %d.",
              DataLayout::kMKLDNN, filter->layout()));
      PADDLE_ENFORCE_NE(filter->format(), MKLDNNMemoryFormat::undef,
                        platform::errors::InvalidArgument(
                            "Got wrong format for Filter tensor."));

      PADDLE_ENFORCE_GE(filter->dims().size(), 4,
                        platform::errors::InvalidArgument(
                            "Filter must be with 4 or 5 dimensions, i.e. OIHW "
                            "or OIDHW, but got dimensions = %d .",
                            filter->dims().size()));
      PADDLE_ENFORCE_LE(filter->dims().size(), 5,
                        platform::errors::InvalidArgument(
                            "Filter must be with 4 or 5 dimensions, i.e. OIHW "
                            "or OIDHW, but got dimensions = %d .",
                            filter->dims().size()));

      PADDLE_ENFORCE_EQ(
          !fuse_residual_conn || !force_fp32_output, true,
          platform::errors::Unimplemented(
              "residual fusion does not support force output with fp32"));

      auto* bias = ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;

      if (bias) {
        PADDLE_ENFORCE_EQ(
            bias->layout(), DataLayout::kMKLDNN,
            platform::errors::InvalidArgument(
                "The bias tensor's layout should be %d, but got %d.",
                DataLayout::kMKLDNN, bias->layout()));
        PADDLE_ENFORCE_NE(bias->format(), MKLDNNMemoryFormat::undef,
                          platform::errors::InvalidArgument(
                              "Got wrong format for Bias tensor."));

        PADDLE_ENFORCE_EQ(bias->dims().size(), 1,
                          platform::errors::InvalidArgument(
                              "Bias must only have 1 dimension, i.e. X, but "
                              "got dimension = %d .",
                              bias->dims().size()));
      }

      std::vector<int> strides_temp = ctx.Attr<std::vector<int>>("strides");
      std::vector<int64_t> strides(begin(strides_temp), end(strides_temp));

      std::vector<int> paddings_temp = ctx.Attr<std::vector<int>>("paddings");
      std::vector<int64_t> paddings(begin(paddings_temp), end(paddings_temp));

      std::vector<int> dilations_temp = ctx.Attr<std::vector<int>>("dilations");
      std::vector<int64_t> dilations(begin(dilations_temp),
                                     end(dilations_temp));

      std::string padding_algorithm =
          ctx.Attr<std::string>("padding_algorithm");

      bool is_conv3d = strides.size() == 3U;

      PADDLE_ENFORCE_NE(is_conv3d, true,
                        platform::errors::Unimplemented(
                            "int8 does not support conv3d currently"));

      auto input_dims = input->dims();
      auto data_dims = framework::slice_ddim(input_dims, 2, input_dims.size());
      auto filter_dims = filter->dims();
      auto filter_data_dims =
          framework::slice_ddim(filter_dims, 2, filter_dims.size());

      auto ksize = framework::vectorize(filter_data_dims);

      UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                               data_dims, strides, ksize);

      int groups = ctx.Attr<int>("groups");
      auto weights_tz = paddle::framework::vectorize(filter->dims());
      int g = std::max(groups, 1);

      platform::GetGroupConvWeightsTz(weights_tz, g);
      auto dst_tz = paddle::framework::vectorize(output->dims());

      std::transform(dilations.begin(), dilations.end(), dilations.begin(),
                     [](int64_t i) { return i - 1; });

      const K* filter_data = filter->data<K>();
      auto scale_in_data = ctx.Attr<float>("Scale_in");
      auto scale_in_eltwise_data = ctx.Attr<float>("Scale_in_eltwise");
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      auto scale_out_data =
          force_fp32_output ? 1.0f : ctx.Attr<float>("Scale_out");
      float sum_scale =
          fuse_residual_conn ? scale_out_data / scale_in_eltwise_data : 1.0f;

      bool is_multi_channel = scale_weights_data.size() > 1;

      int count = is_multi_channel ? (g > 1 ? (weights_tz)[1] * (weights_tz)[0]
                                            : (weights_tz)[0])
                                   : 1;
      std::vector<float> output_shift_scale(count);
#pragma omp parallel for if (count > 1)
      for (int i = 0; i < count; i++) {
        if (scale_weights_data[i] == 0.0)
          output_shift_scale[i] =
              scale_out_data;  // weights data will contain 0
                               // in some models, then weights
                               // scale couldn't be calculated
        else
          output_shift_scale[i] =
              static_cast<float>(static_cast<double>(scale_out_data) /
                                 (static_cast<double>(scale_in_data) *
                                  static_cast<double>(scale_weights_data[i])));
      }

      auto user_src_md =
          platform::MKLDNNMemDesc({src_tz}, src_dt, input->format());
      auto user_weights_md = platform::MKLDNNMemDesc(
          {weights_tz}, platform::MKLDNNGetDataType<K>(),
          ((g) == 1) ? MKLDNNMemoryFormat::oihw : MKLDNNMemoryFormat::goihw);

      /* create memory descriptor for convolution without specified format
       * ('any') which lets a primitive (convolution in this case) choose
       * the memory format preferred for best performance
       */
      auto chosen_memory_format = MKLDNNMemoryFormat::any;

      std::vector<int64_t> bias_tz;

      auto src_md =
          platform::MKLDNNMemDesc(src_tz, src_dt, chosen_memory_format);
      auto weights_md = platform::MKLDNNMemDesc(
          weights_tz, memory::data_type::s8, chosen_memory_format);
      auto dst_md = platform::MKLDNNMemDesc(
          dst_tz, platform::MKLDNNGetDataType<T_out>(), chosen_memory_format);

      handler.reset(
          new platform::ConvMKLDNNHandler(dev_ctx, mkldnn_engine, key));
      // create a conv primitive descriptor and save it for usage in backward
      auto propagation = is_test ? mkldnn::prop_kind::forward_scoring
                                 : mkldnn::prop_kind::forward_training;

      if (bias) {
        bias_tz = paddle::framework::vectorize(bias->dims());
        auto bias_md = platform::MKLDNNMemDesc(bias_tz, memory::data_type::s32,
                                               MKLDNNMemoryFormat::x);
        conv_pd = handler->AcquireConvolutionPrimitiveDescriptor(
            src_md, weights_md, bias_md, dst_md, strides, dilations, paddings,
            mkldnn_engine, fuse_activation, fuse_alpha, fuse_beta,
            fuse_residual_conn, propagation, output_shift_scale, sum_scale);
      } else {
        conv_pd = handler->AcquireConvolutionPrimitiveDescriptor(
            src_md, weights_md, boost::none, dst_md, strides, dilations,
            paddings, mkldnn_engine, fuse_activation, fuse_alpha, fuse_beta,
            fuse_residual_conn, propagation, output_shift_scale, sum_scale);
      }

      // create mkldnn memory from input tensors (data/weights)
      user_src_memory_p =
          handler->AcquireSrcMemory(user_src_md, to_void_cast<T>(input_data));
      auto user_weights_memory_p = handler->AcquireWeightsMemory(
          user_weights_md, to_void_cast<K>(filter_data));

      // create reorder primitive if the input format is not the preferred one
      src_memory_p =
          handler->AcquireSrcMemoryFromPrimitive(user_src_memory_p, pipeline);

      std::shared_ptr<mkldnn::memory> weights_memory_p;
      int mask_reorder =
          is_multi_channel ? ((g != 1) ? (1 << 1) + (1 << 0) : 1 << 0) : 0;
      weights_memory_p = handler->AcquireWeightsMemoryFromPrimitive(
          user_weights_memory_p, pipeline, is_test, true, scale_weights_data,
          mask_reorder);

      if (fuse_residual_conn) {
        auto residual_param = ctx.Input<Tensor>("ResidualData");
        PADDLE_ENFORCE_EQ(
            output->dims(), residual_param->dims(),
            platform::errors::InvalidArgument(
                "Output and elementwise parameter need to have the "
                "same dimension sizes, but got output's dimension = %d"
                " and residual param's dimension =%d .",
                output->dims().size(), residual_param->dims().size()));
        auto residual_dt =
            paddle::framework::ToMKLDNNDataType(residual_param->type());
        if (residual_param->format() != handler->GetDstFormat()) {
          auto residual_data_tz =
              paddle::framework::vectorize(residual_param->dims());
          auto user_residual_md = platform::MKLDNNMemDesc(
              residual_data_tz, residual_dt, residual_param->format());
          dst_memory_p = platform::SetDstMemory<T_out>(
              ctx, output, residual_param, user_residual_md, handler,
              &pipeline);
        } else {
          output->ShareDataWith(*residual_param);
          dst_memory_p = platform::SetDstMemory<T_out>(ctx, output, handler);
        }
        need_s8_to_u8 =
            (platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8) &&
            unsigned_output;
      } else {
        dst_memory_p = platform::SetDstMemory<T_out>(ctx, output, handler);
      }

      // create convolution op primitive
      auto scale_bias_key = key + "@scale_bias";
      conv_p = handler->AcquireConvolution();
      if (bias) {
        const K* bias_data = bias->data<K>();
        auto user_bias_md = platform::MKLDNNMemDesc(
            {bias_tz}, platform::MKLDNNGetDataType<K>(), MKLDNNMemoryFormat::x);
        auto user_bias_memory_p = handler->AcquireBiasMemory(
            user_bias_md, to_void_cast<K>(bias_data));
        std::shared_ptr<mkldnn::memory> bias_memory_p;
        int mask_reorder = is_multi_channel ? 1 << 0 : 1;
        int count =
            is_multi_channel
                ? (g > 1 ? (weights_tz)[1] * (weights_tz)[0] : (weights_tz)[0])
                : 1;
        std::vector<float> scale_bias_data(count);
#pragma omp parallel for if (count > 1)
        for (int i = 0; i < count; i++) {
          scale_bias_data[i] = scale_in_data * scale_weights_data[i];
        }
        bias_memory_p = handler->AcquireBiasMemoryFromPrimitive(
            user_bias_memory_p, pipeline, is_test, true, scale_bias_data,
            mask_reorder);
        conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                  {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                  {MKLDNN_ARG_BIAS, *bias_memory_p},
                                  {MKLDNN_ARG_DST, *dst_memory_p}});
      } else {
        conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                  {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                  {MKLDNN_ARG_DST, *dst_memory_p}});
      }
    } else {
      auto src_memory_reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx.GetBlob(src_reorder_key));
      src_memory_p =
          std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(src_key));
      if (src_memory_reorder_p) {
        user_src_memory_p = std::static_pointer_cast<mkldnn::memory>(
            dev_ctx.GetBlob(user_src_key));
        user_src_memory_p->set_data_handle(to_void_cast<T>(input_data));
        {
          platform::RecordEvent record_reorder("int_reorder",
                                               platform::EventRole::kUniqueOp);
          src_memory_reorder_p->execute(astream, *user_src_memory_p,
                                        *src_memory_p);
          astream.wait();
        }
      } else if (src_memory_p) {
        src_memory_p->set_data_handle(to_void_cast<T>(input_data));
      }
      auto weights_memory_p = std::static_pointer_cast<mkldnn::memory>(
          dev_ctx.GetBlob(weights_key));
      dst_memory_p =
          std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(dst_key));
      conv_pd =
          std::static_pointer_cast<mkldnn::convolution_forward::primitive_desc>(
              dev_ctx.GetBlob(key_conv_pd));
      if (conv_pd) {
        handler.reset(new platform::ConvMKLDNNHandler(conv_pd, dev_ctx,
                                                      mkldnn_engine, key));
      }

      if (fuse_residual_conn) {
        auto residual_param = ctx.Input<Tensor>("ResidualData");
        output->ShareDataWith(*residual_param);
        need_s8_to_u8 =
            (platform::MKLDNNGetDataType<T_out>() == memory::data_type::s8) &&
            unsigned_output;
      }
      platform::SetDstMemoryHandler<T_out>(ctx, output, handler, dst_memory_p);

      auto residual_reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx.GetBlob(residual_reorder_key));
      if (residual_reorder_p) {
        auto user_residual_data_p = std::static_pointer_cast<mkldnn::memory>(
            dev_ctx.GetBlob(user_residual_key));
        {
          platform::RecordEvent record_reorder("int_reorder",
                                               platform::EventRole::kUniqueOp);
          residual_reorder_p->execute(astream, *user_residual_data_p,
                                      *dst_memory_p);
          astream.wait();
        }
      }

      auto bias_memory_p =
          std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(bias_key));

      if (bias_memory_p) {
        conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                  {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                  {MKLDNN_ARG_BIAS, *bias_memory_p},
                                  {MKLDNN_ARG_DST, *dst_memory_p}});
      } else {
        conv_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                                  {MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                  {MKLDNN_ARG_DST, *dst_memory_p}});
      }
    }
    astream.wait();
    if (need_s8_to_u8) {
      output->mutable_data<uint8_t>(ctx.GetPlace());
    }
    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(GetMKLDNNFormat(*dst_memory_p));
  }
};

template <typename T, typename K>
class ConvMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL ConvGrad must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    const Tensor* input = ctx.Input<Tensor>("Input");
    const Tensor* filter = ctx.Input<Tensor>("Filter");
    const Tensor* bias =
        ctx.HasInput("Bias") ? ctx.Input<Tensor>("Bias") : nullptr;
    const Tensor* output_grad =
        ctx.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad = ctx.Output<Tensor>(framework::GradVarName("Filter"));

    if (!input_grad && !filter_grad) return;

    // TODO(jczaja): Are all tensors really needed?
    ConvMKLDNNHandlerT<T, K, T> handler(
        ctx, dev_ctx, ctx.GetPlace(), input, filter, bias, output_grad,
        filter_grad, input_grad,
        ctx.InputName("Input") + ctx.InputName("Filter"));

    // create mkldnn memory from input tensors (data/weights)
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (filter_grad) {
      auto src_memory_p =
          handler.AcquireSrcMemoryWithReorderFromWeightsPrimitive(input);
      auto diff_dst_memory_p =
          handler.AcquireDiffDstMemoryWithReorderFromWeightsPrimitive(
              output_grad);

      // For convoluition with groups write filter grad into
      // oneDNN buffer and then we reorder it into filter_grad tensor
      int g = std::max(ctx.Attr<int>("groups"), 1);
      auto diff_weights_memory_p =
          g > 1 ? handler.AcquireDiffWeightsMemory()
                : handler.AcquireDiffWeightsMemory(filter_grad);

      auto conv_bwd_weights_p = handler.AcquireBackwardWeightsPrimitive();

      // TODO(grygielski) why no bias_diff?
      conv_bwd_weights_p->execute(
          astream, {{MKLDNN_ARG_SRC, *src_memory_p},
                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                    {MKLDNN_ARG_DIFF_WEIGHTS, *diff_weights_memory_p}});
      astream.wait();

      filter_grad->set_layout(DataLayout::kMKLDNN);
      // in OneDNN groups in convolution are treated as separate dimension
      // which is not the case in paddlepaddle
      auto filter_fmt = GetMKLDNNFormat(*diff_weights_memory_p);

      // For convolution with groups convert from blocked to NCHW
      // otherwise there will be problems in next operators working on this data
      if (g > 1) {
        memory::data_type in_type = framework::ToMKLDNNDataType(filter->type());
        // for 3d conv with groups (six dimensional data reorder to goidhw)
        // for 2d conv with groups (five dimensional data reorder to goihw)
        // auto weights_tz = paddle::framework::vectorize(filter->dims());

        auto weights_tz = diff_weights_memory_p->get_desc().dims();
        mkldnn::memory::format_tag out_format =
            weights_tz.size() == 6 ? mkldnn::memory::format_tag::goidhw
                                   : mkldnn::memory::format_tag::goihw;
        std::string key = platform::CreateKey(dev_ctx, weights_tz, filter_fmt,
                                              out_format, in_type);
        key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

        platform::ReorderMKLDNNHandler handler(
            weights_tz, filter->type(), in_type, dev_ctx, mkldnn_engine, key);
        auto reorder_dst_memory_p =
            handler.AcquireDstMemory(filter_grad, out_format, ctx.GetPlace());

        auto reorder_p =
            handler.AcquireReorder(reorder_dst_memory_p, diff_weights_memory_p);

        {
          platform::RecordEvent record_reorder("int_reorder",
                                               platform::EventRole::kUniqueOp);
          reorder_p->execute(astream, *diff_weights_memory_p,
                             *reorder_dst_memory_p);
          astream.wait();
        }

        // So here we have a data in goihw , which can be interpreted as OIHW
        // (OIDHW for conv3d)
        // because filter_grad shape is set for OIHW (OIDHW for conv3d)
        mkldnn::memory::format_tag target_format =
            weights_tz.size() == 6 ? mkldnn::memory::format_tag::oidhw
                                   : mkldnn::memory::format_tag::oihw;
        filter_grad->set_format(target_format);
      } else {
        filter_grad->set_format(filter_fmt);
      }
    }
    if (input_grad) {
      auto weights_memory_p =
          handler.AcquireWeightsMemoryWithReorderFromDataPrimitive(
              filter, ctx.Attr<int>("groups"),
              ctx.Attr<std::vector<int>>("strides").size() == 3U);

      auto diff_dst_memory_p =
          handler.AcquireDiffDstMemoryWithReorderMemoryFromDataPrimitive(
              output_grad);
      auto diff_src_memory_p = handler.AcquireDiffSrcMemory(input_grad);

      auto conv_bwd_data_p = handler.AcquireBackwardPrimitive();

      conv_bwd_data_p->execute(astream,
                               {{MKLDNN_ARG_WEIGHTS, *weights_memory_p},
                                {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                                {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
      astream.wait();

      input_grad->set_layout(DataLayout::kMKLDNN);
      input_grad->set_format(GetMKLDNNFormat(*diff_src_memory_p));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    conv2d, MKLDNN, ::paddle::platform::CPUPlace, BF16, ops::kConvMKLDNNFP32,
    ops::ConvMKLDNNOpKernel<paddle::platform::bfloat16, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, U8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d, MKLDNN,
                                    ::paddle::platform::CPUPlace, S8,
                                    ops::kConvMKLDNNINT8,
                                    ops::ConvMKLDNNOpKernel<int8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv2d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(conv3d_grad, MKLDNN,
                                    ::paddle::platform::CPUPlace, FP32,
                                    ops::kConvMKLDNNFP32,
                                    ops::ConvMKLDNNGradOpKernel<float, float>);
