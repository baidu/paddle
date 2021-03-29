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

#include <memory>

#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Tensor;
}  // namespace framework
namespace platform {
class MKLDNNDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::LoDTensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

template <typename T_in, typename T_w, typename T_out>
class FCPrimitiveFactory {
 public:
  explicit FCPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  void ExecuteFcPrimitive(const LoDTensor* input, const Tensor* weights,
                          const Tensor* bias, LoDTensor* output,
                          const MKLDNNDeviceContext& dev_ctx,
                          const ExecutionContext& ctx) {
    RecomputeOutputDims(ctx, input, weights, output);
    // If primitive has already been created and cached, don't create new one,
    // but update input and output data pointers and return it.
    if (fc_) {
      UpdateDataPointers(ctx, output, input);
      this->Execute();
      return;
    }  // Otherwise, create a new one.

    auto in_col_dims = ctx.Attr<int>("in_num_col_dims");
    PADDLE_ENFORCE_LE(
        in_col_dims, 2,
        platform::errors::Unimplemented(
            "DNNL FC doesn't support in_num_col_dims parameter to "
            "be higher than "
            "2."));
    if (in_col_dims == 2) {
      PADDLE_ENFORCE_EQ(
          input->dims().size(), 3,
          platform::errors::Unimplemented(
              "DNNL FC only supports in_num_col_dims equal to 2 when "
              "3 dim input is provided."));
      PADDLE_ENFORCE_EQ(
          input->format(), MKLDNNMemoryFormat::ncw,
          platform::errors::Unimplemented(
              "DNNL FC only supports in_num_col_dims equal to 2 when "
              "input format is equal to ncw."));
    }

    weights_ = CreateWeightsMemory(weights);

    // Since MKL-DNN has a lot of limitations on what the input/weights/output
    // dimensions should be, to simplify the code, the creation of primitive
    // descriptor has been divided into separate cases, based on the number
    // of input dimensions.
    size_t input_dim_num = input->dims().size();
    boost::optional<mkldnn::inner_product_forward::primitive_desc> fc_prim_desc;
    memory::desc usr_weights_desc = {};
    switch (input_dim_num) {
      case 2:
        fc_prim_desc =
            Create2DFcPrimDescriptor(input, weights, bias, output, ctx);
        usr_weights_desc = Create2DUserWeightsDesc();
        break;
      case 3:
        fc_prim_desc =
            Create3DFcPrimDescriptor(input, weights, bias, output, ctx);
        usr_weights_desc = Create3DUserWeightsDesc(weights);
        break;
      case 4:
        fc_prim_desc =
            Create4DFcPrimDescriptor(input, weights, bias, output, ctx);
        usr_weights_desc = Create4DUserWeightsDesc(input, weights);
        break;
      default:
        PADDLE_THROW(platform::errors::Unimplemented(
            "DNNL FC doesn't support input dims different than 2, 3, 4."));
        break;
    }
    input_ = CreateMemory<T_in>(fc_prim_desc->src_desc(), input);
    // Update weights format inside of its memory
    weights_ = Reorder(usr_weights_desc, usr_weights_desc,
                       weights_->get_data_handle());

    // Quantize weights and reorder to format chosen by FC primitive descriptor.
    QuantizeWeights(ctx, fc_prim_desc->weights_desc());

    bias_ = CreateMemoryToBeCached<float>(fc_prim_desc->bias_desc(), bias);
    // If int8 is desired, quantize bias into 32-bit signed int
    QuantizeBias(*fc_prim_desc, ctx);

    // Store weights and bias in the mkldnn cache
    CacheWeightsAndBias(dev_ctx, ctx);

    // Based on format determined by inner_product, create output in desired
    // memory format
    output_ = CreateDstMemory(*fc_prim_desc, ctx, output);

    // Return MKL-DNN primitive ready to be fed into pipeline and executed
    fc_ = inner_product_forward(*fc_prim_desc);
    this->Execute();
  }

  void Execute() {
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (bias_) {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_BIAS, *bias_},
                             {MKLDNN_ARG_DST, *output_}});
    } else {
      fc_->execute(astream, {{MKLDNN_ARG_SRC, *input_},
                             {MKLDNN_ARG_WEIGHTS, *weights_},
                             {MKLDNN_ARG_DST, *output_}});
    }
    astream.wait();
  }

 private:
  // DNNL always returns 2-dimensional data block as a result of computing
  // inner product. Hence the format 'nc' is always set for its output
  // primitive. Therefore, function SetOutputFormat is needed to choose
  // an appropriate format based on the number of input dimensions and
  // format of an input tensor.
  void SetOutputFormat(MKLDNNMemoryFormat in_format, Tensor* out) {
    int dim_num = out->dims().size();
    // In case of 2 dims, we set the only possible format, nc
    if (dim_num == 2) {
      out->set_format(MKLDNNMemoryFormat::nc);
      // In case of 3 dims, we generate a format that is based on number
      // of output dims and the layout of input format (nchw or nhwc).
    } else if (dim_num == 3) {
      if (in_format == MKLDNNMemoryFormat::nwc ||
          in_format == MKLDNNMemoryFormat::nhwc) {
        out->set_format(
            platform::MKLDNNFormatForSize(dim_num, MKLDNNMemoryFormat::nhwc));
      } else {
        out->set_format(
            platform::MKLDNNFormatForSize(dim_num, MKLDNNMemoryFormat::nchw));
      }
      // In any other case we overwrite the output format with the input one.
    } else {
      out->set_format(in_format);
    }
  }

  void UpdateDataPointers(const ExecutionContext& ctx, Tensor* out,
                          const Tensor* in) {
    input_->set_data_handle(to_void_cast(in->data<T_in>()));
    output_->set_data_handle(out->mutable_data<T_out>(ctx.GetPlace()));
    // If the primitive exists, but the output tensor has changed its
    // variable, update its format to what has been determined in first
    // call to CreateFcPrimitive method.
    if (out->format() == MKLDNNMemoryFormat::undef) {
      SetOutputFormat(in->format(), out);
    }
  }

  mkldnn::inner_product_forward::primitive_desc Create2DFcPrimDescriptor(
      const LoDTensor* input, const Tensor* weights, const Tensor* bias,
      LoDTensor* output, const ExecutionContext& ctx) {
    auto src_desc = CreateMemDescriptor<T_in>(input, input->format());
    auto weight_dims = Get2DWeightDimsForDNNL(weights);
    platform::MKLDNNMemoryDescriptor weights_desc;
    const auto& dev_ctx = ctx.device_context<platform::MKLDNNDeviceContext>();
    if (dev_ctx.UseInputMemFormat()) {
      weights_desc =
          CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::oi);
    } else {
      weights_desc =
          CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::any);
    }
    auto bias_desc = CreateMemDescriptor<float>(bias, MKLDNNMemoryFormat::x);
    auto dst_desc = CreateMemDescriptor<T_out>(output, MKLDNNMemoryFormat::any);
    const auto attrs = CreatePostOps(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
  }

  std::vector<int64_t> Get2DWeightDimsForDNNL(const Tensor* weights) {
    auto dims = framework::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // swap input dim with output dim
    return dims;
  }

  memory::desc Create2DUserWeightsDesc() { return weights_->get_desc(); }

  mkldnn::inner_product_forward::primitive_desc Create3DFcPrimDescriptor(
      const LoDTensor* input, const Tensor* weights, const Tensor* bias,
      LoDTensor* output, const ExecutionContext& ctx) {
    auto input_dims = framework::vectorize(input->dims());
    std::vector<int64_t> new_input_dims = {input_dims[0] * input_dims[1],
                                           input_dims[2], 1};
    auto src_desc = CreateMemDescriptor<T_in>(new_input_dims, input->format());

    auto weight_dims = Get3DWeightDimsForDNNL(weights);
    platform::MKLDNNMemoryDescriptor weights_desc;
    const auto& dev_ctx = ctx.device_context<platform::MKLDNNDeviceContext>();
    if (dev_ctx.UseInputMemFormat()) {
      weights_desc =
          CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::oiw);
    } else {
      weights_desc =
          CreateMemDescriptor<T_w>(weight_dims, MKLDNNMemoryFormat::any);
    }

    auto bias_desc = CreateMemDescriptor<float>(bias, MKLDNNMemoryFormat::x);

    auto dst_dims = {input_dims[0] * input_dims[1], weight_dims[0]};
    auto dst_desc =
        CreateMemDescriptor<T_out>(dst_dims, MKLDNNMemoryFormat::any);
    const auto attrs = CreatePostOps(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
  }

  std::vector<int64_t> Get3DWeightDimsForDNNL(const Tensor* weights) {
    auto paddle_w_dims = framework::vectorize(weights->dims());
    return {paddle_w_dims[1], paddle_w_dims[0], 1};
  }

  memory::desc Create3DUserWeightsDesc(const Tensor* weights) {
    auto dims = Get3DWeightDimsForDNNL(weights);
    return CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oiw);
  }

  mkldnn::inner_product_forward::primitive_desc Create4DFcPrimDescriptor(
      const LoDTensor* input, const Tensor* weights, const Tensor* bias,
      LoDTensor* output, const ExecutionContext& ctx) {
    auto src_desc = CreateMemDescriptor<T_in>(input, input->format());
    // Since MKL-DNN doesn't support 4D column-major data formats in
    // inner_product primitive, transpose the weights to be in
    // row-major format
    auto dims = Get4DWeightDimsForDNNL(input, weights);
    platform::MKLDNNMemoryDescriptor weights_desc;
    const auto& dev_ctx = ctx.device_context<platform::MKLDNNDeviceContext>();
    if (dev_ctx.UseInputMemFormat()) {
      weights_desc = CreateMemDescriptor<T_w>(dims, MKLDNNMemoryFormat::oihw);
    } else {
      weights_desc = CreateMemDescriptor<T_w>(dims, MKLDNNMemoryFormat::any);
    }
    auto bias_desc = CreateMemDescriptor<float>(bias, MKLDNNMemoryFormat::x);
    auto dst_desc = CreateMemDescriptor<T_out>(output, MKLDNNMemoryFormat::any);
    const auto attrs = CreatePostOps(ctx);
    return CreateFcPrimDesc(src_desc, weights_desc, bias_desc, dst_desc, attrs);
  }

  std::vector<int64_t> Get4DWeightDimsForDNNL(const LoDTensor* input,
                                              const Tensor* weights) {
    auto old_w_dims = framework::vectorize(weights->dims());
    auto old_in_dims = framework::vectorize(input->dims());
    auto dims = {old_w_dims[1], old_in_dims[1], old_in_dims[2], old_in_dims[3]};
    return dims;
  }

  memory::desc Create4DUserWeightsDesc(const LoDTensor* input,
                                       const Tensor* weights) {
    auto dims = Get4DWeightDimsForDNNL(input, weights);
    return CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oihw);
  }

  // Convert data from one data format to another
  std::shared_ptr<mkldnn::memory> Reorder(const memory::desc& src_desc,
                                          const memory::desc& dst_desc,
                                          void* src_data) {
    auto src_mem = memory(src_desc, engine_, src_data);
    auto dst_mem = std::make_shared<memory>(dst_desc, engine_);

    auto reorder = mkldnn::reorder(src_mem, *dst_mem);
    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    {
      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);
      reorder.execute(astream, src_mem, *dst_mem);
      astream.wait();
    }

    return dst_mem;
  }

  // Convert data from one data format to another and rescale it.
  // If the desired data type is (un)signed int8, quantization occurs here.
  std::shared_ptr<mkldnn::memory> ReorderWithScale(
      const std::shared_ptr<memory> src_mem, const memory::desc& dst_md,
      const std::vector<float>& scale_data) {
    auto dst_mem = std::make_shared<mkldnn::memory>(dst_md, engine_);
    mkldnn::primitive_attr attributes;
    // According to MKL-DNN's documentation mask determines along which
    // dimensions should the scale be applied.
    // 0 - Single scale applied to whole tensor
    // 1 - Apply Scale along a slice of each dimension which index is 1.
    //     In case of weights quantization, that dimension is output,
    //     becuase we perform per-output-channel quantization
    int mask = CreateMask(0, scale_data.size() > 1);
    attributes.set_output_scales(mask, scale_data);
    auto reorder = mkldnn::reorder(*src_mem, *dst_mem, attributes);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    {
      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);
      reorder.execute(astream,
                      {{MKLDNN_ARG_FROM, *src_mem}, {MKLDNN_ARG_TO, *dst_mem}});
      astream.wait();
    }

    return dst_mem;
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(
      const std::vector<int64_t>& dims, MKLDNNMemoryFormat format) {
    return platform::MKLDNNMemDesc(dims, platform::MKLDNNGetDataType<T>(),
                                   format);
  }

  template <typename T>
  static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                  MKLDNNMemoryFormat format) {
    auto dims = framework::vectorize(tensor->dims());
    return CreateMemDescriptor<T>(dims, format);
  }

  template <typename T>
  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, platform::to_void_cast<T>(tensor->data<T>()));
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc, void* data) {
    return memory(desc, engine_, data);
  }

  template <typename T>
  std::shared_ptr<mkldnn::memory> CreateMemoryToBeCached(
      const mkldnn::memory::desc& desc, const Tensor* tensor) {
    return CreateMemoryToBeCached(desc,
                                  platform::to_void_cast<T>(tensor->data<T>()));
  }

  std::shared_ptr<mkldnn::memory> CreateMemoryToBeCached(
      const mkldnn::memory::desc& desc, void* data) {
    return std::make_shared<memory>(desc, engine_, data);
  }

  // Create weights memory and transform to default MKL-DNN format
  std::shared_ptr<mkldnn::memory> CreateWeightsMemory(const Tensor* weights) {
    auto dims = framework::vectorize(weights->dims());
    std::swap(dims[0], dims[1]);  // Correct output dimensions
    auto src_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::io);
    auto dst_desc = CreateMemDescriptor<float>(dims, MKLDNNMemoryFormat::oi);
    // Transpose weights through MKL-DNN's reorder from io to oi format.
    return Reorder(src_desc, dst_desc,
                   platform::to_void_cast<float>(weights->data<float>()));
  }

  void CacheWeightsAndBias(const MKLDNNDeviceContext& dev_ctx,
                           const ExecutionContext& ctx) {
    std::string key = platform::CreateKey(dev_ctx);
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    const std::string weights_key = key + ctx.InputName("W");
    const std::string bias_key = key + ctx.InputName("Bias");
    dev_ctx.SetBlob(weights_key, weights_);
    dev_ctx.SetBlob(bias_key, bias_);
  }

  // Compute the bias scales so that its values correspond to the
  // scale of data being an output of weights and input multiplication
  std::vector<float> ComputeBiasScales(const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    const size_t weight_scales_num = scale_weights_data.size();
    std::vector<float> bias_scales(weight_scales_num);

#pragma omp parallel for
    for (size_t i = 0; i < weight_scales_num; i++) {
      if (scale_weights_data[i] == 0.0)
        bias_scales[i] = 1.0f;
      else
        bias_scales[i] = scale_in_data * scale_weights_data[i];
    }

    return bias_scales;
  }

  // Correct output scale, to take into account scaling of input and weights
  // Since the data that comes out of input and weight multiplication is
  // scaled with its own scales, this data needs to be divided by
  // those scales to normalise them back to what their floating-point range
  // was. Then we multiply them by desired output scale we want on the output.
  std::vector<float> ComputeOutputShiftScale(const ExecutionContext& ctx) {
    auto scale_in_data = ctx.Attr<float>("Scale_in");
    auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
    // If the output will be in floats, we don't multiply by scale_out.
    auto scale_out_data = ctx.Attr<bool>("force_fp32_output")
                              ? 1.0f
                              : ctx.Attr<float>("Scale_out");
    const size_t weight_scales_num = scale_weights_data.size();
    std::vector<float> output_shift_scale(weight_scales_num);

#pragma omp parallel for
    for (size_t i = 0; i < weight_scales_num; i++) {
      if (scale_weights_data[i] == 0.0)
        output_shift_scale[i] = scale_out_data;
      else
        output_shift_scale[i] =
            scale_out_data / (scale_in_data * scale_weights_data[i]);
    }

    return output_shift_scale;
  }

  // Computing MKL-DNN's scaling mask which determines along which dimension
  // slice should the scaling be applied. For more data plase refer to:
  // https://intel.github.io/mkl-dnn/group__c__api__attributes.html
  // Section dnnl_status_t DNNL_API dnnl_primitive_attr_set_output_scales
  int CreateMask(int slice_dimension, bool is_multi_channel_quantizied) {
    return is_multi_channel_quantizied ? 1 << slice_dimension : 0;
  }

  void QuantizeWeights(const ExecutionContext& ctx, memory::desc dst) {
    weights_ = ReorderWithScale(weights_, dst,
                                ctx.Attr<std::vector<float>>("Scale_weights"));
  }

  void QuantizeBias(const inner_product_forward::primitive_desc& fc_prim_desc,
                    const ExecutionContext& ctx) {
    auto bias_scales = ComputeBiasScales(ctx);
    bias_ = ReorderWithScale(bias_, fc_prim_desc.bias_desc(), bias_scales);
  }

  // Fuse relu into FC with activation type attribute has been set to 'relu'
  mkldnn::primitive_attr CreatePostOps(const ExecutionContext& ctx) {
    mkldnn::primitive_attr attributes;
    mkldnn::post_ops post_operations;

    auto output_shift_scale = ComputeOutputShiftScale(ctx);
    int mask = CreateMask(1, output_shift_scale.size() > 1);
    attributes.set_output_scales(mask, output_shift_scale);

    if (ctx.Attr<std::string>("activation_type") == "relu") {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 1.0f;  // beta
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_gelu,
                                     alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu_tanh") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, mkldnn::algorithm::eltwise_gelu_tanh, alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "gelu_erf") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_gelu_erf,
                                     alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "tanh") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_tanh,
                                     alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "sigmoid") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_logistic,
                                     alpha, beta);
    } else if (ctx.Attr<std::string>("activation_type") == "hard_swish") {
      constexpr float scale = 1.0f;
      constexpr float alpha = 0.0f;
      constexpr float beta = 0.0f;
      post_operations.append_eltwise(
          scale, mkldnn::algorithm::eltwise_hardswish, alpha, beta);
    }

    attributes.set_post_ops(post_operations);
    return attributes;
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& bias_desc,
      const mkldnn::memory::desc& dst_desc,
      const mkldnn::primitive_attr& attrs) {
    auto fc_desc =
        inner_product_forward::desc(prop_kind::forward_scoring, input_desc,
                                    weights_desc, bias_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, attrs, engine_);
  }

  // Create output memory based on output tensor and inner_product
  // primitive descriptor format chosen for output
  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_desc = fc_prim_desc.dst_desc();
    auto buffer_size = dst_desc.get_size();
    T_out* output_data =
        output->mutable_data<T_out>(ctx.GetPlace(), buffer_size);
    memory dst_mem(dst_desc, engine_, to_void_cast<T_out>(output_data));
    SetOutputFormat(ctx.Input<LoDTensor>("Input")->format(), output);

    return dst_mem;
  }

  void RecomputeOutputDims(const ExecutionContext& ctx, const LoDTensor* input,
                           const Tensor* w, LoDTensor* output) {
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    bool padding_weights = ctx.Attr<bool>("padding_weights");
    PADDLE_ENFORCE_EQ(padding_weights, false,
                      platform::errors::PermissionDenied(
                          "Weight padding in fc can not be used in MKLDNN."));
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims,
                 padding_weights);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  std::shared_ptr<memory> bias_;
  std::shared_ptr<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};

// Attempt to fetch cached primitive factory based on provided parameters
// of input format, weight dimensions and output name.
// If not cached, create a new one.
template <typename T_in, typename T_w, typename T_out>
static std::shared_ptr<FCPrimitiveFactory<T_in, T_w, T_out>>
GetPrimitiveFactory(const MKLDNNDeviceContext& dev_ctx,
                    const std::string& key) {
  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T_in, T_w, T_out>>(
          dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator = std::make_shared<FCPrimitiveFactory<T_in, T_w, T_out>>(
        dev_ctx.GetEngine());
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

// Choose appropriate primitive factory implementation based on inferred
// output type (uint8, int8 or float).
template <typename T_in, typename T_w>
static void ExecuteFc(const ExecutionContext& ctx, const LoDTensor* input,
                      const Tensor* w, const Tensor* bias, LoDTensor* output,
                      bool fuse_relu, bool force_fp32_output) {
  auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  std::string prim_key = platform::CreateKey(
      dev_ctx, input->format(), input->dims()[0],
      framework::vectorize<int>(w->dims()), ctx.OutputName("Out"));
  prim_key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, prim_key);

  constexpr bool is_int8 =
      std::is_same<T_in, int8_t>::value || std::is_same<T_in, uint8_t>::value;
  bool is_bfloat16 = std::is_same<T_in, paddle::platform::bfloat16>::value;
  if ((!is_int8 && !is_bfloat16) || force_fp32_output) {
    GetPrimitiveFactory<T_in, T_w, float>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else if (is_bfloat16) {
    GetPrimitiveFactory<T_in, T_w, platform::bfloat16>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else if (fuse_relu) {
    GetPrimitiveFactory<T_in, T_w, uint8_t>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  } else {
    GetPrimitiveFactory<T_in, T_w, int8_t>(dev_ctx, prim_key)
        ->ExecuteFcPrimitive(input, w, bias, output, dev_ctx, ctx);
  }
}

template <typename T_in, typename T_w>
class FCMKLDNNOpKernel : public framework::OpKernel<T_in> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        platform::errors::PreconditionNotMet("FC MKL-DNN must use CPUPlace."));
    platform::MKLDNNDeviceContext::tls().log_lib_version();
    auto input = ctx.Input<LoDTensor>("Input");
    auto w = ctx.Input<Tensor>("W");
    auto bias = ctx.Input<Tensor>("Bias");
    auto output = ctx.Output<LoDTensor>("Out");

    bool fuse_relu = ctx.Attr<std::string>("activation_type") == "relu";
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    ExecuteFc<T_in, T_w>(ctx, input, w, bias, output, fuse_relu,
                         force_fp32_output);

    output->set_layout(DataLayout::kMKLDNN);
  }
};
}  // namespace operators
}  // namespace paddle

// Weights of FC are by default stored using fp32, template argument of weight
// data type implies their destination data type. (What's eventually going to
// be used during computations of kernel).
namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(
    fc, MKLDNN, ::paddle::platform::CPUPlace, BF16, ops::kFCMKLDNNFP32,
    ops::FCMKLDNNOpKernel<paddle::platform::bfloat16,
                          paddle::platform::bfloat16>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, int8_t>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, int8_t>);
