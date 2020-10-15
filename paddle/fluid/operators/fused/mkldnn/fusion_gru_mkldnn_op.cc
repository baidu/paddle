/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/fused/fusion_gru_op.h"
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

template <typename T, typename T_out = T>
class GRUMKLDNNHandler : public platform::MKLDNNHandlerT<T, dnnl::gru_forward> {
 public:
  GRUMKLDNNHandler(const paddle::framework::ExecutionContext& ctx,
                   const platform::MKLDNNDeviceContext& dev_ctx,
                   const mkldnn::engine mkldnn_engine,
                   platform::Place cpu_place, const LoDTensor* input,
                   const Tensor* weight_h, const Tensor* h0,
                   const bool is_reverse, const int64_t N, const int64_t Ti,
                   const int64_t IC, const int64_t OC,
                   const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, dnnl::gru_forward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            CreateKey(unique_name, MKLDNNGetDataType<T>(), Ti)),
        N(N),
        Ti(Ti),
        IC(IC),
        OC(OC) {
    // Create memory key without Ti because weights, bias and h0 memories
    // do not depend on Ti size but primitive and input/output memory do
    if (platform::MKLDNNDeviceContext::tls().get_cur_mkldnn_session_id() !=
        platform::MKLDNNDeviceContextThreadLocals::kMKLDNNSessionID_Default) {
      memory_key_ = CreateKey(unique_name, MKLDNNGetDataType<T>());
    } else {
      memory_key_ = CreateKey(unique_name, MKLDNNGetDataType<T>(), "-t:",
                              platform::ThreadIDasStr());
    }

    // Is it int8 kernel
    const bool is_INT8 = std::is_same<T, uint8_t>::value;

    if (is_INT8) {
      // Int8 attributes
      const float scale_data = ctx.Attr<float>("Scale_data");
      const float shift_data = ctx.Attr<float>("Shift_data");
      const auto scale_weights = ctx.Attr<std::vector<float>>("Scale_weights");

      const int weights_scale_mask =
          0 +
          (1 << 3)  // bit, indicating the unique scales for `g` dim in `ldigo`
          +
          (1 << 4);  // bit, indicating the unique scales for `o` dim in `ldigo`

      attr_.set_rnn_data_qparams(scale_data, shift_data);
      attr_.set_rnn_weights_qparams(weights_scale_mask, scale_weights);
    }

    if (!this->isCached()) {
      // oneDNN kernel has hardcoded activation functions
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("gate_activation"), "sigmoid",
          platform::errors::Unimplemented(
              "oneDNN fusion_gru supports only sigmoid as a gate activation."));
      PADDLE_ENFORCE_EQ(
          ctx.Attr<std::string>("activation"), "tanh",
          platform::errors::Unimplemented(
              "oneDNN fusion_gru supports only tanh as an activation."));

      // Weights for int8 kernel are of a type s8
      const auto weights_dt =
          is_INT8 ? dnnl::memory::data_type::s8 : MKLDNNGetDataType<T>();

      // oneDNN RNN dimensions
      const int64_t D = 1;  // Directions
      const int64_t L = 1;  // Layers (PP supports only 1 stacked layer)
      const int64_t G = 3;  // Number of Gates, 3 for GRU

      // Create memory descriptors
      auto input_md = MKLDNNMemDesc({Ti, N, IC}, MKLDNNGetDataType<T>(),
                                    MKLDNNMemoryFormat::ntc);
      auto weight_x_md =
          MKLDNNMemDesc({L, D, IC, G, OC}, weights_dt, MKLDNNMemoryFormat::any);
      auto weight_h_md =
          MKLDNNMemDesc({L, D, OC, G, OC}, weights_dt, MKLDNNMemoryFormat::any);
      auto bias_md = MKLDNNMemDesc({L, D, G, OC}, MKLDNNGetDataType<float>(),
                                   MKLDNNMemoryFormat::ldgo);
      auto hidden_md = MKLDNNMemDesc({Ti, N, OC}, MKLDNNGetDataType<T_out>(),
                                     MKLDNNMemoryFormat::ntc);
      auto h0_md = MKLDNNMemDesc({L, D, N, OC}, MKLDNNGetDataType<T>(),
                                 MKLDNNMemoryFormat::ldnc);

      // Create GRU oneDNN primitive
      const auto direction =
          is_reverse ? dnnl::rnn_direction::unidirectional_right2left
                     : dnnl::rnn_direction::unidirectional_left2right;

      this->AcquireForwardPrimitiveDescriptor(
          attr_, dnnl::prop_kind::forward_inference, direction, input_md, h0_md,
          weight_x_md, weight_h_md, bias_md, hidden_md, dnnl::memory::desc());
    }
  }

  bool is_NTC() {
    return (platform::GetMKLDNNFormat(this->fwd_pd_->dst_desc()) ==
            dnnl::memory::format_tag::ntc);
  }

  void reorderRNNdata(void* input_data, void* output_data,
                      std::vector<size_t> lod, const bool is_reverse,
                      platform::RNNReorderType reorder_type) {
    switch (reorder_type) {
      // Reorder input memory [WORDS, C] + LoD -> [N, T, C]
      case platform::RNNReorderType::PP_NTC: {
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * IC;
          const auto offset = is_reverse ? (Ti * IC - num_elements) : 0;
          memcpy(output_data_iter + n * Ti * IC + offset, input_data_iter,
                 sizeof(T) * num_elements);
          input_data_iter += num_elements;
        }
      } break;
      // Reorder input memory [WORDS, C] + LoD -> [T, N, C]
      case platform::RNNReorderType::PP_TNC: {
        auto* input_data_iter = reinterpret_cast<T*>(input_data);
        auto* output_data_iter = reinterpret_cast<T*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]);
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter + (t + offset) * N * IC + n * IC,
                   input_data_iter, sizeof(T) * IC);
            input_data_iter += IC;
          }
        }
      } break;
      // Reorder output values to PP format [N, T, C] -> [WORDS, C]
      case platform::RNNReorderType::NTC_PP: {
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = (lod[n + 1] - lod[n]) * OC;
          const auto offset = is_reverse ? (Ti * OC - num_elements) : 0;
          memcpy(output_data_iter, input_data_iter + n * Ti * OC + offset,
                 sizeof(T_out) * num_elements);
          output_data_iter += num_elements;
        }
      } break;
      // Reorder output values to PP format [T, N, C] -> [WORDS, C]
      case platform::RNNReorderType::TNC_PP: {
        auto* input_data_iter = reinterpret_cast<T_out*>(input_data);
        auto* output_data_iter = reinterpret_cast<T_out*>(output_data);
        for (int n = 0; n < N; ++n) {
          const auto num_elements = lod[n + 1] - lod[n];
          const auto offset = is_reverse ? (Ti - num_elements) : 0;
          for (size_t t = 0; t < num_elements; ++t) {
            memcpy(output_data_iter,
                   input_data_iter + (t + offset) * N * OC + n * OC,
                   sizeof(T_out) * OC);
            output_data_iter += OC;
          }
        }
      } break;
    }
  }

  std::shared_ptr<dnnl::memory> AcquireInputMemoryWithReorder(
      const LoDTensor* input, const bool is_reverse) {
    const auto name = this->key_ + "@input_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }

    const auto& input_lod = input->lod()[0];
    auto* x_data = to_void_cast(input->data<T>());

    auto* x_onednn_data = memory_p->get_data_handle();
    memset(x_onednn_data, 0, sizeof(T) * N * Ti * IC);

    if (platform::GetMKLDNNFormat(this->fwd_pd_->src_desc()) ==
        dnnl::memory::format_tag::ntc) {
      reorderRNNdata(x_data, x_onednn_data, input_lod, is_reverse,
                     platform::RNNReorderType::PP_NTC);
    } else {
      reorderRNNdata(x_data, x_onednn_data, input_lod, is_reverse,
                     platform::RNNReorderType::PP_TNC);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireOutputMemory() {
    const auto name = this->key_ + "@output_mem";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(name));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->dst_desc(),
                                                this->engine_);
      this->dev_ctx_.SetBlob(name, memory_p);
    }
    return memory_p;
  }

  // TODO(grygielski) H0 is for now persistable
  // TODO(jczaja) H0 should be updated each iter and of T type (Fusion pass does
  // not support in yet)
  std::shared_ptr<dnnl::memory> AcquireH0Memory(const Tensor* h0) {
    const std::string h0_key = memory_key_ + "@h0";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(h0_key));

    if (!memory_p) {
      auto user_h0_memory = dnnl::memory();
      if (h0) {
        user_h0_memory =
            dnnl::memory({{1, 1, N, OC},
                          MKLDNNGetDataType<float>(),
                          MKLDNNMemoryFormat::ldnc},
                         this->engine_, to_void_cast(h0->data<float>()));
      } else {
        user_h0_memory = dnnl::memory({{1, 1, N, OC},
                                       MKLDNNGetDataType<float>(),
                                       MKLDNNMemoryFormat::ldnc},
                                      this->engine_);
        memset(user_h0_memory.get_data_handle(), 0, sizeof(float) * N * OC);
      }
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->src_iter_desc(),
                                                this->engine_);

      dnnl::stream astream(this->engine_);
      dnnl::reorder(user_h0_memory, *memory_p, attr_)
          .execute(astream, user_h0_memory, *memory_p);

      this->dev_ctx_.SetBlob(h0_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireWeightXMemory(const Tensor* weight_x,
                                                     const bool origin_mode) {
    const std::string wx_key = memory_key_ + "@weight_x";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wx_key));

    if (!memory_p) {
      auto user_md =
          MKLDNNMemDesc({1, 1, IC, 3, OC}, MKLDNNGetDataType<float>(),
                        MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      auto* weight_x_data =
          reinterpret_cast<float*>(user_memory.get_data_handle());
      memcpy(weight_x_data, weight_x->data<float>(),
             sizeof(float) * IC * 3 * OC);

      if (origin_mode == false) {
        for (int64_t i = 0; i < IC; ++i) {
          for (int64_t j = 0; j < OC; ++j) {
            weight_x_data[j] *= -1;
          }
          weight_x_data += 3 * OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_layer_desc(), this->engine_);

      dnnl::stream astream(this->engine_);
      dnnl::reorder(user_memory, *memory_p, attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wx_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireWeightHMemory(const Tensor* weight_h,
                                                     const bool origin_mode) {
    const std::string wh_key = memory_key_ + "@weight_h";
    auto memory_p =
        std::static_pointer_cast<dnnl::memory>(this->dev_ctx_.GetBlob(wh_key));

    if (!memory_p) {
      auto user_md =
          MKLDNNMemDesc({1, 1, OC, 3, OC}, MKLDNNGetDataType<float>(),
                        MKLDNNMemoryFormat::ldigo);
      auto user_memory = dnnl::memory(user_md, this->engine_);

      // Reorder weights_h from PP format [OC, 2OC] + [OC, OC] to
      // oneDNN format [OC, 3OC]
      auto* weight_h_data =
          reinterpret_cast<float*>(user_memory.get_data_handle());
      auto* user_weight_h_data = weight_h->data<float>();

      auto src1_iter = user_weight_h_data;
      auto src2_iter = user_weight_h_data + 2 * OC * OC;

      for (int64_t c = 0; c < OC; ++c) {
        memcpy(weight_h_data, src1_iter, 2 * OC * sizeof(float));
        memcpy(weight_h_data + 2 * OC, src2_iter, OC * sizeof(float));

        src1_iter += 2 * OC;
        src2_iter += OC;
        weight_h_data += 3 * OC;
      }

      weight_h_data = reinterpret_cast<float*>(user_memory.get_data_handle());

      if (origin_mode == false) {
        for (int64_t i = 0; i < OC; ++i) {
          for (int64_t j = 0; j < OC; ++j) {
            weight_h_data[j] *= -1;
          }
          weight_h_data += 3 * OC;
        }
      }

      memory_p = std::make_shared<dnnl::memory>(
          this->fwd_pd_->weights_iter_desc(), this->engine_);

      dnnl::stream astream(this->engine_);
      dnnl::reorder(user_memory, *memory_p, attr_)
          .execute(astream, user_memory, *memory_p);

      this->dev_ctx_.SetBlob(wh_key, memory_p);
    }
    return memory_p;
  }

  std::shared_ptr<dnnl::memory> AcquireBiasMemory(const Tensor* bias,
                                                  const bool origin_mode) {
    const std::string bias_key = memory_key_ + "@bias";
    auto memory_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(bias_key));

    if (!memory_p) {
      memory_p = std::make_shared<dnnl::memory>(this->fwd_pd_->bias_desc(),
                                                this->engine_);
      auto* bias_data = reinterpret_cast<float*>(memory_p->get_data_handle());
      if (bias) {
        const float* user_bias_data =
            bias->data<float>();  // Bias in oneDNN is always float
        memcpy(bias_data, user_bias_data, sizeof(float) * 3 * OC);
      } else {
        // oneDNN always need bias memory, if it's not provided in PP, let
        // oneDNN allocate memory and set it to 0
        memset(bias_data, 0, sizeof(float) * 3 * OC);
      }

      if (origin_mode == false && bias) {
        for (int64_t i = 0; i < OC; ++i) {
          bias_data[i] *= -1;
        }
      }
      this->dev_ctx_.SetBlob(bias_key, memory_p);
    }
    return memory_p;
  }

 private:
  // RNN dimensions
  // N - Batch Size
  // Ti - Max sentence length
  // IC - Input Channels
  // OC - Output Channels
  const int64_t N, Ti, IC, OC;

  // Memory size of weights, bias and h0 does not depend
  // on Ti size, thus we need another key to cache them
  std::string memory_key_;
  dnnl::primitive_attr attr_;
};

template <typename T>
class FusionGRUMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const bool is_bf16 = std::is_same<T, paddle::platform::bfloat16>::value;
    const bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");

    // BF16 does not support force output
    if (!is_bf16 && force_fp32_output) {
      RunKernel<float>(ctx);
    } else {
      RunKernel<T>(ctx);
    }
  }

  template <typename Tout = T>
  void RunKernel(const framework::ExecutionContext& ctx) const {
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    // Get Tensors
    const auto* input = ctx.Input<LoDTensor>("X");
    const auto* h0 = ctx.Input<Tensor>("H0");
    const auto* weight_x = ctx.Input<Tensor>("WeightX");
    const auto* weight_h = ctx.Input<Tensor>("WeightH");
    const auto* bias = ctx.Input<Tensor>("Bias");
    auto* hidden = ctx.Output<LoDTensor>("Hidden");
    auto x_dims = input->dims();
    auto x_mat_dims = (x_dims.size() == 3 && x_dims[1] == 1)
                          ? framework::flatten_to_2d(x_dims, 1)
                          : x_dims;
    // Get attributes
    const bool is_reverse = ctx.Attr<bool>("is_reverse");
    const bool origin_mode = ctx.Attr<bool>("origin_mode");

    // Get tensor dimensions
    const auto x_mat_dims_vec = framework::vectorize(x_mat_dims);
    const auto weight_h_dims = framework::vectorize(weight_h->dims());
    const auto& input_lod = input->lod()[0];

    // Calculate RNN dimensions
    const int64_t N = input_lod.size() - 1;  // Number of sentences (batches)
    const int64_t Ti =  // Max length of the sentence in a batch
        [&input_lod]() {
          size_t res = 0;
          for (size_t i = 0; i < (input_lod.size() - 1); ++i) {
            res = std::max(res, input_lod[i + 1] - input_lod[i]);
          }
          return res;
        }();
    const int64_t IC = x_mat_dims_vec[1];  // Input channels
    const int64_t OC = weight_h_dims[0];   // Output channels

    GRUMKLDNNHandler<T, Tout> handler(
        ctx, dev_ctx, mkldnn_engine, ctx.GetPlace(), input, weight_h, h0,
        is_reverse, N, Ti, IC, OC,
        ctx.InputName("X") + ctx.InputName("WeightH"));

    auto input_memory_p =
        handler.AcquireInputMemoryWithReorder(input, is_reverse);
    auto h0_memory_p = handler.AcquireH0Memory(h0);
    auto weight_x_memory_p =
        handler.AcquireWeightXMemory(weight_x, origin_mode);
    auto weight_h_memory_p =
        handler.AcquireWeightHMemory(weight_h, origin_mode);
    auto bias_memory_p = handler.AcquireBiasMemory(bias, origin_mode);
    auto hidden_onednn_memory_p = handler.AcquireOutputMemory();

    std::unordered_map<int, dnnl::memory> gru_args = {
        {DNNL_ARG_SRC_LAYER, *input_memory_p},
        {DNNL_ARG_SRC_ITER, *h0_memory_p},
        {DNNL_ARG_WEIGHTS_LAYER, *weight_x_memory_p},
        {DNNL_ARG_WEIGHTS_ITER, *weight_h_memory_p},
        {DNNL_ARG_BIAS, *bias_memory_p},
        {DNNL_ARG_DST_LAYER, *hidden_onednn_memory_p}};

    auto gru_forward_p = handler.AcquireForwardPrimitive();

    dnnl::stream astream(mkldnn_engine);
    gru_forward_p->execute(astream, gru_args);
    astream.wait();

    auto* hidden_onednn_data = hidden_onednn_memory_p->get_data_handle();
    auto* hidden_data =
        to_void_cast(hidden->mutable_data<Tout>(ctx.GetPlace()));
    if (handler.is_NTC()) {
      handler.reorderRNNdata(hidden_onednn_data, hidden_data, input_lod,
                             is_reverse, platform::RNNReorderType::NTC_PP);
    } else {
      handler.reorderRNNdata(hidden_onednn_data, hidden_data, input_lod,
                             is_reverse, platform::RNNReorderType::TNC_PP);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(fusion_gru, MKLDNN, paddle::platform::CPUPlace,
                   ops::FusionGRUMKLDNNKernel<float>,
                   ops::FusionGRUMKLDNNKernel<paddle::platform::bfloat16>,
                   ops::FusionGRUMKLDNNKernel<uint8_t>);
