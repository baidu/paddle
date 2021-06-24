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

#include "paddle/fluid/operators/fused/fusion_lstm_op.h"
#include <string>
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/sequence2batch.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

void FusionLSTMOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasInput("WeightX"), "Input", "WeightX", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasInput("WeightH"), "Input", "WeightH", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasOutput("XX"), "Output", "XX", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "fusion_lstm");
  OP_INOUT_CHECK(ctx->HasOutput("Cell"), "Output", "Cell", "fusion_lstm");

  auto x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "Input(X)'s rank must be 2, but received x's rank "
                        "is:%d, x dim is:[%s]",
                        x_dims.size(), x_dims));

  if (ctx->HasInput("H0")) {
    OP_INOUT_CHECK(ctx->HasInput("C0"), "Input", "C0", "fusion_lstm");
    auto h_dims = ctx->GetInputDim("H0");
    auto c_dims = ctx->GetInputDim("C0");
    PADDLE_ENFORCE_EQ(h_dims, c_dims,
                      platform::errors::InvalidArgument(
                          "The dimension of Input(H0) and Input(C0) should be "
                          "same, but received h0 dims is:[%s], c0 dims is:[%s]",
                          h_dims, c_dims));
  }

  auto wx_dims = ctx->GetInputDim("WeightX");
  PADDLE_ENFORCE_EQ(wx_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightX) should be 2, but received "
                        "WeightX's rank is:%d, WeightX dim is:[%s]",
                        wx_dims.size(), wx_dims));
  PADDLE_ENFORCE_EQ(wx_dims[0], x_dims[1],
                    platform::errors::InvalidArgument(
                        "The first dimension of Input(WeightX) "
                        "should equal to second dimension of Input(X), but "
                        "received WeightX first dim is:%d, X second dim is:%d",
                        wx_dims[0], x_dims[1]));

  int frame_size = wx_dims[1] / 4;
  auto wh_dims = ctx->GetInputDim("WeightH");

  PADDLE_ENFORCE_EQ(wh_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(WeightH) should be 2, but received "
                        "WeightH rank is:%d, WeightH dim is:[%s]",
                        wh_dims.size(), wh_dims));
  PADDLE_ENFORCE_EQ(wh_dims[0], frame_size,
                    platform::errors::InvalidArgument(
                        "The first dimension of Input(WeightH) "
                        "should equal to frame size, but received WeightH "
                        "first dim is:%d, frame size is:%d.",
                        wh_dims[0], frame_size));

  PADDLE_ENFORCE_EQ(wh_dims[1], 4 * frame_size,
                    platform::errors::InvalidArgument(
                        "The second dimension of Input(WeightH) "
                        "should equal to 4 * frame_size, but received WeightH "
                        "second dimension is:%d, frame size is:%d.",
                        wh_dims[1], frame_size));

  auto b_dims = ctx->GetInputDim("Bias");
  PADDLE_ENFORCE_EQ(b_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "The rank of Input(Bias) should be 2, but received "
                        "Bias rank is:%d, Bias dim is:[%s]",
                        b_dims.size(), b_dims));
  PADDLE_ENFORCE_EQ(b_dims[0], 1,
                    platform::errors::InvalidArgument(
                        "The first dimension of Input(Bias) should be 1, but "
                        "received Bias's dimension is:[%s]",
                        b_dims));

  if (ctx->Attrs().Get<bool>("use_peepholes")) {
    PADDLE_ENFORCE_EQ(b_dims[1], 7 * frame_size,
                      platform::errors::InvalidArgument(
                          "The second dimension of Input(Bias) should be "
                          "7 * %d if enable peepholes connection, but received "
                          "Bias dim is:[%s]",
                          frame_size, b_dims));
    ctx->SetOutputDim("CheckedCell", {2, frame_size});
  } else {
    PADDLE_ENFORCE_EQ(
        b_dims[1], 4 * frame_size,
        platform::errors::InvalidArgument(
            "The second dimension of Input(Bias) should be "
            "4 * %d if disable peepholes, but received Bias dim is:[%s]",
            frame_size, b_dims));
  }

  framework::DDim out_dims({x_dims[0], frame_size});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->ShareLoD("X", "Hidden");
  ctx->ShareLoD("X", "Cell");
  int xx_width;
  if (ctx->Attrs().Get<bool>("use_seq")) {
    xx_width = wx_dims[1];
  } else {
    xx_width = x_dims[1] > wx_dims[1] ? wx_dims[1] : x_dims[1];

    OP_INOUT_CHECK(ctx->HasOutput("BatchedInput"), "Output", "BatchedInput",
                   "fusion_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedHidden"), "Output", "BatchedHidden",
                   "fusion_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("BatchedCell"), "Output", "BatchedCell",
                   "fusion_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedH0"), "Output", "ReorderedH0",
                   "fusion_lstm");
    OP_INOUT_CHECK(ctx->HasOutput("ReorderedC0"), "Output", "ReorderedC0",
                   "fusion_lstm");

    ctx->SetOutputDim("BatchedInput", {x_dims[0], wx_dims[1]});
    ctx->SetOutputDim("BatchedHidden", out_dims);
    ctx->SetOutputDim("BatchedCell", out_dims);
  }
  ctx->SetOutputDim("XX", {x_dims[0], xx_width});
  ctx->ShareLoD("X", "XX");
}

framework::OpKernelType FusionLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library = framework::LibraryType::kPlain;
  framework::DataLayout layout = framework::DataLayout::kAnyLayout;
  auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
#ifdef PADDLE_WITH_MKLDNN
  if (this->CanMKLDNNBeUsed(ctx, data_type)) {
    library = framework::LibraryType::kMKLDNN;
    layout = framework::DataLayout::kMKLDNN;
  }
#endif
  return framework::OpKernelType(data_type, ctx.GetPlace(), layout, library);
}

void FusionLSTMOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("WeightX",
           "(Tensor) the learnable weights of X."
           " - The shape is (M x 4D), where M is the dim size of x, D is the "
           "hidden size. "
           " - Weight = {W_cx, W_ix, W_fx, W_ox}");
  AddInput("WeightH",
           "(Tensor) same as LSTMOp, the learnable hidden-hidden weights."
           " - The shape is (D x 4D), where D is the hidden size. "
           " - Weight = {W_ch, W_ih, W_fh, W_oh}");
  AddInput("Bias",
           "(Tensor) the learnable weights. Almost same as LSTMOp"
           "Note: we should add the fc bias into this (1x4D) in bias."
           "input-hidden bias weight and peephole connections weight if "
           "setting `use_peepholes` True. "
           "1. `use_peepholes = False` "
           " - The shape is (1 x 4D). "
           " - Bias = {b_c, b_i, b_f, b_o}."
           "2. `use_peepholes = True` "
           " - The shape is (1 x 7D). "
           " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.");
  AddInput("H0",
           "(Tensor, optional) (same as LSTMOp) the initial hidden state is an "
           "optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size and D is the hidden size.")
      .AsDispensable();
  AddInput("C0",
           "(Tensor, optional) (same as LSTMOp) (the initial cell state is an "
           "optional "
           "input. This is a tensor with shape (N x D), where N is the "
           "batch size. `H0` and `C0` can be NULL but only at the same time.")
      .AsDispensable();
  AddOutput("Hidden",
            "(LoDTensor) (same as LSTMOp) the hidden state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("Cell",
            "(LoDTensor) (same as LSTMOp) the cell state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("XX",
            "(LoDTensor) the result after X * WeightX (size is T x 4D)"
            " or batched_X (size is T x M), this will be automatically chosen,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size, M is the dim size of x input.")
      .AsIntermediate();
  AddOutput("BatchedInput", "(LoDTensor) (T x 4D).").AsIntermediate();
  AddOutput("BatchedHidden", "(LoDTensor) (T x D).").AsIntermediate();
  AddOutput("BatchedCell", "(LoDTensor) (T x D).").AsIntermediate();
  AddOutput("ReorderedH0", "(LoDTensor) (N x D).").AsIntermediate();
  AddOutput("ReorderedC0", "(LoDTensor) (N x D).").AsIntermediate();
  AddOutput("CheckedCell", "(Tensor) (2 x D) only for peephole.")
      .AsIntermediate();
  AddAttr<bool>("use_peepholes",
                "(bool, default: True) "
                "whether to enable diagonal/peephole connections.")
      .SetDefault(true);
  AddAttr<bool>("is_reverse",
                "(bool, default: False) "
                "whether to compute reversed LSTM.")
      .SetDefault(false);
  AddAttr<bool>("use_seq",
                "(bool, default: True) "
                "whether to use seq mode to compute.")
      .SetDefault(true);
  AddAttr<std::string>("gate_activation",
                       "(string, default: sigmoid)"
                       "The activation for input gate, forget gate and output "
                       "gate, `sigmoid` by default.")
      .SetDefault("sigmoid")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<std::string>("cell_activation",
                       "(string, default: tanh)"
                       "The activation for cell output, `tanh` by default.")
      .SetDefault("tanh")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<std::string>("candidate_activation",
                       "(string, default: tanh)"
                       "The activation for candidate hidden state, "
                       "`tanh` by default.")
      .SetDefault("tanh")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddAttr<bool>("use_mkldnn",
                "(bool, default false) Only used in mkldnn kernel")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  AddAttr<float>("Scale_data",
                 "Scale to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(1.0f);
  AddAttr<float>("Shift_data",
                 "Shift to be used for int8 input/output data."
                 "Only used with MKL-DNN INT8.")
      .SetDefault(0.0f);
  AddAttr<std::vector<float>>("Scale_weights",
                              "Scale_weights to be used for int8 weights data."
                              "Only used with MKL-DNN INT8.")
      .SetDefault({1.0f});
  AddAttr<bool>("force_fp32_output",
                "(bool, default false) Force INT8 kernel output FP32, only "
                "used in MKL-DNN INT8")
      .SetDefault(false);
  AddComment(R"DOC(
Fusion Long-Short Term Memory (LSTM) Operator.
This operator fuse the X into LSTM, more details can refer to LSTM op.
)DOC");
}

template <typename T>
class FuisonLSTMKernel : public framework::OpKernel<T> {
 public:
#define INIT_BASE_DEFINES                                   \
  using DeviceContext = paddle::platform::CPUDeviceContext; \
  auto* x = ctx.Input<LoDTensor>("X");                      \
  auto* h0 = ctx.Input<Tensor>("H0");                       \
  auto* c0 = ctx.Input<Tensor>("C0");                       \
  auto* wx = ctx.Input<Tensor>("WeightX");                  \
  auto* wh = ctx.Input<Tensor>("WeightH");                  \
  auto* bias = ctx.Input<Tensor>("Bias");                   \
  auto* xx = ctx.Output<LoDTensor>("XX");                   \
  auto* hidden_out = ctx.Output<LoDTensor>("Hidden");       \
  auto* cell_out = ctx.Output<LoDTensor>("Cell");           \
  bool is_reverse = ctx.Attr<bool>("is_reverse");           \
  bool use_peepholes = ctx.Attr<bool>("use_peepholes");     \
  auto x_dims = x->dims();   /* T x M*/                     \
  auto wh_dims = wh->dims(); /* D x 4D*/                    \
  const int M = x_dims[1];                                  \
  const int D = wh_dims[0];                                 \
  const int D4 = wh_dims[1]

#define INIT_OTHER_DEFINES                                                     \
  const T* x_data = x->data<T>();                                              \
  const T* wx_data = wx->data<T>();                                            \
  const T* wh_data = wh->data<T>();                                            \
  /* diagonal weight*/                                                         \
  const T* wp_data = bias->data<T>() + D4;                                     \
  /* for peephole only*/                                                       \
  T* checked_cell_data = nullptr;                                              \
  auto place = ctx.GetPlace();                                                 \
  if (use_peepholes) {                                                         \
    /* w_ic * Ct-1, w_fc * Ct-1  ; w_oc * Ct => ih*/                           \
    auto* checked_cell = ctx.Output<Tensor>("CheckedCell");                    \
    checked_cell_data = checked_cell->mutable_data<T>(place);                  \
  }                                                                            \
  const jit::lstm_attr_t attr(                                                 \
      D, jit::to_kerneltype(ctx.Attr<std::string>("gate_activation")),         \
      jit::to_kerneltype(ctx.Attr<std::string>("candidate_activation")),       \
      jit::to_kerneltype(ctx.Attr<std::string>("cell_activation")),            \
      use_peepholes);                                                          \
  jit::lstm_t one_step;                                                        \
  one_step.wp = wp_data;                                                       \
  one_step.checked = checked_cell_data;                                        \
  auto ComputeC1H1 =                                                           \
      jit::KernelFuncs<jit::LSTMC1H1Tuple<T>, platform::CPUPlace>::Cache().At( \
          attr);                                                               \
  auto ComputeCtHt =                                                           \
      jit::KernelFuncs<jit::LSTMCtHtTuple<T>, platform::CPUPlace>::Cache().At( \
          attr)

// Wh GEMM
#define GEMM_WH_ADDON(bs, prev, out)                                           \
  blas.GEMM(CblasNoTrans, CblasNoTrans, bs, D4, D, static_cast<T>(1), prev, D, \
            wh_data, D4, static_cast<T>(1), out, D4)

  void SeqCompute(const framework::ExecutionContext& ctx) const {
    INIT_BASE_DEFINES;
    INIT_OTHER_DEFINES;
    auto x_lod = x->lod();
    const int total_T = x_dims[0];
    const int N = x_lod[0].size() - 1;
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* c0_data = c0 ? c0->data<T>() : nullptr;
    T* xx_data = xx->mutable_data<T>(place);
    T* h_out_data = hidden_out->mutable_data<T>(place);
    T* c_out_data = cell_out->mutable_data<T>(place);
    auto blas = math::GetBlas<DeviceContext, T>(ctx);

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    math::FCFunctor<DeviceContext, T> fc;
    fc(dev_ctx, total_T, D4, M, x_data, wx_data, xx_data, bias->data<T>());

    int xx_offset = D4;
    int gate_offset = D;
    if (is_reverse) {
      const int offset = (total_T - 1) * D;
      xx_data = xx_data + offset * 4;
      h_out_data = h_out_data + offset;
      c_out_data = c_out_data + offset;
      xx_offset = -D4;
      gate_offset = -D;
    }

    for (int i = 0; i < N; ++i) {
      int bid = is_reverse ? N - 1 - i : i;
      int seq_len = x_lod[0][bid + 1] - x_lod[0][bid];
      const T* prev_c_data = nullptr;
      const T* prev_h_data = nullptr;
      int tstart = 0;
      if (h0_data) {
        prev_h_data = h0_data + bid * D;
        prev_c_data = c0_data + bid * D;
      } else {
        one_step.gates = xx_data;
        one_step.ct = c_out_data;
        one_step.ht = h_out_data;
        ComputeC1H1(&one_step, &attr);
        tstart = 1;
        // move one step
        prev_h_data = h_out_data;
        prev_c_data = c_out_data;
        xx_data = xx_data + xx_offset;
        h_out_data = h_out_data + gate_offset;
        c_out_data = c_out_data + gate_offset;
      }
      for (int step = tstart; step < seq_len; ++step) {
        GEMM_WH_ADDON(1, prev_h_data, xx_data);

        one_step.gates = xx_data;
        one_step.ct_1 = prev_c_data;
        one_step.ct = c_out_data;
        one_step.ht = h_out_data;
        ComputeCtHt(&one_step, &attr);
        // move one step
        prev_h_data = h_out_data;
        prev_c_data = c_out_data;
        xx_data = xx_data + xx_offset;
        h_out_data = h_out_data + gate_offset;
        c_out_data = c_out_data + gate_offset;
      }
    }
  }

  void BatchCompute(const framework::ExecutionContext& ctx) const {
    INIT_BASE_DEFINES;
    if (x->lod()[0].size() == 2) {
      xx->Resize({x_dims[0], D4});
      SeqCompute(ctx);
      return;
    }
    INIT_OTHER_DEFINES;

    auto* reordered_h0 = ctx.Output<Tensor>("ReorderedH0");
    auto* reordered_c0 = ctx.Output<Tensor>("ReorderedC0");
    auto* batched_input = ctx.Output<LoDTensor>("BatchedInput");
    auto* batched_c_out = ctx.Output<LoDTensor>("BatchedCell");
    auto* batched_h_out = ctx.Output<LoDTensor>("BatchedHidden");
    T* xx_data = xx->mutable_data<T>(place);
    T* batched_input_data = batched_input->mutable_data<T>(place);
    T* batched_c_out_data = batched_c_out->mutable_data<T>(place);
    T* batched_h_out_data = batched_h_out->mutable_data<T>(place);
    hidden_out->mutable_data<T>(place);
    cell_out->mutable_data<T>(place);

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    math::FCFunctor<DeviceContext, T> fc;
    if (M > D4) {
      fc(dev_ctx, x_dims[0], D4, M, x_data, wx_data, xx_data, bias->data<T>());
      to_batch(dev_ctx, *xx, batched_input, true, is_reverse);
    } else {
      to_batch(dev_ctx, *x, xx, true, is_reverse);
      batched_input->set_lod(xx->lod());
      fc(dev_ctx, x_dims[0], D4, M, xx_data, wx_data, batched_input_data,
         bias->data<T>());
    }

    auto batched_lod = batched_input->lod();
    const auto& seq_order = batched_lod[2];
    const int max_bs = seq_order.size();
    reordered_h0->Resize({max_bs, D});
    reordered_c0->Resize({max_bs, D});

    int tstart = 0;
    T* prev_h_data = nullptr;
    T* prev_c_data = nullptr;
    if (h0) {
      // reorder h0, c0
      T* reordered_h0_data = reordered_h0->mutable_data<T>(place);
      T* reordered_c0_data = reordered_c0->mutable_data<T>(place);
      const T* h0_data = h0->data<T>();
      const T* c0_data = c0->data<T>();
      prev_h_data = reordered_h0_data;
      prev_c_data = reordered_c0_data;
      size_t sz = D;
      for (int i = 0; i < max_bs; ++i) {
        blas.VCOPY(sz, h0_data + seq_order[i] * D, reordered_h0_data);
        blas.VCOPY(sz, c0_data + seq_order[i] * D, reordered_c0_data);
        reordered_h0_data += D;
        reordered_c0_data += D;
      }
    } else {
      // compute without h0, c0
      T* cur_in_data = batched_input_data;
      T* cur_h_out_data = batched_h_out_data;
      T* cur_c_out_data = batched_c_out_data;
      for (int i = 0; i < max_bs; ++i) {
        one_step.gates = cur_in_data;
        one_step.ct = cur_c_out_data;
        one_step.ht = cur_h_out_data;
        ComputeC1H1(&one_step, &attr);

        cur_in_data += D4;
        cur_c_out_data += D;
        cur_h_out_data += D;
      }
      tstart = 1;
      prev_h_data = batched_h_out_data;
      prev_c_data = batched_c_out_data;
    }

    // compute kernel part
    const auto& batch_starts = batched_lod[0];
    const int max_seq_len = batch_starts.size() - 1;
    const int offset = tstart * max_bs * D;
    batched_input_data = batched_input_data + offset * 4;
    batched_h_out_data = batched_h_out_data + offset;
    batched_c_out_data = batched_c_out_data + offset;
    for (int step = tstart; step < max_seq_len; ++step) {
      const int cur_bs = batch_starts[step + 1] - batch_starts[step];
      GEMM_WH_ADDON(cur_bs, prev_h_data, batched_input_data);
      T* cur_in_data = batched_input_data;
      T* cur_prev_c_data = prev_c_data;
      T* cur_c_out_data = batched_c_out_data;
      T* cur_h_out_data = batched_h_out_data;
      for (int i = 0; i < cur_bs; ++i) {
        one_step.gates = cur_in_data;
        one_step.ct_1 = cur_prev_c_data;
        one_step.ct = cur_c_out_data;
        one_step.ht = cur_h_out_data;
        ComputeCtHt(&one_step, &attr);

        // move one batch
        cur_in_data += D4;
        cur_prev_c_data += D;
        cur_c_out_data += D;
        cur_h_out_data += D;
      }
      // move one step
      prev_c_data = batched_c_out_data;
      prev_h_data = batched_h_out_data;
      batched_c_out_data = cur_c_out_data;
      batched_h_out_data = cur_h_out_data;
      batched_input_data = cur_in_data;
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batched_h_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_h_out, hidden_out);
    batched_c_out->set_lod(batched_lod);
    to_seq(dev_ctx, *batched_c_out, cell_out);
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    if (ctx.Attr<bool>("use_seq")) {
      SeqCompute(ctx);
    } else {
      BatchCompute(ctx);
    }
  }

#undef GEMM_WH_ADDON
#undef INIT_OTHER_DEFINES
#undef INIT_BASE_DEFINES
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_lstm, ops::FusionLSTMOp, ops::FusionLSTMOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_lstm, ops::FuisonLSTMKernel<float>,
                       ops::FuisonLSTMKernel<double>);
