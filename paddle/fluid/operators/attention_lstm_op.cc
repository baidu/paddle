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

#include "paddle/fluid/operators/attention_lstm_op.h"

#include <string>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {

void AttentionLSTMOp::InferShape(framework::InferShapeContext* ctx) const {
  OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasInput("C0"), "Input", "C0", "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasInput("LSTMWeight"), "Input", "LSTMWeight",
                 "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasInput("LSTMBias"), "Input", "LSTMBias",
                 "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasInput("AttentionWeight"), "Input", "AttentionWeight",
                 "AttentionLstm");

  OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasOutput("Cell"), "Output", "Cell", "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasOutput("AttentionedX"), "Output", "AttentionedX",
                 "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasOutput("AttentionFCOut"), "Output", "AttentionFCOut",
                 "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasOutput("LSTMX"), "Output", "LSTMX", "AttentionLstm");
  OP_INOUT_CHECK(ctx->HasOutput("LSTMOUT"), "Output", "LSTMOUT",
                 "AttentionLstm");

  auto x_dims = ctx->GetInputDim("X");
  const int M = x_dims[1];
  PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "Expected input(X)'s dimension is 2. But received %d.",
                        x_dims.size()));

  auto w_dims = ctx->GetInputDim("LSTMWeight");
  const int D = w_dims[1] / 4;
  PADDLE_ENFORCE_EQ(
      w_dims.size(), 2,
      platform::errors::InvalidArgument(
          "Expected input(LSTMWeight)'s dimension is 2.But received %d.",
          w_dims.size()));
  PADDLE_ENFORCE_EQ(
      w_dims[0], D + M,
      platform::errors::InvalidArgument(
          "LSTMWeight dims should be (%d + %d) * %d.", D, M, 4 * D));

  auto b_dims = ctx->GetInputDim("LSTMBias");
  PADDLE_ENFORCE_EQ(
      b_dims.size(), 2,
      platform::errors::InvalidArgument("Input(LSTMBias)'s rank must be 2."));
  PADDLE_ENFORCE_EQ(b_dims[0], 1,
                    platform::errors::InvalidArgument(
                        "LSTMBias dims should be 1 x %d.", 4 * D));
  PADDLE_ENFORCE_EQ(b_dims[1], 4 * D,
                    platform::errors::InvalidArgument(
                        "LSTMBias dims should be 1 x %d.", 4 * D));

  auto c_dims = ctx->GetInputDim("C0");
  PADDLE_ENFORCE_EQ(
      c_dims.size(), 2,
      platform::errors::InvalidArgument("Input(C0)'s rank must be 2."));
  if (ctx->IsRuntime()) {
    PADDLE_ENFORCE_EQ(
        c_dims[1], D,
        platform::errors::InvalidArgument("C0 dims should be N x %d.", D));
  }

  if (ctx->HasInput("H0")) {
    auto h_dims = ctx->GetInputDim("H0");
    PADDLE_ENFORCE_EQ(
        h_dims.size(), 2UL,
        platform::errors::InvalidArgument(
            "Expected input(H0)'s dimension is 2. But received %d.",
            h_dims.size()));
    if (ctx->IsRuntime() ||
        (framework::product(c_dims) > 0 && framework::product(h_dims) > 0)) {
      PADDLE_ENFORCE_EQ(h_dims, c_dims,
                        platform::errors::InvalidArgument(
                            "The dimension of Input(H0) and Input(C0) "
                            "should be the same."));
    }
  }

  auto atten_w_dims = ctx->GetInputDim("AttentionWeight");
  PADDLE_ENFORCE_EQ(atten_w_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "Input(AttentionWeight)'s rank must be 2."));
  PADDLE_ENFORCE_EQ(atten_w_dims[0], M + D,
                    platform::errors::InvalidArgument(
                        "Expected `AttentionWeight` shape is [(%d + %d), 1]. "
                        "But received shape = [%d, 1], shape[0] is not %d.",
                        M, D, atten_w_dims[0], M + D));
  PADDLE_ENFORCE_EQ(atten_w_dims[1], 1,
                    platform::errors::InvalidArgument(
                        "AttentionWeight shapes must be (%d + %d) * 1.", M, D));

  if (ctx->HasInput("AttentionBias")) {
    auto atten_b_dims = ctx->GetInputDim("AttentionBias");
    PADDLE_ENFORCE_EQ(atten_b_dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "Input(AttentionBias)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(atten_b_dims[0], 1,
                      platform::errors::InvalidArgument(
                          "AttentionBias shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(atten_b_dims[1], 1,
                      platform::errors::InvalidArgument(
                          "AttentionBias shapes must be 1 * 1."));
  }

  if (ctx->HasInput("AttentionScalar")) {
    auto dims = ctx->GetInputDim("AttentionScalar");
    PADDLE_ENFORCE_EQ(dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "Input(AttentionScalar)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(dims[0], 1,
                      platform::errors::InvalidArgument(
                          "AttentionScalar shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(dims[1], 1,
                      platform::errors::InvalidArgument(
                          "AttentionScalar shapes must be 1 * 1."));
  }

  if (ctx->HasInput("AttentionScalarBias")) {
    auto dims = ctx->GetInputDim("AttentionScalarBias");
    OP_INOUT_CHECK(ctx->HasInput("AttentionScalar"), "Input", "AttentionScalar",
                   "AttentionLstm");
    PADDLE_ENFORCE_EQ(dims.size(), 2,
                      platform::errors::InvalidArgument(
                          "Input(AttentionScalarBias)'s rank must be 2."));
    PADDLE_ENFORCE_EQ(dims[0], 1,
                      platform::errors::InvalidArgument(
                          "AttentionScalarBias shapes must be 1 * 1."));
    PADDLE_ENFORCE_EQ(dims[1], 1,
                      platform::errors::InvalidArgument(
                          "AttentionScalarBias shapes must be 1 * 1."));
  }

  framework::DDim out_dims({x_dims[0], D});
  ctx->SetOutputDim("Hidden", out_dims);
  ctx->SetOutputDim("Cell", out_dims);
  ctx->SetOutputDim("AttentionedX", {x_dims[0], 1});
  ctx->SetOutputDim("LSTMX", {1, M});
  ctx->SetOutputDim("LSTMOUT", {1, 4 * D});
  // AttentionFCOut should be reshape as (maxseqlen,1) in runtime
  ctx->ShareLoD("X", "Hidden");
  ctx->ShareLoD("X", "Cell");
}

framework::OpKernelType AttentionLSTMOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context());
}

void AttentionLSTMOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) the input is a LodTensor, which support "
           "variable-time length input sequence. The underlying tensor in "
           "this LoDTensor is a matrix with shape (T X M), where T is the "
           "total time steps in this mini-batch, M is the dim size of x.");
  AddInput("C0",
           "(Tensor) LSTM C0"
           "This is a tensor with shape (N x D), where N is the batch size, D "
           "is the gate size."
           "C0 is necessary because of attention.");
  AddInput("H0",
           "(Tensor, optional) LSTM H0"
           "This is a tensor with shape (N x D), where N is the "
           "batch size and D is the gate size.")
      .AsDispensable();
  AddInput("AttentionWeight",
           "(Tensor) the weights of attention fc. Always relu the fc result."
           "The shape is ((M+D) x 1), where M is the dim size of x, D is the "
           "gate size of LSTM.");
  AddInput("AttentionBias",
           "(Tensor, optional) the bias of attention fc."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("AttentionScalar",
           "(Tensor, optional) the scalar on the result of attentioned fc. "
           "Always relu the Scalar."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("AttentionScalarBias",
           "(Tensor, optional) the scalar bias of attention fc."
           "The shape is (1 x 1)")
      .AsDispensable();
  AddInput("LSTMWeight",
           "(Tensor) the combined weight of LSTM"
           " - The shape is ((D+M) x 4D), where D is the hidden gate size, M "
           "is the dim size of x"
           " - Weight = {W_forget, W_input, W_output, W_cell}");
  AddInput("LSTMBias",
           "(Tensor) the combined bias of LSTM, shape (1x4D)."
           "Note: we should add the bias of hidden and context accorindg to "
           "the same gate: "
           "{B_forget, B_input, B_output, B_cell}");
  AddOutput("Hidden",
            "(LoDTensor) (same as LSTMOp) the hidden state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("Cell",
            "(LoDTensor) (same as LSTMOp) the cell state of LSTM operator. "
            "The shape is (T x D), and lod is the same with the `Input`.");
  AddOutput("AttentionedX",
            "(Tensor) shape is (T x 1), the result after X * AttentionWeight,"
            " where T is the total time steps in this mini-batch,"
            " D is the hidden size.")
      .AsIntermediate();
  AddOutput("AttentionFCOut",
            "(Tensor) (max_seq_len, 1), compute at each step.")
      .AsIntermediate();
  AddOutput("LSTMX",
            "(Tensor) the input X of LSTM for each step."
            "Shape is (1 x M), where M is the x frame size")
      .AsIntermediate();
  AddOutput(
      "LSTMOUT",
      "(Tensor) the output of LSTM X(1*(D+M))* weight((D+M)*4D) for each step."
      "Shape is (1 x 4D), where M is the x frame size")
      .AsIntermediate();
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
  AddComment(R"DOC(
Attention Long-Short Term Memory (LSTM) Operator.

Attention part:
concat( x(seqlen * M), expand( cell_t-1(1,D) ) ) => tmp(seqlen*(M+D))

tmp(seqlen*(M+D)) * fc((M+D)*1) => fcout(seqlen*1) with bias, relu

fcout(seqlen*1) * scalar => fcout(seqlen*1) with bias, relu

dotmul and sum pool ( fcout(seqlen*1), x(seqlen * M) ) => lstm_x_t(1, M) 

LSTM part:
use lstm_x_t as input and compute as standard LSTM.

)DOC");
}

// y[i] = (x[i] + bias[0]) > 0 ? (x[i] + bias[0]) : 0;
template <typename T>
inline void bias_relu(const int n, const T* x, const T* bias, T* y) {
  if (bias) {
    math::vec_add_bias<T, platform::avx>(n, *bias, x, y);
    math::vec_relu<T, platform::avx>(n, y, y);
  } else {
    math::vec_relu<T, platform::avx>(n, x, y);
  }
}

template <typename T>
inline void vec_softmax(const int n, const T* x, T* y) {
  T scalar = x[0];
  // max
  for (int i = 1; i < n; ++i) {
    scalar = scalar < x[i] ? x[i] : scalar;
  }
  math::vec_add_bias<T, platform::avx>(n, -scalar, x, y);  // sub
  math::vec_exp<T>(n, y, y);                               // exp
  // sum
  scalar = T(0);
  for (int i = 0; i < n; ++i) {
    scalar += y[i];
  }
  math::vec_scal<T>(n, static_cast<T>(1) / scalar, y);  // scale
}

template <typename T>
class AttentionLSTMKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using DeviceContext = paddle::platform::CPUDeviceContext;

    auto* x = ctx.Input<LoDTensor>("X");
    auto* h0 = ctx.Input<Tensor>("H0");
    auto* c0 = ctx.Input<Tensor>("C0");
    auto* atten_w = ctx.Input<Tensor>("AttentionWeight");
    auto* atten_b = ctx.Input<Tensor>("AttentionBias");
    auto* atten_scalar = ctx.Input<Tensor>("AttentionScalar");
    auto* atten_scalar_bias = ctx.Input<Tensor>("AttentionScalarBias");
    auto* lstm_w = ctx.Input<Tensor>("LSTMWeight");
    auto* lstm_b = ctx.Input<Tensor>("LSTMBias");

    auto* hidden_out = ctx.Output<LoDTensor>("Hidden");
    auto* cell_out = ctx.Output<LoDTensor>("Cell");
    auto* atted_x = ctx.Output<Tensor>("AttentionedX");
    auto* fc_out = ctx.Output<Tensor>("AttentionFCOut");
    auto* lstm_x = ctx.Output<Tensor>("LSTMX");
    auto* lstm_out = ctx.Output<Tensor>("LSTMOUT");

    // some shape should be reshape here since infershape can not get lod info
    auto x_lod = x->lod();
    const int N = x_lod[0].size() - 1;  // batch size
    auto x_dims = x->dims();            // T x M
    auto w_dims = lstm_w->dims();       // (D+M) x 4D
    const int total_T = x_dims[0];
    const int M = x_dims[1];      // x frame size
    const int D = w_dims[1] / 4;  // gate frame size
    const int D2 = D * 2;
    const int D3 = D * 3;
    const int D4 = w_dims[1];
    int max_seq_len = x_lod[0][1];
    for (int i = 1; i < N; ++i) {
      int len = x_lod[0][i + 1] - x_lod[0][i];
      max_seq_len = max_seq_len < len ? len : max_seq_len;
    }
    PADDLE_ENFORCE_EQ(
        x_lod.size(), 1UL,
        platform::errors::InvalidArgument("Input(X)'s lod size must be 1."));
    PADDLE_ENFORCE_EQ(
        c0->dims()[0], N,
        platform::errors::InvalidArgument("C0 dims should be %d x %d.", N, D));
    fc_out->Resize({max_seq_len, 1});

    std::function<void(const int, const T*, T*)> act_gate, act_cell, act_cand;
    auto& act_gate_str = ctx.Attr<std::string>("gate_activation");
    auto& act_cell_str = ctx.Attr<std::string>("cell_activation");
    auto& act_cand_str = ctx.Attr<std::string>("candidate_activation");
    if (platform::MayIUse(platform::avx)) {
      math::VecActivations<T, platform::avx> act_functor;
      act_gate = act_functor(act_gate_str);
      act_cell = act_functor(act_cell_str);
      act_cand = act_functor(act_cand_str);
    } else {
      math::VecActivations<T, platform::isa_any> act_functor;
      act_gate = act_functor(act_gate_str);
      act_cell = act_functor(act_cell_str);
      act_cand = act_functor(act_cand_str);
    }

    const T* x_data = x->data<T>();
    const T* h0_data = h0 ? h0->data<T>() : nullptr;
    const T* c0_data = c0->data<T>();
    const T* lstm_w_data = lstm_w->data<T>();
    const T* lstm_b_data = lstm_b->data<T>();
    const T* atten_w_data = atten_w->data<T>();
    const T* atten_b_data = atten_b ? atten_b->data<T>() : nullptr;
    const T* atten_scalar_data =
        atten_scalar ? atten_scalar->data<T>() : nullptr;
    const T* atten_scalar_bias_data =
        atten_scalar_bias ? atten_scalar_bias->data<T>() : nullptr;

    T* hidden_out_data = hidden_out->mutable_data<T>(ctx.GetPlace());
    T* cell_out_data = cell_out->mutable_data<T>(ctx.GetPlace());
    T* atted_x_data = atted_x->mutable_data<T>(ctx.GetPlace());
    T* fc_out_data = fc_out->mutable_data<T>(ctx.GetPlace());
    T* lstm_x_data = lstm_x->mutable_data<T>(ctx.GetPlace());
    T* lstm_out_data = lstm_out->mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);

    // x(TxM) * fc (Mx1) part of atten_wgt(M+D)x1
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    math::FCFunctor<DeviceContext, T> fc;
    fc(dev_ctx, total_T, 1, M, x_data, atten_w_data, atted_x_data,
       atten_b_data);

    const T* cur_atten_x_data = atted_x_data;
    const T* cur_x_data = x_data;
    const T* prev_cell_data = nullptr;
    const T* prev_hidden_data = nullptr;
    T* cur_cell_out_data = cell_out_data;
    T* cur_hidden_out_data = hidden_out_data;
    for (int i = 0; i < N; ++i) {
      int seq_len = x_lod[0][i + 1] - x_lod[0][i];
      prev_cell_data = c0_data + i * D;
      prev_hidden_data = h0_data ? h0_data + i * D : nullptr;
      for (int step = 0; step < seq_len; ++step) {
        /// 1. compute attention vector
        // 1a. prev_cell(1xD) * fc(D) rest part of atten_wgt
        T prev_cell_bias = blas.DOT(D, prev_cell_data, atten_w_data + M);
        // 1b. add cell bias and relu
        bias_relu<T>(seq_len, cur_atten_x_data, &prev_cell_bias, fc_out_data);
        // 1c. fc scalar
        if (atten_scalar_data) {
          blas.SCAL(seq_len, *atten_scalar_data, fc_out_data);
          bias_relu<T>(seq_len, fc_out_data, atten_scalar_bias_data,
                       fc_out_data);
        }
        // 1d. softmax
        vec_softmax<T>(seq_len, fc_out_data, fc_out_data);
        // mul x(seq_len*M) and sum pool
        fc(dev_ctx, 1, M, seq_len, fc_out_data, cur_x_data, lstm_x_data);

        /// 2. compute LSTM step
        // lstm weight : concat[forget , input , output , tilde]
        // shape : (D + M) x (4 * D)
        // fc inputX(1xM) * weightX(M*(4D))  => 1 x 4D
        blas.MatMul(1, D4, M, lstm_x_data, lstm_w_data + D * D4, lstm_out_data);
        if (prev_hidden_data) {
          blas.GEMM(CblasNoTrans, CblasNoTrans, 1, D4, D, static_cast<T>(1),
                    prev_hidden_data, D, lstm_w_data, D4, static_cast<T>(1),
                    lstm_out_data, D4);
        }
        // since input is 1xM, so can use add bias
        blas.VADD(D4, lstm_b_data, lstm_out_data, lstm_out_data);

        // gate act: sigmoid
        act_gate(D3, lstm_out_data, lstm_out_data);
        // candicate act: tanh
        act_cand(D, lstm_out_data + D3, lstm_out_data + D3);

        // a = forget * prev_cell
        blas.VMUL(D, lstm_out_data, prev_cell_data, lstm_out_data);

        // b = input * tilde
        blas.VMUL(D, lstm_out_data + D, lstm_out_data + D3, lstm_out_data + D);

        // cell_out = a + b
        blas.VADD(D, lstm_out_data, lstm_out_data + D, cur_cell_out_data);

        // state act tanh(cell_out) * output_gate
        act_cell(D, cur_cell_out_data, lstm_out_data);
        blas.VMUL(D, lstm_out_data, lstm_out_data + D2, cur_hidden_out_data);

        prev_hidden_data = cur_hidden_out_data;
        prev_cell_data = cur_cell_out_data;
        cur_cell_out_data = cur_cell_out_data + D;
        cur_hidden_out_data = cur_hidden_out_data + D;
      }
      cur_x_data = cur_x_data + seq_len * M;
      cur_atten_x_data = cur_atten_x_data + seq_len;
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(attention_lstm, ops::AttentionLSTMOp,
                  ops::AttentionLSTMOpMaker);

REGISTER_OP_CPU_KERNEL(attention_lstm, ops::AttentionLSTMKernel<float>,
                       ops::AttentionLSTMKernel<double>);
