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
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

class CudnnLSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasInput("InitH"), "Input", "InitH", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasInput("InitC"), "Input", "InitC", "CudnnLSTM");

    OP_INOUT_CHECK(ctx->HasOutput("Reserve"), "Output", "Reserve", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("StateOut"), "Output", "StateOut",
                   "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("LastH"), "Output", "LastH", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("LastC"), "Output", "LastC", "CudnnLSTM");

    auto in_dims = ctx->GetInputDim("Input");
    auto init_h_dims = ctx->GetInputDim("InitH");
    auto init_c_dims = ctx->GetInputDim("InitC");

    PADDLE_ENFORCE_EQ(in_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of Input in CudnnLSTM  must be 3. But "
                          "received Input's rank is %d.",
                          in_dims.size()));
    PADDLE_ENFORCE_EQ(init_h_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of InitH in CudnnLSTM  must be 3. But "
                          "received InitH's rank is %d.",
                          init_h_dims.size()));

    if (ctx->HasInput("SequenceLength")) {
      auto seq_dims = ctx->GetInputDim("SequenceLength");
      PADDLE_ENFORCE_EQ(
          in_dims[1], seq_dims[0],
          platform::errors::InvalidArgument(
              "The size of SequenceLength has to equal the batch_size. But "
              "received batch_size is %d and the size of SequenceLength is %d.",
              in_dims[1], seq_dims[0]));
    }

    PADDLE_ENFORCE_EQ(
        in_dims[1], init_h_dims[1],
        platform::errors::InvalidArgument(
            "The in_dims[1] (Input dims) and init_h_dims[1] (InitH "
            "dims) should be equal. But "
            "received in_dims[1] is %d and init_h_dims[1] is %d.",
            in_dims[1], init_h_dims[1]));

    PADDLE_ENFORCE_EQ(init_c_dims, init_h_dims,
                      platform::errors::InvalidArgument(
                          "The InitC dims and InitH "
                          "dims should be equal. But "
                          "received init_c_dims is %d and init_h_dims is %d.",
                          init_c_dims, init_h_dims));

    auto out_dims = in_dims;
    auto hidden_size = ctx->Attrs().Get<int>("hidden_size");
    bool is_bidirec = ctx->Attrs().Get<bool>("is_bidirec");
    out_dims[2] = is_bidirec ? hidden_size * 2 : hidden_size;
    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("LastH", init_c_dims);
    ctx->SetOutputDim("LastC", init_h_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class CudnnLSTMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) RNN input tensor, which support variable-time length input "
        "sequence."
        "The shape of the Tensor MUST be ( seq_len * batch_size * input_size)"
        "seq_len is the total time step in this mini-batch (CAN be change in "
        "different batch)"
        "batch_size is the instance number of this batch"
        "input_size is the hidden size of the input."
        "input_size and the hidden_size in the next may not be same");
    AddInput("InitH",
             "(Tensor) the initial hidden state of the LSTM"
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)");
    AddInput("InitC",
             "(Tensor) the initial cell state of the LSTm "
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)");
    AddInput("W",
             "(Tensor) the learnable hidden-hidden weights."
             " The shape is (N), where N is total weight size of the LSTM. "
             " cudnn concatenate all the weight to one Tensor")
        .AsDispensable();
    AddInput("WeightList",
             "(vector<Tensor>), stores weight and bias data when the weight "
             "use the list format. ")
        .AsDispensable()
        .AsDuplicable();
    AddInput("SequenceLength",
             "(Tensor) When the input data is padding, "
             "set this parameter. This parameter represents "
             "the variable sequence lengths in a batch. "
             "The size of the vector has to equal the batch_size.")
        .AsDispensable();
    AddOutput("Reserve",
              "(Tensor, a temporary output Tensor to store the reserve_data "
              "of cudnn kernel.")
        .AsIntermediate();
    AddOutput("StateOut",
              "Share memory with State. "
              "Store the global drop state when training");
    AddOutput("Out",
              "(Tensor) the hidden state of LSTM operator. "
              "The shape is ( seq_len x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be ( seq_len x "
              "batch_size x hidden_size * 2) ");
    AddOutput("LastH",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)");
    AddOutput("LastC",
              "(Tensor) the cell state of the last step"
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirect is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size*2)");
    AddAttr<float>(
        "dropout_prob",
        "dropout prob of the dropout op"
        "the dropout ONLY work between lstm layers, not between time steps"
        "There is no dropout work on the Out tensor")
        .SetDefault(0.0);
    AddAttr<bool>("is_bidirec",
                  "is_bidirec"
                  "if it is bidirectional rnn"
                  "The will affect the shape of the Out, LastH, and LastC")
        .SetDefault(false);
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(10);
    AddAttr<int>("hidden_size", "hidden size of the LSTM").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number of the LSTM")
        .SetDefault(1);
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(0);
    AddComment(R"DOC(
CUDNN LSTM implementation

A four-gate Long Short-Term Memory network with no peephole connections.
In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1, 
the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:

$$ i_t = sigmoid(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i) $$

$$ f_t = sigmoid(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f) $$

$$ o_t = sigmoid(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o) $$

$$ \\tilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c) $$

$$ c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t} $$

$$ h_t = o_t \\odot tanh(c_t) $$

- W terms denote weight matrices (e.g. $W_{ix}$ is the matrix
  of weights from the input gate to the input)
- The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
- sigmoid is the logistic sigmoid function.
- $i, f, o$ and $c$ are the input gate, forget gate, output gate,
  and cell activation vectors, respectively, all of which have the same size as
  the cell output activation vector $h$.
- The $\odot$ is the element-wise product of the vectors.
- `tanh` is the activation functions.
- $\tilde{c_t}$ is also called candidate hidden state,
  which is computed based on the current input and the previous hidden state.

Where sigmoid is the sigmoid operator: sigmoid(x) = 1 / (1 + e^-x), * represents a point-wise multiplication, 
X represensts a matrix multiplication


)DOC");
  }
};

class CudnnLSTMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "CudnnLSTMGrad");
    OP_INOUT_CHECK(ctx->HasInput("InitH"), "Input", "InitH", "CudnnLSTMGrad");
    OP_INOUT_CHECK(ctx->HasInput("InitC"), "Input", "InitC", "CudnnLSTMGrad");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name)) {
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
      }
    };

    SetOutGradDim("Input");
    if (ctx->HasInputs("WeightList")) {
      ctx->SetOutputsDim(framework::GradVarName("WeightList"),
                         ctx->GetInputsDim("WeightList"));
    }
    SetOutGradDim("InitH");
    SetOutGradDim("InitC");
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class CudnnLSTMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("cudnn_lstm_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("InitH", this->Input("InitH"));
    op->SetInput("InitC", this->Input("InitC"));
    if (this->HasInput("WeightList")) {
      op->SetInput("WeightList", this->Input("WeightList"));
    }
    if (this->HasInput("SequenceLength")) {
      op->SetInput("SequenceLength", this->Input("SequenceLength"));
    }
    op->SetInput("Reserve", this->Output("Reserve"));
    op->SetInput("StateOut", this->Output("StateOut"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput(framework::GradVarName("LastC"), this->OutputGrad("LastC"));
    op->SetInput(framework::GradVarName("LastH"), this->OutputGrad("LastH"));

    if (this->HasInput("WeightList")) {
      op->SetOutput(framework::GradVarName("WeightList"),
                    this->InputGrad("WeightList", false));
    }

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("InitH"), this->InputGrad("InitH"));
    op->SetOutput(framework::GradVarName("InitC"), this->InputGrad("InitC"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class NotImpleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unimplemented(
        "CPU is not support for this kernel now. Will be add in the future"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cudnn_lstm, ops::CudnnLSTMOp, ops::CudnnLSTMOpMaker,
                  ops::CudnnLSTMGradOpMaker<paddle::framework::OpDesc>,
                  ops::CudnnLSTMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(cudnn_lstm_grad, ops::CudnnLSTMGradOp);

REGISTER_OP_CPU_KERNEL(cudnn_lstm, ops::NotImpleKernel<float>);
REGISTER_OP_CPU_KERNEL(cudnn_lstm_grad, ops::NotImpleKernel<float>);

// TODO(Shixiaowei02) Add ModifyInput support
REGISTER_OP_VERSION(cudnn_lstm)
    .AddCheckpoint(
        R"ROC(
              Upgrade cudnn_lstm add a new input [WeightList] and modify input [W] to dispensable.)ROC",
        paddle::framework::compatible::OpVersionDesc()
            .NewInput(
                "WeightList",
                "The WeightList stores weight and bias data. WeightList is "
                "dispensable.")
            .NewInput("SequenceLength",
                      "When the input data is padding, set this parameter. "
                      "SequenceLength is dispensable.")
            .NewOutput("StateOut", "Store the global drop state when training")
            .NewOutput("Reserve",
                       "A temporary output Tensor to store the reserve_data"));
