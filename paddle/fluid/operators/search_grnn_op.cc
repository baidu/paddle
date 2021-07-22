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

#include "paddle/fluid/operators/search_grnn_op.h"
#include <algorithm>
#include <cmath>
#include <vector>
#ifndef WIN32
//#include "naive_gemm.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/dynload/mklml.h"
#endif
#include "paddle/fluid/operators/search_compute.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using LoD = framework::LoD;

#define SIGMOID(z) (sigmoid(z))
#define SIGMOID_D(a) ((a) * (1 - (a)))
#define TANHD(a) (1 - (a) * (a))

template <typename T>
T sigmoid(T z) {
  return 1 / (1 + std::exp(-z));
}

class SearchGrnnOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "X (LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("Wi", "Wi (Tensor)");
    AddInput("Wh", "Wh (Tensor)");
    AddAttr<int>("num_input", "num_input: the embedding size").SetDefault(0);
    AddAttr<int>("num_hidden", "num_hidden: the hidden size").SetDefault(0);

    AddOutput("Out",
              "Out (LoDTensor, default LoDTensor<float>) Output variable");
    AddOutput("tmp_buffer",
              "tmp_buffer (LoDTensor, default LoDTensor<float>) tmp variable");
    AddOutput("idx_sorted_by_width",
              "idx_sorted_by_width (Tensor, Tensor<int>) tmp variable");
    AddOutput(
        "layout_input",
        "layout_input (LoDTensor, default LoDTensor<float>) tmp variable");

    AddComment(R"DOC(
      SearchGrnn
      
      NOTE: only support 'float32' data type now.

    )DOC");
  }
};

class SearchGrnnOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Wi"), "Wi(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Wh"), "Wh(Input) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"), "Out(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("tmp_buffer"),
                   "tmp_buffer(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("idx_sorted_by_width"),
                   "idx_sorted_by_width(Output) should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("layout_input"),
                   "layout_input(Output) should not be null.");

    int _cap_h = ctx->Attrs().Get<int>("num_hidden");
    int _cap_e = ctx->Attrs().Get<int>("num_input");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                      "The rank of X(Input) can't be less than 2.");
    PADDLE_ENFORCE_EQ(x_dims[1], _cap_e, "x_dims[1] should be equal to _cap_e");

    auto wi_dims = ctx->GetInputDim("Wi");
    PADDLE_ENFORCE_EQ(wi_dims.size(), 3, "Wi should be 3-D tensor");
    PADDLE_ENFORCE_EQ(wi_dims[0], 3, "Wi dim[0] should be equal to 3");
    PADDLE_ENFORCE_EQ(wi_dims[1], _cap_h,
                      "wi_dims[1] should be equal to _cap_h");
    PADDLE_ENFORCE_EQ(wi_dims[2], _cap_e,
                      "wi_dims[2] should be equal to _cap_e");

    auto wh_dims = ctx->GetInputDim("Wh");
    PADDLE_ENFORCE_EQ(wh_dims.size(), 3, "Wi should be 3-D tensor");
    PADDLE_ENFORCE_EQ(wh_dims[0], 3, "Wh dim[0] should be equal to 3");
    PADDLE_ENFORCE_EQ(wh_dims[1], _cap_h,
                      "wh_dims[1] should be equal to _cap_h");
    PADDLE_ENFORCE_EQ(wh_dims[2], _cap_h,
                      "wh_dims[2] should be equal to _cap_h");

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      const auto& x_lod = x_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!x_lod.empty(), "The Input(X) must hold lod info.");

      PADDLE_ENFORCE_EQ(
          x_dims[0], static_cast<int64_t>(x_lod[0].back()),
          "The Input(X)'s lod info mismatches the actual tensor shape.");
    } else {
      std::vector<int64_t> out_dims_vec{-1};
      out_dims_vec.push_back(_cap_h);
      std::vector<int64_t> tmp_buffer_shape{20};
      tmp_buffer_shape.push_back(-1);
      tmp_buffer_shape.push_back(_cap_h);
      ctx->SetOutputDim("Out", framework::make_ddim(out_dims_vec));
      ctx->SetOutputDim("tmp_buffer", framework::make_ddim(tmp_buffer_shape));
    }

    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class CPUSearchGrnnOPKernel : public framework::OpKernel<T> {
 public:
  void prepare_layout(const framework::ExecutionContext& ctx,
                      const LoDTensor* input_blob) const {
    auto* _idx_sorted_by_width = ctx.Output<Tensor>("idx_sorted_by_width");
    auto* _layout_input = ctx.Output<LoDTensor>("layout_input");

    auto _input = input_blob;

    // usually total length
    int dim0 = _input->dims()[0];
    // if it is id only sequence
    int dim1 = 1;

    // if its a embedding like sequence (dim1 would be embedding_size)
    if (_input->dims().size() > 1) {
      dim1 = _input->dims()[1];
    }

    int batch = _input->lod()[0].size() - 1;

    auto& offset = _input->lod()[0];

    Tensor _width;
    _width.Resize(framework::make_ddim({batch}));
    _idx_sorted_by_width->Resize(framework::make_ddim({batch}));
    int* width_data = _width.mutable_data<int>(ctx.GetPlace());
    int* idx_sorted_by_width_data =
        _idx_sorted_by_width->mutable_data<int>(ctx.GetPlace());
    // sort sequence by width (descending) and find the largest width in the
    // batch
    for (int i = 0; i < batch; i++) {
      width_data[i] = offset[i + 1] - offset[i];
      idx_sorted_by_width_data[i] = i;
    }
    std::sort(idx_sorted_by_width_data, idx_sorted_by_width_data + batch,
              [&_width](int a, int b) {
                return _width.data<int>()[a] > _width.data<int>()[b];
              });
    int max_width = width_data[idx_sorted_by_width_data[0]];

    // start of reorganizing the input
    std::vector<size_t> new_offset;
    new_offset.resize(max_width + 1);

    new_offset[0] = 0;
    int j = batch - 1;
    int last_width = 0;
    int sub_row = 0;
    int sub_col = 0;

    for (int i = 1; i <= max_width;) {
      for (int k = j; k >= 0; --k) {
        if (width_data[idx_sorted_by_width_data[k]] > last_width) {
          sub_row = width_data[idx_sorted_by_width_data[k]] - last_width;
          sub_col = k + 1;

          for (int s = 0; s < sub_row; s++) {
            new_offset[i] = new_offset[i - 1] + sub_col;
            i++;
          }

          // move on
          last_width = width_data[idx_sorted_by_width_data[k]];
          j = k - 1;
          break;
        }
      }
    }

    // copying to the reorganized buffer
    if (_input->dims().size() == 1) {
      //_layout_input.reshape_batch_sequence({dim0}, new_offset);
    } else {
      //_layout_input.reshape_batch_sequence({dim0, dim1}, new_offset);

      framework::LoD new_lod;
      new_lod.push_back(new_offset);
      _layout_input->set_lod(new_lod);
      _layout_input->Resize(framework::make_ddim({dim0, dim1}));
    }

    auto* new_emb = _layout_input->mutable_data<T>(ctx.GetPlace());
    for (int i = 0; i < max_width; i++) {
      int w = new_offset[i + 1] - new_offset[i];
      auto* emb_start = new_emb + dim1 * new_offset[i];
      for (int j = 0; j < w; ++j) {
        memcpy(emb_start + dim1 * j,
               _input->data<T>() + dim1 * offset[idx_sorted_by_width_data[j]] +
                   dim1 * i,
               dim1 * sizeof(T));
      }
    }
    // end of reorganizing the input
  }

  void copy_back(const framework::ExecutionContext& ctx, T* from, T* to,
                 int step) const {
    auto* _input = ctx.Input<LoDTensor>("X");
    auto* _layout_input = ctx.Output<LoDTensor>("layout_input");
    auto* _idx_sorted_by_width = ctx.Output<Tensor>("idx_sorted_by_width");

    const auto& offset = _input->lod()[0];
    const auto& new_offset = _layout_input->lod()[0];
    const auto* idx_sorted_by_width_data = _idx_sorted_by_width->data<int>();
    for (size_t i = 0; i < _layout_input->lod()[0].size() - 1; ++i) {
      int w = new_offset[i + 1] - new_offset[i];
      for (int j = 0; j < w; j++) {
        memcpy(to + step * (offset[idx_sorted_by_width_data[j]] + i),
               from + (new_offset[i] + j) * step, step * sizeof(T));
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* wi = ctx.Input<LoDTensor>("Wi");
    auto* wh = ctx.Input<Tensor>("Wh");
    auto* top = ctx.Output<LoDTensor>("Out");
    auto* _buffer = ctx.Output<LoDTensor>("tmp_buffer");

    // std::vector<const LoDTensor*> _blobs{wi, wh};

    int _cap_h = ctx.Attr<int>("num_hidden");
    int _cap_e = ctx.Attr<int>("num_input");

    int _cap_l = bottom->dims()[0];
    int batch = bottom->lod()[0].size() - 1;

    const auto& offset = bottom->lod()[0];
    framework::LoD top_lod;
    top_lod.push_back(offset);
    top->set_lod(top_lod);
    std::vector<int64_t> top_dims_vec{_cap_l, _cap_h};
    auto* top_hidden = top->mutable_data<T>(framework::make_ddim(top_dims_vec),
                                            ctx.GetPlace());

    const auto* dense_e2h = wi->data<T>();
    const auto* dense_h2h = wh->data<T>();

    const auto* e2h = dense_e2h;
    const auto* e2hr = dense_e2h + 1 * _cap_e * _cap_h;
    const auto* e2hz = dense_e2h + 2 * _cap_e * _cap_h;
    const auto* h2h = dense_h2h;
    const auto* h2hr = dense_h2h + 1 * _cap_h * _cap_h;
    const auto* h2hz = dense_h2h + 2 * _cap_h * _cap_h;

    prepare_layout(ctx, bottom);
    auto* _layout_input = ctx.Output<LoDTensor>("layout_input");
    auto* new_emb = _layout_input->mutable_data<T>(ctx.GetPlace());
    const auto& new_offset = _layout_input->lod()[0];
    int max_width = _layout_input->lod()[0].size() - 1;

    // this buffer is used for book keeping info which will be used in bp
    // buffer also needed in bp, so make it larger
    _buffer->Resize(framework::make_ddim({20, _cap_l, _cap_h}));
    auto* buffer_data = _buffer->mutable_data<T>(ctx.GetPlace());
    auto* w_x_e = buffer_data + 0 * _cap_l * _cap_h;
    auto* wr_x_e = buffer_data + 1 * _cap_l * _cap_h;
    auto* wz_x_e = buffer_data + 2 * _cap_l * _cap_h;

    auto* u_x_h = buffer_data + 3 * _cap_l * _cap_h;
    auto* ur_x_h = buffer_data + 4 * _cap_l * _cap_h;
    auto* uz_x_h = buffer_data + 5 * _cap_l * _cap_h;

    auto* r = buffer_data + 6 * _cap_l * _cap_h;
    auto* z = buffer_data + 7 * _cap_l * _cap_h;
    auto* tilde = buffer_data + 8 * _cap_l * _cap_h;
    // the internal hidden
    auto* hidden = buffer_data + 19 * _cap_l * _cap_h;

    // precompute embedding to hidden
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    call_gemm(blas, CblasNoTrans, CblasTrans, _cap_l, _cap_h, _cap_e, 1.0f,
              new_emb, e2h, 0.0f, w_x_e);
    call_gemm(blas, CblasNoTrans, CblasTrans, _cap_l, _cap_h, _cap_e, 1.0f,
              new_emb, e2hr, 0.0f, wr_x_e);
    call_gemm(blas, CblasNoTrans, CblasTrans, _cap_l, _cap_h, _cap_e, 1.0f,
              new_emb, e2hz, 0.0f, wz_x_e);

    // precompute hidden0
    for (int i = 0; i < batch * _cap_h; i++) {
      tilde[i] = std::tanh(w_x_e[i]);
      z[i] = sigmoid<T>(wz_x_e[i]);
      hidden[i] = (1. - z[i]) * tilde[i];
    }

    // recurrence
    for (int i = 1; i < max_width; i++) {
      int w_tm1 = new_offset[i] - new_offset[i - 1];
      int w = new_offset[i + 1] - new_offset[i];

      // precompute hidden i-1 to hidden i
      auto* htm1 = hidden + new_offset[i - 1] * _cap_h;

      call_gemm(blas, CblasNoTrans, CblasTrans, w, _cap_h, _cap_h, 1.0f, htm1,
                h2h, 0.0f, u_x_h + new_offset[i] * _cap_h);
      call_gemm(blas, CblasNoTrans, CblasTrans, w, _cap_h, _cap_h, 1.0f, htm1,
                h2hr, 0.0f, ur_x_h + new_offset[i] * _cap_h);
      call_gemm(blas, CblasNoTrans, CblasTrans, w, _cap_h, _cap_h, 1.0f, htm1,
                h2hz, 0.0f, uz_x_h + new_offset[i] * _cap_h);

      // compute the gate and hidden
      for (size_t j = new_offset[i] * _cap_h; j < (new_offset[i] + w) * _cap_h;
           j++) {
        r[j] = sigmoid(wr_x_e[j] + ur_x_h[j]);
        z[j] = sigmoid(wz_x_e[j] + uz_x_h[j]);
        tilde[j] = std::tanh(w_x_e[j] + r[j] * u_x_h[j]);

        hidden[j] = z[j] * hidden[j - _cap_h * w_tm1] + (1.0 - z[j]) * tilde[j];
      }
    }

    // copy back to top
    copy_back(ctx, hidden, top_hidden, _cap_h);
  }
};

class SearchGrnnOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Wi"), "Input(Wi) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Wh"), "Input(Wh) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) of SequencePadGradOp should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Wi"))) {
      ctx->SetOutputDim(framework::GradVarName("Wi"), ctx->GetInputDim("Wi"));
    }
    if (ctx->HasOutput(framework::GradVarName("Wh"))) {
      ctx->SetOutputDim(framework::GradVarName("Wh"), ctx->GetInputDim("Wh"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename DeviceContext, typename T>
class CPUSearchGrnnOPGradKernel : public framework::OpKernel<T> {
 public:
  void do_same_layout(const framework::ExecutionContext& ctx, const T* from,
                      T* to, int step) const {
    auto* _input = ctx.Input<LoDTensor>("X");
    auto* _layout_input = ctx.Input<LoDTensor>("layout_input");
    auto& offset = _input->lod()[0];
    const auto& new_offset = _layout_input->lod()[0];
    auto* _idx_sorted_by_width = ctx.Input<Tensor>("idx_sorted_by_width");
    const int* idx_sorted_by_width_data = _idx_sorted_by_width->data<int>();

    for (int i = 0; i < _layout_input->lod()[0].size() - 1; i++) {
      int w = new_offset[i + 1] - new_offset[i];
      for (int j = 0; j < w; j++) {
        memcpy(to + (new_offset[i] + j) * step,
               from + step * (offset[idx_sorted_by_width_data[j]] + i),
               step * sizeof(T));
      }
    }
  }

  void copy_back(const framework::ExecutionContext& ctx, T* from, T* to,
                 int step) const {
    auto* _input = ctx.Input<LoDTensor>("X");
    auto* _layout_input = ctx.Input<LoDTensor>("layout_input");
    auto* _idx_sorted_by_width = ctx.Input<Tensor>("idx_sorted_by_width");

    const auto& offset = _input->lod()[0];
    const auto& new_offset = _layout_input->lod()[0];
    const auto* idx_sorted_by_width_data = _idx_sorted_by_width->data<int>();
    for (size_t i = 0; i < _layout_input->lod()[0].size() - 1; ++i) {
      int w = new_offset[i + 1] - new_offset[i];
      for (int j = 0; j < w; j++) {
        memcpy(to + step * (offset[idx_sorted_by_width_data[j]] + i),
               from + (new_offset[i] + j) * step, step * sizeof(T));
      }
    }
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* bottom = ctx.Input<LoDTensor>("X");
    auto* wi = ctx.Input<LoDTensor>("Wi");
    auto* wh = ctx.Input<Tensor>("Wh");
    auto* _buffer = ctx.Input<LoDTensor>("tmp_buffer");
    auto* _layout_input = ctx.Input<LoDTensor>("layout_input");

    // std::vector<const LoDTensor*> _blobs{wi, wh};

    int _cap_h = ctx.Attr<int>("num_hidden");
    int _cap_e = ctx.Attr<int>("num_input");
    int _cap_l = bottom->dims()[0];

    auto* d_bottom = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto* d_top = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* d_wi = ctx.Output<LoDTensor>(framework::GradVarName("Wi"));
    auto* d_wh = ctx.Output<LoDTensor>(framework::GradVarName("Wh"));

    int batch = bottom->lod()[0].size() - 1;

    const auto& new_offset = _layout_input->lod()[0];
    int max_width = _layout_input->lod()[0].size() - 1;

    // the original top and bottom pointers
    auto* top_diff = d_top->data<T>();
    auto* ediff = d_bottom->mutable_data<T>(ctx.GetPlace());

    const auto* dense_e2h = wi->data<T>();
    const auto* dense_h2h = wh->data<T>();

    auto* dense_e2h_diff = d_wi->mutable_data<T>(ctx.GetPlace());
    auto* dense_h2h_diff = d_wh->mutable_data<T>(ctx.GetPlace());
    // init parameter's diff
    memset(dense_e2h_diff, 0, 3 * _cap_e * _cap_h * sizeof(T));
    memset(dense_h2h_diff, 0, 3 * _cap_h * _cap_h * sizeof(T));

    const auto* e2h = dense_e2h;
    const auto* e2hr = dense_e2h + 1 * _cap_e * _cap_h;
    const auto* e2hz = dense_e2h + 2 * _cap_e * _cap_h;
    const auto* h2h = dense_h2h;
    const auto* h2hr = dense_h2h + 1 * _cap_h * _cap_h;
    const auto* h2hz = dense_h2h + 2 * _cap_h * _cap_h;

    auto* e2h_diff = dense_e2h_diff;
    auto* e2hr_diff = dense_e2h_diff + 1 * _cap_e * _cap_h;
    auto* e2hz_diff = dense_e2h_diff + 2 * _cap_e * _cap_h;
    auto* h2h_diff = dense_h2h_diff;
    auto* h2hr_diff = dense_h2h_diff + 1 * _cap_h * _cap_h;
    auto* h2hz_diff = dense_h2h_diff + 2 * _cap_h * _cap_h;

    auto u_x_h = _buffer->data<T>() + 3 * _cap_l * _cap_h;

    Tensor buffer_diff;
    buffer_diff.Resize(framework::make_ddim({20, _cap_l, _cap_h}));
    auto* buffer_diff_data = buffer_diff.mutable_data<T>(ctx.GetPlace());

    auto e2hdiff = buffer_diff_data + 0 * _cap_l * _cap_h;
    auto e2hrdiff = buffer_diff_data + 1 * _cap_l * _cap_h;
    auto e2hzdiff = buffer_diff_data + 2 * _cap_l * _cap_h;

    auto h2hdiff = buffer_diff_data + 3 * _cap_l * _cap_h;
    auto h2hrdiff = buffer_diff_data + 4 * _cap_l * _cap_h;
    auto h2hzdiff = buffer_diff_data + 5 * _cap_l * _cap_h;

    auto* buffer_data = _buffer->data<T>();
    auto r = buffer_data + 6 * _cap_l * _cap_h;
    auto z = buffer_data + 7 * _cap_l * _cap_h;
    auto tilde = buffer_data + 8 * _cap_l * _cap_h;

    auto d_r = buffer_diff_data + 9 * _cap_l * _cap_h;
    auto d_z = buffer_diff_data + 10 * _cap_l * _cap_h;
    auto d_tilde = buffer_diff_data + 11 * _cap_l * _cap_h;

    auto tmp_buffer = buffer_diff_data + 12 * _cap_l * _cap_h;

    auto hidden = buffer_data + 19 * _cap_l * _cap_h;
    auto hidden_diff = buffer_diff_data + 19 * _cap_l * _cap_h;
    auto embedding = _layout_input->data<T>();
    Tensor _layout_input_grad;
    _layout_input_grad.Resize(_layout_input->dims());
    auto embedding_diff = _layout_input_grad.mutable_data<T>(ctx.GetPlace());

    // copy top_hiddden diff back to the reorganized hidden, so we can use
    // segemm to back-prop the sequence
    do_same_layout(ctx, top_diff, hidden_diff, _cap_h);

    // precompute nonlinear diff
    for (int k = 0; k < new_offset[1] * _cap_h; k++) {
      d_z[k] = SIGMOID_D(z[k]);
      d_tilde[k] = TANHD(tilde[k]);
    }

    for (int k = new_offset[1] * _cap_h; k < new_offset[max_width] * _cap_h;
         k++) {
      d_r[k] = SIGMOID_D(r[k]);
      d_z[k] = SIGMOID_D(z[k]);
      d_tilde[k] = TANHD(tilde[k]);
    }

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    // back prop
    for (int i = max_width - 1; i > 0; i--) {
      int w_tm1 = new_offset[i] - new_offset[i - 1];
      int w = new_offset[i + 1] - new_offset[i];

      for (int j = new_offset[i]; j < (new_offset[i] + w); j++) {
        for (int k = 0; k < _cap_h; k++) {
          int ht = j * _cap_h + k;
          int htm1 = ht - _cap_h * w_tm1;

          T common = (1.0 - z[ht]) * d_tilde[ht] * hidden_diff[ht];

          h2hdiff[htm1] = common * r[ht];
          h2hrdiff[htm1] = common * u_x_h[ht] * d_r[ht];
          h2hzdiff[htm1] =
              (hidden[htm1] - tilde[ht]) * d_z[ht] * hidden_diff[ht];

          e2hdiff[ht] = common;
          e2hrdiff[ht] = h2hrdiff[htm1];
          e2hzdiff[ht] = h2hzdiff[htm1];
        }
      }

      auto* hidden_htm1 = hidden + new_offset[i - 1] * _cap_h;
      auto* h2hdiff_htm1 = h2hdiff + new_offset[i - 1] * _cap_h;
      auto* h2hrdiff_htm1 = h2hrdiff + new_offset[i - 1] * _cap_h;
      auto* h2hzdiff_htm1 = h2hzdiff + new_offset[i - 1] * _cap_h;

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_h, w, (T)1.0,
                h2hdiff_htm1, hidden_htm1, (T)1.0, h2h_diff);

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_h, w, (T)1.0,
                h2hrdiff_htm1, hidden_htm1, (T)1.0, h2hr_diff);

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_h, w, (T)1.0,
                h2hzdiff_htm1, hidden_htm1, (T)1.0, h2hz_diff);

      auto* embedding_et = embedding + new_offset[i] * _cap_e;
      auto* e2hdiff_ht = e2hdiff + new_offset[i] * _cap_h;
      auto* e2hrdiff_ht = e2hrdiff + new_offset[i] * _cap_h;
      auto* e2hzdiff_ht = e2hzdiff + new_offset[i] * _cap_h;

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_e, w, (T)1.0,
                e2hdiff_ht, embedding_et, (T)1.0, e2h_diff);

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_e, w, (T)1.0,
                e2hrdiff_ht, embedding_et, (T)1.0, e2hr_diff);

      call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_e, w, (T)1.0,
                e2hzdiff_ht, embedding_et, (T)1.0, e2hz_diff);

      sse_eltmul(z + new_offset[i] * _cap_h,
                 hidden_diff + new_offset[i] * _cap_h,
                 tmp_buffer + new_offset[i - 1] * _cap_h, _cap_h * w);
      // add this with diff from top
      sse_eltadd(hidden_diff + new_offset[i - 1] * _cap_h,
                 tmp_buffer + new_offset[i - 1] * _cap_h,
                 hidden_diff + new_offset[i - 1] * _cap_h, _cap_h * w);

      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_h, _cap_h, (T)1.0,
                h2hdiff_htm1, h2h, (T)1.0,
                hidden_diff + new_offset[i - 1] * _cap_h);
      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_h, _cap_h, (T)1.0,
                h2hrdiff_htm1, h2hr, (T)1.0,
                hidden_diff + new_offset[i - 1] * _cap_h);
      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_h, _cap_h, (T)1.0,
                h2hzdiff_htm1, h2hz, (T)1.0,
                hidden_diff + new_offset[i - 1] * _cap_h);

      // bp embedding diff
      auto* embedding_diff_et = embedding_diff + new_offset[i] * _cap_e;

      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_e, _cap_h, (T)1.0,
                e2hdiff_ht, e2h, (T)0.0, embedding_diff_et);

      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_e, _cap_h, (T)1.0,
                e2hrdiff_ht, e2hr, (T)1.0, embedding_diff_et);

      call_gemm(blas, CblasNoTrans, CblasNoTrans, w, _cap_e, _cap_h, (T)1.0,
                e2hzdiff_ht, e2hz, (T)1.0, embedding_diff_et);
    }

    for (int i = 0; i < batch * _cap_h; i++) {
      e2hdiff[i] = (1. - z[i]) * d_tilde[i] * hidden_diff[i];
      e2hzdiff[i] = (-tilde[i]) * d_z[i] * hidden_diff[i];
    }
    call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_e, batch, (T)1.0,
              e2hdiff, embedding, (T)1.0, e2h_diff);
    call_gemm(blas, CblasTrans, CblasNoTrans, _cap_h, _cap_e, batch, (T)1.0,
              e2hzdiff, embedding, (T)1.0, e2hz_diff);

    call_gemm(blas, CblasNoTrans, CblasNoTrans, batch, _cap_e, _cap_h, (T)1.0,
              e2hdiff, e2h, (T)0.0, embedding_diff);
    call_gemm(blas, CblasNoTrans, CblasNoTrans, batch, _cap_e, _cap_h, (T)1.0,
              e2hzdiff, e2hz, (T)1.0, embedding_diff);

    // copy back to original embedding diff, and hidden diff (probablly no use,
    // but for safety)
    copy_back(ctx, embedding_diff, ediff, _cap_e);
    //_layout_helper.copy_back(hidden_diff, top_diff, _cap_h);
  }
};

template <typename T>
class SearchGrnnOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("search_grnn_grad");
    retv->SetInput("X", this->Input("X"));
    retv->SetInput("Wi", this->Input("Wi"));
    retv->SetInput("Wh", this->Input("Wh"));
    retv->SetInput("layout_input", this->Output("layout_input"));
    retv->SetInput("tmp_buffer", this->Output("tmp_buffer"));
    retv->SetInput("idx_sorted_by_width", this->Output("idx_sorted_by_width"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));

    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Wi"), this->InputGrad("Wi"));
    retv->SetOutput(framework::GradVarName("Wh"), this->InputGrad("Wh"));

    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;
namespace frm = paddle::framework;
REGISTER_OPERATOR(search_grnn, ops::SearchGrnnOP, ops::SearchGrnnOpMaker,
                  ops::SearchGrnnOpGradMaker<paddle::framework::OpDesc>,
                  ops::SearchGrnnOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(search_grnn_grad, ops::SearchGrnnOpGrad);

REGISTER_OP_CPU_KERNEL(
    search_grnn, ops::CPUSearchGrnnOPKernel<plt::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    search_grnn_grad,
    ops::CPUSearchGrnnOPGradKernel<plt::CPUDeviceContext, float>);