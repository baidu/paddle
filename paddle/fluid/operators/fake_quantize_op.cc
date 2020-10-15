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

#include "paddle/fluid/operators/fake_quantize_op.h"
#include <algorithm>
#include <string>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/clip_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
struct Compare {
 public:
  bool operator()(const T a, const T b) { return (std::abs(a) < std::abs(b)); }
};

template <typename T>
struct FindAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx, const T* in,
                  const int num, T* out) {
    *out = std::abs(*(std::max_element(in + 0, in + num, Compare<T>())));
  }
};

template struct FindAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct FindChannelAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in_tensor, const int quant_axis,
                  T* out_abs_max) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    auto* in_data = in_tensor.data<T>();
    auto in_dims = in_tensor.dims();
    const int64_t channel = in_dims[quant_axis];
    if (quant_axis == 0) {
      const int64_t channel_size = in_tensor.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        auto* start = in_data + i * channel_size;
        auto* end = in_data + (i + 1) * channel_size;
        out_abs_max[i] =
            std::abs(*(std::max_element(start, end, Compare<T>())));
      }
    } else if (quant_axis == 1) {
      for (int64_t i = 0; i < channel; i++) {
        out_abs_max[i] = 0;
      }
      const int64_t step_i = in_tensor.numel() / in_dims[0];
      const int64_t step_j = in_tensor.numel() / (in_dims[0] * in_dims[1]);
      for (int64_t i = 0; i < in_dims[0]; i++) {
        for (int64_t j = 0; j < in_dims[1]; j++) {
          auto* start = in_data + i * step_i + j * step_j;
          auto* end = in_data + i * step_i + (j + 1) * step_j;
          T abs_max = std::abs(*(std::max_element(start, end, Compare<T>())));
          out_abs_max[j] = std::max(out_abs_max[j], abs_max);
        }
      }
    }
  }
};

template struct FindChannelAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    T s = scale.data<T>()[0];
    T inv_s = inverse(s);
    platform::Transform<platform::CPUDeviceContext> trans;
    trans(ctx, in.data<T>(), in.data<T>() + in.numel(),
          out->mutable_data<T>(ctx.GetPlace()), ClipFunctor<T>(-s, s));
    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
  }
};

template struct ClipAndFakeQuantFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct ClipAndFakeQuantDequantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, framework::Tensor* out) {
    T s = scale.data<T>()[0];
    T inv_s = inverse(s);

    platform::Transform<platform::CPUDeviceContext> trans;
    trans(ctx, in.data<T>(), in.data<T>() + in.numel(),
          out->mutable_data<T>(ctx.GetPlace()), ClipFunctor<T>(-s, s));
    auto out_e = framework::EigenVector<T>::Flatten(*out);
    out_e.device(*ctx.eigen_device()) =
        (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
  }
};
template struct ClipAndFakeQuantDequantFunctor<platform::CPUDeviceContext,
                                               float>;

template <typename T>
struct ChannelClipAndFakeQuantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    // At present, channelwise quantization supports conv2d, depthwise_conv2d
    // conv2d_transpose and mul
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));
    auto* scale_data = scale.data<T>();
    auto* in_data = in.data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in_dims = in.dims();
    const int64_t channel = in_dims[quant_axis];
    platform::Transform<platform::CPUDeviceContext> trans;
    if (quant_axis == 0) {
      const int64_t channel_size = in.numel() / channel;
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        auto* start = in_data + i * channel_size;
        auto* end = in_data + (i + 1) * channel_size;
        trans(ctx, start, end, out_data + i * channel_size,
              ClipFunctor<T>(-s, s));
      }
      for (int64_t i = 0; i < channel; i++) {
        T s = scale_data[i];
        T inv_s = inverse(s);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        auto out_e = framework::EigenVector<T>::Flatten(one_channel_out);
        out_e.device(*ctx.eigen_device()) = (bin_cnt * inv_s * out_e).round();
      }
    } else if (quant_axis == 1) {
      const int64_t step_i = in.numel() / in_dims[0];
      const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
      for (int i = 0; i < in_dims[0]; i++) {
        for (int j = 0; j < in_dims[1]; j++) {
          T s = scale_data[j];
          T inv_s = inverse(s);
          auto* start = in_data + i * step_i + j * step_j;
          auto* end = in_data + i * step_i + (j + 1) * step_j;
          auto* cur_out_data = out_data + i * step_i + j * step_j;
          trans(ctx, start, end, cur_out_data, ClipFunctor<T>(-s, s));
          for (int k = 0; k < step_j; k++) {
            cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]);
          }
        }
      }
    }
  }
};

template struct ChannelClipAndFakeQuantFunctor<platform::CPUDeviceContext,
                                               float>;
template <typename T>
struct ChannelClipFakeQuantDequantFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in, const framework::Tensor& scale,
                  const int bin_cnt, const int quant_axis,
                  framework::Tensor* out) {
    PADDLE_ENFORCE_EQ(
        quant_axis == 0 || quant_axis == 1, true,
        platform::errors::InvalidArgument("'quant_axis' should be 0 or 1, but "
                                          "the received is %d",
                                          quant_axis));

    auto* scale_data = scale.data<T>();
    auto* in_data = in.data<T>();
    auto* out_data = out->mutable_data<T>(ctx.GetPlace());
    auto in_dims = in.dims();
    const int64_t channel = in_dims[quant_axis];
    platform::Transform<platform::CPUDeviceContext> trans;
    if (quant_axis == 0) {
      const int64_t channel_size = in.numel() / channel;
      for (int i = 0; i < channel; i++) {
        T s = scale_data[i];
        auto* start = in_data + i * channel_size;
        auto* end = in_data + (i + 1) * channel_size;
        trans(ctx, start, end, out_data + i * channel_size,
              ClipFunctor<T>(-s, s));
      }
      for (int i = 0; i < channel; i++) {
        T s = scale_data[i];
        T inv_s = inverse(s);
        framework::Tensor one_channel_out = out->Slice(i, i + 1);
        auto out_e = framework::EigenVector<T>::Flatten(one_channel_out);
        out_e.device(*ctx.eigen_device()) =
            (bin_cnt * inv_s * out_e).round() * s / static_cast<T>(bin_cnt);
      }
    } else if (quant_axis == 1) {
      const int64_t step_i = in.numel() / in_dims[0];
      const int64_t step_j = in.numel() / (in_dims[0] * in_dims[1]);
      for (int i = 0; i < in_dims[0]; i++) {
        for (int j = 0; j < in_dims[1]; j++) {
          T s = scale_data[j];
          T inv_s = inverse(s);
          auto* start = in_data + i * step_i + j * step_j;
          auto* end = in_data + i * step_i + (j + 1) * step_j;
          auto* cur_out_data = out_data + i * step_i + j * step_j;
          trans(ctx, start, end, cur_out_data, ClipFunctor<T>(-s, s));
          for (int k = 0; k < step_j; k++) {
            cur_out_data[k] = std::round(bin_cnt * inv_s * cur_out_data[k]) *
                              s / static_cast<T>(bin_cnt);
          }
        }
      }
    }
  }
};

template struct ChannelClipFakeQuantDequantFunctor<platform::CPUDeviceContext,
                                                   float>;
template <typename T>
struct FindRangeAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& cur_scale,
                  const framework::Tensor& last_scale,
                  const framework::Tensor& iter, const int window_size,
                  framework::Tensor* scales_arr, framework::Tensor* out_scale) {
    T* scale_arr = scales_arr->mutable_data<T>(ctx.GetPlace());
    int64_t it = iter.data<int64_t>()[0];
    int idx = it % window_size;
    T removed = scale_arr[idx];
    T cur = cur_scale.data<T>()[0];
    scale_arr[idx] = cur;

    T max = last_scale.data<T>()[0];
    if (max < cur) {
      max = cur;
    } else if (fabs(removed - max) < 1e-6) {
      int size = (it > window_size) ? window_size : it;
      FindAbsMaxFunctor<platform::CPUDeviceContext, T>()(ctx, scale_arr, size,
                                                         &max);
    }
    out_scale->mutable_data<T>(ctx.GetPlace())[0] = max;
  }
};

template struct FindRangeAbsMaxFunctor<platform::CPUDeviceContext, float>;

template <typename T>
struct FindMovingAverageAbsMaxFunctor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& ctx,
                  const framework::Tensor& in_accum,
                  const framework::Tensor& in_state, const T* cur_scale,
                  const float rate, framework::Tensor* out_state,
                  framework::Tensor* out_accum, framework::Tensor* out_scale) {
    T accum = in_accum.data<T>()[0];
    T state = in_state.data<T>()[0];
    T scale = cur_scale[0];

    state = rate * state + 1;
    accum = rate * accum + scale;
    scale = accum / state;

    out_state->mutable_data<T>(ctx.GetPlace())[0] = state;
    out_accum->mutable_data<T>(ctx.GetPlace())[0] = accum;
    out_scale->mutable_data<T>(ctx.GetPlace())[0] = scale;
  }
};

template struct FindMovingAverageAbsMaxFunctor<platform::CPUDeviceContext,
                                               float>;

class FakeQuantOrWithDequantAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantOrWithDequantAbsMaxOp(const std::string& type,
                                 const framework::VariableNameMap& inputs,
                                 const framework::VariableNameMap& outputs,
                                 const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FakeQuantOrWithDequantAbsMaxOp");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "FakeQuantOrWithDequantAbsMaxOp");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FakeQuantOrWithDequantAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current scale");
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
This is a Base Op which supports FakeQuantAbsMaxOpMaker and FakeQuantDequantAbsMaxOpMaker.
FakeQuantAbsMaxOp operator is used in the dynamic quantization.

$$scale = max(abs(X))$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

FakeQuantDequantAbsMaxOp operator does the abs_max quantization and then dequantization.

$$scale = max(abs(X))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range) * scale / range$$

)DOC");
  }
};

class FakeChannelWiseQuantizeAbsMaxOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FakeChannelWiseQuantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FakeChannelWiseQuantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "FakeChannelWiseQuantizeAbsMax");
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FakeChannelWiseQuantizeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized low level tensor, "
              "but also saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current channel wise scale");
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int& quant_axis) {
          PADDLE_ENFORCE_EQ(quant_axis == 0 || quant_axis == 1, true,
                            platform::errors::InvalidArgument(
                                "'quant_axis' should be 0 or 1, but "
                                "the received is %d",
                                quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
The scale of FakeChannelWiseQuantize operator is a vector.
In detail, each channel of the input X has a scale value.

$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c})$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};

class FakeChannelWiseQuantizeDequantizeAbsMaxOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "FakeChannelWiseQuantizeDequantizeAbsMax");
    int quant_axis = ctx->Attrs().Get<int>("quant_axis");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {ctx->GetInputDim("X")[quant_axis]});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class FakeChannelWiseQuantizeDequantizeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddOutput("Out",
              "(Tensor) Output of quantized and dequantized low level tensor, "
              "saved as float data type.");
    AddOutput("OutScale", "(Tensor) Current channel wise scale");
    AddAttr<int>("quant_axis",
                 "(int, default 0) The axis for quantization. "
                 "For conv2d, depthwise_conv2d, conv2d_transpose "
                 "and mul, the quant_axis is equal to the cout axis.")
        .SetDefault(0)
        .AddCustomChecker([](const int& quant_axis) {
          PADDLE_ENFORCE_EQ(quant_axis == 0 || quant_axis == 1, true,
                            platform::errors::InvalidArgument(
                                "'quant_axis' should be 0 or 1, but "
                                "the received is %d",
                                quant_axis));
        });
    AddAttr<int>("bit_length", "(int, default 8)")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddComment(R"DOC(
The scale of FakeChannelWiseQuantize operator is a vector.
In detail, each channel of the input X has a scale value.

$$scale_c = max(abs(X_c))$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out_c = round(\frac{X_c * range} {scale_c}) * \frac{scale_c} {range}$$
In above three formulas, the range value of c is as follow:
$$0 \leq c \lt \ the\ channel\ number\ of\ X$$
)DOC");
  }
};

class FakeQuantizeRangeAbsMaxOp : public framework::OperatorWithKernel {
 public:
  FakeQuantizeRangeAbsMaxOp(const std::string& type,
                            const framework::VariableNameMap& inputs,
                            const framework::VariableNameMap& outputs,
                            const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "FakeQuantizeRangeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FakeQuantizeRangeAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "FakeQuantizeRangeAbsMax");
    if (ctx->HasOutput("OutScales")) {
      int window_size = ctx->Attrs().Get<int>("window_size");
      ctx->SetOutputDim("OutScales", {window_size});
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FakeQuantizeRangeAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InScale", "Last scale.");
    AddInput("Iter", "Global step iteration.").AsDispensable();
    AddOutput("Out", "(Tensor) Output of quantized low level tensor.");
    AddOutput("OutScale", " Current scale");
    AddOutput("OutScales", "(Tensor) scale buffer.").AsDispensable();
    AddAttr<int>("window_size", "(int, default 10000) window range size.")
        .SetDefault(10000);
    AddAttr<int>("bit_length", "(int, default 8), quantization bit number.")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
FakeQuantize operator is used in static quantization.

$$scale = max(max(abs(x)), history_abs_max)$$
$$range = 2^{bit_length - 1} - 1$$
$$Out = round(X/scale * range)$$

)DOC");
  }
};

class FakeQuantOrWithDequantMovingAverageAbsMaxOp
    : public framework::OperatorWithKernel {
 public:
  FakeQuantOrWithDequantMovingAverageAbsMaxOp(
      const std::string& type, const framework::VariableNameMap& inputs,
      const framework::VariableNameMap& outputs,
      const framework::AttributeMap& attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "FakeQuantOrWithDequantMovingAverageAbsMax");
    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->SetOutputDim("OutScale", {1});
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FakeQuantOrWithDequantMovingAverageAbsMaxOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InScale", "Last scale.");
    AddInput("InAccum", "Last accum.").AsDispensable();
    AddInput("InState", "Last state.").AsDispensable();
    AddOutput("Out", "(Tensor) Output of quantized low level tensor.");
    AddOutput("OutScale", " Current scale");
    AddOutput("OutState", "(Tensor) state buffer.").AsDispensable();
    AddOutput("OutAccum", "(Tensor) accum buffer.").AsDispensable();
    AddAttr<float>("moving_rate", "(float, default 0.9) moving rate.")
        .SetDefault(0.9);
    AddAttr<int>("bit_length", "(int, default 8), quantization bit number.")
        .SetDefault(8)
        .AddCustomChecker([](const int& bit_length) {
          PADDLE_ENFORCE_EQ(bit_length >= 1 && bit_length <= 16, true,
                            platform::errors::InvalidArgument(
                                "'bit_length' should be between 1 and 16, but "
                                "the received is %d",
                                bit_length));
        });
    AddAttr<bool>("is_test",
                  "(bool, default false) Set to true for inference only, false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
This is a Base Op which supports FakeQuantMovingAverageAbsMaxOp and FakeQuantDequantMovingAverageAbsMaxOp.
FakeQuantMovingAverageAbsMaxOp operator is used in the static quantization.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range)$$

FakeQuantDequantMovingAverageAbsMaxOp operator does the moving_average_abs_max quant and then dequant.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$range = 2^{bit\_length - 1} - 1$$
$$Out = round(X/scale * range) * scale / range$$

)DOC");
  }
};

class MovingAverageAbsMaxScaleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X",
                   "MovingAverageAbsMaxScale");
    OP_INOUT_CHECK(ctx->HasOutput("OutScale"), "Output", "OutScale",
                   "MovingAverageAbsMaxScale");
    if (ctx->HasOutput("OutState")) {
      ctx->SetOutputDim("OutState", {1});
    }
    if (ctx->HasOutput("OutAccum")) {
      ctx->SetOutputDim("OutAccum", {1});
    }
    ctx->SetOutputDim("OutScale", {1});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class MovingAverageAbsMaxScaleOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input is float data type.");
    AddInput("InAccum", "Last accum.").AsDispensable();
    AddInput("InState", "Last state.").AsDispensable();
    AddOutput("OutScale", " Current scale");
    AddOutput("OutState", "(Tensor) state buffer.").AsDispensable();
    AddOutput("OutAccum", "(Tensor) accum buffer.").AsDispensable();
    AddAttr<float>("moving_rate", "(float, default 0.9) moving rate.")
        .SetDefault(0.9);
    AddAttr<bool>("is_test",
                  "(bool, default false) Set true for inference only and false "
                  "for training. Some layers may run faster when this is true.")
        .SetDefault(false);
    AddComment(R"DOC(
MovingAverageAbsMaxScale operator is only used for calculating the quantization scale.
And it will not quantize the input tensor.

$$scale = (moving\_rate*accum+max(abs(x)))/(moving\_rate*state+1)$$
$$Out = X$$

)DOC");
  }
};

class FakeQuantDequantGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    auto out_grad_name = framework::GradVarName("Out");
    auto x_grad_name = framework::GradVarName("X");
    OP_INOUT_CHECK(ctx->HasInput(out_grad_name), "Input", out_grad_name,
                   "FakeQuantDequantGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(x_grad_name), "Output", x_grad_name,
                   "FakeQuantDequantGradOp");

    ctx->SetOutputDim(x_grad_name, ctx->GetInputDim(out_grad_name));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

template <typename T>
class FakeQuantDequantGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("fake_quantize_dequantize_grad");
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPU = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(
    fake_quantize_abs_max, ops::FakeQuantOrWithDequantAbsMaxOp,
    ops::FakeQuantOrWithDequantAbsMaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_quantize_abs_max,
                       ops::FakeQuantizeAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(fake_quantize_dequantize_abs_max,
                  ops::FakeQuantOrWithDequantAbsMaxOp,
                  ops::FakeQuantOrWithDequantAbsMaxOpMaker,
                  ops::FakeQuantDequantGradMaker<paddle::framework::OpDesc>,
                  ops::FakeQuantDequantGradMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_quantize_dequantize_abs_max,
                       ops::FakeQuantizeDequantizeAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(
    fake_quantize_range_abs_max, ops::FakeQuantizeRangeAbsMaxOp,
    ops::FakeQuantizeRangeAbsMaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_quantize_range_abs_max,
                       ops::FakeQuantizeRangeAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(
    fake_quantize_moving_average_abs_max,
    ops::FakeQuantOrWithDequantMovingAverageAbsMaxOp,
    ops::FakeQuantOrWithDequantMovingAverageAbsMaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_quantize_moving_average_abs_max,
                       ops::FakeQuantizeMovingAverageAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(fake_quantize_dequantize_moving_average_abs_max,
                  ops::FakeQuantOrWithDequantMovingAverageAbsMaxOp,
                  ops::FakeQuantOrWithDequantMovingAverageAbsMaxOpMaker,
                  ops::FakeQuantDequantGradMaker<paddle::framework::OpDesc>,
                  ops::FakeQuantDequantGradMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fake_quantize_dequantize_moving_average_abs_max,
    ops::FakeQuantizeDequantizeMovingAverageAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(
    fake_channel_wise_quantize_abs_max, ops::FakeChannelWiseQuantizeAbsMaxOp,
    ops::FakeChannelWiseQuantizeAbsMaxOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(fake_channel_wise_quantize_abs_max,
                       ops::FakeChannelWiseQuantizeAbsMaxKernel<CPU, float>);

REGISTER_OPERATOR(
    moving_average_abs_max_scale, ops::MovingAverageAbsMaxScaleOp,
    ops::MovingAverageAbsMaxScaleOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(moving_average_abs_max_scale,
                       ops::MovingAverageAbsMaxScaleKernel<CPU, float>);

REGISTER_OPERATOR(fake_quantize_dequantize_grad, ops::FakeQuantDequantGradOp);
REGISTER_OP_CPU_KERNEL(fake_quantize_dequantize_grad,
                       ops::FakeQuantDequantGradKernel<CPU, float>);

REGISTER_OPERATOR(fake_channel_wise_quantize_dequantize_abs_max,
                  ops::FakeChannelWiseQuantizeDequantizeAbsMaxOp,
                  ops::FakeChannelWiseQuantizeDequantizeAbsMaxOpMaker,
                  ops::FakeQuantDequantGradMaker<paddle::framework::OpDesc>,
                  ops::FakeQuantDequantGradMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    fake_channel_wise_quantize_dequantize_abs_max,
    ops::FakeChannelWiseQuantizeDequantizeAbsMaxKernel<CPU, float>);
