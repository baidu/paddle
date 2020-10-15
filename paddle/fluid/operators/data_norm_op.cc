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

#include "paddle/fluid/operators/data_norm_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/framework/data_layout.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

template <typename T>
using EigenArrayMap =
    Eigen::Map<Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using EigenVectorArrayMap = Eigen::Map<Eigen::Array<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<T, Eigen::Dynamic, 1>>;

class DataNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DataNorm");
    OP_INOUT_CHECK(ctx->HasInput("BatchSize"), "Input", "BatchSize",
                   "DataNorm");
    OP_INOUT_CHECK(ctx->HasInput("BatchSum"), "Input", "BatchSum", "DataNorm");
    OP_INOUT_CHECK(ctx->HasInput("BatchSquareSum"), "Input", "BatchSquareSum",
                   "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Means"), "Output", "Means", "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Scales"), "Output", "Scales", "DataNorm");
    OP_INOUT_CHECK(ctx->HasOutput("Y"), "Output", "Y", "DataNorm");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("scale_w"), true,
          platform::errors::InvalidArgument(
              "Input(scale_w) of DataNormOp should not be null."));
      PADDLE_ENFORCE_EQ(ctx->HasInput("bias"), true,
                        platform::errors::InvalidArgument(
                            "Input(bias) of DataNormOp should not be null."));
    }

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));

    PADDLE_ENFORCE_EQ(x_dims.size() >= 2 && x_dims.size() <= 5, true,
                      platform::errors::InvalidArgument(
                          "Input X must have 2 to 5 dimensions."));

    const int64_t C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSize shouold be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSum shouold be 1"));
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "The input dim of BatchSquareSum shouold be 1"));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSize")[0], C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSize shouold be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSum")[0], C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSum shouold be C"));
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("BatchSquareSum")[0], C,
                        platform::errors::InvalidArgument(
                            "The input dim[0] of BatchSqureSum shouold be C"));
    }

    if (enable_scale_and_shift) {
      auto scale_dim = ctx->GetInputDim("scale_w");
      auto bias_dim = ctx->GetInputDim("bias");

      PADDLE_ENFORCE_EQ(
          scale_dim.size(), 1UL,
          platform::errors::InvalidArgument("the dimensionof scale"
                                            "must equal to 1. But received: "
                                            "the shape of scale is [%s], "
                                            "the dimensionof scale is [%d]",
                                            scale_dim, scale_dim.size()));
      PADDLE_ENFORCE_EQ(
          bias_dim.size(), 1UL,
          platform::errors::InvalidArgument("the dimension of bias"
                                            "must equal to 1. But received: "
                                            "the shape of bias is [%s],"
                                            "the dimension of bias is [%d]",
                                            bias_dim, bias_dim.size()));

      bool check = true;
      if ((!ctx->IsRuntime()) && (framework::product(scale_dim) <= 0 ||
                                  framework::product(bias_dim) <= 0)) {
        check = false;
      }

      if (check) {
        PADDLE_ENFORCE_EQ(scale_dim[0], C,
                          platform::errors::InvalidArgument(
                              "the shape of scale must equal to [%d]"
                              "But received: the shape of scale is [%d]",
                              C, scale_dim[0]));
        PADDLE_ENFORCE_EQ(bias_dim[0], C,
                          platform::errors::InvalidArgument(
                              "the shape of bias must equal to [%d]"
                              "But received: the shape of bias is [%d]",
                              C, bias_dim[0]));
      }
    }

    ctx->SetOutputDim("Y", x_dims);
    ctx->SetOutputDim("Means", {C});
    ctx->SetOutputDim("Scales", {C});
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    // By default, the type of the scale, bias, mean,
    // and var tensors should both be float. (For float or float16 input tensor)
    // or double (For double input tensor).
    auto dn_param_type = framework::proto::VarType::FP32;
    if (input_data_type == framework::proto::VarType::FP64) {
      dn_param_type = framework::proto::VarType::FP64;
    }
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSize"),
                      platform::errors::InvalidArgument(
                          "BatchSize input should be of float type"));
    PADDLE_ENFORCE_EQ(dn_param_type,
                      OperatorWithKernel::IndicateVarDataType(ctx, "BatchSum"),
                      platform::errors::InvalidArgument(
                          "BatchSum input should be of float type"));
    PADDLE_ENFORCE_EQ(dn_param_type, OperatorWithKernel::IndicateVarDataType(
                                         ctx, "BatchSquareSum"),
                      platform::errors::InvalidArgument(
                          "BatchSquareSum input should be of float type"));

    bool enable_scale_and_shift = ctx.Attr<bool>("enable_scale_and_shift");
    if (enable_scale_and_shift) {
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "scale_w"),
                        platform::errors::InvalidArgument(
                            "scale_w input should be of float type"));
      PADDLE_ENFORCE_EQ(dn_param_type,
                        OperatorWithKernel::IndicateVarDataType(ctx, "bias"),
                        platform::errors::InvalidArgument(
                            "bias input should be of float type"));
    }
    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library);
  }
};

class DataNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    // AddAttr<bool>("is_test", "").SetDefault(false);
    AddAttr<float>("epsilon", "")
        .SetDefault(1e-4)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 0.001f, true,
                            platform::errors::InvalidArgument(
                                "'epsilon' should be between 0.0 and 0.001."));
        });
    AddAttr<int>("slot_dim",
                 "(int, default -1) Dimension of one slot if set, "
                 "when the input is concated by slot-wise embeddings")
        .SetDefault(-1);
    AddAttr<float>(
        "summary_decay_rate",
        "(float, default 0.9999999) The decay rate when update the summary")
        .SetDefault(0.9999999);
    AddAttr<bool>(
        "enable_scale_and_shift",
        "(bool, default false) Set to true to enable scale and shift such as "
        "batch_norm op")
        .SetDefault(false);
    AddInput("scale_w",
             "scale_w is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddInput("bias",
             "bias is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddAttr<std::string>("data_layout", "").SetDefault("NCHW");
    AddAttr<bool>("sync_stats", "(bool, default false) only used in multi-GPU")
        .SetDefault(false);
    AddAttr<bool>("use_mkldnn",
                  "(bool, default false) Only used in mkldnn kernel")
        .SetDefault(false);
    AddInput("X", "The input tensor");
    AddInput("BatchSize",
             "BatchSize is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSum",
             "BatchSum is a 1-dimensional tensor of size C "
             "that is applied to the output");
    AddInput("BatchSquareSum",
             "The global BatchSquareSum (for training) or "
             "estimated BatchSquareSum (for testing)");
    AddOutput("Y", "result after normalization");
    AddOutput("Means",
              "Mean of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddOutput("Scales",
              "Scales of the history data batch, "
              "will apply to output when training")
        .AsIntermediate();
    AddComment(R"DOC(
Data Normalization.

Can be used as a normalizer function for data
The required data format for this layer is one of the following:
1. NHWC `[batch, in_height, in_width, in_channels]`
2. NCHW `[batch, in_channels, in_height, in_width]`

)DOC");
  }
};

template <typename T>
class DataNormKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    // const bool is_test = ctx.Attr<bool>("is_test");
    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    const auto *x = ctx.Input<Tensor>("X");
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::InvalidArgument(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    auto *y = ctx.Output<Tensor>("Y");
    auto *mean_out = ctx.Output<Tensor>("Means");
    auto *scales = ctx.Output<Tensor>("Scales");

    // alloc memory
    T *y_data = y->mutable_data<T>(ctx.GetPlace());

    ConstEigenVectorArrayMap<T> b_size_arr(
        ctx.Input<Tensor>("BatchSize")->data<T>(), C);
    ConstEigenVectorArrayMap<T> b_sum_arr(
        ctx.Input<Tensor>("BatchSum")->data<T>(), C);
    ConstEigenVectorArrayMap<T> b_square_sum_arr(
        ctx.Input<Tensor>("BatchSquareSum")->data<T>(), C);
    EigenVectorArrayMap<T> means_arr(mean_out->mutable_data<T>(ctx.GetPlace()),
                                     C);
    EigenVectorArrayMap<T> scales_arr(scales->mutable_data<T>(ctx.GetPlace()),
                                      C);
    means_arr = b_sum_arr / b_size_arr;
    scales_arr = (b_size_arr / b_square_sum_arr).sqrt();

    const T *means_data = mean_out->data<T>();
    const T *x_data = x->data<T>();

    const T *scales_data = scales->data<T>();
    const int slot_dim = ctx.Attr<int>("slot_dim");
    T min_precision = 1e-7f;
    switch (data_layout) {
      case DataLayout::kNCHW:  // It's two dimensions, so make no difference
      case DataLayout::kNHWC: {
        // if slot_dim is set and batch size is larger than zero, we choose
        // to check if show number is zero, if so, skip normalization.
        if (slot_dim > 0 && N > 0 &&
            (!ctx.Attr<bool>("enable_scale_and_shift"))) {
          const int item_size = x->numel() / N;
          // location of show number in one embedding
          int offset = 0;
          for (int k = 0; k < N; ++k) {
            for (int i = 0; i < item_size; i += slot_dim) {
              if (x_data[offset + i] > -min_precision &&
                  x_data[offset + i] < min_precision) {
                // show = 0
                memset(y_data + offset + i, 0, sizeof(T) * slot_dim);
              } else {
                for (int j = i; j < i + slot_dim; ++j) {
                  y_data[offset + j] =
                      (x_data[offset + j] - means_data[j]) * scales_data[j];
                }
              }
            }

            offset += item_size;
          }
        } else {
          if (!ctx.Attr<bool>("enable_scale_and_shift") && slot_dim <= 0) {
            EigenArrayMap<T>(y_data, C, N) =
                (ConstEigenArrayMap<T>(x->data<T>(), C, N).colwise() -
                 means_arr)
                    .colwise() *
                scales_arr;
          } else if (ctx.Attr<bool>("enable_scale_and_shift") &&
                     slot_dim <= 0) {
            const auto *scale_w = ctx.Input<Tensor>("scale_w");
            const auto *bias = ctx.Input<Tensor>("bias");
            ConstEigenVectorArrayMap<T> scale_w_arr(scale_w->data<T>(), C);
            ConstEigenVectorArrayMap<T> bias_arr(bias->data<T>(), C);

            Eigen::Array<T, Eigen::Dynamic, 1> new_scale =
                scales_arr * scale_w_arr;
            Eigen::Array<T, Eigen::Dynamic, 1> new_bias =
                bias_arr - means_arr * scales_arr * scale_w_arr;
            EigenArrayMap<T>(y_data, C, N) =
                (ConstEigenArrayMap<T>(x->data<T>(), C, N).colwise() *
                 new_scale)
                    .colwise() +
                new_bias;

          } else {
            const int item_size = x->numel() / N;
            const auto *scale_w = ctx.Input<Tensor>("scale_w");
            const auto *bias = ctx.Input<Tensor>("bias");
            const T *scale_w_data = scale_w->data<T>();
            const T *bias_data = bias->data<T>();
            // location of show number in one embedding
            int offset = 0;
            for (int k = 0; k < N; ++k) {
              for (int i = 0; i < item_size; i += slot_dim) {
                if (x_data[offset + i] > -min_precision &&
                    x_data[offset + i] < min_precision) {
                  // show = 0
                  memset(y_data + offset + i, 0, sizeof(T) * slot_dim);
                } else {
                  for (int j = i; j < i + slot_dim; ++j) {
                    y_data[offset + j] = ((x_data[offset + j] - means_data[j]) *
                                          scales_data[j]) *
                                             scale_w_data[j] +
                                         bias_data[j];
                  }
                }
              }  // end for i

              offset += item_size;
            }  // end for k
          }
        }
        break;
      }
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unknown storage order: %d", data_layout));
    }
  }
};

class DataNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")), "Input",
                   framework::GradVarName("Y"), "DataNormGrad");
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSize"), true,
        platform::errors::NotFound(
            "Output(BatchSize) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSum"), true,
        platform::errors::NotFound(
            "Output(BatchSum) of DataNormGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("BatchSquareSum"), true,
        platform::errors::NotFound(
            "Output(BatchSquareSum) of DataNormGradOp should not be null."));
    OP_INOUT_CHECK(ctx->HasInput("Means"), "Input", "Means", "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Scales"), "Input", "Scales", "DataNormGrad");
    bool enable_scale_and_shift =
        ctx->Attrs().Get<bool>("enable_scale_and_shift");
    // check output
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSize")),
                   "Output", framework::GradVarName("BatchSize"),
                   "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSum")), "Output",
                   framework::GradVarName("BatchSum"), "DataNormGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("BatchSquareSum")),
                   "Output", framework::GradVarName("BatchSquareSum"),
                   "DataNormGrad");

    const auto x_dims = ctx->GetInputDim("X");
    const DataLayout data_layout = framework::StringToDataLayout(
        ctx->Attrs().Get<std::string>("data_layout"));
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
    ctx->SetOutputDim(framework::GradVarName("BatchSize"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSum"), {C});
    ctx->SetOutputDim(framework::GradVarName("BatchSquareSum"), {C});
    if (enable_scale_and_shift) {
      const bool has_scale_grad =
          ctx->HasOutput(framework::GradVarName("scale_w"));
      const bool has_bias_grad = ctx->HasOutput(framework::GradVarName("bias"));

      PADDLE_ENFORCE_EQ((has_scale_grad == has_bias_grad), true,
                        platform::errors::InvalidArgument(
                            "Output(Scale@GRAD) and Output(Bias@GRAD)"
                            "must be null or not be null at same time. "
                            "But now, has Scale@Grad=[%d], has Bias@GRAD=[%d]",
                            has_scale_grad, has_bias_grad));
      if (has_scale_grad) {
        ctx->SetOutputDim(framework::GradVarName("scale_w"), {C});
        ctx->SetOutputDim(framework::GradVarName("bias"), {C});
      }
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Y@GRAD can not be found for computation"));
    }

    // TODO(pzelazko-intel): enable MKLDNN layout when it's ready
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;

#ifdef PADDLE_WITH_MKLDNN
    if (library == framework::LibraryType::kPlain &&
        platform::CanMKLDNNBeUsed(ctx)) {
      library = framework::LibraryType::kMKLDNN;
      layout = framework::DataLayout::kMKLDNN;
    }
#endif

    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
        layout, library);
  }
};

template <typename T>
class DataNormGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    const auto *d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto *scales = ctx.Input<Tensor>("Scales");
    const auto *means = ctx.Input<Tensor>("Means");

    const std::string data_layout_str = ctx.Attr<std::string>("data_layout");
    const DataLayout data_layout =
        framework::StringToDataLayout(data_layout_str);

    // Get the size for each dimension.
    // NCHW [batch_size, in_channels, in_height, in_width]
    const auto &x_dims = x->dims();
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, platform::errors::InvalidArgument(
                                            "The Input dim size should be 2"));
    const int N = x_dims[0];
    const int C =
        (data_layout == DataLayout::kNCHW ? x_dims[1]
                                          : x_dims[x_dims.size() - 1]);
    // init output
    Tensor *d_x = nullptr;
    if (ctx.HasOutput(framework::GradVarName("X"))) {
      d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    }

    auto *d_batch_size =
        ctx.Output<Tensor>(framework::GradVarName("BatchSize"));
    auto *d_batch_sum = ctx.Output<Tensor>(framework::GradVarName("BatchSum"));
    auto *d_batch_square_sum =
        ctx.Output<Tensor>(framework::GradVarName("BatchSquareSum"));

    const T *mean_data = means->data<T>();
    const T *inv_var_data = scales->data<T>();
    ConstEigenVectorArrayMap<T> mean_arr(mean_data, C);
    ConstEigenVectorArrayMap<T> inv_var_arr(inv_var_data, C);

    T *d_batch_size_data = d_batch_size->mutable_data<T>(ctx.GetPlace());
    T *d_batch_sum_data = d_batch_sum->mutable_data<T>(ctx.GetPlace());
    T *d_batch_square_sum_data =
        d_batch_square_sum->mutable_data<T>(ctx.GetPlace());
    EigenVectorArrayMap<T> d_batch_size_arr(d_batch_size_data, C);
    EigenVectorArrayMap<T> d_batch_sum_arr(d_batch_sum_data, C);
    EigenVectorArrayMap<T> d_batch_square_sum_arr(d_batch_square_sum_data, C);
    d_batch_size_arr.setZero();
    d_batch_sum_arr.setZero();
    d_batch_square_sum_arr.setZero();
    const T *x_data = x->data<T>();
    const T *means_data = means->data<T>();

    const float epsilon = ctx.Attr<float>("epsilon");
    T min_precision = 1e-7f;
    const int slot_dim = ctx.Attr<int>("slot_dim");
    switch (data_layout) {  // it's two dimensions, make no difference
      case DataLayout::kNCHW:
      case DataLayout::kNHWC: {
        ConstEigenVectorArrayMap<T> scales_arr(scales->data<T>(), C);
        ConstEigenVectorArrayMap<T> means_arr(means->data<T>(), C);
        ConstEigenArrayMap<T> x_arr(x->data<T>(), C, N);
        ConstEigenArrayMap<T> d_y_arr(d_y->data<T>(), C, N);
        if (d_x != nullptr) {
          EigenArrayMap<T> d_x_arr(d_x->mutable_data<T>(ctx.GetPlace()), C, N);
          d_x_arr.setZero();
          if (!ctx.Attr<bool>("enable_scale_and_shift")) {
            for (int nc = 0; nc < N; ++nc) {
              d_x_arr.col(nc) = d_y_arr.col(nc) * scales_arr;
            }
          } else {
            const auto *scale_w = ctx.Input<Tensor>("scale_w");
            auto *d_scale =
                ctx.Output<Tensor>(framework::GradVarName("scale_w"));
            auto *d_bias = ctx.Output<Tensor>(framework::GradVarName("bias"));
            ConstEigenVectorArrayMap<T> scale_arr(scale_w->data<T>(), C);
            T *d_bias_data = nullptr;
            T *d_scale_data = nullptr;

            d_scale->mutable_data<T>(ctx.GetPlace());
            d_bias->mutable_data<T>(ctx.GetPlace());
            d_bias_data = d_bias->mutable_data<T>(ctx.GetPlace());
            d_scale_data = d_scale->mutable_data<T>(ctx.GetPlace());

            EigenVectorArrayMap<T> d_bias_arr(d_bias_data, C);
            EigenVectorArrayMap<T> d_scale_arr(d_scale_data, C);
            Tensor dy_sum;
            dy_sum.Resize({C});
            dy_sum.mutable_data<T>(ctx.GetPlace());
            EigenVectorArrayMap<T> dy_sum_arr(
                dy_sum.mutable_data<T>(ctx.GetPlace()), C);
            Tensor dy_mul_x_sub_mean_mul_invstd_sum;
            dy_mul_x_sub_mean_mul_invstd_sum.Resize({C});
            dy_mul_x_sub_mean_mul_invstd_sum.mutable_data<T>(ctx.GetPlace());
            EigenVectorArrayMap<T> dy_mul_x_sub_mean_mul_invstd_sum_arr(
                dy_mul_x_sub_mean_mul_invstd_sum.mutable_data<T>(
                    ctx.GetPlace()),
                C);

            dy_sum_arr.setZero();
            dy_mul_x_sub_mean_mul_invstd_sum_arr.setZero();

            if (slot_dim <= 0) {
              for (int n = 0; n < N; ++n) {
                dy_sum_arr += d_y_arr.col(n);
                dy_mul_x_sub_mean_mul_invstd_sum_arr +=
                    ((x_arr.col(n) - mean_arr) * inv_var_arr * d_y_arr.col(n));
              }
              if (d_scale && d_bias) {
                d_bias_arr = dy_sum_arr;
                d_scale_arr = dy_mul_x_sub_mean_mul_invstd_sum_arr;
              }
              for (int nc = 0; nc < N; ++nc) {
                d_x_arr.col(nc) = d_y_arr.col(nc) * scales_arr * scale_arr;
              }
            } else {
              int offset = 0;
              const int item_size = x->numel() / N;
              T *d_x_data = d_x->mutable_data<T>(ctx.GetPlace());
              T *d_scale_data = d_scale->mutable_data<T>(ctx.GetPlace());
              T *d_bias_data = d_bias->mutable_data<T>(ctx.GetPlace());
              const T *dy_data = d_y->data<T>();
              const T *scales_data = scales->data<T>();
              const T *scale_w_data = scale_w->data<T>();
              const T *x_data = x->data<T>();
              for (int i = 0; i < item_size; i++) {
                d_bias_data[i] = 0;
                d_scale_data[i] = 0;
              }
              for (int k = 0; k < N; ++k) {
                for (int i = 0; i < item_size; i += slot_dim) {
                  if (!(x_data[offset + i] > -min_precision &&
                        x_data[offset + i] < min_precision)) {
                    // show != 0
                    for (int j = i; j < i + slot_dim; ++j) {
                      d_x_data[offset + j] = dy_data[offset + j] *
                                             scales_data[j] * scale_w_data[j];
                      d_bias_data[j] += dy_data[offset + j];
                      d_scale_data[j] += (x_data[offset + j] - mean_data[j]) *
                                         inv_var_data[j] * dy_data[offset + j];
                    }
                  }
                }
                offset += item_size;
              }
            }
          }
        }

        if (slot_dim > 0 && N > 0) {
          // if slot_dim is set and batch size is larger than zero, we choose
          // to check if show number is zero, if so, skip update statistics.
          int offset = 0;
          const int item_size = x->numel() / N;
          for (int k = 0; k < N; ++k) {
            for (int i = 0; i < item_size; i += slot_dim) {
              if (!(x_data[offset + i] > -min_precision &&
                    x_data[offset + i] < min_precision)) {
                // show != 0
                for (int j = i; j < i + slot_dim; ++j) {
                  d_batch_size_data[j] += 1;
                  d_batch_sum_data[j] += x_data[offset + j];
                  d_batch_square_sum_data[j] +=
                      (x_data[offset + j] - means_data[j]) *
                      (x_data[offset + j] - means_data[j]);
                }
              }
            }
            offset += item_size;
          }

          for (int i = 0; i < item_size; i += slot_dim) {
            for (int j = i; j < i + slot_dim; ++j) {
              if (d_batch_size_data[j] >= 1) {
                d_batch_sum_data[j] /= d_batch_size_data[j];
                d_batch_square_sum_data[j] =
                    d_batch_square_sum_data[j] / d_batch_size_data[j] +
                    d_batch_size_data[j] * epsilon;
                d_batch_size_data[j] = 1;
              }
            }
          }
        } else {
          // calculate data sum and squre sum
          Eigen::Array<T, Eigen::Dynamic, 1> sample_sum(C);
          Eigen::Array<T, Eigen::Dynamic, 1> sample_square_sum(C);
          // calculate data sample sum and square sum
          sample_sum.setZero();
          sample_square_sum.setZero();
          for (int nc = 0; nc < N; ++nc) {
            sample_sum += x_arr.col(nc);
            sample_square_sum += (x_arr.col(nc) - means_arr).square();
          }
          // calculate gradient
          d_batch_size_arr.setConstant(N);
          d_batch_sum_arr = sample_sum;
          d_batch_square_sum_arr =
              sample_square_sum + d_batch_size_arr * epsilon;
        }
        break;
      }
      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Unknown storage order: %s", data_layout_str));
    }
  }
};

template <typename T>
class DataNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("data_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetInput("scale_w", this->Input("scale_w"));
    op->SetInput("bias", this->Input("bias"));
    op->SetOutput("BatchSize", this->Input("BatchSize"));
    op->SetOutput("BatchSum", this->Input("BatchSum"));
    op->SetOutput("BatchSquareSum", this->Input("BatchSquareSum"));
    op->SetInput("Scales", this->Output("Scales"));
    op->SetInput("Means", this->Output("Means"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("BatchSize"),
                  this->InputGrad("BatchSize"));
    op->SetOutput(framework::GradVarName("BatchSum"),
                  this->InputGrad("BatchSum"));
    op->SetOutput(framework::GradVarName("BatchSquareSum"),
                  this->InputGrad("BatchSquareSum"));
    op->SetOutput(framework::GradVarName("scale_w"),
                  this->InputGrad("scale_w"));
    op->SetOutput(framework::GradVarName("bias"), this->InputGrad("bias"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(data_norm, ops::DataNormOp, ops::DataNormOpMaker,
                  ops::DataNormGradMaker<paddle::framework::OpDesc>,
                  ops::DataNormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(data_norm_grad, ops::DataNormGradOp);

REGISTER_OP_CPU_KERNEL(
    data_norm, ops::DataNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DataNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    data_norm_grad,
    ops::DataNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DataNormGradKernel<paddle::platform::CPUDeviceContext, double>);
