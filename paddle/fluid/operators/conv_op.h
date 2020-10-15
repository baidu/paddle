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

#pragma once

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/layout_utils.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;
constexpr int MaxKeyLength = 256;

// Base convolution operator definations for other conv
// like operators to reuse the implementation.
inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + 2 * padding - (dilation * (filter_size - 1) + 1)) / "
          "stride + 1), where input_size is %d, padding is %d, "
          "filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding, filter_size, dilation, stride));

  return output_size;
}

inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding_1, int padding_2, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding_1 + padding_2 - dkernel) / stride + 1;
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "The output's size is expected to be greater than 0. "
          "But recieved: output's size is %d. The output's size is computed by "
          "((input_size + padding_1 + padding_2 - (dilation * (filter_size - "
          "1) + 1)) / stride + 1), where input_size is %d, padding is "
          "(%d, %d), filter_size is %d, dilation is %d, stride is %d.",
          output_size, input_size, padding_1, padding_2, filter_size, dilation,
          stride));

  return output_size;
}

template <typename T = int>
inline void UpdatePaddingAndDilation(std::vector<T>* paddings,
                                     std::vector<T>* dilation,
                                     const std::string padding_algorithm,
                                     const framework::DDim data_dims,
                                     const std::vector<T>& strides,
                                     const std::vector<T>& ksize) {
  // set padding size == data_dims.size() * 2
  auto data_shape = framework::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(
        data_dims.size() * 2, paddings->size(),
        platform::errors::InvalidArgument(
            "Attribute padding's size should be the same or twice as the "
            "input's dimension. "
            "But recieved: padding's size is %d, padding is [%s]; input's "
            "dimension is %d, input's shape is [%s].",
            paddings->size(), framework::make_ddim(*paddings), data_dims.size(),
            data_dims));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + ksize[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;

      // dilation
      *(dilation->begin() + i) = 1;
    }

  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
}

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  if (paddings.size() != strides.size()) {
    for (size_t j = 0; j < paddings.size(); ++j) {
      padding_0 = padding_0 && (paddings[j] == 0);
    }
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

// Define Op classes in .h file so that other conv
// operator implementations can reuse the code.
class Conv2DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class Conv3DOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final;

 protected:
  virtual void Apply() {}
};

class ConvOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{
        {"Input", /*->*/ "Output"}};
    return m;
  }
};

class ConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override {
    std::vector<int64_t> output_shape = ComputeOutputShape(ctx);

    OP_INOUT_CHECK(ctx->HasOutput("Output"), "Output", "Output", "Conv");
    ctx->SetOutputDim("Output", framework::make_ddim(output_shape));
    ctx->ShareLoD("Input", "Output");
  }

 protected:
  std::vector<int64_t> ComputeOutputShape(
      framework::InferShapeContext* ctx) const;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override;
};

class ConvOpDoubleGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

template <typename DeviceContext, typename T>
class GemmConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    // The filter will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    const int groups = context.Attr<int>("groups");
    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");
    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output,
                                             &transformed_output);

    } else {
      transformed_input = *input;
      transformed_output = *output;
    }

    // update padding and dilation
    auto trans_in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims =
        framework::slice_ddim(trans_in_dims, 2, trans_in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());

    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    const int batch_size = static_cast<int>(transformed_input.dims()[0]);

    // filter_shape_vec:
    // {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));

    // output_shape_vec:
    // {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(transformed_output.dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec:
    // {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w,
    // o_d,o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;

    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = trans_in_dims[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }

    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size:
    // (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d * o_h *
    // o_w)

    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim);

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    framework::DDim in_matrix_shape = framework::slice_ddim(
        transformed_input.dims(), 1, transformed_input.dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        transformed_output.dims()[1],
        transformed_output.numel() /
            (transformed_output.dims()[0] * transformed_output.dims()[1])};

    // convolution operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
    int out_step = static_cast<int>(transformed_output.dims()[1]) / groups;

    math::Vol2ColFunctor<DeviceContext, T> vol2col;
    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;

    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch =
          transformed_input.Slice(i, i + 1).Resize(in_matrix_shape);
      Tensor out_batch =
          transformed_output.Slice(i, i + 1).Resize(output_matrix_shape);

      for (int g = 0; g < groups; g++) {
        Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          im2col(dev_ctx, in_slice, dilations, strides,
                 std::vector<int>{paddings[0], paddings[2], paddings[1],
                                  paddings[3]},
                 &col);

        } else if (data_dim == 3U) {
          vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
        }

        // gemm
        Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(filter_slice, false, col_matrix, false, T(1.0), &out_slice,
                    T(0.0));
      }
    }
    if (channel_last) {
      TransToChannelLast<DeviceContext, T>(context, &transformed_output,
                                           output);
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    // The filter and filter_grad will be reshaped in the calculations,
    // so here use an assignment operation,
    // that avoids modifying the variable in the Scope.
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    int groups = context.Attr<int>("groups");
    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    Tensor transformed_input(input->type());
    Tensor transformed_output_grad(output_grad->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output_grad,
                                             &transformed_output_grad);
      TransToChannelFirst<DeviceContext, T>(context, output_grad,
                                            &transformed_output_grad);
    } else {
      transformed_input = *input;
      transformed_output_grad = *output_grad;
    }

    // update padding and dilation
    auto in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();
    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(transformed_input.dims()[0]);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(transformed_output_grad.dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = transformed_input.dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (i_c/g * k_h * k_w, o_h * o_w)
    // or
    // (i_c/g * k_d * k_h * k_w, o_d * o_h * o_w)
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);

    framework::DDim input_shape = framework::slice_ddim(
        transformed_input.dims(), 1, transformed_input.dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        transformed_output_grad.dims()[1],
        transformed_output_grad.numel() / (transformed_output_grad.dims()[0] *
                                           transformed_output_grad.dims()[1])};

    // convolution backward input operator:  gemm + col2im(or col2vol)
    // convolution backward weight operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(transformed_input.dims()[1]) / groups;
    int out_step = static_cast<int>(transformed_output_grad.dims()[1]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);

    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      col = context.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      Tensor transformed_input_grad(input_grad->type());
      if (channel_last) {
        ResizeToChannelFirst<DeviceContext, T>(context, input_grad,
                                               &transformed_input_grad);

      } else {
        transformed_input_grad = *input_grad;
      }
      // if is_expand is false, the operation of set_zero is unnecessary,
      // because math::matmul will reset input_grad.
      if (is_expand) {
        set_zero(dev_ctx, &transformed_input_grad, static_cast<T>(0));
      }
      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            transformed_output_grad.Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_grad_batch =
            transformed_input_grad.Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // gemm
          Tensor out_grad_slice =
              out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);

          Tensor in_grad_slice =
              in_grad_batch.Slice(g * in_step, (g + 1) * in_step);

          if (!is_expand) {
            col_matrix.ShareDataWith(in_grad_slice);
            col_matrix.Resize(col_matrix_shape);
          }
          blas.MatMul(filter_slice, true, out_grad_slice, false, T(1.0),
                      &col_matrix, T(0.0));

          if (is_expand && data_dim == 2U) {
            col2im(dev_ctx, col, dilations, strides,
                   std::vector<int>{paddings[0], paddings[2], paddings[1],
                                    paddings[3]},
                   &in_grad_slice);
          } else if (is_expand && data_dim == 3U) {
            col2vol(dev_ctx, col, dilations, strides, paddings, &in_grad_slice);
          }
        }
      }
      if (channel_last) {
        TransToChannelLast<DeviceContext, T>(context, &transformed_input_grad,
                                             input_grad);
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      Tensor filter_grad_ = *filter_grad;
      filter_grad_.Resize(filter_matrix_shape);
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            transformed_output_grad.Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_batch = transformed_input.Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // im2col
          Tensor out_grad_slice =
              out_grad_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

          if (!is_expand) {
            col.ShareDataWith(in_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx, in_slice, dilations, strides,
                   std::vector<int>{paddings[0], paddings[2], paddings[1],
                                    paddings[3]},
                   &col);

          } else if (data_dim == 3U) {
            vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
          }

          // gemm
          Tensor filter_grad_slice =
              filter_grad_.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(out_grad_slice, false, col_matrix, true, T(1.0),
                      &filter_grad_slice, T(1.0));
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class GemmConvDoubleGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(ctx.GetPlace()), true,
        paddle::platform::errors::PreconditionNotMet("It must use CPUPlace."));
    const Tensor* X = ctx.Input<Tensor>("Input");
    const Tensor* dY = ctx.Input<Tensor>("DOutput");
    const Tensor* ddX = ctx.Input<Tensor>("DDInput");
    const Tensor* ddW_in = ctx.Input<Tensor>("DDFilter");

    Tensor* ddY = ctx.Output<Tensor>("DDOutput");
    Tensor* dW = ctx.Output<Tensor>("DFilter");
    Tensor* dX = ctx.Output<Tensor>("DInput");
    Tensor W = GET_DATA_SAFELY(ctx.Input<Tensor>("Filter"), "Input", "Filter",
                               "GemmConvDoubleGrad");
    if (!ddY && !dW && !dX) return;

    const int groups = ctx.Attr<int>("groups");
    const std::vector<int> strides = ctx.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = ctx.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = ctx.Attr<std::vector<int>>("dilations");
    const std::string padding_algorithm =
        ctx.Attr<std::string>("padding_algorithm");
    const std::string data_format = ctx.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // transform Tensor
    Tensor transformed_X(X->type());
    Tensor transformed_dY(dY->type());
    Tensor transformed_ddX(X->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(ctx, X, &transformed_X);
      TransToChannelFirst<DeviceContext, T>(ctx, X, &transformed_X);

      ResizeToChannelFirst<DeviceContext, T>(ctx, dY, &transformed_dY);
      TransToChannelFirst<DeviceContext, T>(ctx, dY, &transformed_dY);

      if (ddX) {
        ResizeToChannelFirst<DeviceContext, T>(ctx, ddX, &transformed_ddX);
        TransToChannelFirst<DeviceContext, T>(ctx, ddX, &transformed_ddX);
      }
    } else {
      transformed_X = *X;
      transformed_dY = *dY;
      if (ddX) {
        transformed_ddX = *ddX;
      }
    }

    // update padding and dilation
    auto in_dims = transformed_X.dims();
    auto filter_dims = W.dims();

    framework::DDim in_data_dims =
        framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    const int batch_size = static_cast<int>(transformed_X.dims()[0]);
    std::vector<int64_t> filter_shape_vec(framework::vectorize(W.dims()));
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(transformed_dY.dims()));

    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    // col_shape [in_channel/group, kh, kw, oh, ow]
    col_shape_vec[0] = transformed_X.dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + data_dim + 1] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));
    // col_matrix_shape [in_channel/group * kh * kw, oh * ow]
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);
    // input_shape [Cin, H, W]
    framework::DDim input_shape = framework::slice_ddim(
        transformed_X.dims(), 1, transformed_X.dims().size());
    // filter_matrix_shape [Cout, Cin * kh * kw]
    framework::DDim filter_matrix_shape = {W.dims()[0],
                                           W.numel() / W.dims()[0]};

    W.Resize(filter_matrix_shape);
    framework::DDim output_matrix_shape = {
        transformed_dY.dims()[1],
        transformed_dY.numel() /
            (transformed_dY.dims()[0] * transformed_dY.dims()[1])};
    int in_step = static_cast<int>(transformed_X.dims()[1]) / groups;
    int out_step = static_cast<int>(transformed_dY.dims()[1]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    Tensor col_matrix;
    if (is_expand) {
      col = ctx.AllocateTmpTensor<T, DeviceContext>(col_shape, dev_ctx);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    // dx convolution double grad:  gemm + col2im(col2vol)
    // dx = ddw * dy  ==> dx(N, Cin, H, W), ddw(Cout, Cin, kh, kw), dy(N, Cout,
    // oH, oW)
    if (dX && ddW_in) {
      Tensor ddW;
      ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);
      dX->mutable_data<T>(ctx.GetPlace());

      Tensor transformed_dX(dX->type());

      if (channel_last) {
        ResizeToChannelFirst<DeviceContext, T>(ctx, dX, &transformed_dX);

      } else {
        transformed_dX = *dX;
      }
      // if is_expand is false, the operation of set_zero is unnecessary
      // because math::matmul will reset dx
      if (is_expand) {
        set_zero(dev_ctx, &transformed_dX, static_cast<T>(0));
      }
      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor dy_batch =
            transformed_dY.Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor dx_batch = transformed_dX.Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; g++) {
          // gemm
          Tensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
          Tensor dx_slice = dx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col_matrix.ShareDataWith(dx_slice);
            col_matrix.Resize(col_matrix_shape);
          }
          blas.MatMul(ddw_slice, true, dy_slice, false, T(1.0), &col_matrix,
                      T(0.0));

          if (is_expand && data_dim == 2U) {
            col2im(dev_ctx, col, dilations, strides,
                   std::vector<int>{paddings[0], paddings[2], paddings[1],
                                    paddings[3]},
                   &dx_slice);
          } else if (is_expand && data_dim == 3U) {
            col2vol(dev_ctx, col, dilations, strides, paddings, &dx_slice);
          }
        }
      }
      if (channel_last) {
        TransToChannelLast<DeviceContext, T>(ctx, &transformed_dX, dX);
      }
    }

    // dw = ddx * dy  ==> dw(Cout, Cin, kh, kw), ddx(N, Cin, H, W), dy(N, Cout,
    // oH, oW)
    // dw convolution double grad:  im2col(vol2col) + gemm
    if (dW && ddX) {
      dW->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, dW, static_cast<T>(0));
      Tensor dW_arr = *dW;
      dW_arr.Resize(filter_matrix_shape);
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; ++i) {
        Tensor dy_batch =
            transformed_dY.Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor ddx_batch = transformed_ddX.Slice(i, i + 1).Resize(input_shape);
        for (int g = 0; g < groups; ++g) {
          // im2col
          Tensor dy_slice = dy_batch.Slice(g * out_step, (g + 1) * out_step);
          Tensor ddx_slice = ddx_batch.Slice(g * in_step, (g + 1) * in_step);
          if (!is_expand) {
            col.ShareDataWith(ddx_slice);
            col_matrix.ShareDataWith(col);
            col_matrix.Resize(col_matrix_shape);
          } else if (data_dim == 2U) {
            im2col(dev_ctx, ddx_slice, dilations, strides,
                   std::vector<int>{paddings[0], paddings[2], paddings[1],
                                    paddings[3]},
                   &col);
          } else if (data_dim == 3U) {
            vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
          }

          Tensor dw_slice = dW_arr.Slice(g * out_step, (g + 1) * out_step);
          blas.MatMul(dy_slice, false, col_matrix, true, T(1.0), &dw_slice,
                      T(1.0));
        }
      }
    }

    // ddy = w * ddx + x * ddw ==> ddy(N, Cout, oH, oW), x/ddx(N, Cin, H, W),
    // w/ddw(Cout, Cin, kh, kw)
    // ddy convolution double grad: im2col(vol2col) + gemm
    if (ddY) {
      ddY->mutable_data<T>(ctx.GetPlace());

      Tensor transformed_ddY(ddY->type());
      if (channel_last) {
        ResizeToChannelFirst<DeviceContext, T>(ctx, ddY, &transformed_ddY);
      } else {
        transformed_ddY = *ddY;
      }

      set_zero(dev_ctx, &transformed_ddY, static_cast<T>(0));
      math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;
      math::Vol2ColFunctor<DeviceContext, T> vol2col;
      for (int i = 0; i < batch_size; ++i) {
        Tensor ddy_batch =
            transformed_ddY.Slice(i, i + 1).Resize(output_matrix_shape);
        for (int g = 0; g < groups; ++g) {
          // gemm
          Tensor ddy_slice = ddy_batch.Slice(g * out_step, (g + 1) * out_step);

          if (ddX) {
            Tensor ddx_batch =
                transformed_ddX.Slice(i, i + 1).Resize(input_shape);
            Tensor ddx_slice = ddx_batch.Slice(g * in_step, (g + 1) * in_step);
            if (!is_expand) {
              col.ShareDataWith(ddx_slice);
              col_matrix.ShareDataWith(col);
              col_matrix.Resize(col_matrix_shape);
            } else if (data_dim == 2U) {
              im2col(dev_ctx, ddx_slice, dilations, strides,
                     std::vector<int>{paddings[0], paddings[2], paddings[1],
                                      paddings[3]},
                     &col);
            } else if (data_dim == 3U) {
              vol2col(dev_ctx, ddx_slice, dilations, strides, paddings, &col);
            }
            Tensor w_slice = W.Slice(g * out_step, (g + 1) * out_step);
            blas.MatMul(w_slice, false, col_matrix, false, T(1.0), &ddy_slice,
                        T(0.0));
          }

          if (ddW_in) {
            Tensor x_batch = transformed_X.Slice(i, i + 1).Resize(input_shape);
            Tensor x_slice = x_batch.Slice(g * in_step, (g + 1) * in_step);

            Tensor ddW;
            ddW.ShareDataWith(*ddW_in).Resize(filter_matrix_shape);
            if (!is_expand) {
              col.ShareDataWith(x_slice);
              col_matrix.ShareDataWith(col);
              col_matrix.Resize(col_matrix_shape);
            } else if (data_dim == 2U) {
              im2col(dev_ctx, x_slice, dilations, strides,
                     std::vector<int>{paddings[0], paddings[2], paddings[1],
                                      paddings[3]},
                     &col);
            } else if (data_dim == 3U) {
              vol2col(dev_ctx, x_slice, dilations, strides, paddings, &col);
            }

            // gemm
            Tensor ddw_slice = ddW.Slice(g * out_step, (g + 1) * out_step);
            blas.MatMul(ddw_slice, false, col_matrix, false, T(1.0), &ddy_slice,
                        T(1.0));
          }
        }
      }
      if (channel_last) {
        TransToChannelLast<DeviceContext, T>(ctx, &transformed_ddY, ddY);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    const std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");

    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");
    if (channel_last) {
      PADDLE_ENFORCE_EQ(
          output->dims()[output->dims().size() - 1] %
              input->dims()[input->dims().size() - 1],
          0, platform::errors::InvalidArgument(
                 "ShapeError: The output channels must be a multiple of the "
                 "input channels. But receivced output channel number is %d "
                 "and input channel number is %d",
                 output->dims()[output->dims().size() - 1],
                 input->dims()[input->dims().size() - 1]));
    } else {
      PADDLE_ENFORCE_EQ(
          output->dims()[1] % input->dims()[1], 0,
          platform::errors::InvalidArgument(
              "ShapeError: The output channels must be a multiple of the "
              "input channels. But receivced output channel number is %d "
              "and input channel number is %d",
              output->dims()[1], input->dims()[1]));
    }
    // transform tensor
    Tensor transformed_input(input->type());
    Tensor transformed_output(output->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output,
                                             &transformed_output);

    } else {
      transformed_input = *input;
      transformed_output = *output;
    }

    // update padding and dilation
    auto in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());

    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
    if (!is_sys_pad) {
      for (size_t i = 0; i < strides.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }

    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (fuse_relu) {
      math::DepthwiseConvFunctor<DeviceContext, T, true> depthwiseConv;
      depthwiseConv(dev_ctx, transformed_input, filter, strides, paddings,
                    dilations, &transformed_output);
    } else {
      math::DepthwiseConvFunctor<DeviceContext, T, false> depthwiseConv;
      depthwiseConv(dev_ctx, transformed_input, filter, strides, paddings,
                    dilations, &transformed_output);
    }
    if (channel_last) {
      TransToChannelLast<DeviceContext, T>(context, &transformed_output,
                                           output);
    }
  }
};

template <typename DeviceContext, typename T>
class DepthwiseConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    const Tensor* output_grad =
        context.Input<Tensor>(framework::GradVarName("Output"));
    Tensor* input_grad =
        context.Output<Tensor>(framework::GradVarName("Input"));
    Tensor* filter_grad =
        context.Output<Tensor>(framework::GradVarName("Filter"));
    Tensor filter = *context.Input<Tensor>("Filter");

    if (!input_grad && !filter_grad) return;

    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");
    bool fuse_relu = context.Attr<bool>("fuse_relu_before_depthwise_conv");
    const std::string padding_algorithm =
        context.Attr<std::string>("padding_algorithm");
    const std::string data_format = context.Attr<std::string>("data_format");

    const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

    // transform Tensor
    Tensor transformed_input(input->type());
    Tensor transformed_output_grad(output_grad->type());

    if (channel_last) {
      ResizeToChannelFirst<DeviceContext, T>(context, input,
                                             &transformed_input);
      TransToChannelFirst<DeviceContext, T>(context, input, &transformed_input);

      ResizeToChannelFirst<DeviceContext, T>(context, output_grad,
                                             &transformed_output_grad);
      TransToChannelFirst<DeviceContext, T>(context, output_grad,
                                            &transformed_output_grad);

    } else {
      transformed_input = *input;
      transformed_output_grad = *output_grad;
    }

    // update padding and dilation
    auto in_dims = transformed_input.dims();
    auto filter_dims = filter.dims();

    framework::DDim in_data_dims;
    in_data_dims = framework::slice_ddim(in_dims, 2, in_dims.size());
    framework::DDim filter_data_dims =
        framework::slice_ddim(filter_dims, 2, filter_dims.size());
    std::vector<int> ksize = framework::vectorize<int>(filter_data_dims);
    UpdatePaddingAndDilation(&paddings, &dilations, padding_algorithm,
                             in_data_dims, strides, ksize);

    bool is_sys_pad = strides.size() * 2 == paddings.size() ? false : true;
    if (!is_sys_pad) {
      for (size_t i = 0; i < strides.size(); ++i) {
        paddings.erase(paddings.begin() + i + 1);
      }
    }
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      Tensor transformed_input_grad(input_grad->type());
      if (channel_last) {
        ResizeToChannelFirst<DeviceContext, T>(context, input_grad,
                                               &transformed_input_grad);

      } else {
        transformed_input_grad = *input_grad;
      }

      set_zero(dev_ctx, &transformed_input_grad, static_cast<T>(0));

      if (fuse_relu) {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, true>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, transformed_input, filter,
                               transformed_output_grad, strides, paddings,
                               dilations, &transformed_input_grad);
      } else {
        math::DepthwiseConvInputGradFunctor<DeviceContext, T, false>
            depthwiseConvInputGrad;
        depthwiseConvInputGrad(dev_ctx, transformed_input, filter,
                               transformed_output_grad, strides, paddings,
                               dilations, &transformed_input_grad);
      }
      if (channel_last) {
        TransToChannelLast<DeviceContext, T>(context, &transformed_input_grad,
                                             input_grad);
      }
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      if (fuse_relu) {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, true>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, transformed_input,
                                transformed_output_grad, strides, paddings,
                                dilations, filter_grad);
      } else {
        math::DepthwiseConvFilterGradFunctor<DeviceContext, T, false>
            depthwiseConvFilterGrad;
        depthwiseConvFilterGrad(dev_ctx, transformed_input,
                                transformed_output_grad, strides, paddings,
                                dilations, filter_grad);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
