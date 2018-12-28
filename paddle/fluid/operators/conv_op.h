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

#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/depthwise_conv.h"
#include "paddle/fluid/operators/math/im2col.h"
#include "paddle/fluid/operators/math/vol2col.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
constexpr int kConvMKLDNNFP32 = 1;
constexpr int kConvMKLDNNINT8 = 2;

// Base convolution operator definations for other conv
// like operators to reuse the implementation.
inline int ConvOutputSize(int input_size, int filter_size, int dilation,
                          int padding, int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  PADDLE_ENFORCE(
      output_size > 0,
      "Due to the settings of padding(%d), filter_size(%d), dilation(%d) and "
      "stride(%d), the output size is less than 0, please check "
      "again. Input_size:%d",
      padding, filter_size, dilation, stride, input_size);

  return output_size;
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
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return std::unordered_map<std::string, std::string>{
        {"Input", /*->*/ "Output"}};
  }
};

// calculate floating point operations,
// not floating point operations per second.
class ConvEstimateFlops : public framework::EstimateFlopsBase {
 public:
  size_t operator()(const framework::InferShapeContext* ctx) const override {
    auto filter_dims = ctx->GetInputDim("Filter");
    auto out_dims = ctx->GetOutputDim("Output");
    int groups = ctx->Attrs().Get<int>("groups");
    auto omap_dims = framework::slice_ddim(out_dims, 2, out_dims.size());
    size_t flops = 2 * framework::product(filter_dims) *
                   framework::product(omap_dims) / groups;
    if (!ctx->HasInput("Bias")) {
      flops -= framework::product(out_dims) / out_dims[0];
    }
    return flops;
  }
};

class ConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext* ctx) const override;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override;
};

class ConvOpGrad : public framework::OperatorWithKernel {
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

    int groups = context.Attr<int>("groups");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");

    auto& dev_ctx = context.template device_context<DeviceContext>();

    const int batch_size = static_cast<int>(input->dims()[0]);

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = input->dims()[1] / groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    framework::DDim col_shape(framework::make_ddim(col_shape_vec));

    // use col_matrix_shape in the gemm calculation
    // size: (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w, o_d *
    // o_h * o_w)
    framework::DDim col_matrix_shape =
        framework::flatten_to_2d(col_shape, data_dim + 1);

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      auto tmp_allocation_ptr =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              framework::product(col_shape) * sizeof(T));
      col = framework::GetTensor<T>(std::move(tmp_allocation_ptr), col_shape);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        output->dims()[1],
        output->numel() / (output->dims()[0] * output->dims()[1])};

    // convolution operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output->dims()[1]) / groups;

    math::Vol2ColFunctor<DeviceContext, T> vol2col;
    math::Im2ColFunctor<math::ColFormat::kCFO, DeviceContext, T> im2col;

    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
      Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

      for (int g = 0; g < groups; g++) {
        Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U) {
          // im2col
          im2col(dev_ctx, in_slice, dilations, strides,
                 std::vector<int>{paddings[0], paddings[1], paddings[0],
                                  paddings[1]},
                 &col);
        } else if (data_dim == 3U) {
          // vol2col
          vol2col(dev_ctx, in_slice, dilations, strides, paddings, &col);
        }

        // gemm
        Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
        Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);
        blas.MatMul(filter_slice, false, col_matrix, false, T(1.0), &out_slice,
                    T(0.0));
      }
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
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");

    const int batch_size = static_cast<int>(input->dims()[0]);

    auto& dev_ctx = context.template device_context<DeviceContext>();

    // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h, k_w}
    std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
    // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h, o_w}
    std::vector<int64_t> output_shape_vec(
        framework::vectorize(output_grad->dims()));

    // use col_shape in the im2col calculation
    // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h, k_w, o_d,
    // o_h, o_w}
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = input->dims()[1] / groups;
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

    framework::DDim input_shape =
        framework::slice_ddim(input->dims(), 1, input->dims().size());

    framework::DDim filter_matrix_shape = {filter.dims()[0],
                                           filter.numel() / filter.dims()[0]};
    filter.Resize(filter_matrix_shape);

    framework::DDim output_matrix_shape = {
        output_grad->dims()[1],
        output_grad->numel() /
            (output_grad->dims()[0] * output_grad->dims()[1])};

    // convolution backward input operator:  gemm + col2im(or col2vol)
    // convolution backward weight operator: im2col(or vol2col) + gemm
    int in_step = static_cast<int>(input->dims()[1]) / groups;
    int out_step = static_cast<int>(output_grad->dims()[1]) / groups;

    bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
    Tensor col;
    // col_matrix shares the same piece of data with col,
    // but will be reshaped into a two-dimensional matrix shape
    // to call the matrix multiplication interface.
    Tensor col_matrix;
    if (is_expand) {
      auto tmp_allocation_ptr =
          platform::DeviceTemporaryAllocator::Instance().Get(dev_ctx).Allocate(
              framework::product(col_shape) * sizeof(T));
      col = framework::GetTensor<T>(std::move(tmp_allocation_ptr), col_shape);
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }

    math::SetConstant<DeviceContext, T> set_zero;
    auto blas = math::GetBlas<DeviceContext, T>(dev_ctx);

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());

      // if is_expand is false, the operation of set_zero is unnecessary,
      // because math::matmul will reset input_grad.
      if (is_expand) {
        set_zero(dev_ctx, input_grad, static_cast<T>(0));
      }
      math::Col2VolFunctor<DeviceContext, T> col2vol;
      math::Col2ImFunctor<math::ColFormat::kCFO, DeviceContext, T> col2im;

      for (int i = 0; i < batch_size; i++) {
        Tensor out_grad_batch =
            output_grad->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_grad_batch = input_grad->Slice(i, i + 1).Resize(input_shape);
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
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
                   &in_grad_slice);
          } else if (is_expand && data_dim == 3U) {
            col2vol(dev_ctx, col, dilations, strides, paddings, &in_grad_slice);
          }
        }
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
            output_grad->Slice(i, i + 1).Resize(output_matrix_shape);
        Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
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
                   std::vector<int>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
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
class DepthwiseConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* input = context.Input<Tensor>("Input");
    Tensor filter = *context.Input<Tensor>("Filter");
    Tensor* output = context.Output<Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    PADDLE_ENFORCE_EQ(
        output->dims()[1] % input->dims()[1], 0,
        "The output channels must be a multiple of the input channels");
    std::vector<int> strides = context.Attr<std::vector<int>>("strides");
    std::vector<int> paddings = context.Attr<std::vector<int>>("paddings");
    std::vector<int> dilations = context.Attr<std::vector<int>>("dilations");

    math::DepthwiseConvFunctor<DeviceContext, T> depthwiseConv;

    auto& dev_ctx = context.template device_context<DeviceContext>();
    depthwiseConv(dev_ctx, *input, filter, strides, paddings, dilations,
                  output);
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

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();

    math::DepthwiseConvInputGradFunctor<DeviceContext, T>
        depthwiseConvInputGrad;
    math::DepthwiseConvFilterGradFunctor<DeviceContext, T>
        depthwiseConvFilterGrad;

    if (input_grad) {
      input_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, input_grad, static_cast<T>(0));
      depthwiseConvInputGrad(dev_ctx, *input, filter, *output_grad, strides,
                             paddings, dilations, input_grad);
    }

    if (filter_grad) {
      filter_grad->mutable_data<T>(context.GetPlace());
      set_zero(dev_ctx, filter_grad, static_cast<T>(0));
      depthwiseConvFilterGrad(dev_ctx, *input, *output_grad, strides, paddings,
                              dilations, filter_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle
