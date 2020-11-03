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

#include "paddle/fluid/operators/pool_op.h"
#include <unordered_map>
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

int PoolOutputSize(int input_size, int filter_size, int padding_1,
                   int padding_2, int stride, bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size =
        (input_size - filter_size + padding_1 + padding_2) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + padding_1 + padding_2 + stride - 1) /
            stride +
        1;
  }
  PADDLE_ENFORCE_GT(
      output_size, 0,
      platform::errors::InvalidArgument(
          "the output size must be greater than 0. But received: "
          "output_size = %d due to the settings of input_size(%d), "
          "padding(%d,%d), "
          "k_size(%d) and stride(%d). Please check again!",
          output_size, input_size, padding_1, padding_2, filter_size, stride));
  return output_size;
}

void PoolOp::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(
      ctx->HasInput("X"), true,
      platform::errors::NotFound("Input(X) of Pool operator is not found."));
  PADDLE_ENFORCE_EQ(
      ctx->HasOutput("Out"), true,
      platform::errors::NotFound("Output(Out) of Pool operator is not found."));

  std::string pooling_type = ctx->Attrs().Get<std::string>("pooling_type");
  std::vector<int> ksize = ctx->Attrs().Get<std::vector<int>>("ksize");
  std::vector<int> strides = ctx->Attrs().Get<std::vector<int>>("strides");
  std::vector<int> paddings = ctx->Attrs().Get<std::vector<int>>("paddings");
  bool ceil_mode = ctx->Attrs().Get<bool>("ceil_mode");
  bool adaptive = ctx->Attrs().Get<bool>("adaptive");
  bool global_pooling = ctx->Attrs().Get<bool>("global_pooling");
  std::string data_format = ctx->Attrs().Get<std::string>("data_format");
  std::string padding_algorithm =
      ctx->Attrs().Get<std::string>("padding_algorithm");

  auto in_x_dims = ctx->GetInputDim("X");
  PADDLE_ENFORCE_EQ(
      in_x_dims.size() == 4 || in_x_dims.size() == 5, true,
      platform::errors::InvalidArgument(
          "the input of Op(pool) should be 4-D or 5-D Tensor. But "
          "received: %u-D Tensor and it's shape is [%s].",
          in_x_dims.size(), in_x_dims));

  PADDLE_ENFORCE_EQ(
      in_x_dims.size() - ksize.size(), 2U,
      platform::errors::InvalidArgument(
          "the dimension of input minus the size of "
          "Attr(ksize) must be euqal to 2 in Op(pool). "
          "But received: the dimension of input minus the size "
          "of Attr(ksize) is %d, the "
          "input's dimension is %d, the shape of input "
          "is [%s], the Attr(ksize)'s size is %d, the Attr(ksize) is [%s].",
          in_x_dims.size() - ksize.size(), in_x_dims.size(), in_x_dims,
          ksize.size(), framework::make_ddim(ksize)));

  PADDLE_ENFORCE_EQ(
      ksize.size(), strides.size(),
      platform::errors::InvalidArgument(
          "the size of Attr(ksize) and Attr(strides) in "
          "Op(pool) must be equal. "
          "But received: Attr(ksize)'s size is %d, Attr(strides)'s "
          "size is %d, Attr(ksize) is [%s], Attr(strides)is [%s].",
          ksize.size(), strides.size(), framework::make_ddim(ksize),
          framework::make_ddim(strides)));

  // MKL-DNN Kernels are using NCHW order of dims description
  // so we ignore data_format consideration for MKL-DNN kernel
  const bool channel_last = (this->IsMKLDNNType() == false) &&
                            (data_format == "NHWC" || data_format == "NDHWC");

  // update paddings if "SAME" or global_pooling
  framework::DDim data_dims;
  if (channel_last) {
    data_dims = framework::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
  } else {
    data_dims = framework::slice_ddim(in_x_dims, 2, in_x_dims.size());
  }
  UpdatePadding(&paddings, global_pooling, adaptive, padding_algorithm,
                data_dims, strides, ksize);

  if (global_pooling) {
    UpdateKsize(&ksize, data_dims);
  }

  std::vector<int64_t> output_shape;
  if (adaptive) {
    output_shape.insert(output_shape.end(), ksize.begin(), ksize.end());
  } else {
    for (int i = 0; i < data_dims.size(); ++i) {
      if ((!ctx->IsRuntime()) && (data_dims[i] < 0)) {
        output_shape.push_back(data_dims[i]);
      } else {
        output_shape.push_back(
            PoolOutputSize(data_dims[i], ksize[i], paddings[2 * i],
                           paddings[2 * i + 1], strides[i], ceil_mode));
      }
    }
  }

  // output_N = input_N
  output_shape.insert(output_shape.begin(), in_x_dims[0]);
  // output_C = input_C
  if (channel_last) {
    output_shape.push_back(in_x_dims[in_x_dims.size() - 1]);
  } else {
    output_shape.insert(output_shape.begin() + 1, in_x_dims[1]);
  }

  ctx->SetOutputDim("Out", framework::make_ddim(output_shape));
  ctx->ShareLoD("X", "Out");
}

framework::OpKernelType PoolOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = "AnyLayout";
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
#endif

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace(),
      layout_, library_);
}

framework::OpKernelType PoolOp::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  if ((expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    auto dl = framework::StringToDataLayout(data_format);
    // Some models may have intentionally set "AnyLayout" for pool
    // op. Treat this as NCHW (default data_format value)
    if (dl != framework::DataLayout::kAnyLayout) {
      return framework::OpKernelType(expected_kernel_type.data_type_,
                                     tensor.place(), dl);
    }
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

void PoolOpGrad::InferShape(framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                    platform::errors::NotFound(
                        "Input(X) of Pool Gradoperator is not found."));
  PADDLE_ENFORCE_EQ(ctx->HasOutput(framework::GradVarName("X")), true,
                    platform::errors::NotFound(
                        "Input(X@GRAD) of Pool Gradoperator is not found."));
  ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
}

framework::OpKernelType PoolOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_{framework::LibraryType::kPlain};
  std::string data_format = "AnyLayout";
  framework::DataLayout layout_ = framework::StringToDataLayout(data_format);

#ifdef PADDLE_WITH_CUDA
  if (platform::CanCUDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kCUDNN;
  }
#endif
#ifdef PADDLE_WITH_MKLDNN
  if (library_ == framework::LibraryType::kPlain &&
      platform::CanMKLDNNBeUsed(ctx)) {
    library_ = framework::LibraryType::kMKLDNN;
    layout_ = framework::DataLayout::kMKLDNN;
  }
#endif

  auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
  if (input_data_type == framework::proto::VarType::FP16) {
    PADDLE_ENFORCE_EQ(library_, framework::LibraryType::kCUDNN,
                      platform::errors::InvalidArgument(
                          "Float16 can only be used when CUDNN is used"));
  }
  return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout_,
                                 library_);
}

framework::OpKernelType PoolOpGrad::GetKernelTypeForVar(
    const std::string& var_name, const Tensor& tensor,
    const framework::OpKernelType& expected_kernel_type) const {
#ifdef PADDLE_WITH_MKLDNN
  if ((expected_kernel_type.data_layout_ == framework::DataLayout::kMKLDNN) &&
      (tensor.layout() != framework::DataLayout::kMKLDNN)) {
    auto attrs = Attrs();
    auto ar = paddle::framework::AttrReader(attrs);
    const std::string data_format = ar.Get<std::string>("data_format");
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(),
                                   framework::StringToDataLayout(data_format));
  }
#endif
  return framework::OpKernelType(expected_kernel_type.data_type_,
                                 tensor.place(), tensor.layout());
}

void Pool2dOpMaker::Make() {
  AddInput(
      "X",
      "(Tensor) The input tensor of pooling operator. "
      "The format of input tensor is NCHW, where N is batch size, C is the "
      "number of channels, H is the height of the feature, "
      "and W is the width of the feature.");
  AddOutput("Out",
            "(Tensor) The output tensor of pooling operator. "
            "The format of output tensor is also NCHW, "
            "where N is batch size, C is the number of channels, "
            "H is the height of the feature, "
            "and W is the width of the feature.");

  AddAttr<std::string>("pooling_type",
                       "(string), pooling type, can be \"max\" for max-pooling "
                       "and \"avg\" for average-pooling.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>("ksize",
                            "(vector<int>) The pooling window "
                            "size(height, width) of the pooling operator. "
                            "If global_pooling = true, ksize and paddings will "
                            "be ignored.");  // TODO(Chengduo): Add checker.
                                             // (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "global_pooling",
      "(bool) Whether to use the global pooling. "
      "If global_pooling = true, kernel size and paddings will be ignored. "
      "Default False.")
      .SetDefault(false);
  AddAttr<std::vector<int>>("strides",
                            "(vector<int>, default {1, 1}), strides(height, "
                            "width) of pooling operator.")
      .SetDefault({1, 1});
  // TODO(Chengduo): Add checker. (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int>, default {0,0}), paddings(height_top, height_bottom, "
      "width_left, wifth_right) of pooling operator."
      "If global_pooling = true, paddings and kernel size will be ignored.")
      .SetDefault({0, 0});
  AddAttr<bool>(
      "exclusive",
      "(bool) When true, will exclude the zero-padding in the "
      "averaging calculating, otherwise, include the zero-padding. Note, it "
      "is only used when pooling_type is avg. The default is True. "
      "Default True.")
      .SetDefault(true);
  AddAttr<bool>(
      "adaptive",
      "(bool) When true, will perform adaptive pooling instead, "
      "output shape in H and W dimensions will be same as ksize, input data "
      "will be divided into grids specify by ksize averagely and perform "
      "pooling in each grid area to get output pooling value. "
      "Default False.")
      .SetDefault(false);

  AddAttr<bool>(
      "use_cudnn",
      "(bool) Only used in cudnn kernel, need install cudnn. Default False")
      .SetDefault(false);
  AddAttr<bool>(
      "ceil_mode",
      "(bool) Whether to use the ceil function to calculate "
      "output height and width. False is the default. If it is set to False, "
      "the floor function will be used. Default False")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool) Only used in mkldnn kernel. Default False")
      .SetDefault(false);
  AddAttr<bool>(
      "use_quantizer",
      "(bool, default false) "
      "This parameter is no longer used. Use 'mkldnn_data_type' instead.")
      .SetDefault(false);
  AddAttr<std::string>(
      "mkldnn_data_type",
      "(string, default \"float32\"). Data type of mkldnn kernel")
      .SetDefault("float32")
      .InEnum({"float32", "int8", "bfloat16"});
  AddAttr<std::string>(
      "data_format",
      "(string, default NCHW) Only used in "
      "An optional string from: \"NHWC\", \"NCHW\". "
      "Defaults to \"NHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCHW");
  AddAttr<bool>("is_test",
                "(bool, default false) Set to true for inference only, false "
                "for training. Some layers may run faster when this is true.")
      .SetDefault(false);

  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  // TODO(dzhwinter): need to registered layout transform function

  AddComment(R"DOC(
This operation calculates the pooling output based on
the input, pooling_type and pool_size, pool_stride, pool_padding parameters.
Input(X) and Output(Out) are in NCHW or NHWC format, where N is batch size, C is the
number of channels, H is the height of the feature, and W is the width of the feature.
Parameters(pool_size, pool_stride, pool_padding) hold two integer elements.
These two elements represent height and width, respectively.
The input(X) size and output(Out) size may be different.

Example:

  Input:

       X shape: $(N, C, H_{in}, W_{in})$

  Output:

       Out shape: $(N, C, H_{out}, W_{out})$

  For pool_padding = "SAME":
       $$
       H_{out} = \\frac{(H_{in} + strides[0] - 1)}{strides[0]}
       $$
       $$
       W_{out} = \\frac{(W_{in} + strides[1] - 1)}{strides[1]}
       $$

  For pool_padding = "VALID":
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + strides[0])}{strides[0]}
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + strides[1])}{strides[1]}
       $$

  For ceil_mode = false:
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + pad_height_top + pad_height_bottom}{strides[0]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + pad_width_left + pad_width_right}{strides[1]} + 1
       $$

  For ceil_mode = true:
       $$
       H_{out} = \\frac{(H_{in} - ksize[0] + pad_height_top + pad_height_bottom + strides[0] - 1)}{strides[0]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[1] + pad_width_left + pad_width_right + strides[1] - 1)}{strides[1]} + 1
       $$

  For exclusive = false:
       $$
       hstart = i * strides[0] - pad_height_top
       $$
       $$
       hend = hstart + ksize[0]
       $$
       $$
       wstart = j * strides[1] - pad_width_left
       $$
       $$
       wend = wstart + ksize[1]
       $$
       $$
       Output(i ,j) = \\frac{sum(Input[hstart:hend, wstart:wend])}{ksize[0] * ksize[1]}
       $$

  For exclusive = true:
       $$
       hstart = max(0, i * strides[0] - pad_height_top)
       $$
       $$
       hend = min(H, hstart + ksize[0])
       $$
       $$
       wstart = max(0, j * strides[1] - pad_width_left)
       $$
       $$
       wend = min(W, wstart + ksize[1])
       $$
       $$
       Output(i ,j) = \\frac{sum(Input[hstart:hend, wstart:wend])}{(hend - hstart) * (wend - wstart)}
       $$

)DOC");
}

class PoolOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

void Pool3dOpMaker::Make() {
  AddInput("X",
           "(Tensor) The input tensor of pooling operator. "
           "The format of input tensor is NCDHW or NDHWC, where N is batch "
           "size, C is "
           "the number of channels, and D, H and W is the depth, height and "
           "width of "
           "the feature, respectively.");
  AddOutput("Out",
            "(Tensor) The output tensor of pooling operator."
            "The format of output tensor is also NCDHW or NDHWC, "
            "where N is batch size, C is "
            "the number of channels, and D, H and W is the depth, height and "
            "width of the feature, respectively.");

  AddAttr<std::string>("pooling_type",
                       "(string) Pooling type, can be \"max\" for max-pooling "
                       "and \"avg\" for average-pooling.")
      .InEnum({"max", "avg"});
  AddAttr<std::vector<int>>(
      "ksize",
      "(vector<int>) The pooling window size(depth, height, "
      "width) of pooling operator. "
      "If global_pooling = true, ksize and paddings will "
      "be ignored.");  // TODO(Chengduo): Add checker.
                       // (Currently,
  // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "global_pooling",
      "(bool) Whether to use the global pooling. "
      "If global_pooling = true, kernel size and paddings will be ignored. "
      "Default False")
      .SetDefault(false);
  AddAttr<std::vector<int>>(
      "strides",
      "(vector<int>, default {1,1,1}) Strides(depth, height, "
      "width) of the pooling operator.")
      .SetDefault({1, 1, 1});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)
  AddAttr<std::vector<int>>(
      "paddings",
      "(vector<int>, default {0,0,0}), paddings(pad_depth_front, "
      "pad_depth_back, "
      "pad_height_top, pad_height_bottom, pad_width_left, pad_width_right"
      ") of pooling operator. "
      "If global_pooling = true, ksize and paddings will be ignored.")
      .SetDefault({0, 0, 0});  // TODO(Chengduo): Add checker. (Currently,
                               // TypedAttrChecker don't support vector type.)
  AddAttr<bool>(
      "exclusive",
      "(bool) When true, will exclude the zero-padding in the "
      "averaging calculating, otherwise, include the zero-padding. Note, it "
      "is only used when pooling_type is avg. The default is True. "
      "Default True")
      .SetDefault(true);
  AddAttr<bool>(
      "adaptive",
      "(bool) When true, will perform adaptive pooling instead, "
      "output shape in H and W dimensions will be same as ksize, input data "
      "will be divided into grids specify by ksize averagely and perform "
      "pooling in each grid area to get output pooling value. "
      "Default False")
      .SetDefault(false);

  AddAttr<bool>(
      "use_cudnn",
      "(bool) Only used in cudnn kernel, need install cudnn. Default False")
      .SetDefault(false);
  AddAttr<bool>(
      "ceil_mode",
      "(bool) Whether to use the ceil function to calculate "
      "output height and width. False is the default. If it is set to False, "
      "the floor function will be used. Default False")
      .SetDefault(false);
  AddAttr<bool>("use_mkldnn",
                "(bool) Only used in mkldnn kernel. Default False")
      .SetDefault(false);
  AddAttr<std::string>(
      "data_format",
      "(string, default NCDHW) Only used in "
      "An optional string from: \"NDHWC\", \"NCDHW\". "
      "Defaults to \"NDHWC\". Specify the data format of the output data, "
      "the input will be transformed automatically. ")
      .SetDefault("NCDHW");
  AddAttr<std::string>(
      "padding_algorithm",
      "(string, default \"EXPLICIT\") An optional string from: \"EXPLICIT\","
      "\"SAME\",\"VALID\". Set to \"EXPLICIT\" for explicit padding. "
      "Set to \"SAME\" or \"VALID\" for algorithm of padding. ")
      .SetDefault("EXPLICIT");
  // TODO(dzhwinter): need to registered layout transform function

  AddComment(R"DOC(
This operation calculates the output based on
the input, pooling_type, pool_size, pool_stride, and pool_padding parameters.
Input(X) and output(Out) are in NCDHW or NDHWC format, where N is batch
size, C is the number of channels, and D, H and W are the depth, height and
width of the feature, respectively. Parameters(pool_size, pool_stride, pool_padding)
hold three integer elements. These three elements represent depth, height and
width, respectively. The input(X) size and output(Out) size may be different.

Example:
  Input:
       X shape: $(N, C, D_{in}, H_{in}, W_{in})$
  Output:
       Out shape: $(N, C, D_{out}, H_{out}, W_{out})$

  For pool_padding = "SAME":
       $$
       D_{out} = \\frac{(D_{in} + strides[0] - 1)}{strides[0]}
       $$
       $$
       H_{out} = \\frac{(H_{in} + strides[1] - 1)}{strides[1]}
       $$
       $$
       W_{out} = \\frac{(W_{in} + strides[2] - 1)}{strides[2]}
       $$

  For pool_padding = "VALID":
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + strides[0])}{strides[0]}
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + strides[1])}{strides[1]}
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + strides[2])}{strides[2]}
       $$

  For ceil_mode = false:
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + pad_depth_front + pad_depth_back)}{strides[0]} + 1
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + pad_height_top + pad_height_bottom)}{strides[1]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + pad_width_left + pad_width_right)}{strides[2]} + 1
       $$
  For ceil_mode = true:
       $$
       D_{out} = \\frac{(D_{in} - ksize[0] + pad_depth_front + pad_depth_back + strides[0] -1)}{strides[0]} + 1
       $$
       $$
       H_{out} = \\frac{(H_{in} - ksize[1] + pad_height_top + pad_height_bottom + strides[1] -1)}{strides[1]} + 1
       $$
       $$
       W_{out} = \\frac{(W_{in} - ksize[2] + pad_width_left + pad_width_right + strides[2] -1)}{strides[2]} + 1
       $$

  For exclusive = false:
       $$
       dstart = i * strides[0] - pad_depth_front
       $$
       $$
       dend = dstart + ksize[0]
       $$
       $$
       hstart = j * strides[1] - pad_height_top
       $$
       $$
       hend = hstart + ksize[1]
       $$
       $$
       wstart = k * strides[2] -  pad_width_left
       $$
       $$
       wend = wstart + ksize[2]
       $$
       $$
       Output(i ,j, k) = \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{ksize[0] * ksize[1] * ksize[2]}
       $$

  For exclusive = true:
       $$
       dstart = max(0, i * strides[0] - pad_depth_front)
       $$
       $$
       dend = min(D, dstart + ksize[0])
       $$
       $$
       hstart = max(0, j * strides[1] - pad_height_top)
       $$
       $$
       hend = min(H, hstart + ksize[1])
       $$
       $$
       wstart = max(0, k * strides[2] - pad_width_left)
       $$
       $$
       wend = min(W, wstart + ksize[2])
       $$
       $$
       Output(i ,j, k) = \\frac{sum(Input[dstart:dend, hstart:hend, wstart:wend])}{(dend - dstart) * (hend - hstart) * (wend - wstart)}
       $$

)DOC");
}
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    pool2d, ops::PoolOp, ops::Pool2dOpMaker, ops::PoolOpInferVarType,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(pool2d_grad, ops::PoolOpGrad);

REGISTER_OP_CPU_KERNEL(
    pool2d, ops::PoolKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PoolKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    pool2d_grad, ops::PoolGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PoolGradKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OPERATOR(
    pool3d, ops::PoolOp, ops::Pool3dOpMaker, ops::PoolOpInferVarType,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(pool3d_grad, ops::PoolOpGrad);

REGISTER_OP_CPU_KERNEL(
    pool3d, ops::PoolKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PoolKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    pool3d_grad, ops::PoolGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PoolGradKernel<paddle::platform::CPUDeviceContext, double>);
