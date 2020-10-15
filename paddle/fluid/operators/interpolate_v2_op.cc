/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/interpolate_v2_op.h"
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using framework::Tensor;
using DataLayout = framework::DataLayout;

static void Interpolate1DInferShapeCheck(framework::InferShapeContext* ctx) {
  auto dim_x = ctx->GetInputDim("X");
  auto interp_method = ctx->Attrs().Get<std::string>("interp_method");

  PADDLE_ENFORCE_EQ("linear", interp_method,
                    platform::errors::InvalidArgument(
                        "Interpolation method can only be \"linear\" when"
                        "Input(X) dimension is 3, but got method = %s .",
                        interp_method));
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->HasInputs("SizeTensor")) {
    // top prority size
    auto inputs_name = ctx->Inputs("SizeTensor");
    PADDLE_ENFORCE_EQ(
        inputs_name.size(), 1,
        platform::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 1. "
            "Attr(out_shape)'s length must be 1 for 3-D input tensor, but got "
            "size = %d .",
            inputs_name.size()));
    int out_w = ctx->Attrs().Get<int>("out_w");
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_w};
    } else {
      dim_out = {dim_x[0], out_w, dim_x[2]};
    }
    ctx->SetOutputDim("Out", dim_out);

    return;
  }

  int out_w;
  if (ctx->HasInput("Scale")) {
    auto scale_tensor = ctx->GetInputDim("Scale");
    PADDLE_ENFORCE_EQ(
        scale_tensor.size(), 1,
        platform::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got dimension = %d .",
            scale_tensor.size()));
    PADDLE_ENFORCE_EQ(
        scale_tensor[0], 1,
        platform::errors::InvalidArgument(
            "Scale's shape must be 1, but got shape = %d .", scale_tensor[0]));
    out_w = -1;
  } else {
    auto scale = ctx->Attrs().Get<std::vector<float>>("scale");
    if (scale.size() > 0) {
      float scale_w = -1;
      scale_w = scale[0];
      PADDLE_ENFORCE_EQ(scale_w > 0, true, platform::errors::InvalidArgument(
                                               "scale  of Op(interpolate) "
                                               "should be greater than 0."));
      if (scale_w > 0.) {
        // round down
        out_w = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[2] * scale_w)
                     : static_cast<int>(dim_x[1] * scale_w));
        // protect when input shape is -1
        out_w = out_w > 0 ? out_w : -1;
      }
    } else {
      out_w = ctx->Attrs().Get<int>("out_w");
    }
  }

  if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
    auto out_size_dim = ctx->GetInputDim("OutSize");
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(), 1,
        platform::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimention = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(out_size_dim[0], 1, platform::errors::InvalidArgument(
                                              "OutSize's dim[0] must be 1"));
    ctx->ShareLoD("X", "Out");
    return;
  }

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_w};
  } else {
    dim_out = {dim_x[0], out_w, dim_x[2]};
  }
  ctx->SetOutputDim("Out", dim_out);
}

static void Interpolate2DInferShapeCheck(framework::InferShapeContext* ctx) {
  auto dim_x = ctx->GetInputDim("X");
  auto interp_method = ctx->Attrs().Get<std::string>("interp_method");

  PADDLE_ENFORCE(
      "bilinear" == interp_method || "nearest" == interp_method ||
          "bicubic" == interp_method,
      platform::errors::InvalidArgument(
          "Interpolation method can only be \"bilinear\" or \"nearest\" when "
          "Input(X) dimension is 4, but got method = %s.",
          interp_method));
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->HasInputs("SizeTensor")) {
    // top prority size
    auto inputs_name = ctx->Inputs("SizeTensor");
    PADDLE_ENFORCE_EQ(
        inputs_name.size(), 2,
        platform::errors::InvalidArgument(
            "Input(SizeTensor)'size of Op(interpolate) must be 2. "
            "Attr(out_shape)'s length must be 2 for 4-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    int out_h = ctx->Attrs().Get<int>("out_h");
    int out_w = ctx->Attrs().Get<int>("out_w");
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
    }
    ctx->SetOutputDim("Out", dim_out);

    return;
  }

  int out_h, out_w;
  if (ctx->HasInput("Scale")) {
    auto scale_tensor = ctx->GetInputDim("Scale");
    PADDLE_ENFORCE_EQ(
        scale_tensor.size(), 1,
        platform::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got dimension = %d .",
            scale_tensor.size()));
    PADDLE_ENFORCE_EQ(scale_tensor[0] == 2 || scale_tensor[0] == 1, true,
                      platform::errors::InvalidArgument(
                          "Scale's shape must be 2 or 1, but got shape = %d .",
                          scale_tensor[0]));
    out_h = -1;
    out_w = -1;
  } else {
    auto scale = ctx->Attrs().Get<std::vector<float>>("scale");
    if (scale.size() > 0) {
      float scale_h = -1;
      float scale_w = -1;
      scale_h = scale[0];
      scale_w = scale[1];
      PADDLE_ENFORCE_EQ(
          scale_w > 0 && scale_h > 0, true,
          platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                            "should be greater than 0."));
      if (scale_h > 0. && scale_w > 0.) {
        // round down
        out_h = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[2] * scale_h)
                     : static_cast<int>(dim_x[1] * scale_h));
        out_w = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[3] * scale_w)
                     : static_cast<int>(dim_x[2] * scale_w));
        // protect when input shape is -1
        out_h = out_h > 0 ? out_h : -1;
        out_w = out_w > 0 ? out_w : -1;
      }
    } else {
      out_h = ctx->Attrs().Get<int>("out_h");
      out_w = ctx->Attrs().Get<int>("out_w");
    }
  }

  if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
    auto out_size_dim = ctx->GetInputDim("OutSize");
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(), 1,
        platform::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got dimension = %d .",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(
        out_size_dim[0], 2,
        platform::errors::InvalidArgument(
            "OutSize's dim[0] must be 2, but got dimention = %d .",
            out_size_dim[0]));
    ctx->ShareLoD("X", "Out");
    return;
  }

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_h, out_w};
  } else {
    dim_out = {dim_x[0], out_h, out_w, dim_x[3]};
  }
  ctx->SetOutputDim("Out", dim_out);
}

static void Interpolate3DInferShapeCheck(framework::InferShapeContext* ctx) {
  auto dim_x = ctx->GetInputDim("X");
  auto interp_method = ctx->Attrs().Get<std::string>("interp_method");

  PADDLE_ENFORCE_EQ(
      "trilinear", interp_method,
      platform::errors::InvalidArgument(
          "Interpolation method can only be \"trilinear\" when Input(X) "
          "dimension is 5, but got method = %s .",
          interp_method));
  const DataLayout data_layout = framework::StringToDataLayout(
      ctx->Attrs().Get<std::string>("data_layout"));

  if (ctx->HasInputs("SizeTensor")) {
    // top prority size
    auto inputs_name = ctx->Inputs("SizeTensor");
    PADDLE_ENFORCE_EQ(
        inputs_name.size(), 3,
        platform::errors::InvalidArgument(
            "Input(SizeTensor)'s size of Op(interpolate) must be 3. "
            "Attr(out_shape)'s length must be 3 for 5-D input "
            "tensor, but got size = %d .",
            inputs_name.size()));
    int out_d = ctx->Attrs().Get<int>("out_d");
    int out_h = ctx->Attrs().Get<int>("out_h");
    int out_w = ctx->Attrs().Get<int>("out_w");
    framework::DDim dim_out;
    if (data_layout == DataLayout::kNCHW) {
      dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
    } else {
      dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
    }
    ctx->SetOutputDim("Out", dim_out);

    return;
  }

  int out_d, out_h, out_w;
  if (ctx->HasInput("Scale")) {
    auto scale_tensor = ctx->GetInputDim("Scale");
    PADDLE_ENFORCE_EQ(
        scale_tensor.size(), 1,
        platform::errors::InvalidArgument(
            "Scale's dimension size must be 1, but got size = %d .",
            scale_tensor.size()));
    PADDLE_ENFORCE_EQ(scale_tensor[0] == 3 || scale_tensor[0] == 1, true,
                      platform::errors::InvalidArgument(
                          "Scale's shape must be 3 or 1, but got shape = %d .",
                          scale_tensor[0]));
    out_d = -1;
    out_h = -1;
    out_w = -1;
  } else {
    auto scale = ctx->Attrs().Get<std::vector<float>>("scale");
    if (scale.size() > 0) {
      float scale_d = -1;
      float scale_h = -1;
      float scale_w = -1;
      scale_d = scale[0];
      scale_h = scale[1];
      scale_w = scale[2];
      PADDLE_ENFORCE_EQ(
          scale_w > 0 && scale_h > 0 && scale_d > 0, true,
          platform::errors::InvalidArgument("scale  of Op(interpolate) "
                                            "should be greater than 0."));
      if (scale_d > 0. && scale_h > 0. && scale_w > 0.) {
        // round down
        out_d = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[2] * scale_d)
                     : static_cast<int>(dim_x[1] * scale_d));
        out_h = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[3] * scale_h)
                     : static_cast<int>(dim_x[2] * scale_h));
        out_w = (data_layout == DataLayout::kNCHW
                     ? static_cast<int>(dim_x[4] * scale_w)
                     : static_cast<int>(dim_x[3] * scale_w));
        // protect when input shape is -1
        out_d = out_d > 0 ? out_d : -1;
        out_h = out_h > 0 ? out_h : -1;
        out_w = out_w > 0 ? out_w : -1;
      }
    } else {
      out_d = ctx->Attrs().Get<int>("out_d");
      out_h = ctx->Attrs().Get<int>("out_h");
      out_w = ctx->Attrs().Get<int>("out_w");
    }
  }

  if (ctx->HasInput("OutSize") && ctx->IsRuntime()) {
    auto out_size_dim = ctx->GetInputDim("OutSize");
    PADDLE_ENFORCE_EQ(
        out_size_dim.size(), 1,
        platform::errors::InvalidArgument(
            "OutSize's dimension size must be 1, but got size is %d.",
            out_size_dim.size()));
    PADDLE_ENFORCE_EQ(out_size_dim[0], 3,
                      platform::errors::InvalidArgument(
                          "OutSize's dim[0] must be 3, but got size is %d.",
                          out_size_dim[0]));
    ctx->ShareLoD("X", "Out");
    return;
  }

  framework::DDim dim_out;
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {dim_x[0], dim_x[1], out_d, out_h, out_w};
  } else {
    dim_out = {dim_x[0], out_d, out_h, out_w, dim_x[4]};
  }
  ctx->SetOutputDim("Out", dim_out);
}

class InterpolateV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Interpolate");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Interpolate");

    auto dim_x = ctx->GetInputDim("X");  // NCHW format
    PADDLE_ENFORCE(
        dim_x.size() == 3 || dim_x.size() == 4 || dim_x.size() == 5,
        platform::errors::Unimplemented(
            "Input(X) dimension must be 3, 4 or 5, but got dimension = %d .",
            dim_x.size()));

    if (dim_x.size() == 3) {
      // shape check for 1D interpolate for input tensor shape NCHW
      Interpolate1DInferShapeCheck(ctx);
    } else if (dim_x.size() == 4) {
      // shape check for 2D interpolate for input tensor shape NCHW
      Interpolate2DInferShapeCheck(ctx);
    } else {  // dim_x.size() == 5
      // shape check for 3D interpolate for input tensor shape NCDHW
      Interpolate3DInferShapeCheck(ctx);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "SizeTensor" || var_name == "Scale") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

class InterpolateV2OpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input tensor of interpolate operator, "
             "This is a 4-D tensor with shape of [N, C, H, W] or a "
             "5-D tensor with shape of [N, C, D, H, W].");
    AddInput("OutSize",
             "This is a 1-D tensor with two numbers to specify output size. "
             "It should be [output_height, output_width] when input is a 4-D "
             "tensor and should be [output_depth, output_height, output_width] "
             "when input is a 5-D tensor. It has a higher priority than "
             "the attr(out_d), attr(out_h), attr(out_w) and attr(scale).")
        .AsDispensable();
    AddInput("SizeTensor",
             "(vector<Tensor<int32>>, optional). If provided, interpolate will "
             "use this. The shape of the tensor in vector MUST BE [1]. "
             "It has the highest priority compare with Input(OutSize) and "
             "attr(out_d), attr(out_h), attr(out_w) and attr(scale).")
        .AsDuplicable()
        .AsDispensable();
    AddInput("Scale",
             "This is a 1-D tensor with one number to specify output scale. "
             "It has the higher priority compare with attr(scale).")
        .AsDispensable();
    AddOutput("Out",
              "The output tensor of interpolate operator, "
              "This is a tensor in same rank with Input(X).");

    AddAttr<std::string>(
        "data_layout",
        "(string, default NCHW) Only used in "
        "an optional string from: \"NHWC\", \"NCHW\". "
        "Specify that the data format of the input and output data is "
        "channel_first or channel_last.")
        .SetDefault("NCHW");
    AddAttr<int>("out_d", "output depth of interpolate op.").SetDefault(0);
    AddAttr<int>("out_h", "output height of interpolate op.").SetDefault(0);
    AddAttr<int>("out_w", "output width of interpolate op.").SetDefault(0);
    AddAttr<std::vector<float>>("scale", "scale_d factor of interpolate op.")
        .SetDefault(std::vector<float>{});
    AddAttr<std::string>("interp_method",
                         "(string, default \"bilinear\"), interpolation "
                         "method, can be \"linear\" for linear interpolation"
                         ",\"bilinear\" for "
                         "bilinear interpolation, \"trilinear\" for trilinear "
                         "interpolation and \"nearest\" for nearest "
                         "neighbor interpolation, and \"bicubic\" for bicubic"
                         "interpolation.")
        .SetDefault("bilinear");
    AddAttr<bool>(
        "align_corners",
        "an optional bool. Defaults to True. "
        "If True, the centers of 4 corner pixels of the input and output "
        "tensors are aligned, preserving the values at the corner pixels, "
        "If False, are not aligned")
        .SetDefault(true);
    AddAttr<int>("align_mode",
                 "(int, default \'1\'), optional for bilinear interpolation, "
                 "can be \'0\' for src_idx = scale*(dst_indx+0.5)-0.5 , "
                 "can be \'1\' for src_idx = scale*dst_index .")
        .SetDefault(1);
    AddComment(R"DOC(
          This operator samples input X to given output shape by using specified
          interpolation method, the interpolation methods can be \"nearest\"
          for nearest neighbor interpolation and \"bilinear\" for bilinear 
          interpolation and \"linear\" for linear interpolation..

          Nearest neighbor interpolation is to perform nearest neighbor interpolation
          in both the 3rd dimension(in height direction) and the 4th dimension(in width 
          direction) on input tensor.
           
          Linear interpolation is the method of using a line connecting two known quantities 
          to determine the value of an unknown quantity between the two known quantities. 
          
          Bilinear interpolation is an extension of linear interpolation for 
          interpolating functions of two variables (e.g. H-direction and 
          W-direction in this op) on a rectilinear 2D grid. The key idea is 
          to perform linear interpolation first in one direction, and then 
          again in the other direction.

          Trilinear interpolation is an extension of linear interpolation for 
          interpolating functions of three variables (e.g. D-direction, 
          H-direction and W-direction in this op) on a rectilinear 3D grid. 
          The linear interpolation is performed on three directions.

          Bicubic interpolation is an extension of cubic interpolation for interpolating
          data points on a two-dimensional regular grid. The interpolated surface is
          smoother than corresponding surfaces obtained by bilinear interpolation or
          nearest-neighbor interpolation.

          Align_corners and align_mode are optional parameters,the calculation method 
          of interpolation can be selected by them.
          
          Example:

          For scale:
          
            if align_corners = True and out_{size}>1 :

              scale_{factor} = (in_{size}-1.0)/(out_{size}-1.0)
            
            else:
              
              scale_{factor} = float(in_{size}/out_{size})
            
          
          Nearest neighbor interpolation:
          
          if:
              align_corners = False

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = \left \lfloor {H_{in} * scale_{}factor}} \right \rfloor
              W_out = \left \lfloor {W_{in} * scale_{}factor}} \right \rfloor

          else:
              align_corners = True

              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = round(H_{in} * scale_{factor})
              W_out = round(W_{in} * scale_{factor})

          Bilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:

              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

          Trilinear interpolation:

          if:
              align_corners = False , align_mode = 0
              
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:
              
              D_out = (D_{in}+0.5) * scale_{factor} - 0.5
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5


          else:
           
              input : (N,C,D_in,H_in,W_in)
              output: (N,C,D_out,H_out,W_out) where:

              D_out = D_{in} * scale_{factor}
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

          Bicubic interpolation:

          if:
              align_corners = False
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = (H_{in}+0.5) * scale_{factor} - 0.5
              W_out = (W_{in}+0.5) * scale_{factor} - 0.5
          else:
              input : (N,C,H_in,W_in)
              output: (N,C,H_out,W_out) where:
              H_out = H_{in} * scale_{factor}
              W_out = W_{in} * scale_{factor}

          For details of nearest neighbor interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation

          For details of bilinear interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Bilinear_interp_v2olation

          For details of trilinear interpolation, please refer to Wikipedia: 
          https://en.wikipedia.org/wiki/Trilinear_interp_v2olation

          For details of bicubic interpolation, please refer to Wikipedia:
          https://en.wikipedia.org/wiki/Bicubic_interpolation
         )DOC");
  }
};

class InterpolateV2OpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "InterpolateGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "InterpolateGrad");

    auto dim_x = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), dim_x);
    }
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const override {
    if (var_name == "SizeTensor" || var_name == "Scale") {
      return expected_kernel_type;
    }
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place(), tensor.layout());
  }
};

template <typename T>
class InterpolateV2GradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType(this->ForwardOpType() + "_grad");
    op->SetInput("X", this->Input("X"));
    if (this->HasInput("SizeTensor") > 0) {
      op->SetInput("SizeTensor", this->Input("SizeTensor"));
    }
    if (this->HasInput("OutSize") > 0) {
      op->SetInput("OutSize", this->Input("OutSize"));
    }
    if (this->HasInput("Scale") > 0) {
      op->SetInput("Scale", this->Input("Scale"));
    }
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(InterpolateV2GradNoNeedBufferVarsInferer,
                                    "X");

}  // namespace operators
}  // namespace paddle

// interp_v2 support scale_factor whose input type is list, this operation is
// not
// compatible with interp_op, so a new one is added in paddle2.0
namespace ops = paddle::operators;
REGISTER_OPERATOR(bilinear_interp_v2, ops::InterpolateV2Op,
                  ops::InterpolateV2OpMaker,
                  ops::InterpolateV2GradMaker<paddle::framework::OpDesc>,
                  ops::InterpolateV2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(bilinear_interp_v2_grad, ops::InterpolateV2OpGrad,
                  ops::InterpolateV2GradNoNeedBufferVarsInferer);
REGISTER_OPERATOR(nearest_interp_v2, ops::InterpolateV2Op,
                  ops::InterpolateV2OpMaker,
                  ops::InterpolateV2GradMaker<paddle::framework::OpDesc>,
                  ops::InterpolateV2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(nearest_interp_v2_grad, ops::InterpolateV2OpGrad,
                  ops::InterpolateV2GradNoNeedBufferVarsInferer);
REGISTER_OPERATOR(trilinear_interp_v2, ops::InterpolateV2Op,
                  ops::InterpolateV2OpMaker,
                  ops::InterpolateV2GradMaker<paddle::framework::OpDesc>,
                  ops::InterpolateV2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(trilinear_interp_v2_grad, ops::InterpolateV2OpGrad,
                  ops::InterpolateV2GradNoNeedBufferVarsInferer);
REGISTER_OPERATOR(bicubic_interp_v2, ops::InterpolateV2Op,
                  ops::InterpolateV2OpMaker,
                  ops::InterpolateV2GradMaker<paddle::framework::OpDesc>,
                  ops::InterpolateV2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(bicubic_interp_v2_grad, ops::InterpolateV2OpGrad,
                  ops::InterpolateV2GradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(bilinear_interp_v2, ops::InterpolateV2Kernel<float>,
                       ops::InterpolateV2Kernel<double>,
                       ops::InterpolateV2Kernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(bilinear_interp_v2_grad,
                       ops::InterpolateV2GradKernel<float>,
                       ops::InterpolateV2GradKernel<double>);
REGISTER_OP_CPU_KERNEL(nearest_interp_v2, ops::InterpolateV2Kernel<float>,
                       ops::InterpolateV2Kernel<double>,
                       ops::InterpolateV2Kernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(nearest_interp_v2_grad,
                       ops::InterpolateV2GradKernel<float>,
                       ops::InterpolateV2GradKernel<double>);
REGISTER_OP_CPU_KERNEL(trilinear_interp_v2, ops::InterpolateV2Kernel<float>,
                       ops::InterpolateV2Kernel<double>,
                       ops::InterpolateV2Kernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(trilinear_interp_v2_grad,
                       ops::InterpolateV2GradKernel<float>,
                       ops::InterpolateV2GradKernel<double>);
REGISTER_OPERATOR(linear_interp_v2, ops::InterpolateV2Op,
                  ops::InterpolateV2OpMaker,
                  ops::InterpolateV2GradMaker<paddle::framework::OpDesc>,
                  ops::InterpolateV2GradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(linear_interp_v2_grad, ops::InterpolateV2OpGrad,
                  ops::InterpolateV2GradNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(linear_interp_v2, ops::InterpolateV2Kernel<float>,
                       ops::InterpolateV2Kernel<double>,
                       ops::InterpolateV2Kernel<uint8_t>);
REGISTER_OP_CPU_KERNEL(linear_interp_v2_grad,
                       ops::InterpolateV2GradKernel<float>,
                       ops::InterpolateV2GradKernel<double>);
REGISTER_OP_CPU_KERNEL(bicubic_interp_v2, ops::InterpolateV2Kernel<float>,
                       ops::InterpolateV2Kernel<double>);
REGISTER_OP_CPU_KERNEL(bicubic_interp_v2_grad,
                       ops::InterpolateV2GradKernel<float>,
                       ops::InterpolateV2GradKernel<double>);
