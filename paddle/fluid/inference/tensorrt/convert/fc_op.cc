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

#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"

namespace paddle {
namespace framework {
class Scope;

namespace proto {
class OpDesc;
}  // namespace proto
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace inference {
namespace tensorrt {

/*
 * FC converter convert a MUL op in Fluid to a FC layer in TRT.
 */
class FcOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid fc op to tensorrt fc layer without bias";
    framework::OpDesc op_desc(op, nullptr);
    auto output_name = op_desc.Output("Out").front();
    auto input_names = op_desc.InputNames();
    bool with_bias = input_names.size() >= 3;
    std::string w_name = "Y";
    std::string i_name = "X";
    if (with_bias) {
      w_name = "W";
      i_name = "Input";
    }
    // Declare inputs
    auto* X = engine_->GetITensor(op_desc.Input(i_name).front());
    auto x_dim = X->getDimensions();
    // Declare weights
    auto* Y_v = scope.FindVar(op_desc.Input(w_name).front());
    PADDLE_ENFORCE_NOT_NULL(
        Y_v, platform::errors::NotFound(
                 "Can not find %s presistale var of fc in scope.", w_name));
    auto* Y_t = Y_v->GetMutable<framework::LoDTensor>();
    int x_num_col_dims =
        op_desc.HasAttr("x_num_col_dims")
            ? BOOST_GET_CONST(int, op_desc.GetAttr("x_num_col_dims"))
            : (op_desc.HasAttr("in_num_col_dims")
                   ? BOOST_GET_CONST(int, op_desc.GetAttr("in_num_col_dims"))
                   : 1);
    const std::string activation_type =
        op_desc.HasAttr("activation_type")
            ? BOOST_GET_CONST(std::string, op_desc.GetAttr("activation_type"))
            : "";
    // This may trigger a GPU->CPU copy, because TRT's weight can only be
    // assigned from CPU memory, which can't be avoided.
    float* weight_data = nullptr;
    bool enable_int8 = op_desc.HasAttr("enable_int8");
    float in_scale = 0.;
    if (enable_int8) {
#if IS_TRT_VERSION_GE(5000)
      CHECK(op_desc.HasAttr(i_name + "_scale"));
      in_scale =
          BOOST_GET_CONST(float, op_desc.GetAttr(i_name + "_scale")) * 127;
      auto weight_scale =
          BOOST_GET_CONST(std::vector<float>, op_desc.GetAttr("weight_scale"));
      weight_data = engine_->GetWeightCPUData(op_desc.Input(w_name).front(),
                                              Y_t, true, weight_scale);
      engine_->SetTensorDynamicRange(X, in_scale);
#endif
    } else {
      weight_data =
          engine_->GetWeightCPUData(op_desc.Input(w_name).front(), Y_t, false);
    }

    PADDLE_ENFORCE_EQ(Y_t->dims().size(), 2UL,
                      platform::errors::InvalidArgument(
                          "The fc's weight should be a matrix with 2 dims, but "
                          "it's %d-dimensional.",
                          Y_t->dims().size()));  // a matrix
    size_t n_output = Y_t->dims()[1];

    int m = Y_t->dims()[0];
    int n = Y_t->dims()[1];

    auto tranpose_weight = [](const float* src, float* dst, int m, int n) {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          dst[j * m + i] = src[i * n + j];
        }
      }
    };

    auto regist_fc = [&](nvinfer1::ITensor* inputs, int n_output,
                         TensorRTEngine::Weight& weight,
                         TensorRTEngine::Weight& bias) {
      if (enable_int8) {
        // add conv layer
        PADDLE_ENFORCE_EQ(
            op_desc.HasAttr("out_threshold"), true,
            platform::errors::InvalidArgument(
                "must have out threshold in fc layers in int8 mode"));
        float out_scale =
            BOOST_GET_CONST(float, op_desc.GetAttr("out_threshold"));
        nvinfer1::DimsHW nv_ksize(1, 1);
        auto* fc_layer_int8 =
            TRT_ENGINE_ADD_LAYER(engine_, Convolution, *inputs, n_output,
                                 nv_ksize, weight.get(), bias.get());
        engine_->SetTensorDynamicRange(fc_layer_int8->getOutput(0), out_scale);
        if (activation_type == "relu") {
          nvinfer1::IActivationLayer* relu_layer_int8 = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_layer_int8->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_int8, "relu_after_fc_shuffle",
                                   {output_name}, test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_int8, "shuffle_after_fc",
                                   {output_name}, test_mode);
        }
      } else {
        // add fc layer
        auto* fc_layer_before =
            TRT_ENGINE_ADD_LAYER(engine_, FullyConnected, *inputs, n_output,
                                 weight.get(), bias.get());
        fc_layer_before->setName(
            ("fc_layer_before(Output: " + output_name + ")").c_str());
        // add shuffle after fc
        nvinfer1::Dims reshape_after_fc_dim;
        if (engine_->use_oss() && engine_->with_ernie() && x_dim.nbDims == 4 &&
            x_dim.d[2] == 1 && x_dim.d[3] == 1 && x_num_col_dims == 1) {
          // If use tensorrt'oss, the x_dim and x_num_col_dims need change
          reshape_after_fc_dim.nbDims = 4;
        } else {
          reshape_after_fc_dim.nbDims = x_num_col_dims + 1;
        }
        for (int i = 0; i < reshape_after_fc_dim.nbDims; i++) {
          reshape_after_fc_dim.d[i] = 0;
        }
        auto* fc_layer_float = TRT_ENGINE_ADD_LAYER(
            engine_, Shuffle, *fc_layer_before->getOutput(0));
        fc_layer_float->setReshapeDimensions(reshape_after_fc_dim);
        if (activation_type == "relu") {
          nvinfer1::IActivationLayer* relu_layer_float = TRT_ENGINE_ADD_LAYER(
              engine_, Activation, *(fc_layer_float->getOutput(0)),
              nvinfer1::ActivationType::kRELU);
          RreplenishLayerAndOutput(relu_layer_float, "relu_after_fc_shuffle",
                                   {output_name}, test_mode);
        } else {
          RreplenishLayerAndOutput(fc_layer_float, "shuffle_after_fc",
                                   {output_name}, test_mode);
        }
      }
    };

    std::vector<float> weight_data_tmp;
    weight_data_tmp.reserve(Y_t->numel());
    memcpy(weight_data_tmp.data(), weight_data, Y_t->numel() * sizeof(float));
    tranpose_weight(weight_data_tmp.data(), weight_data, m, n);

    TensorRTEngine::Weight weight{nvinfer1::DataType::kFLOAT,
                                  static_cast<void*>(weight_data),
                                  static_cast<size_t>(Y_t->numel())};
    weight.dims.assign({n, m});

    float* bias_data = nullptr;
    int bias_num = 0;
    if (with_bias) {
      auto* b_v = scope.GetVar(op_desc.Input("Bias").front());
      auto* b_t = b_v->GetMutable<framework::LoDTensor>();
      bias_data =
          engine_->GetWeightCPUData(op_desc.Input("Bias").front(), b_t, false);
      bias_num = b_t->numel();
    }
    TensorRTEngine::Weight bias{nvinfer1::DataType::kFLOAT,
                                static_cast<void*>(bias_data),
                                static_cast<size_t>(bias_num)};

    // Running the TRT Static Shape mode: x_num_col_dims-1
    if (!engine_->with_dynamic_shape()) {
      x_num_col_dims--;
    }
    // If use tensorrt'oss, the x_dim and x_num_col_dims need change
    if (engine_->use_oss() && engine_->with_ernie() && x_dim.nbDims == 4 &&
        x_dim.d[2] == 1 && x_dim.d[3] == 1 && x_num_col_dims == 2) {
      x_num_col_dims = 1;
    }
    PADDLE_ENFORCE_GT(
        x_dim.nbDims, x_num_col_dims,
        platform::errors::InvalidArgument(
            "Params and input dims mismatch. Paddle-TRT FC "
            "converter expects x_dim.nbDims > x_num_col_dims, but "
            "x_dim.nbDims : %d, x_num_col_dims : %d.",
            x_dim.nbDims, x_num_col_dims));
    // add shuffle before fc
    nvinfer1::Dims reshape_before_fc_dim;
    reshape_before_fc_dim.nbDims = x_num_col_dims + 3;
    // padding shape "* x q x 1 x 1"
    for (int i = 0; i < reshape_before_fc_dim.nbDims; i++) {
      reshape_before_fc_dim.d[i] = 1;
    }
    for (int i = 0; i < x_dim.nbDims; i++) {
      if (i < x_num_col_dims) {
        reshape_before_fc_dim.d[i] = 0;
      } else {
        if (x_dim.d[i] < 0) {
          reshape_before_fc_dim.d[x_num_col_dims] = -1;
          break;
        }
        reshape_before_fc_dim.d[x_num_col_dims] *= x_dim.d[i];
      }
    }
    auto* reshape_before_fc_layer = TRT_ENGINE_ADD_LAYER(engine_, Shuffle, *X);
    reshape_before_fc_layer->setReshapeDimensions(reshape_before_fc_dim);
    reshape_before_fc_layer->setName(
        ("shuffle_before_fc(Output: " + output_name + ")").c_str());
    auto* reshape_itensor = reshape_before_fc_layer->getOutput(0);
    if (enable_int8) {
      engine_->SetTensorDynamicRange(reshape_itensor, in_scale);
    }
    regist_fc(reshape_itensor, n_output, weight, bias);
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(fc, FcOpConverter);
