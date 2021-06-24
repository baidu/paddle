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
 * MatMulOp, IMatrixMultiplyLayer in TRT. This Layer doesn't has weights.
 */
class MatMulOpConverter : public OpConverter {
 public:
  void operator()(const framework::proto::OpDesc& op,
                  const framework::Scope& scope, bool test_mode) override {
    VLOG(3) << "convert a fluid matmul op to tensorrt mul layer without bias";

    framework::OpDesc op_desc(op, nullptr);
    // Declare inputs
    auto* input1 = engine_->GetITensor(op_desc.Input("X")[0]);
    auto* input2 = engine_->GetITensor(op_desc.Input("Y")[0]);

    bool transpose_X = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_X"));
    bool transpose_Y = BOOST_GET_CONST(bool, op_desc.GetAttr("transpose_Y"));

    nvinfer1::MatrixOperation matrix_operation_X =
        transpose_X ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;
    nvinfer1::MatrixOperation matrix_operation_Y =
        transpose_Y ? nvinfer1::MatrixOperation::kTRANSPOSE
                    : nvinfer1::MatrixOperation::kNONE;

    auto* layer =
        TRT_ENGINE_ADD_LAYER(engine_, MatrixMultiply, *input1,
                             matrix_operation_X, *input2, matrix_operation_Y);

    float alpha = BOOST_GET_CONST(float, op_desc.GetAttr("alpha"));
    auto output_name = op_desc.Output("Out")[0];
    if (fabs(alpha - 1.0) < std::numeric_limits<float>::epsilon()) {
      engine_->SetITensor(output_name, layer->getOutput(0));
    } else {
      auto create_weights = [&](float data, const std::string& type) -> float* {
        std::unique_ptr<framework::Tensor> tmp_tensor(new framework::Tensor());
        tmp_tensor->Resize({1});
        auto* tmp_data = tmp_tensor->mutable_data<float>(platform::CPUPlace());
        tmp_data[0] = data;
        engine_->SetWeights(output_name + "_add_scale_op_" + type,
                            std::move(tmp_tensor));
        return tmp_data;
      };
      float* alpha_data = create_weights(alpha, "alpha");
      float* shift_data = create_weights(0.0, "shift");
      float* power_data = create_weights(1.0, "power");
      TensorRTEngine::Weight nv_alpha{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(alpha_data), 1};
      TensorRTEngine::Weight nv_shift{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(shift_data), 1};
      TensorRTEngine::Weight nv_power{nvinfer1::DataType::kFLOAT,
                                      static_cast<void*>(power_data), 1};
      auto* scale_layer = TRT_ENGINE_ADD_LAYER(
          engine_, Scale, *layer->getOutput(0), nvinfer1::ScaleMode::kUNIFORM,
          nv_shift.get(), nv_alpha.get(), nv_power.get());
      engine_->SetITensor(output_name, scale_layer->getOutput(0));
    }
    if (test_mode) {  // the test framework can not determine which is the
                      // output, so place the declaration inside.
      engine_->DeclareOutput(output_name);
    }
  }
};

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle

REGISTER_TRT_OP_CONVERTER(matmul, MatMulOpConverter);
