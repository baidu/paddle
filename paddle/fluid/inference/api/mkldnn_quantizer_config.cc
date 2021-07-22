// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "paddle/fluid/inference/api/paddle_mkldnn_quantizer_config.h"

namespace paddle {

MkldnnQuantizerConfig::MkldnnQuantizerConfig() {
  // The default configuration of scale computing algorightms
  rules_["conv2d"]["Input"] = ScaleAlgo::KL;
  rules_["conv2d"]["Filter"] = ScaleAlgo::MAX_CH;
  rules_["conv2d"]["Bias"] = ScaleAlgo::NONE;  // do not compute scale
  rules_["conv2d"]["ResidualData"] = ScaleAlgo::KL;
  rules_["conv2d"]["Output"] = ScaleAlgo::KL;

  rules_["pool2d"]["X"] = ScaleAlgo::KL;
  rules_["pool2d"]["Out"] = ScaleAlgo::KL;

  rules_["concat"]["X"] = ScaleAlgo::KL;
  rules_["concat"]["Out"] = ScaleAlgo::KL;

  rules_["prior_box"]["Input"] = ScaleAlgo::KL;
  rules_["prior_box"]["Image"] = ScaleAlgo::NONE;
  rules_["prior_box"]["Boxes"] = ScaleAlgo::NONE;
  rules_["prior_box"]["Variances"] = ScaleAlgo::NONE;

  // Transpose2 does not perform calculation on the data. Scale is calculated on
  // input data and assign to Quantize and Dequantize scale.
  rules_["transpose2"]["X"] = ScaleAlgo::KL;
  rules_["transpose2"]["Out"] = ScaleAlgo::NONE;

  rules_["fc"]["Input"] = ScaleAlgo::KL;
  rules_["fc"]["W"] = ScaleAlgo::MAX_CH_T;
  rules_["fc"]["Bias"] = ScaleAlgo::NONE;
  rules_["fc"]["Out"] = ScaleAlgo::KL;

  rules_["matmul"]["X"] = ScaleAlgo::KL;
  rules_["matmul"]["Y"] = ScaleAlgo::KL;
  rules_["matmul"]["Out"] = ScaleAlgo::KL;

  rules_["elementwise_add"]["X"] = ScaleAlgo::KL;
  rules_["elementwise_add"]["Y"] = ScaleAlgo::KL;
  rules_["elementwise_add"]["Out"] = ScaleAlgo::KL;

  rules_["elementwise_mul"]["X"] = ScaleAlgo::KL;
  rules_["elementwise_mul"]["Y"] = ScaleAlgo::KL;
  rules_["elementwise_mul"]["Out"] = ScaleAlgo::KL;

  // Reshape2 does not perform calculation on the data and shapes are not
  // changed. Scale is calculated on input data and assign to Quantize and
  // Dequantize scale.
  rules_["reshape2"]["X"] = ScaleAlgo::KL;
  rules_["reshape2"]["Shape"] = ScaleAlgo::NONE;
  rules_["reshape2"]["ShapeTensor"] = ScaleAlgo::NONE;
  rules_["reshape2"]["XShape"] = ScaleAlgo::NONE;
  rules_["reshape2"]["Out"] = ScaleAlgo::NONE;

  rules_["fusion_gru"]["X"] = ScaleAlgo::KL;
  rules_["fusion_gru"]["H0"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["Bias"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["WeightX"] = ScaleAlgo::NONE;  // Weights will be handled
  rules_["fusion_gru"]["WeightH"] = ScaleAlgo::NONE;  // separately
  rules_["fusion_gru"]["ReorderedH0"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["XX"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["BatchedInput"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["BatchedOut"] = ScaleAlgo::NONE;
  rules_["fusion_gru"]["Hidden"] = ScaleAlgo::KL;

  rules_["multi_gru"]["X"] = ScaleAlgo::KL;
  rules_["multi_gru"]["Bias"] = ScaleAlgo::NONE;
  rules_["multi_gru"]["WeightX"] = ScaleAlgo::NONE;  // Weights will be handled
  rules_["multi_gru"]["WeightH"] = ScaleAlgo::NONE;  // separately
  rules_["multi_gru"]["Scale_weights"] = ScaleAlgo::NONE;
  rules_["multi_gru"]["Hidden"] = ScaleAlgo::KL;
}

ScaleAlgo MkldnnQuantizerConfig::scale_algo(
    const std::string& op_type_name, const std::string& conn_name) const {
  if (rules_.find(op_type_name) != rules_.end()) {
    auto op_rule = rules_.at(op_type_name);
    if (op_rule.find(conn_name) != op_rule.end()) return op_rule.at(conn_name);
  }
  return default_scale_algo_;
}

}  // namespace paddle
