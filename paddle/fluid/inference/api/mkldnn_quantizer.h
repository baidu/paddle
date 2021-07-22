// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once
#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif

namespace paddle {

/*
 * Map variable name to tensor of scaling factors scaling it to MAX=1.0.
 * bool denotes whether quantization of the variable should be done to unsigned
 * type.
 */
using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, framework::LoDTensor>>;

class AnalysisPredictor::MkldnnQuantizer {
 public:
  explicit MkldnnQuantizer(AnalysisPredictor& predictor,  // NOLINT
                           const MkldnnQuantizerConfig* qconfig)
      : predictor_(predictor), qconfig_(qconfig) {}

  // Execute full quantization procedure.
  bool Quantize();

#ifdef PADDLE_WITH_TESTING
  friend class MkldnnQuantizerTest;
#endif

 private:
  // Run single warmup iteration
  bool RunWarmup() const;
  // Gather data from variables and calculate scales for them.
  bool CalculateScales();
  // Calculate a scale for tensor based on ScaleAlgo rules.
  void CalculateSingleScale(const std::string& op_name,
                            const std::string& conn_name,
                            const std::string& var_name,
                            const framework::LoDTensor& var_tensor,
                            bool is_unsigned);
  void CalculateSingleGRUWeightsScale(const std::string& var_name,
                                      const framework::LoDTensor& var_tensor);
  void CalculateScalesForGRUWeights(const paddle::framework::OpDesc* op);
  void CalculateScalesForOpOutputs(const paddle::framework::OpDesc* op);
  void CalculateScalesForOpInputs(const paddle::framework::OpDesc* op);
  void PrepareArgument() const;
  void ClearDeviceContext() const;
  bool RunQuantizePasses() const;

  std::vector<int> ExpandQuantizedBins(std::vector<int> quantized_bins,
                                       std::vector<int> reference_bins) const;

  // Using the KL-divergence method get the most precise scaling factor.
  std::pair<bool, framework::LoDTensor> GetKLScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const;

  std::pair<bool, framework::LoDTensor> GetMaxChScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned,
      bool is_transposed) const;

  std::pair<bool, framework::LoDTensor> GetMaxChGRUScalingFactor(
      const framework::LoDTensor& wx_tensor,
      const framework::LoDTensor& wh_tensor) const;

  std::pair<bool, framework::LoDTensor> GetMaxScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const;

  // Returns histogram and bin width
  std::pair<std::vector<int>, float> Histogram(
      const framework::LoDTensor& var_tensor, float min_val, float max_val,
      size_t num_bins = 2048) const;

  // Calculate the entropy.
  float SafeEntropy(std::vector<int> reference_distr_P, int P_sum,
                    std::vector<int> candidate_distr_Q, int Q_sum) const;

 private:
  AnalysisPredictor& predictor_;
  const MkldnnQuantizerConfig* qconfig_;

  // A map: variable name -> scale
  VarQuantScale scales_;
};

}  // namespace paddle
