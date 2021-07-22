// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#include <gtest/gtest.h>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"

DEFINE_string(dirname, "", "dirname to tests.");

namespace paddle {

class MkldnnQuantizerTest : public testing::Test {
 public:
  MkldnnQuantizerTest() {
    AnalysisConfig config(FLAGS_dirname);
    predictor = std::move(CreatePaddlePredictor(config));
    auto* predictor_p = static_cast<AnalysisPredictor*>(predictor.get());

    auto qconfig = new MkldnnQuantizerConfig();

    mkldnn_quantizer.reset(
        new AnalysisPredictor::MkldnnQuantizer(*predictor_p, qconfig));
  }

  std::pair<std::vector<int>, float> Histogram(
      const framework::LoDTensor& var_tensor, float min_val, float max_val,
      int num_bins) const {
    return mkldnn_quantizer->Histogram(var_tensor, min_val, max_val, num_bins);
  }

  std::pair<bool, framework::LoDTensor> GetMaxScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetMaxScalingFactor(var_tensor, is_unsigned);
  }

  std::pair<bool, framework::LoDTensor> GetMaxChScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetMaxChScalingFactor(var_tensor, is_unsigned, 0);
  }

  std::pair<bool, framework::LoDTensor> GetKLScalingFactor(
      const framework::LoDTensor& var_tensor, bool is_unsigned) const {
    return mkldnn_quantizer->GetKLScalingFactor(var_tensor, is_unsigned);
  }

  std::pair<bool, framework::LoDTensor> GetMaxChGRUScalingFactor(
      const framework::LoDTensor& wx_tensor,
      const framework::LoDTensor& wh_tensor) const {
    return mkldnn_quantizer->GetMaxChGRUScalingFactor(wx_tensor, wh_tensor);
  }

 protected:
  std::unique_ptr<PaddlePredictor> predictor;
  std::unique_ptr<AnalysisPredictor::MkldnnQuantizer> mkldnn_quantizer;
  float abs_error = 1e-6;
  static const std::array<float, 10> non_negative_values;
  static const std::array<float, 10> positive_and_negative_values;
};

const std::array<float, 10> MkldnnQuantizerTest::non_negative_values = {
    0.0158671, 0.026459,   0.0280772,  0.00962479, 0.0131628,
    0.016704,  0.00118407, 0.00765726, 0.0123213,  0.00944741};
const std::array<float, 10> MkldnnQuantizerTest::positive_and_negative_values =
    {-0.0482659, -0.0102493, -0.00794221, -0.00387115, -0.00674586,
     -0.0495346, 0.0629528,  -0.00531285, -0.0230353,  0.0269089};

TEST_F(MkldnnQuantizerTest, histogram_inverted_min_max) {
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, max_val, min_val, 3),
               platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, histogram_non_negative_to_3) {
  // all non-negative values
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_EQ(histogram[0], 4);
  ASSERT_EQ(histogram[1], 4);
  ASSERT_EQ(histogram[2], 2);
}

TEST_F(MkldnnQuantizerTest, histogram_positive_and_negative_to_3) {
  const auto& values = positive_and_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  std::vector<int> histogram;
  float bin_width;

  std::tie(histogram, bin_width) = Histogram(var_tensor, min_val, max_val, 3);

  ASSERT_NEAR(bin_width, std::abs(max_val - min_val) / 3.0f, abs_error)
      << "Improperly calculated bin_width.";

  ASSERT_EQ(histogram[0], 3);
  ASSERT_EQ(histogram[1], 5);
  ASSERT_EQ(histogram[2], 2);
}

TEST_F(MkldnnQuantizerTest, histogram_zero_bins) {
  const auto& values = non_negative_values;
  auto min_val = *std::min_element(values.begin(), values.end());
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  ASSERT_THROW(Histogram(var_tensor, min_val, max_val, 0),
               platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, histogram_empty) {
  // empty tensor
  ASSERT_THROW(Histogram({}, -1, 1, 1), platform::EnforceNotMet);

  // zero tensor
  framework::LoDTensor var_tensor;
  var_tensor.Resize({0});
  var_tensor.mutable_data<double>(platform::CPUPlace());

  ASSERT_THROW(Histogram(var_tensor, -1, 1, 1), platform::EnforceNotMet);
}

TEST_F(MkldnnQuantizerTest, kl_scaling_factor_signed) {
  const auto& values = positive_and_negative_values;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetKLScalingFactor(var_tensor, false);

  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / 0.0899106152344, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_signed) {
  const auto& values = positive_and_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxScalingFactor(var_tensor, false);

  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / max_val, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_unsigned) {
  const auto& values = non_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / max_val, abs_error);
}

TEST_F(MkldnnQuantizerTest, max_scaling_factor_chwise_unsigned) {
  const auto& values = non_negative_values;
  auto max_val = *std::max_element(values.begin(), values.end());
  int channels = 3;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(channels, 1, 1, values.size()));
  for (int i = 0; i < channels; i++)
    std::copy(begin(values), end(values),
              var_tensor.mutable_data<float>(platform::CPUPlace()) +
                  i * values.size());

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetMaxChScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), channels);
  for (int i = 0; i < channels; i++) {
    ASSERT_NEAR(lod_tensor.data<double>()[i], 1.0 / max_val, abs_error);
  }
  printf("--- max_scaling_factor_chwise_unsigned ---\n");
}

TEST_F(MkldnnQuantizerTest, kl_scaling_factor_unsigned) {
  const auto& values = non_negative_values;

  framework::LoDTensor var_tensor;
  var_tensor.Resize(framework::make_dim(values.size()));
  std::copy(begin(values), end(values),
            var_tensor.mutable_data<float>(platform::CPUPlace()));

  bool is_unsigned;
  framework::LoDTensor lod_tensor;

  std::tie(is_unsigned, lod_tensor) = GetKLScalingFactor(var_tensor, true);

  ASSERT_EQ(is_unsigned, true);
  ASSERT_EQ(lod_tensor.numel(), 1);
  ASSERT_NEAR(lod_tensor.data<double>()[0], 1.0 / 0.0252845321362, abs_error);
  printf("--- kl_scaling_factor_unsigned ---\n");
}

const std::vector<std::vector<float>> wx = {
    {0.04347931, -0.5643393, 0.7551297, 0.26713502, 0.8055306, 0.91144973},
    {0.01707571, 0.12741385, 0.15419468, 0.66127586, 0.46821925, 0.9665961},
    {0.40393898, 0.884427, -0.5853097, 0.5840954, 0.9170512, 0.98245513}};
const std::vector<std::vector<float>> wh = {
    {0.42484227, -0.9025513, 0.17087583, 0.8403284, 0.03325734, 0.92331886},
    {0.32630175, 0.41691914, 0.99848574, 0.3504407, 0.06707559, 0.62239844}};

TEST_F(MkldnnQuantizerTest, max_ch_gru_scaling_factor) {
  framework::LoDTensor wx_tensor, wh_tensor, lod_tensor;

  wx_tensor.Resize(framework::make_dim(wx.size(), wx[0].size()));
  for (size_t i = 0; i < wx.size(); i++)
    std::copy(
        begin(wx[i]), end(wx[i]),
        wx_tensor.mutable_data<float>(platform::CPUPlace()) + i * wx[0].size());

  wh_tensor.Resize(framework::make_dim(wh.size(), wh[0].size()));
  for (size_t i = 0; i < wh.size(); i++)
    std::copy(
        begin(wh[i]), end(wh[i]),
        wh_tensor.mutable_data<float>(platform::CPUPlace()) + i * wh[0].size());

  bool is_unsigned;
  std::tie(is_unsigned, lod_tensor) =
      GetMaxChGRUScalingFactor(wx_tensor, wh_tensor);

  std::vector<double> scales = {2.35381475, 1.08304947, 1.32427582,
                                1.19001095, 1.00151656, 1.01785819};
  ASSERT_EQ(is_unsigned, false);
  ASSERT_EQ(lod_tensor.numel(), static_cast<int64_t>(scales.size()));
  for (int64_t i = 0; i < lod_tensor.numel(); i++) {
    ASSERT_NEAR(lod_tensor.data<double>()[i], scales[i], abs_error);
  }
  printf("--- max_ch_gru_scaling_factor ---\n");
  ASSERT_EQ(true, false);  // trigger output printing
}
}  // namespace paddle
