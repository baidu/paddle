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

#include "paddle/fluid/inference/api/mkldnn_quantizer.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/pass.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/analysis_predictor.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {

using platform::CPUPlace;
using framework::LoDTensor;
using framework::Variable;
using framework::ir::Graph;
using ConstEigenVectorArrayMap =
    Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
using EigenMatrixDoubleArray =
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using EigenMatrixArray =
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ConstEigenMatrixArrayMap = Eigen::Map<const EigenMatrixArray>;
using string::PrettyLogH1;
using VariableNameMap = std::map<std::string, std::vector<std::string>>;
static LoDTensor CreateScaleTensor(int64_t channels_num = 1);

static void check_var(const Variable* var, const std::string& var_name) {
  PADDLE_ENFORCE_NOT_NULL(var, platform::errors::PreconditionNotMet(
                                   "%s is not in the scope", var_name));
  PADDLE_ENFORCE_EQ(
      var->IsType<LoDTensor>(), true,
      platform::errors::PreconditionNotMet("Only support lod tensor now."));
}

static void check_tensor(const LoDTensor& tensor) {
  printf("******* check_tensor ********\n");
  PADDLE_ENFORCE_GT(tensor.dims().size(), 0, platform::errors::InvalidArgument(
                                                 "Tensor dimension is empty."));
}

void AnalysisPredictor::MkldnnQuantizer::CalculateScalesForGRUWeights(
    const paddle::framework::OpDesc* op) {
  const auto& wx_names = op->Input("WeightX");
  const auto& wh_names = op->Input("WeightH");
  for (size_t i = 0; i < wx_names.size(); ++i) {
    const auto& wx_name = wx_names[i];
    const auto& wh_name = wh_names[i];
    auto* wx_var = predictor_.sub_scope_->FindVar(wx_name);
    auto* wh_var = predictor_.sub_scope_->FindVar(wh_name);
    check_var(wx_var, wx_name);
    check_var(wh_var, wh_name);
    LoDTensor* wx_tensor = wx_var->GetMutable<LoDTensor>();
    LoDTensor* wh_tensor = wh_var->GetMutable<LoDTensor>();
    scales_[wx_name] = GetMaxChGRUScalingFactor(*wx_tensor, *wh_tensor);
  }
  printf("******* CalculateScalesForGRUWeights ********\n");
}

void AnalysisPredictor::MkldnnQuantizer::CalculateScalesForOpInputs(
    const paddle::framework::OpDesc* op) {
  if (op->Type() == "fusion_gru" || op->Type() == "multi_gru") {
    CalculateScalesForGRUWeights(op);
  }
  for (auto const& input : op->Inputs()) {
    for (const auto& var_name : input.second) {
      // skip if scale already computed
      if (scales_.find(var_name) != scales_.end()) continue;
      auto* var = predictor_.sub_scope_->FindVar(var_name);
      check_var(var, var_name);
      LoDTensor* var_tensor = var->GetMutable<LoDTensor>();
      // force unsigned type if already know it
      bool is_unsigned = false;
      CalculateSingleScale(op->Type(), input.first, var_name, *var_tensor,
                           is_unsigned);
    }
  }
  printf("******* CalculateScalesForOpInputs ********\n");
}

void AnalysisPredictor::MkldnnQuantizer::CalculateScalesForOpOutputs(
    const paddle::framework::OpDesc* op) {
  for (auto const& output : op->Outputs()) {
    for (const auto& var_name : output.second) {
      // skip if scale already computed
      if (scales_.find(var_name) != scales_.end()) continue;
      auto* var = predictor_.sub_scope_->FindVar(var_name);
      check_var(var, var_name);
      LoDTensor* var_tensor = var->GetMutable<LoDTensor>();
      // force unsigned type if already know it
      bool is_unsigned = false;
      bool compute_scale = true;
      if (op->Type() == "conv2d" || op->Type() == "fc") {
        // output of conv2d with relu must be unsigned
        std::string fuse_activation =
            op->GetAttrIfExists<std::string>("fuse_activation");
        is_unsigned = (fuse_activation == "relu" || fuse_activation == "relu6");
      } else if (op->Type() == "relu") {
        is_unsigned = true;
      } else if (op->Type() == "transpose2" || op->Type() == "reshape2" ||
                 op->Type() == "pool2d") {
        auto input_var_name = op->Input("X")[0];
        PADDLE_ENFORCE_NE(scales_.find(input_var_name), scales_.end(),
                          platform::errors::PreconditionNotMet(
                              "Input scales must be calculated before the "
                              "output scales to infer if output is unsigned."));
        if (scales_.find(input_var_name) != scales_.end()) {
          scales_[var_name] = scales_[input_var_name];
        }
        compute_scale = false;
      } else if (op->Type() == "concat") {
        // output of ops with unsigned input must be unsigned
        is_unsigned = true;
        double min_scale = std::numeric_limits<double>::max();
        for (auto input_var_name : op->Input("X")) {
          PADDLE_ENFORCE_NE(
              scales_.find(input_var_name), scales_.end(),
              platform::errors::PreconditionNotMet(
                  "Input scales must be calculated before the "
                  "output scales to infer if output is unsigned."));
          is_unsigned = is_unsigned && scales_[input_var_name].first;
          min_scale = std::min(
              min_scale, scales_[input_var_name].second.data<double>()[0]);
        }
        auto scale_tensor = CreateScaleTensor();
        scale_tensor.data<double>()[0] = min_scale;
        scales_[var_name] = {is_unsigned, scale_tensor};
        compute_scale = false;
      }
      if (compute_scale) {
        CalculateSingleScale(op->Type(), output.first, var_name, *var_tensor,
                             is_unsigned);
      }
    }
  }
  printf("******* CalculateScalesForOpOutputs ********\n");
}

bool AnalysisPredictor::MkldnnQuantizer::CalculateScales() {
  PrettyLogH1("--- Calculating scales for quantization");
  std::map<std::string, std::map<std::string, LoDTensor>> gathered_data;
  for (const auto* op : predictor_.inference_program_->Block(0).AllOps()) {
    if (platform::HasOpINT8DataType(op)) {
      // handle inputs first to let is_unsigned be inferred for the outputs
      printf("******* CalculateScales ********\n");
      CalculateScalesForOpInputs(op);
      CalculateScalesForOpOutputs(op);
    }
  }
  return true;
}

void AnalysisPredictor::MkldnnQuantizer::CalculateSingleScale(
    const std::string& op_type_name, const std::string& conn_name,
    const std::string& var_name, const LoDTensor& var_tensor,
    bool is_unsigned) {
  auto rule = qconfig_->scale_algo(op_type_name, conn_name);
  if (rule == ScaleAlgo::NONE) return;

  PADDLE_ENFORCE_GT(
      var_tensor.numel(), 0,
      platform::errors::InvalidArgument(
          "MkldnnQuantizer: LoDTensor of variable %s for quantization of op "
          "%s of connection %s should not be empty.",
          var_name, op_type_name, conn_name));

  switch (rule) {
    case ScaleAlgo::MAX:
      scales_[var_name] = GetMaxScalingFactor(var_tensor, is_unsigned);
      break;
    case ScaleAlgo::MAX_CH:
      scales_[var_name] = GetMaxChScalingFactor(var_tensor, is_unsigned,
                                                /*is_transposed*/ false);
      break;
    case ScaleAlgo::MAX_CH_T:
      scales_[var_name] = GetMaxChScalingFactor(var_tensor, is_unsigned,
                                                /*is_transposed*/ true);
      break;
    case ScaleAlgo::KL:
      scales_[var_name] = GetKLScalingFactor(var_tensor, is_unsigned);
      break;
    default:
      throw std::runtime_error(
          "MkldnnQuantizer: Unexpected ScaleAlgo specified.");
  }
}

static LoDTensor CreateScaleTensor(int64_t channels_num) {
  LoDTensor scale_tensor;
  scale_tensor.Resize({channels_num});
  scale_tensor.mutable_data<double>(CPUPlace());
  return scale_tensor;
}

std::vector<int> AnalysisPredictor::MkldnnQuantizer::ExpandQuantizedBins(
    std::vector<int> quantized_bins, std::vector<int> reference_bins) const {
  std::vector<int> expanded_quantized_bins(reference_bins.size(), 0);
  int num_merged_bins = reference_bins.size() / quantized_bins.size();
  int j_start = 0;
  int j_end = num_merged_bins;
  for (size_t idx = 0; idx < quantized_bins.size(); idx++) {
    int zero_count =
        std::count(&reference_bins[j_start], &reference_bins[j_end], 0);
    num_merged_bins = j_end - j_start;
    int avg_bin_ele;
    if (zero_count == num_merged_bins) {
      avg_bin_ele = 0;
    } else {
      avg_bin_ele = quantized_bins[idx] / (num_merged_bins - zero_count + 0.0);
    }
    for (int idx1 = j_start; idx1 < j_end; idx1++) {
      expanded_quantized_bins[idx1] =
          (reference_bins[idx1] == 0) ? 0 : avg_bin_ele;
    }
    j_start += num_merged_bins;
    j_end += num_merged_bins;
    if ((idx + 1) == quantized_bins.size() - 1) {
      j_end = reference_bins.size();
    }
  }
  return expanded_quantized_bins;
}

std::pair<bool, LoDTensor>
AnalysisPredictor::MkldnnQuantizer::GetKLScalingFactor(
    const LoDTensor& var_tensor, bool is_unsigned) const {
  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  int precision_hist_num_bins = 2048;
  float max_val = eigen_tensor.maxCoeff();
  float min_val = eigen_tensor.minCoeff();
  bool is_positive = min_val >= 0.0f;
  if (is_unsigned)
    PADDLE_ENFORCE_EQ(
        is_positive, true,
        platform::errors::InvalidArgument(
            "Tensor is claimed to be unsigned, but its min value (%f) is < 0.0",
            min_val));

  int num_quantized_bins = 255;

  std::vector<int> hist;
  float bin_width;
  int starting_iter;
  int ending_iter = precision_hist_num_bins - 1;
  if (is_positive) {
    std::tie(hist, bin_width) =
        Histogram(var_tensor, min_val, max_val, precision_hist_num_bins);
    starting_iter = static_cast<int>(ending_iter * 0.7);
  } else {
    float th = std::max(std::abs(max_val), std::abs(min_val));
    std::tie(hist, bin_width) =
        Histogram(var_tensor, -th, th, precision_hist_num_bins);
    starting_iter = 0;
    if (std::abs(max_val) > std::abs(min_val)) {
      while (starting_iter < ending_iter) {
        if (hist[starting_iter] == 0) {
          ++starting_iter;
          continue;
        } else {
          break;
        }
      }
      starting_iter += static_cast<int>((ending_iter - starting_iter) * 0.6);
    } else {
      while (ending_iter > 0) {
        if (hist[ending_iter] == 0) {
          --ending_iter;
          continue;
        } else {
          break;
        }
      }
      starting_iter = static_cast<int>(0.6 * ending_iter);
    }
  }
  auto P_sum = eigen_tensor.size();
  int min_kl_divergence = 0;
  int min_kl_index = 0;
  bool kl_inited = false;
  for (int i = starting_iter; i <= ending_iter; i++) {
    std::vector<int> reference_distr_P(&hist[0], &hist[i]);
    auto outliers_count =
        std::accumulate(&hist[i], &hist[precision_hist_num_bins], 0);
    if (reference_distr_P[i - 1] == 0) {
      continue;
    }
    reference_distr_P[i - 1] += outliers_count;
    auto reference_distr_bins = reference_distr_P;
    std::vector<int> candidate_distr_Q(&hist[0], &hist[i]);
    int num_merged_bins = i / num_quantized_bins;
    std::vector<int> candidate_distr_Q_quantized(num_quantized_bins, 0);
    int j_start = 0;
    int j_end = num_merged_bins;
    for (int idx = 0; idx < num_quantized_bins; idx++) {
      candidate_distr_Q_quantized[idx] = std::accumulate(
          &candidate_distr_Q[j_start], &candidate_distr_Q[j_end], 0);
      j_start += num_merged_bins;
      j_end += num_merged_bins;
      if ((idx + 1) == num_quantized_bins - 1) {
        j_end = i;
      }
    }
    candidate_distr_Q =
        ExpandQuantizedBins(candidate_distr_Q_quantized, reference_distr_bins);
    int Q_sum =
        std::accumulate(candidate_distr_Q.begin(), candidate_distr_Q.end(), 0);
    auto kl_divergence =
        SafeEntropy(reference_distr_P, P_sum, candidate_distr_Q, Q_sum);
    if (!kl_inited) {
      min_kl_divergence = kl_divergence;
      min_kl_index = i;
      kl_inited = true;
    } else if (kl_divergence < min_kl_divergence) {
      min_kl_divergence = kl_divergence;
      min_kl_index = i;
    } else {
    }
  }
  if (min_kl_index == 0) {
    while (starting_iter > 0) {
      if (hist[starting_iter] == 0) {
        starting_iter -= 1;
        continue;
      } else {
        break;
      }
    }
    min_kl_index = starting_iter;
  }

  LoDTensor scale_tensor = CreateScaleTensor();
  scale_tensor.data<double>()[0] = 1.0 / ((min_kl_index + 0.5) * bin_width);

  return std::make_pair(is_unsigned, scale_tensor);
}

std::pair<bool, LoDTensor>
AnalysisPredictor::MkldnnQuantizer::GetMaxScalingFactor(
    const LoDTensor& var_tensor, bool is_unsigned) const {
  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  float max_abs = eigen_tensor.abs().maxCoeff();
  float min_val = eigen_tensor.minCoeff();
  if (is_unsigned)
    PADDLE_ENFORCE_GE(
        min_val, 0.0f,
        platform::errors::InvalidArgument(
            "Tensor is claimed to be unsigned, but its min value (%f) is < 0.0",
            min_val));

  LoDTensor scale_tensor = CreateScaleTensor();
  scale_tensor.data<double>()[0] = 1.0 / max_abs;

  return std::make_pair(is_unsigned, scale_tensor);
}

std::pair<bool, LoDTensor>
AnalysisPredictor::MkldnnQuantizer::GetMaxChScalingFactor(
    const LoDTensor& var_tensor, bool is_unsigned, bool is_transposed) const {
  check_tensor(var_tensor);

  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  float min_val = eigen_tensor.minCoeff();
  if (is_unsigned)
    PADDLE_ENFORCE_GE(
        min_val, 0.0f,
        platform::errors::InvalidArgument(
            "Tensor is claimed to be unsigned, but its min value (%f) is < 0.0",
            min_val));

  auto dims = var_tensor.dims();
  constexpr int num_col_dims = 1;
  auto flattened_dims = framework::flatten_to_2d(dims, num_col_dims);
  ConstEigenMatrixArrayMap eigen_tensor_mat{
      var_tensor.data<float>(), flattened_dims[0], flattened_dims[1]};

  EigenMatrixDoubleArray scales;
  if (is_transposed) {
    scales = 1.0 / eigen_tensor_mat.cast<double>().abs().colwise().maxCoeff();
  } else {
    scales = 1.0 / eigen_tensor_mat.cast<double>().abs().rowwise().maxCoeff();
  }
  int output_channel_axis = is_transposed;
  int channels = dims[output_channel_axis];
  LoDTensor scale_tensor = CreateScaleTensor(channels);
  auto* scale_ptr = scale_tensor.mutable_data<double>(CPUPlace());
  std::copy(scales.data(), scales.data() + scales.size(), scale_ptr);

  return std::make_pair(is_unsigned, scale_tensor);
}

std::pair<bool, LoDTensor>
AnalysisPredictor::MkldnnQuantizer::GetMaxChGRUScalingFactor(
    const LoDTensor& wx_tensor, const LoDTensor& wh_tensor) const {
  check_tensor(wx_tensor);
  check_tensor(wh_tensor);

  int OC = wh_tensor.dims()[0];
  std::vector<float> scale_ur(2 * OC);
  std::vector<float> scale_o(OC);

  for (int row_id = 0; row_id < wx_tensor.dims()[0]; row_id++) {
    for (int col_id = 0; col_id < 2 * OC; col_id++) {
      int idx = (row_id * wx_tensor.dims()[1]) + col_id;
      auto abs_value = std::abs(wx_tensor.data<float>()[idx]);
      if (row_id == 0) {
        scale_ur[col_id] = abs_value;
      } else {
        if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
      }
    }
  }

  for (int i = 0; i < 2 * OC * OC; i++) {
    int col_id = i % (2 * OC);
    auto abs_value = std::abs(wh_tensor.data<float>()[i]);
    if (abs_value > scale_ur[col_id]) scale_ur[col_id] = abs_value;
  }

  for (int row_id = 0; row_id < wx_tensor.dims()[0]; row_id++) {
    for (int col_id = 2 * OC; col_id < wx_tensor.dims()[1]; col_id++) {
      int idx = (row_id * wx_tensor.dims()[1]) + col_id;
      auto abs_value = std::abs(wx_tensor.data<float>()[idx]);
      if (row_id == 0) {
        scale_o[col_id % OC] = abs_value;
      } else {
        if (abs_value > scale_o[col_id]) scale_o[col_id % OC] = abs_value;
      }
    }
  }

  for (int i = 2 * OC * OC; i < OC * wh_tensor.dims()[1]; i++) {
    int col_id = i % OC;
    auto abs_value = std::abs(wh_tensor.data<float>()[i]);
    if (abs_value > scale_o[col_id]) scale_o[col_id] = abs_value;
  }
  scale_ur.insert(scale_ur.end(), scale_o.begin(), scale_o.end());
  transform(scale_ur.begin(), scale_ur.end(), scale_ur.begin(),
            [](float& c) { return 1 / c; });
  LoDTensor scale_tensor = CreateScaleTensor(scale_ur.size());
  auto* scale_ptr = scale_tensor.mutable_data<double>(CPUPlace());
  std::copy(scale_ur.begin(), scale_ur.end(), scale_ptr);
  bool is_unsigned = false;
  printf("******* GetMaxChGRUScalingFactor ********\n");
  return std::make_pair(is_unsigned, scale_tensor);
}

std::pair<std::vector<int>, float>
AnalysisPredictor::MkldnnQuantizer::Histogram(
    const framework::LoDTensor& var_tensor, float min_val, float max_val,
    size_t num_bins) const {
  PADDLE_ENFORCE_GT(num_bins, 0,
                    platform::errors::InvalidArgument(
                        "MkldnnQuantizer: To calculate Histogram, num_bins (" +
                        std::to_string(num_bins) + ") must be positive."));
  PADDLE_ENFORCE_GT(var_tensor.numel(), 0,
                    platform::errors::InvalidArgument(
                        "MkldnnQuantizer: To calculate Histogram, the tensor "
                        "must not be empty."));
  PADDLE_ENFORCE_GE(max_val, min_val,
                    platform::errors::InvalidArgument(
                        "MkldnnQuantizer: To calculate Histogram, max_val (" +
                        std::to_string(max_val) + ") must be greater or equal"
                                                  "to min_val (" +
                        std::to_string(min_val) + ")."));
  ConstEigenVectorArrayMap eigen_tensor{var_tensor.data<float>(),
                                        var_tensor.numel(), 1};
  auto bin_width = std::abs(max_val - min_val) / num_bins;
  std::vector<int> hist(num_bins);

  for (int i = 0; i < eigen_tensor.size(); i++) {
    int bin = std::min(
        num_bins - 1,
        static_cast<size_t>(floor((eigen_tensor[i] - min_val) / bin_width)));
    ++hist[bin];
  }

  return std::make_pair(std::move(hist), std::move(bin_width));
}

void AnalysisPredictor::MkldnnQuantizer::ClearDeviceContext() const {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  platform::MKLDNNDeviceContext* dev_ctx =
      (platform::MKLDNNDeviceContext*)pool.Get(predictor_.place_);
  dev_ctx->ResetBlobMap(
      paddle::platform::MKLDNNDeviceContext::tls().get_curr_exec());
}

void AnalysisPredictor::MkldnnQuantizer::PrepareArgument() const {
  auto& arg = predictor_.argument_;
  if (!arg.scope_valid()) arg.SetScope(new framework::Scope);
  arg.SetMainProgramNotOwned(predictor_.inference_program_.get());
  auto graph = std::unique_ptr<Graph>(new Graph(arg.main_program()));
  arg.SetMainGraph(graph.release());
  auto* scope_ptr = arg.scope_ptr();
  PADDLE_ENFORCE_NOT_NULL(scope_ptr, platform::errors::PreconditionNotMet(
                                         "The scope should not be nullptr."));
  arg.main_graph().SetNotOwned(framework::ir::kParamScopeAttr, scope_ptr);

  auto* builder = predictor_.config_.pass_builder();
  builder->SetPasses({
      "cpu_quantize_pass", "cpu_quantize_squash_pass",
  });
  if (predictor_.config_.ir_debug_) builder->TurnOnDebug();
  auto passes = builder->AllPasses();
  predictor_.argument_.SetIrAnalysisPasses(passes);
  predictor_.argument_.SetAnalysisPasses(
      {"ir_graph_clean_pass", "ir_analysis_pass", "memory_optimize_pass",
       "ir_graph_to_program_pass"});
  predictor_.argument_.SetQuantVarScales(scales_);
}

bool AnalysisPredictor::MkldnnQuantizer::Quantize() {
  if (!RunWarmup()) return false;
  if (!CalculateScales()) return false;
  ClearDeviceContext();
  predictor_.PrepareScope(predictor_.scope_);
  predictor_.CreateExecutor();
  if (!RunQuantizePasses()) return false;
  predictor_.PrepareExecutor();
  predictor_.PrepareFeedFetch();
  return true;
}

bool AnalysisPredictor::MkldnnQuantizer::RunQuantizePasses() const {
  predictor_.executor_->CreateVariables(*predictor_.inference_program_, 0, true,
                                        predictor_.sub_scope_);
  PrepareArgument();
  auto& arg = predictor_.argument_;
  Analyzer().Run(&arg);
  PADDLE_ENFORCE_EQ(
      arg.scope_valid(), true,
      platform::errors::PreconditionNotMet("The scope should be valid."));
  VLOG(5) << "to prepare executor";
  ARGUMENT_CHECK_FIELD((&arg), ir_analyzed_program);
  predictor_.inference_program_.reset(
      new framework::ProgramDesc(arg.ir_analyzed_program()));
  LOG(INFO) << "== optimize 2 end ==";
  predictor_.executor_->CreateVariables(*predictor_.inference_program_, 0,
                                        false, predictor_.sub_scope_);
  return true;
}

bool AnalysisPredictor::MkldnnQuantizer::RunWarmup() const {
  VLOG(3) << "Predictor: run a quantization warmup iteration";
  auto warmup_data = qconfig_->warmup_data();
  PADDLE_ENFORCE_NOT_NULL(warmup_data,
                          platform::errors::PreconditionNotMet(
                              "Warmup data cannot be NULL in the config."));
  PrettyLogH1("--- Running warmup iteration for quantization");

  // Run the inference program
  std::vector<PaddleTensor> output_slots;
  predictor_.Run(*warmup_data, &output_slots, qconfig_->warmup_batch_size());

  return true;
}

float AnalysisPredictor::MkldnnQuantizer::SafeEntropy(
    std::vector<int> reference_distr_P, int P_sum,
    std::vector<int> candidate_distr_Q, int Q_sum) const {
  PADDLE_ENFORCE_EQ(reference_distr_P.size(), candidate_distr_Q.size(),
                    platform::errors::InvalidArgument(
                        "The P size %d should be equal to Q size %d",
                        reference_distr_P.size(), candidate_distr_Q.size()));
  float tmp_sum1 = 0;
  float tmp_sum2 = 0;
  for (size_t idx = 0; idx < reference_distr_P.size(); idx++) {
    int p_idx = reference_distr_P[idx];
    int q_idx = candidate_distr_Q[idx];
    if (p_idx == 0) {
      tmp_sum1 += 0;
      tmp_sum2 += 0;
    } else {
      PADDLE_ENFORCE_NE(
          q_idx, 0,
          platform::errors::PreconditionNotMet(
              "MkldnnQuantizer: Fatal error!, idx = " + std::to_string(idx) +
              " qindex = 0! p_idx = " + std::to_string(p_idx)));
    }
    tmp_sum1 += p_idx * (log(Q_sum * p_idx));
    tmp_sum2 += p_idx * (log(P_sum * q_idx));
  }
  return (tmp_sum1 - tmp_sum2) / P_sum;
}

}  // namespace paddle
