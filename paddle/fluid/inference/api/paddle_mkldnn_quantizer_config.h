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

///
/// \file paddle_mkldnn_quantizer_config.h
///
/// \brief Mkldnn quantizer config.
///
/// \author paddle-infer@baidu.com
/// \date 2020-01-01
/// \since 1.7.0
///

#pragma once

#include <cassert>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "paddle_api.h"            // NOLINT
#include "paddle_infer_declare.h"  // NOLINT

namespace paddle {

///
/// \brief Algorithms for finding scale of quantized Tensors.
///
enum class ScaleAlgo {
  NONE,      ///< Do not compute scale
  MAX,       ///< Find scale based on the max absolute value
  MAX_CH,    ///< Find scale based on the max absolute value per output channel
  MAX_CH_T,  ///< Find scale based on the max absolute value per output channel
             ///< of a transposed tensor
  MAX_CH_GRU,  ///< Find scale based on the max absolute value per output
               /// channel for fusion_gru/multi_gru operators
  KL,          ///< Find scale based on KL Divergence
};

///
/// \class MkldnnQuantizerConfig
///
/// \brief Config for mkldnn quantize.
///
/// The MkldnnQuantizerConfig is used to configure Mkldnn's quantization
/// parameters, including scale algorithm, warmup data, warmup batch size,
/// quantized op list, etc.
///
/// It is not recommended to use this config directly, please refer to
/// AnalysisConfig::mkldnn_quantizer_config()
///
struct PD_INFER_DECL MkldnnQuantizerConfig {
  ///
  /// \brief Construct a new Mkldnn Quantizer Config object
  ///
  MkldnnQuantizerConfig();

  ///
  /// \brief Set the scale algo
  ///
  /// Specify a quantization algorithm for a connection (input/output) of the
  /// operator type.
  /// \param[in] op_type_name the operator's name.
  /// \param[in] conn_name name of the connection (input/output) of the
  /// operator.
  /// \param[in] algo the algorithm for computing scale.
  ///
  void SetScaleAlgo(std::string op_type_name, std::string conn_name,
                    ScaleAlgo algo) {
    rules_[op_type_name][conn_name] = algo;
  }

  ///
  /// \brief Get the scale algo
  ///
  /// Get the quantization algorithm for a connection (input/output) of the
  /// operator type.
  ///
  /// \param[in] op_type_name the operator's name.
  /// \param[in] conn_name name of the connection (input/output) of the
  /// operator.
  /// \return the scale algo.
  ///
  ScaleAlgo scale_algo(const std::string& op_type_name,
                       const std::string& conn_name) const;

  ///
  /// \brief Set the warmup data
  ///
  /// Set the batch of data to be used for warm-up iteration.
  ///
  /// \param[in] data batch of data.
  ///
  void SetWarmupData(std::shared_ptr<std::vector<PaddleTensor>> data) {
    warmup_data_ = data;
  }

  ///
  /// \brief Get the warmup data
  ///
  /// Get the batch of data used for warm-up iteration.
  ///
  /// \return the warm up data
  ///
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data() const {
    return warmup_data_;
  }

  ///
  /// \brief Set the warmup batch size
  ///
  /// Set the batch size for warm-up iteration.
  ///
  /// \param[in] batch_size warm-up batch size
  ///
  void SetWarmupBatchSize(int batch_size) { warmup_bs_ = batch_size; }

  ///
  /// \brief Get the warmup batch size
  ///
  /// Get the batch size for warm-up iteration.
  ///
  /// \return the warm up batch size
  int warmup_batch_size() const { return warmup_bs_; }

  ///
  /// \brief Set quantized op list
  ///
  /// In the quantization process, set the op list that supports quantization
  ///
  /// \param[in] op_list List of quantized ops
  ///
  void SetEnabledOpTypes(std::unordered_set<std::string> op_list) {
    enabled_op_types_ = op_list;
  }

  ///
  /// \brief Get quantized op list
  ///
  /// \return list of quantized ops
  ///
  const std::unordered_set<std::string>& enabled_op_types() const {
    return enabled_op_types_;
  }

  ///
  /// \brief Set the excluded op ids
  ///
  /// \param[in] op_ids_list excluded op ids
  ///
  void SetExcludedOpIds(std::unordered_set<int> op_ids_list) {
    excluded_op_ids_ = op_ids_list;
  }

  ///
  /// \brief Get the excluded op ids
  ///
  /// \return exclude op ids
  ///
  const std::unordered_set<int>& excluded_op_ids() const {
    return excluded_op_ids_;
  }

  ///
  /// \brief Set default scale algorithm
  ///
  /// \param[in] algo Method for calculating scale in quantization process
  ///
  void SetDefaultScaleAlgo(ScaleAlgo algo) { default_scale_algo_ = algo; }

  ///
  /// \brief Get default scale algorithm
  ///
  /// \return Method for calculating scale in quantization
  /// process
  ///
  ScaleAlgo default_scale_algo() const { return default_scale_algo_; }

 protected:
  std::map<std::string, std::map<std::string, ScaleAlgo>> rules_;
  std::unordered_set<std::string> enabled_op_types_;
  std::unordered_set<int> excluded_op_ids_;
  std::shared_ptr<std::vector<PaddleTensor>> warmup_data_;
  int warmup_bs_{1};
  ScaleAlgo default_scale_algo_{ScaleAlgo::MAX};
};

}  // namespace paddle
