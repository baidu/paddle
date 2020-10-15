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
#include <vector>
#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_compatible_info.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/api/api_impl.h"
#include "paddle/fluid/inference/api/details/reset_tensor_array.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/string/printf.h"
#ifdef PADDLE_WITH_TESTING
#include <gtest/gtest.h>
#include <gtest/gtest_prod.h>
#endif

///
/// \file analysis_predictor.h
///
/// \brief Compared to NativePredictor, AnalysisPredictor is a high-performance
/// predictor that includes many optimizations
///
/// \author paddle-infer@baidu.com
/// \date 2020-01-01
/// \since 1.7.0
///

namespace paddle {

using inference::analysis::Argument;
using inference::analysis::Analyzer;
using framework::proto::ProgramDesc;
using framework::NaiveExecutor;

///
/// \class AnalysisPredictor
///
/// \brief The analysis predictor is based on the original native predictor with
/// IR and Analysis support. It will optimize IR and Parameters in the runtime.
///
/// The predictor has the following typical uses:
///
/// Get predictor
/// \code{cpp}
///   auto predictor = CreatePaddlePredictor(config);
/// \endcode
///
/// Get input or output names
/// \code{cpp}
///   auto input_names = predictor->GetInputNames();
///   auto output_names = predictor->GetOutputNames();
/// \endcode
///
/// Get input or output tensors
/// \code{cpp}
///   auto input_t = predictor->GetInputTensor(input_names[0]);
///   auto output_t = predictor->GetOutputTensor(output_names[0]);
/// \endcode
///
/// Run predictor
/// \code{cpp}
///   predictor->ZeroCopyRun();
/// \endcode
///
class AnalysisPredictor : public PaddlePredictor {
 public:
  ///
  /// \brief Construct a new Analysis Predictor object
  ///
  /// \param[in] AnalysisConfig config
  ///
  explicit AnalysisPredictor(const AnalysisConfig &config) : config_(config) {
    predictor_id_ = inference::GetUniqueId();
  }
  ///
  /// \brief Destroy the Analysis Predictor object
  ///
  ~AnalysisPredictor();

  ///
  /// \brief Initialize predictor
  ///
  /// Initializing predictor mainly includes the following tasks:
  /// preparing scope, creating executor, preparing program, initializing the
  /// variables required by the executor, getting the feed_target_names and
  /// fetch_target_names, etc.
  ///
  /// \param[in] parent_scope parent scope
  /// \param[in] program program
  /// \return Whether the init function executed successfully
  ///
  bool Init(const std::shared_ptr<framework::Scope> &parent_scope,
            const std::shared_ptr<framework::ProgramDesc> &program = nullptr);

  ///
  /// \brief Run the prediction engine. Deprecated. Please refer to ZeroCopyRun
  ///
  /// \param[in] inputs input tensors
  /// \param[out] output_data output tensors
  /// \param[in] batch_size data's batch size
  /// \return Whether the function executed successfully
  ///
  bool Run(const std::vector<PaddleTensor> &inputs,
           std::vector<PaddleTensor> *output_data,
           int batch_size = -1) override;

  ///
  /// \brief Get the input names
  ///
  /// \return input names
  ///
  std::vector<std::string> GetInputNames();
  ///
  /// \brief Get the output names
  ///
  /// \return output names
  ///
  std::vector<std::string> GetOutputNames();

  ///
  /// \brief Get the Input Tensor object
  ///
  /// \param[in] name input name
  /// \return input tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetInputTensor(
      const std::string &name) override;
  ///
  /// \brief Get the Output Tensor object
  ///
  /// \param[in] name otuput name
  /// \return output tensor
  ///
  std::unique_ptr<ZeroCopyTensor> GetOutputTensor(
      const std::string &name) override;
  ///
  /// \brief Get all input names and their corresponding shapes
  ///
  /// \return the map of input names and shapes
  ///
  std::map<std::string, std::vector<int64_t>> GetInputTensorShape() override;

  ///
  /// \brief Run the prediction engine
  ///
  /// \return Whether the function executed successfully
  ///
  bool ZeroCopyRun() override;

  ///
  /// \brief Create feed fetch variables
  ///
  /// \param[in] scope Scope needed to create variables
  ///
  void CreateFeedFetchVar(framework::Scope *scope);
  ///
  /// \brief Determine the model's inputs and outputs based on the program's
  /// feed fetch op
  ///
  void PrepareFeedFetch();

  ///
  /// \brief Set predictor's argument according to config, which mainly includes
  /// execution information and graph optimization related pass information
  ///
  void PrepareArgument();
  ///
  /// \brief According to argument information, execute the relevant pass
  /// to get the optimized model program
  ///
  void OptimizeInferenceProgram();

  ///
  /// \brief Clear the intermediate tensors of the predictor
  ///
  ///
  void ClearIntermediateTensor();

  ///
  /// \brief Get the argument used by predictor
  ///
  /// \return the argument obtained by config
  ///
  Argument &analysis_argument() { return argument_; }
  ///
  /// \brief Clone to get the new predictor. thread safe.
  ///
  /// \return get a new predictor
  ///
  std::unique_ptr<PaddlePredictor> Clone() override;
  ///
  /// \brief Get the scope used by predictor
  ///
  /// \return scope
  ///
  framework::Scope *scope() { return scope_.get(); }
  ///
  /// \brief Get the inference program
  ///
  /// \return the inference program
  ///
  framework::ProgramDesc &program() { return *inference_program_; }

  ///
  /// \brief Get the serialized program
  ///
  /// \return the serialized program
  ///
  std::string GetSerializedProgram() const override;

  ///
  /// \brief Initialize mkldnn quantizer and execute mkldnn quantization pass
  ///
  /// \return Whether the function executed successfully
  ///
  bool MkldnnQuantize();

  ///
  /// \brief save program to model and save parameters to params
  ///
  /// \param[in] dir path to save the model
  ///
  void SaveOptimModel(const std::string &dir);

 protected:
  ///
  /// \brief Prepare predictor's required programs, including loading model
  /// information, graph optimization, and executor creation variables, etc.
  ///
  /// \param[in] program paddle program
  /// \return Whether the function executed successfully
  ///
  bool PrepareProgram(const std::shared_ptr<framework::ProgramDesc> &program);
  ///
  /// \brief Prepare scope environment, each predictor has its own scope
  ///
  /// \param[in] parent_scope The scope of the predictor to be cloned, or null
  /// \return Whether the function executed successfully
  ///
  bool PrepareScope(const std::shared_ptr<framework::Scope> &parent_scope);
  ///
  /// \brief Create an Executor object
  ///
  /// \return Whether the function executed successfully
  ///
  bool CreateExecutor();
  ///
  /// \brief According to the model's program, the executor creates ops
  ///
  /// \return Whether the function executed successfully
  ///
  bool PrepareExecutor();

  ///
  /// \brief Load model program.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadProgramDesc();
  ///
  /// \brief Load model parameters.
  ///
  /// \return Whether the function executed successfully
  ///
  bool LoadParameters();

  ///
  /// \brief Prepare input data, only used in Run()
  ///
  /// \param[in] input_datas inpute tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool SetFeed(const std::vector<PaddleTensor> &input_datas,
               framework::Scope *scope);
  ///
  /// \brief Get the output data, only used in Run()
  ///
  /// \param[out] output_data output tensors
  /// \param[in] scope the scope used by predictor
  /// \return Whether the function executed successfully
  ///
  bool GetFetch(std::vector<PaddleTensor> *output_data,
                framework::Scope *scope);
  ///
  /// \brief Get the output data, only used in GetFetch()
  ///
  /// \param[in] tensor for fetch op
  /// \param[out] output_data output tensor
  ///
  template <typename T>
  void GetFetchOne(const framework::LoDTensor &fetchs,
                   PaddleTensor *output_data);
  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  /// \param[in] inputs tensors
  ///
  void MkldnnPreSet(const std::vector<PaddleTensor> &inputs);

  ///
  /// \brief PreSet for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  /// \param[in] inputs tensor shape
  ///
  void MkldnnPreSet(const std::vector<std::vector<int>> &inputs_shape);

  ///
  /// \brief PostReset for Mkldnn multi-thread and dynamic shape input.
  ///
  /// Used in AnalysisPredictor::Run(), do not support
  /// AnalysisPredictor::ZeroCopyRun() now.
  ///
  void MkldnnPostReset();

#if PADDLE_WITH_TENSORRT
  ///
  /// \brief save calibration table
  ///
  /// When we use Paddle-TRT INT8 engine, we need to generate calibration table
  /// data first,
  /// the calibration table contains the range for each op's input and output,
  /// this whole process can be divided into several steps:
  /// 1. Builds a 32-bit engine, runs it on the calibration set, and records a
  ///  histogram for each tensor of the distribution of activation values.
  /// 2. Builds a calibration table from the histograms.
  /// After step 2, we need to store the calibration table on disk.
  ///
  /// \return Whether the function executed successfully
  ///
  bool SaveTrtCalibToDisk();
#endif

// Some more detailed tests, they are made the friends of the predictor, so that
// the all the details can be tested.
#if PADDLE_WITH_TESTING
  FRIEND_TEST(AnalysisPredictor, analysis_off);
  FRIEND_TEST(AnalysisPredictor, analysis_on);
  FRIEND_TEST(AnalysisPredictor, with_gpu);
#endif

 private:
  AnalysisConfig config_;
  Argument argument_;
  std::unique_ptr<NaiveExecutor> executor_;
  platform::Place place_;
  std::shared_ptr<framework::Scope> scope_;
  framework::Scope *sub_scope_{nullptr};
  std::shared_ptr<framework::ProgramDesc> inference_program_;
  framework::OpCompatibleMap op_compatible_map_;
  std::vector<framework::OpDesc *> feeds_;
  std::map<std::string, size_t> feed_names_;
  // Sorted according to the idx.
  std::map<size_t, std::string> idx2feeds_;
  std::vector<framework::OpDesc *> fetches_;
  std::map<size_t, std::string> idx2fetches_;

#if PADDLE_WITH_MKLDNN
  // Helper class to perform quantization
  class MkldnnQuantizer;
  MkldnnQuantizer *mkldnn_quantizer_{nullptr};

#if PADDLE_WITH_TESTING
  friend class MkldnnQuantizerTest;
#endif
#endif

  // Memory buffer for feed inputs. The temporary LoDTensor will cause serious
  // concurrency problems, wrong results and memory leak, so cache them.
  std::vector<framework::LoDTensor> feed_tensors_;
  details::TensorArrayBatchCleaner tensor_array_batch_cleaner_;
  // A mutex help to make Clone thread safe.
  std::mutex clone_mutex_;

  // For memory optimization.
  const size_t max_shape_collect_count_{1000};
  int need_collect_var_shapes_{-1};  // -1 for default, 0 for false, 1 for true.
  std::vector<std::map<std::string, std::vector<int>>> batch_var_shapes_;
  int predictor_id_;

 private:
  // Some status here that help to determine the status inside the predictor.
  bool status_is_cloned_{false};
  bool status_use_gpu_{false};
};

}  // namespace paddle
