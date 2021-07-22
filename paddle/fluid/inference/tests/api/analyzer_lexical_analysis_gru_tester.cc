/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/api/paddle_analysis_config.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

// setting iterations to 0 means processing the whole dataset
namespace paddle {
namespace inference {
namespace analysis {

void SetNativeConfig(AnalysisConfig *cfg,
                     const int &num_threads = FLAGS_cpu_num_threads) {
  cfg->SwitchIrOptim(false);
  cfg->DisableGpu();
  cfg->SetModel(FLAGS_infer_model);
  cfg->SetCpuMathLibraryNumThreads(num_threads);
}

void SetAnalysisConfig(AnalysisConfig *cfg,
                       const int &num_threads = FLAGS_cpu_num_threads) {
  cfg->SetModel(FLAGS_infer_model);
  cfg->DisableGpu();
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames(false);
  cfg->SetCpuMathLibraryNumThreads(num_threads);
  cfg->EnableMKLDNN();
}

std::vector<size_t> ReadSentenceLod(std::ifstream &file, size_t offset,
                                    int64_t total_sentences_num) {
  std::vector<size_t> sentence_lod(total_sentences_num);

  file.clear();
  file.seekg(offset);
  file.read(reinterpret_cast<char *>(sentence_lod.data()),
            total_sentences_num * sizeof(size_t));

  if (file.eof()) LOG(ERROR) << "Reached end of stream";
  if (file.fail()) throw std::runtime_error("Failed reading file.");
  return sentence_lod;
}

std::shared_ptr<std::vector<PaddleTensor>> WarmupData(
    const std::vector<std::vector<PaddleTensor>> &test_data,
    int num_images = 1) {
  int data_size = test_data.size();

  PADDLE_ENFORCE_LE(static_cast<size_t>(num_images), data_size,
                    platform::errors::InvalidArgument(
                        "The requested quantization warmup data size must be "
                        "lower or equal to the test data size. But received"
                        "warmup size is %d and test data size is %d",
                        num_images, data_size));
  int words_shape = test_data[0][0].shape[0];
  PaddleTensor words;
  words.name = "words";
  words.shape = {words_shape, 1};
  words.dtype = PaddleDType::INT64;
  words.data.Resize(sizeof(int64_t) * words_shape);

  int target_shape = test_data[0][1].shape[0];
  PaddleTensor targets;
  targets.name = "targets";
  targets.shape = {target_shape, 1};
  targets.dtype = PaddleDType::INT64;
  targets.data.Resize(sizeof(int64_t) * target_shape);

  for (int i = 0; i < num_images; i++) {
    std::copy_n(
        static_cast<int64_t *>(test_data[i][0].data.data()) + i * words_shape,
        words_shape,
        static_cast<int64_t *>(words.data.data()) + i * words_shape);
    words.lod = test_data[i][0].lod;

    std::copy_n(
        static_cast<int64_t *>(test_data[i][1].data.data()) + i * target_shape,
        target_shape,
        static_cast<int64_t *>(targets.data.data()) + i * target_shape);
    targets.lod = test_data[i][1].lod;
  }

  auto warmup_data = std::make_shared<std::vector<PaddleTensor>>(2);
  (*warmup_data)[0] = std::move(words);
  (*warmup_data)[1] = std::move(targets);
  return warmup_data;
}

template <typename T>
class TensorReader {
 public:
  TensorReader(std::ifstream &file, size_t beginning_offset, std::string name)
      : file_(file), position_(beginning_offset), name_(name) {}

  PaddleTensor NextBatch(std::vector<int> shape, std::vector<size_t> lod) {
    int numel =
        std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
    PaddleTensor tensor;
    tensor.name = name_;
    tensor.shape = shape;
    tensor.dtype = GetPaddleDType<T>();
    tensor.data.Resize(numel * sizeof(T));
    if (lod.empty() == false) {
      tensor.lod.clear();
      tensor.lod.push_back(lod);
    }
    file_.seekg(position_);
    if (file_.eof()) LOG(ERROR) << name_ << ": reached end of stream";
    if (file_.fail())
      throw std::runtime_error(name_ + ": failed reading file.");
    file_.read(reinterpret_cast<char *>(tensor.data.data()), numel * sizeof(T));
    position_ = file_.tellg();
    return tensor;
  }

 protected:
  std::ifstream &file_;
  size_t position_;
  std::string name_;
};

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs,
              int32_t batch_size = FLAGS_batch_size) {
  std::ifstream file(FLAGS_infer_data, std::ios::binary);
  if (!file) {
    FAIL() << "Couldn't open file: " << FLAGS_infer_data;
  }

  int64_t total_sentences_num = 0L;
  int64_t total_words_num = 0L;
  file.seekg(0);
  file.read(reinterpret_cast<char *>(&total_sentences_num), sizeof(int64_t));
  LOG(INFO) << "Total sentences in file: " << total_sentences_num;
  file.read(reinterpret_cast<char *>(&total_words_num), sizeof(int64_t));
  LOG(INFO) << "Total words in file: " << total_words_num;
  size_t lods_beginning_offset = static_cast<size_t>(file.tellg());
  auto words_begining_offset =
      lods_beginning_offset + sizeof(size_t) * total_sentences_num;
  auto targets_beginning_offset =
      words_begining_offset + sizeof(int64_t) * total_words_num;

  std::vector<size_t> lod_full =
      ReadSentenceLod(file, lods_beginning_offset, total_sentences_num);

  size_t lods_sum = std::accumulate(lod_full.begin(), lod_full.end(), 0UL);
  EXPECT_EQ(lods_sum, static_cast<size_t>(total_words_num));

  TensorReader<int64_t> words_reader(file, words_begining_offset, "words");
  TensorReader<int64_t> targets_reader(file, targets_beginning_offset,
                                       "targets");
  // If FLAGS_iterations is set to 0, run all batches
  auto iterations_max = total_sentences_num / batch_size;
  auto iterations = iterations_max;
  if (FLAGS_iterations > 0 && FLAGS_iterations < iterations_max) {
    iterations = FLAGS_iterations;
  }

  for (auto i = 0; i < iterations; i++) {
    // Calculate the words num.  Shape=[words_num, 1]
    std::vector<size_t> batch_lod = {0};
    size_t num_words = 0L;
    std::transform(lod_full.begin() + i * FLAGS_batch_size,
                   lod_full.begin() + (i + 1) * FLAGS_batch_size,
                   std::back_inserter(batch_lod),
                   [&num_words](const size_t lodtemp) -> size_t {
                     num_words += lodtemp;
                     return num_words;
                   });
    auto words_tensor = words_reader.NextBatch(
        {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
    if (FLAGS_with_accuracy_layer) {
      auto targets_tensor = targets_reader.NextBatch(
          {static_cast<int>(batch_lod[FLAGS_batch_size]), 1}, batch_lod);
      inputs->emplace_back(std::vector<PaddleTensor>{
          std::move(words_tensor), std::move(targets_tensor)});
    } else {
      inputs->emplace_back(std::vector<PaddleTensor>{std::move(words_tensor)});
    }
  }
}

std::vector<double> Lexical_Test(
    const std::vector<std::vector<PaddleTensor>> &input_slots_all,
    std::vector<std::vector<PaddleTensor>> *outputs, AnalysisConfig *config,
    const bool use_analysis) {
  TestOneThreadPrediction(
      reinterpret_cast<const PaddlePredictor::Config *>(config),
      input_slots_all, outputs, FLAGS_use_analysis);
  std::vector<double> acc_res(3);
  if (FLAGS_with_accuracy_layer) {
    EXPECT_GT(outputs->size(), 0UL);
    EXPECT_EQ(3UL, (*outputs)[0].size());
    std::vector<int64_t> acc_sum(3);
    for (size_t i = 0; i < outputs->size(); i++) {
      for (size_t j = 0; j < 3UL; j++) {
        acc_sum[j] =
            acc_sum[j] + *static_cast<int64_t *>((*outputs)[i][j].data.data());
      }
    }
    // nums_infer, nums_label, nums_correct
    auto precision =
        acc_sum[0]
            ? static_cast<double>(acc_sum[2]) / static_cast<double>(acc_sum[0])
            : 0;
    auto recall =
        acc_sum[1]
            ? static_cast<double>(acc_sum[2]) / static_cast<double>(acc_sum[1])
            : 0;
    auto f1_score =
        acc_sum[2]
            ? static_cast<float>(2 * precision * recall) / (precision + recall)
            : 0;

    LOG(INFO) << "Precision:  " << std::fixed << std::setw(6)
              << std::setprecision(5) << precision;
    LOG(INFO) << "Recall:  " << std::fixed << std::setw(6)
              << std::setprecision(5) << recall;
    LOG(INFO) << "F1 score: " << std::fixed << std::setw(6)
              << std::setprecision(5) << f1_score;

    acc_res = {precision, recall, f1_score};
    // return acc_res;
  } else {
    EXPECT_GT(outputs->size(), 0UL);
    EXPECT_GT(outputs[0].size(), 0UL);
    LOG(INFO) << "No accuracy result. To get accuracy result provide a model "
                 "with accuracy layers in it and use --with_accuracy_layer "
                 "option.";
  }
  return acc_res;
}

TEST(Analyzer_lexical_test, Analyzer_lexical_analysis) {
  AnalysisConfig native_cfg;

  std::vector<std::vector<PaddleTensor>> outputs;
  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  SetNativeConfig(&native_cfg, FLAGS_cpu_num_threads);
  std::vector<double> acc_ref(3);
  acc_ref = Lexical_Test(input_slots_all, &outputs, &native_cfg, false);
  if (FLAGS_use_analysis) {
    AnalysisConfig analysis_cfg;
    SetAnalysisConfig(&analysis_cfg, FLAGS_cpu_num_threads);
    if (FLAGS_enable_bf16) {
      analysis_cfg.EnableMkldnnBfloat16();
    } else if (FLAGS_enable_int8) {
      if (FLAGS_fuse_multi_gru)
        analysis_cfg.pass_builder()->AppendPass("multi_gru_fuse_pass");

      std::shared_ptr<std::vector<PaddleTensor>> warmup_data =
          WarmupData(input_slots_all);
      analysis_cfg.EnableMkldnnQuantizer();
      analysis_cfg.mkldnn_quantizer_config()->SetWarmupData(warmup_data);
      analysis_cfg.mkldnn_quantizer_config()->SetWarmupBatchSize(
          FLAGS_batch_size);
      printf("--- Post-training gru model test ---\n");
    }
    std::vector<double> acc_analysis(3);
    acc_analysis = Lexical_Test(input_slots_all, &outputs, &analysis_cfg, true);
    for (size_t i = 0; i < acc_analysis.size(); i++) {
      CHECK_LE(std::abs(acc_ref[i] - acc_analysis[i]),
               FLAGS_quantized_accuracy);
    }
    ASSERT_EQ(true, false);  // trigger output printing
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
