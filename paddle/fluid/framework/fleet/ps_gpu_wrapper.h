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

#pragma once

#if (defined PADDLE_WITH_NCCL) && (defined PADDLE_WITH_PSLIB)

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/fleet/heter_context.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_ps_base.h"
#include "paddle/fluid/framework/fleet/heter_ps/heter_resource.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

class PSGPUWrapper {
 public:
  virtual ~PSGPUWrapper() { delete HeterPs_; }

  PSGPUWrapper() {
    HeterPs_ = NULL;
    sleep_seconds_before_fail_exit_ = 300;
  }

  void PullSparse(const paddle::platform::Place& place, const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place, const int table_id,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int batch_size);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len);
  void CopyForPull(const paddle::platform::Place& place, uint64_t** gpu_keys,
                   const std::vector<float*>& values,
                   const FeatureValue* total_values_gpu, const int64_t* gpu_len,
                   const int slot_num, const int hidden_size,
                   const int64_t total_length);

  void CopyForPush(const paddle::platform::Place& place,
                   const std::vector<const float*>& grad_values,
                   FeaturePushValue* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size, const int64_t total_length,
                   const int batch_size);

  void BuildGPUPS(const uint64_t table_id, int feature_dim);
  void BuildTask(uint64_t table_id, int feature_dim);
  void InitializeGPU(const std::vector<int>& dev_ids) {
    if (s_instance_ != NULL) {
      VLOG(3) << "PSGPUWrapper Begin InitializeGPU";
      resource_ = std::make_shared<HeterPsResource>(dev_ids);
      resource_->enable_p2p();
      keys_tensor.resize(resource_->total_gpu());
      heter_devices_ = dev_ids;
    }
  }

  void SetSparseSGD(float nonclk_coeff, float clk_coeff, float min_bound,
                    float max_bound, float learning_rate, float initial_g2sum,
                    float initial_range);
  void SetEmbedxSGD(float mf_create_thresholds, float mf_learning_rate,
                    float mf_initial_g2sum, float mf_initial_range,
                    float mf_min_bound, float mf_max_bound);
  void InitializeGPUServer(std::unordered_map<std::string, float> config) {
    float nonclk_coeff = (config.find("nonclk_coeff") == config.end())
                             ? 1.0
                             : config["nonclk_coeff"];
    float clk_coeff =
        (config.find("clk_coeff") == config.end()) ? 1.0 : config["clk_coeff"];
    float min_bound = (config.find("min_bound") == config.end())
                          ? -10000.0
                          : config["min_bound"];
    float max_bound = (config.find("max_bound") == config.end())
                          ? 10000.0
                          : config["max_bound"];
    float learning_rate = (config.find("learning_rate") == config.end())
                              ? 1.0
                              : config["learning_rate"];
    float initial_g2sum = (config.find("initial_g2sum") == config.end())
                              ? 1.0
                              : config["initial_g2sum"];
    float initial_range = (config.find("initial_range") == config.end())
                              ? 1.0
                              : config["initial_range"];

    // mf config settings
    float mf_create_thresholds =
        (config.find("mf_create_thresholds") == config.end())
            ? static_cast<float>(1.0)
            : config["mf_create_thresholds"];
    float mf_learning_rate = (config.find("mf_learning_rate") == config.end())
                                 ? 1.0
                                 : config["mf_learning_rate"];
    float mf_initial_g2sum = (config.find("mf_initial_g2sum") == config.end())
                                 ? 1.0
                                 : config["mf_initial_g2sum"];
    float mf_initial_range = (config.find("mf_initial_range") == config.end())
                                 ? 1.0
                                 : config["mf_initial_range"];
    float mf_min_bound = (config.find("mf_min_bound") == config.end())
                             ? 1.0
                             : config["mf_min_bound"];
    float mf_max_bound = (config.find("mf_max_bound") == config.end())
                             ? 1.0
                             : config["mf_max_bound"];
    for (size_t i = 0; i < heter_devices_.size(); i++) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(heter_devices_[i]));
      this->SetSparseSGD(nonclk_coeff, clk_coeff, min_bound, max_bound,
                         learning_rate, initial_g2sum, initial_range);
      this->SetEmbedxSGD(mf_create_thresholds, mf_learning_rate,
                         mf_initial_g2sum, mf_initial_range, mf_min_bound,
                         mf_max_bound);
    }
  }
  void SetDataset(Dataset* dataset) { dataset_ = dataset; }

  // PSGPUWrapper singleton
  static std::shared_ptr<PSGPUWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PSGPUWrapper());
    }
    return s_instance_;
  }
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>& GetLocalTable(
      int table_id) {
    return local_tables_[table_id];
  }
  void SetSlotVector(const std::vector<int>& slot_vector) {
    slot_vector_ = slot_vector;
  }

 private:
  static std::shared_ptr<PSGPUWrapper> s_instance_;
  Dataset* dataset_;
  std::unordered_map<
      uint64_t, std::vector<std::unordered_map<uint64_t, std::vector<float>>>>
      local_tables_;
  HeterPsBase* HeterPs_;
  std::vector<LoDTensor> keys_tensor;  // Cache for pull_sparse
  std::shared_ptr<HeterPsResource> resource_;
  int32_t sleep_seconds_before_fail_exit_;
  std::vector<int> slot_vector_;
  std::vector<int> heter_devices_;
  std::unordered_set<std::string> gpu_ps_config_keys_;
  HeterObjectPool<HeterContext> gpu_task_pool_;
  std::vector<std::vector<std::vector<uint64_t>>> thread_keys_;
  int thread_keys_thread_num_ = 37;
  int thread_keys_shard_num_ = 37;
  uint64_t max_fea_num_per_pass_ = 5000000000;

 protected:
  static bool is_initialized_;
};

}  // end namespace framework
}  // end namespace paddle
#endif
