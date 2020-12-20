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

#pragma once

#ifdef PADDLE_WITH_PSLIB

#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>


#include "paddle/fluid/framework/fleet/heter_box/hashtable/gpu_resource.h"
#include "paddle/fluid/framework/fleet/heter_box/heter_box_base.h"
#include "paddle/fluid/framework/fleet/gpu_task.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN

namespace paddle {
namespace framework {

class PSGPUWrapper {
 public:
  virtual ~PSGPUWrapper() {
    delete GpuPs_;
  }

  PSGPUWrapper() {
    GpuPs_ = NULL;
    sleep_seconds_before_fail_exit_ = 300;
  }
  
  void PullSparse(const paddle::platform::Place& place,
                  const int table_id,
                  const std::vector<const uint64_t*>& keys,
                  const std::vector<float*>& values,
                  const std::vector<int64_t>& slot_lengths,
                  const int hidden_size);
  void PushSparseGrad(const paddle::platform::Place& place,
                      const int table_id,
                      const std::vector<const uint64_t*>& keys,
                      const std::vector<const float*>& grad_values,
                      const std::vector<int64_t>& slot_lengths,
                      const int hidden_size, const int batch_size);
  void CopyKeys(const paddle::platform::Place& place, uint64_t** origin_keys,
                uint64_t* total_keys, const int64_t* gpu_len, int slot_num,
                int total_len);
  void CopyForPull(const paddle::platform::Place& place, uint64_t** gpu_keys,
                   const std::vector<float*>& values, const FeatureValue* total_values_gpu,
                   const int64_t* gpu_len, const int slot_num,
                   const int hidden_size,
                   const int64_t total_length);

  void CopyForPush(const paddle::platform::Place& place,
                   const std::vector<const float*>& grad_values,
                   FeaturePushValue* total_grad_values_gpu,
                   const std::vector<int64_t>& slot_lengths,
                   const int hidden_size,
                   const int64_t total_length, const int batch_size);
  
  void BuildGPUPS(const uint64_t table_id, int feature_dim, std::shared_ptr<GpuTask> gpu_task);
  void InitializeGPU(const std::vector<int>& dev_ids) {
    if (s_instance_ != NULL) {
      VLOG(3) << "PSGPUWrapper Begin InitializeGPU";
      resource_ = std::make_shared<HeterBoxResource>(dev_ids);
      resource_->enable_p2p();
      keys_tensor.resize(resource_->total_gpu());
    }
  }
  // PSGPUWrapper singleton
  static std::shared_ptr<PSGPUWrapper> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PSGPUWrapper());
    }
    return s_instance_;
  }
  std::vector<std::unordered_map<uint64_t, std::vector<float>>>& GetLocalTable(int table_id) {
    return local_tables_[table_id];
  }
  void SetSlotVector(const std::vector<int>& slot_vector) {
    slot_vector_ = slot_vector;
  }

 private:
  static std::shared_ptr<PSGPUWrapper> s_instance_;
  std::unordered_map<uint64_t, std::vector<std::unordered_map<uint64_t,std::vector<float>>>> local_tables_;
  HeterBoxBase* GpuPs_;
  std::vector<LoDTensor> keys_tensor;  // Cache for pull_sparse
  std::shared_ptr<HeterBoxResource> resource_;
  int32_t sleep_seconds_before_fail_exit_;
  std::vector<int> slot_vector_;

 protected:
  static bool is_initialized_;

};

}  // end namespace framework
}  // end namespace paddle
#endif
