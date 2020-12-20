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

#ifdef PADDLE_WITH_PSLIB
/*
#include <algorithm>
#include <utility>
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
*/
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"


namespace paddle {
namespace framework {

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;

void PSGPUWrapper::BuildGPUPS(uint64_t table_id, int feature_dim, std::shared_ptr<GpuTask> gpu_task) {
  platform::Timer timeline;
  timeline.Start();
  int shard_num = gpu_task->feature_keys_.size();
  if (shard_num == 0) {
    return;
  }
  
  std::vector<size_t> feature_keys_count(shard_num);
  size_t size_max = 0;
  for (int i = 0; i < shard_num; i++) {
    feature_keys_count[i] = gpu_task->feature_keys_[i].size();
    size_max = std::max(size_max, feature_keys_count[i]);
  }
  if (GpuPs_) {
    GpuPs_->show_one_table(0);
    return ;
  }
  GpuPs_ = HeterBoxBase::get_instance(size_max, resource_);
  for (int i = 0; i < shard_num; ++i) {
    std::cout << "building table: " << i << std::endl;
    GpuPs_->build_ps(i, gpu_task->feature_keys_[i].data(), gpu_task->feature_values_[i].data(), feature_keys_count[i], 10000, 2);
    GpuPs_->show_one_table(i);
  }
  timeline.Pause();
  VLOG(0) << "GpuPs build table total costs: " << timeline.ElapsedSec() << " s.";
  
}

void PSGPUWrapper::PullSparse(const paddle::platform::Place& place,
                            const int table_id,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<float*>& values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size) {
  VLOG(3) << "Begine Gpu Ps PullSparse";
  platform::Timer all_timer;
  platform::Timer pull_gpups_timer;
  all_timer.Start();
  int64_t total_length = std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu =
      reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
    int devid_2_index = GpuPs_->get_index_by_devid(device_id);
    LoDTensor& total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys = reinterpret_cast<uint64_t*>(
        total_keys_tensor.mutable_data<int64_t>({total_length, 1}, place));

    // construct slot_level lod info
    auto slot_lengths_lod = slot_lengths;
    for (size_t i = 1; i < slot_lengths_lod.size(); i++) {
      slot_lengths_lod[i] += slot_lengths_lod[i - 1];
    }
    auto buf_key = memory::AllocShared(place, keys.size() * sizeof(uint64_t*));
    auto buf_length =
        memory::AllocShared(place, slot_lengths.size() * sizeof(int64_t));
    uint64_t** gpu_keys = reinterpret_cast<uint64_t**>(buf_key->ptr());
    int64_t* gpu_len = reinterpret_cast<int64_t*>(buf_length->ptr());
    cudaMemcpy(gpu_keys, keys.data(), keys.size() * sizeof(uint64_t*),
               cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_len, slot_lengths_lod.data(),
               slot_lengths.size() * sizeof(int64_t), cudaMemcpyHostToDevice);

    this->CopyKeys(place, gpu_keys, total_keys, gpu_len,
                   static_cast<int>(slot_lengths.size()),
                   static_cast<int>(total_length));
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index << " len: " << total_length;
    pull_gpups_timer.Start();
    GpuPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                              static_cast<int>(total_length));
    //PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
    //                              "PullSparseGPU failed in GPUPS."));
    pull_gpups_timer.Pause();

    VLOG(3) << "Begin Copy result to tensor, total_length[" << total_length
            << "]";
    this->CopyForPull(place, gpu_keys, values, total_values_gpu, gpu_len,
                      static_cast<int>(slot_lengths.size()), hidden_size,
                      total_length);
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GpuPs: PullSparse Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "GpuPs PullSparse total costs: " << all_timer.ElapsedSec()
          << " s, of which GPUPS costs: " << pull_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PullSparse";
  
  
}


void PSGPUWrapper::PushSparseGrad(const paddle::platform::Place& place,
                            const int table_id,
                            const std::vector<const uint64_t*>& keys,
                            const std::vector<const float*>& grad_values,
                            const std::vector<int64_t>& slot_lengths,
                            const int hidden_size, const int batch_size) {
  VLOG(3) << "Begin GPUPS PushSparseGrad";
  platform::Timer all_timer;
  platform::Timer push_gpups_timer;
  all_timer.Start();
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(
      place, total_length * sizeof(FeaturePushValue));
  FeaturePushValue* total_grad_values_gpu = reinterpret_cast<FeaturePushValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (platform::is_gpu_place(place)) {
    int device_id = boost::get<platform::CUDAPlace>(place).GetDeviceId();
    int devid_2_index = GpuPs_->get_index_by_devid(device_id);
    LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to gpups struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index << " len: " << total_length;
    push_gpups_timer.Start();
    GpuPs_->push_sparse(devid_2_index,
        total_keys, total_grad_values_gpu, static_cast<int>(total_length));
    push_gpups_timer.Pause();
  } else {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GPUPS: PushSparseGrad Only Support CUDAPlace Now."));
  }
  all_timer.Pause();
  VLOG(1) << "PushSparseGrad total cost: " << all_timer.ElapsedSec()
          << " s, of which GPUPS cost: " << push_gpups_timer.ElapsedSec()
          << " s";
  VLOG(3) << "End PushSparseGrad";
}


}  // end namespace framework
}  // end namespace paddle
#endif
