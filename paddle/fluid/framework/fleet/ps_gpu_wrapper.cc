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

#if (defined PADDLE_WITH_NCCL) && (defined PADDLE_WITH_PSLIB)

#include <algorithm>
#include <deque>

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/platform/timer.h"

namespace paddle {
namespace framework {

std::shared_ptr<PSGPUWrapper> PSGPUWrapper::s_instance_ = NULL;
bool PSGPUWrapper::is_initialized_ = false;

void PSGPUWrapper::BuildTask(uint64_t table_id, int feature_dim) {
  VLOG(3) << "PSGPUWrapper::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  MultiSlotDataset* dataset = dynamic_cast<MultiSlotDataset*>(dataset_);
  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
  auto input_channel = dataset->GetInputChannel();
  auto& local_keys = gpu_task->feature_keys_;
  auto& local_values = gpu_task->feature_values_;
  auto& local_ptr = gpu_task->value_ptr_;
  std::vector<std::thread> threads;
  auto fleet_ptr = FleetWrapper::GetInstance();

  // data should be in input channel
  thread_keys_.resize(thread_keys_thread_num_);
  for (int i = 0; i < thread_keys_thread_num_; i++) {
    thread_keys_[i].resize(thread_keys_shard_num_);
    for (int j = 0; j < thread_keys_shard_num_; j++) {
      thread_keys_[i][j].reserve(2 * max_fea_num_per_pass_ /
                                 thread_keys_shard_num_ /
                                 thread_keys_thread_num_);
    }
  }
  const std::deque<Record>& vec_data = input_channel->GetData();
  size_t total_len = vec_data.size();
  size_t len_per_thread = total_len / thread_keys_thread_num_;
  int remain = total_len % thread_keys_thread_num_;
  size_t begin = 0;
  auto gen_func = [this](const std::deque<Record>& total_data, int begin_index,
                         int end_index, int i) {
    for (auto iter = total_data.begin() + begin_index;
         iter != total_data.begin() + end_index; iter++) {
      const auto& ins = *iter;
      const auto& feasign_v = ins.uint64_feasigns_;
      for (const auto feasign : feasign_v) {
        uint64_t cur_key = feasign.sign().uint64_feasign_;
        int shard_id = cur_key % thread_keys_shard_num_;
        this->thread_keys_[i][shard_id].push_back(cur_key);
      }
    }
  };
  for (int i = 0; i < thread_keys_thread_num_; i++) {
    threads.push_back(std::thread(gen_func, std::ref(vec_data), begin,
                                  begin + len_per_thread + (i < remain ? 1 : 0),
                                  i));
    begin += len_per_thread + (i < remain ? 1 : 0);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec() << " seconds.";

  timeline.Start();

  // merge thread_keys to shard_keys
  gpu_task->init();
  for (size_t i = 0; i < thread_keys_.size(); i++) {
    gpu_task->batch_add_keys(thread_keys_[i]);
    for (int j = 0; j < thread_keys_thread_num_; j++) {
      thread_keys_[i][j].clear();
    }
  }
  timeline.Pause();

  VLOG(0) << "GpuPs task unique11111 cost " << timeline.ElapsedSec()
          << " seconds.";
  VLOG(0) << "FK1";
  timeline.Start();
  gpu_task->UniqueKeys();
  timeline.Pause();

  VLOG(0) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";

  for (int i = 0; i < thread_keys_shard_num_; i++) {
    local_values[i].resize(local_keys[i].size());
    local_ptr[i].resize(local_keys[i].size());
  }

  auto ptl_func = [this, &local_keys, &local_values, &local_ptr, &table_id,
                   &fleet_ptr](int i) {
    size_t key_size = local_keys[i].size();
    auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
        reinterpret_cast<char**>(local_ptr[i].data()), table_id,
        local_keys[i].data(), key_size);
    tt.wait();
    auto status = tt.get();
    // auto status = 0;
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(300);
      exit(-1);
    } else {
      VLOG(3) << "FleetWrapper Pull sparse to local done with table size: "
              << local_keys[i].size();
    }
    for (size_t num = 0; num < local_ptr[i].size(); ++num) {
      float* ptr_val = local_ptr[i][num]->data();
      FeatureValue& val = local_values[i][num];
      size_t dim = local_ptr[i][num]->size();

      val.delta_score = ptr_val[1];
      val.show = ptr_val[2];
      val.clk = ptr_val[3];
      val.slot = ptr_val[6];
      val.lr = ptr_val[4];
      val.lr_g2sum = ptr_val[5];

      if (dim > 7) {
        val.mf_size = MF_DIM + 1;
        for (int x = 0; x < val.mf_size; x++) {
          val.mf[x] = ptr_val[x + 7];
        }
      } else {
        val.mf_size = 0;
        for (int x = 0; x < MF_DIM + 1; x++) {
          val.mf[x] = 0;
        }
      }
    }
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(ptl_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs pull sparse cost " << timeline.ElapsedSec() << " seconds.";
}

void PSGPUWrapper::BuildGPUPS(uint64_t table_id, int feature_dim) {
  BuildTask(table_id, feature_dim);
  platform::Timer timeline;
  timeline.Start();
  std::shared_ptr<HeterContext> gpu_task = gpu_task_pool_.Get();
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
  if (HeterPs_) {
    HeterPs_->show_one_table(0);
    return;
  }
  std::vector<std::thread> threads(shard_num);
  HeterPs_ = HeterPsBase::get_instance(size_max, resource_);
  auto build_func = [this, &gpu_task, &feature_keys_count](int i) {
    std::cout << "building table: " << i << std::endl;
    this->HeterPs_->build_ps(i, gpu_task->feature_keys_[i].data(),
                             gpu_task->feature_values_[i].data(),
                             feature_keys_count[i], 10000, 2);
    HeterPs_->show_one_table(i);
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(build_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  VLOG(0) << "GpuPs build table total costs: " << timeline.ElapsedSec()
          << " s.";
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
  int64_t total_length =
      std::accumulate(slot_lengths.begin(), slot_lengths.end(), 0UL);
  auto buf = memory::AllocShared(place, total_length * sizeof(FeatureValue));
  FeatureValue* total_values_gpu = reinterpret_cast<FeatureValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GpuPs now."));
  } else if (platform::is_gpu_place(place)) {
    VLOG(3) << "Begin copy keys, key_num[" << total_length << "]";
    int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
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
    VLOG(3) << "Begin call PullSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    pull_gpups_timer.Start();
    HeterPs_->pull_sparse(devid_2_index, total_keys, total_values_gpu,
                          static_cast<int>(total_length));
    // PADDLE_ENFORCE_EQ(ret, 0, platform::errors::PreconditionNotMet(
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
  auto buf =
      memory::AllocShared(place, total_length * sizeof(FeaturePushValue));
  FeaturePushValue* total_grad_values_gpu =
      reinterpret_cast<FeaturePushValue*>(buf->ptr());
  if (platform::is_cpu_place(place)) {
    PADDLE_THROW(platform::errors::Unimplemented(
        "Warning:: CPUPlace is not supported in GPUPS now."));
  } else if (platform::is_gpu_place(place)) {
    int device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();
    int devid_2_index = HeterPs_->get_index_by_devid(device_id);
    LoDTensor& cached_total_keys_tensor = keys_tensor[devid_2_index];
    uint64_t* total_keys =
        reinterpret_cast<uint64_t*>(cached_total_keys_tensor.data<int64_t>());
    VLOG(3) << "Begin copy grad tensor to gpups struct";
    this->CopyForPush(place, grad_values, total_grad_values_gpu, slot_lengths,
                      hidden_size, total_length, batch_size);

    VLOG(3) << "Begin call PushSparseGPU in GPUPS, dev: " << devid_2_index
            << " len: " << total_length;
    push_gpups_timer.Start();
    HeterPs_->push_sparse(devid_2_index, total_keys, total_grad_values_gpu,
                          static_cast<int>(total_length));
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
