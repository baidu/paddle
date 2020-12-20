/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstdlib>
#include <string>
#include <vector>
#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_box/hashtable/feature_value.h"
#include "paddle/fluid/framework/fleet/gpu_task.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#ifdef PADDLE_WITH_PSLIB

#include "paddle/fluid/platform/cuda_device_guard.h"

namespace paddle {
namespace framework {

void PSGPUTrainer::Initialize(const TrainerDesc& trainer_desc,
                                 Dataset* dataset) {
  dataset_ = dataset;
  thread_num_ = trainer_desc.thread_num();
  param_ = trainer_desc.downpour_param();
  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }
  scale_datanorm_ = trainer_desc.scale_datanorm();
  int place_num = trainer_desc.worker_places_size();
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  std::vector<int> dev_ids;
  for (int i = 0; i < place_num; ++i) {
    int num = trainer_desc.worker_places(i);
    platform::CUDAPlace place = platform::CUDAPlace(num);
    places_.push_back(place);
    dev_ids.push_back(num);
  }
  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
  VLOG(3) << "going to initialize pull dense worker";
  pull_dense_worker_ = PullDenseWorker::GetInstance();
  pull_dense_worker_->Initialize(trainer_desc);
  SetDebug(trainer_desc.debug());
  fleet_ptr_ = FleetWrapper::GetInstance();
  trainer_desc_ = trainer_desc;
  workers_.resize(place_num);
  for (int i = 0; i < place_num; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetWorkerNum(place_num);
  }
  auto gpu_ps_wrapper = PSGPUWrapper::GetInstance();
  gpu_ps_wrapper->InitializeGPU(dev_ids);
  return;
}

void PSGPUTrainer::DumpWork(int tid) {}

void PSGPUTrainer::RegisterHeterCallback() {
  /*
  auto fleet_ptr = FleetWrapper::GetInstance();
  fleet_ptr->RegisterHeterCallback([this](int worker, int taskid) {
    // workers_[worker]->Schedule(taskid);
  });
  */
}

void PSGPUTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                     const platform::Place& place) {
  for (size_t i = 0; i < places_.size(); ++i) {
    workers_[i]->SetPlace(places_[i]);
    workers_[i]->SetReaderPlace(places_[i]);
    workers_[i]->SetRootScope(root_scope_);
    workers_[i]->CreateDeviceResource(main_program);  // Program
    workers_[i]->BindingDataFeedMemory();
  }
  for (size_t num = 0; num < places_.size(); ++num) {
    auto place = places_[num];
    Scope* scope = workers_[num]->GetThreadScope();
    auto& block = main_program.Block(0);
    for (auto& var : block.AllVars()) {
      if (var->Persistable()) {
        auto name = var->Name();
        Variable* root_var = root_scope_->FindVar(name);
        if (!root_var) {
          continue;
        }
        LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
        auto* ptr = scope->Var(name);
        InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
        LoDTensor* thread_tensor = ptr->GetMutable<LoDTensor>();
        TensorCopy(*root_tensor, place, thread_tensor);

      }
    }
  }
  place_ = place;
  return;
}

void PSGPUTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  pull_dense_worker_->SetRootScope(root_scope_);
  for (size_t i = 0; i < places_.size(); ++i) {
    pull_dense_worker_->AddThreadScope(workers_[i]->GetThreadScope());
  }
  VLOG(3) << "init other env done.";
}

void PSGPUTrainer::Run() {
  BuildGPUPSTask(0, 8);
  for (size_t thidx = 0; thidx < places_.size(); ++thidx) {
    threads_.push_back(
        std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
  }
}
void PSGPUTrainer::BuildGPUPSTask(int table_id, int feadim) {
  VLOG(3) << "PSGPUTrainer::BuildGPUPSTask begin";
  platform::Timer timeline;
  timeline.Start();
  MultiSlotDataset* dataset = dynamic_cast<MultiSlotDataset*>(dataset_);
  auto fleet_ptr = FleetWrapper::GetInstance();
  std::shared_ptr<GpuTask>  gpu_task = std::make_shared<GpuTask>();
  auto& multi_output_channel = dataset->GetCurOutputChannel();
  auto& input_channel = dataset->GetInputChannelRef();
  int gen_shard_num = multi_output_channel.size();
  int device_num = places_.size();
  auto gpu_ps_wrapper = PSGPUWrapper::GetInstance();
  auto& local_keys = gpu_task->feature_keys_;
  local_keys.resize(device_num);
  auto& local_values = gpu_task->feature_values_;
  local_values.resize(device_num);
  auto& local_ptr = gpu_task->value_ptr_;
  local_ptr.resize(device_num);
  for (auto& ks : local_keys) {
    ks.reserve(100000);
  }
  // read thread
  std::vector<std::thread> threads(gen_shard_num);
  std::vector<std::shared_ptr<ThreadPool>> consume_task_pool(device_num);
  for (size_t i = 0; i < consume_task_pool.size(); i++) {
    consume_task_pool[i].reset(new ::ThreadPool(1));
  }
  auto consume_func = [&local_keys](int shard_id, int feadim,
                                          std::vector<uint64_t>& keys) {
    local_keys[shard_id].insert(local_keys[shard_id].end(), keys.begin(), keys.end());
  };

  if (input_channel->Size() == 0) {
    // output_channel_ should hold one pass instances now
    uint64_t output_channels_data_size = 0;
    for (size_t i = 0; i < multi_output_channel.size(); i++) {
      int cur_channel_size = multi_output_channel[i]->Size();
      output_channels_data_size += cur_channel_size;
      
    }
    CHECK(output_channels_data_size > 0);
    for (auto& ks : local_keys) {
      ks.reserve(output_channels_data_size * 10); // magic number
    }
    auto gen_func = [&dataset, &device_num, &feadim, &consume_task_pool, &multi_output_channel,
                   &consume_func](int i) {
      const std::deque<Record>& vec_data = multi_output_channel[i]->GetData();
      std::vector<std::vector<uint64_t>> task_keys(device_num);
      std::vector<std::future<void>> task_futures;
      for (size_t j = 0; j < vec_data.size(); j++) {
        for (auto& feature : vec_data[j].uint64_feasigns_) {
          int shard = feature.sign().uint64_feasign_ % device_num;
          task_keys[shard].push_back(feature.sign().uint64_feasign_);
        }
      }

      for (int shard_id = 0; shard_id < device_num; shard_id++) {
        task_futures.emplace_back(consume_task_pool[shard_id]->enqueue(
            consume_func, shard_id, feadim, task_keys[shard_id]));
      }

      for (auto& tf : task_futures) {
        tf.wait();
      }
      for (auto& tk : task_keys) {
        tk.clear();
        std::vector<uint64_t>().swap(tk);
      }
      task_keys.clear();
      std::vector<std::vector<uint64_t>>().swap(task_keys);
    };
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(gen_func, i);
    }
    for (std::thread& t : threads) {
      t.join();
    }
  } else {
    int input_channel_size = input_channel->Size();
    CHECK(input_channel_size > 0);
    CHECK(gen_shard_num > 0);
    for (auto& ks : local_keys) {
      ks.reserve(input_channel_size * 10); // magic number
    }
    const std::deque<Record>& vec_data = input_channel->GetData();
    auto gen_func = [&dataset, &vec_data, &device_num, &gen_shard_num, &input_channel_size, &feadim, &consume_task_pool, multi_output_channel,
                   &consume_func](int i) {
      std::vector<std::vector<uint64_t>> task_keys(device_num);
      std::vector<std::future<void>> task_futures;
      size_t per_shard_num = input_channel_size / gen_shard_num + 1;
      size_t total_size = vec_data.size();
      size_t start_index = i * per_shard_num;
      size_t end_index = std::min(start_index+per_shard_num-1, total_size-1);
      for (size_t j = start_index; j <= end_index ; j++) {
        for (auto& feature : vec_data[j].uint64_feasigns_) {
          int shard = feature.sign().uint64_feasign_ % device_num;
          task_keys[shard].push_back(feature.sign().uint64_feasign_);
        }
      }

      for (int shard_id = 0; shard_id < device_num; shard_id++) {
        task_futures.emplace_back(consume_task_pool[shard_id]->enqueue(
            consume_func, shard_id, feadim, task_keys[shard_id]));
      }

      for (auto& tf : task_futures) {
        tf.wait();
      }
      for (auto& tk : task_keys) {
        tk.clear();
        std::vector<uint64_t>().swap(tk);
      }
      task_keys.clear();
      std::vector<std::vector<uint64_t>>().swap(task_keys);
    };
    for (size_t i = 0; i < threads.size(); i++) {
      threads[i] = std::thread(gen_func, i);
    }
    for (std::thread& t : threads) {
      t.join();
    }
  }
  timeline.Pause();
  VLOG(0) << "GpuPs build task cost " << timeline.ElapsedSec() << " seconds.";
  timeline.Start(); 
  auto unique_func = [&local_keys](int i) {
    auto& cur_keys = local_keys[i];
    std::sort(cur_keys.begin(), cur_keys.end());
    cur_keys.erase(std::unique(cur_keys.begin(), cur_keys.end()), cur_keys.end());
  };
  for (size_t i = 0; i < threads.size(); i++) {
    threads[i] = std::thread(unique_func, i);
  }
  for (std::thread& t : threads) {
    t.join();
  }
  timeline.Pause();
  
  VLOG(0) << "GpuPs task unique cost " << timeline.ElapsedSec() << " seconds.";

  timeline.Start();
  for (size_t i = 0; i < consume_task_pool.size(); i++) {
    consume_task_pool[i].reset();
  }
  consume_task_pool.clear();

  for (int i = 0; i < device_num; i++) {
    local_values[i].resize(local_keys[i].size());
    local_ptr[i].resize(local_keys[i].size());
  }
  
  auto ptl_func = [this, &local_keys, &local_values, &local_ptr, &table_id, &fleet_ptr](int i) {
    size_t key_size = local_keys[i].size();
    auto tt = fleet_ptr->pslib_ptr_->_worker_ptr->pull_sparse_ptr(
        (char**)(local_ptr[i].data()), table_id, local_keys[i].data(), key_size);
    tt.wait();
    auto status = tt.get();
    //auto status = 0;
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
  gpu_ps_wrapper->BuildGPUPS(table_id, feadim, gpu_task);
}

Scope* PSGPUTrainer::GetWorkerScope(int thread_id) { return nullptr; }

template <typename T>
void PSGPUTrainer::MergeToRootScope(LoDTensor* root_tensor,
                                       LoDTensor* tensor) {
  LoDTensor tmp_root;
  TensorCopy(*root_tensor, platform::CPUPlace(), &tmp_root);
  T* tmp_root_data = tmp_root.data<T>();
  LoDTensor tmp_tensor;
  TensorCopy(*tensor, platform::CPUPlace(), &tmp_tensor);
  T* data = tmp_tensor.data<T>();
  for (int i = 0; i < tmp_tensor.numel(); i++) {
    tmp_root_data[i] += data[i];
  }
  TensorCopy(tmp_root, platform::CPUPlace(), root_tensor);
}

void PSGPUTrainer::Finalize() {
  for (auto& th : threads_) {
    th.join();
  }
  for (size_t i = 0; i < need_merge_var_names_.size(); i++) {
    Variable* root_var = root_scope_->FindVar(need_merge_var_names_[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();

    for (size_t j = 0; j < places_.size(); j++) {
      Scope* cur_thread_scope = workers_[j]->GetThreadScope();
      Variable* thread_var =
          cur_thread_scope->FindVar(need_merge_var_names_[i]);
      if (thread_var == nullptr) {
        continue;
      }
      LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
#define MergeCallback(cpp_type, proto_type)                                    \
  do {                                                                         \
    if (root_tensor->type() == proto_type) {                                   \
      if (thread_tensor->type() != proto_type) {                               \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names_[" << i \
                << "] " << need_merge_var_names_[i]                            \
                << ", root tensor type=" << root_tensor->type()                \
                << ", thread tensor type=" << thread_tensor->type();           \
        exit(-1);                                                              \
      }                                                                        \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                  \
    }                                                                          \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }
  pull_dense_worker_->MergeDenseParam();
  root_scope_->DropKids();
}
}  // namespace framework
}  // namespace paddle
#endif
