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

#include <cstdlib>
#include <string>
#include <vector>
#include "io/fs.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_context.h"
#include "paddle/fluid/framework/fleet/heter_ps/feature_value.h"
#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/framework/trainer.h"
#if (defined PADDLE_WITH_NCCL) && (defined PADDLE_WITH_PSLIB)
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
  for (size_t thidx = 0; thidx < places_.size(); ++thidx) {
    threads_.push_back(
        std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
  }
}

Scope* PSGPUTrainer::GetWorkerScope(int thread_id) { return nullptr; }

template <typename T>
void PSGPUTrainer::MergeToRootScope(LoDTensor* root_tensor, LoDTensor* tensor) {
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
