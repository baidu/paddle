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

#include <string>
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/trainer.h"

#if defined PADDLE_WITH_PSCORE
#include "paddle/fluid/distributed/service/communicator.h"
#endif

namespace paddle {
namespace framework {

void MultiTrainer::Initialize(const TrainerDesc& trainer_desc,
                              Dataset* dataset) {
  thread_num_ = trainer_desc.thread_num();
  SetDataset(dataset);

  ParseDumpConfig(trainer_desc);
  mpi_rank_ = trainer_desc.mpi_rank();
  mpi_size_ = trainer_desc.mpi_size();
  dump_file_num_ = trainer_desc.dump_file_num();

  for (int i = 0; i < trainer_desc.downpour_param().stat_var_names_size();
       i++) {
    need_merge_var_names_.push_back(
        trainer_desc.downpour_param().stat_var_names(i));
  }
#ifdef PADDLE_WITH_HETERPS
  for (int i = 0; i < thread_num_; ++i) {
    int num = trainer_desc.worker_places(i);
    platform::CUDAPlace place = platform::CUDAPlace(num);
    places_.push_back(place);
  }
#endif
  // get filelist from trainer_desc here
  const std::vector<paddle::framework::DataFeed*> readers =
      dataset->GetReaders();
  VLOG(3) << "readers num: " << readers.size();
  // change thread num to readers num
  thread_num_ = readers.size();
  VLOG(3) << "worker thread num: " << thread_num_;
  workers_.resize(thread_num_);

#if defined PADDLE_WITH_PSCORE
  if (trainer_desc.thread_barrier()) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerReset(
        thread_num_);
  }
#endif

  for (int i = 0; i < thread_num_; ++i) {
    workers_[i] = DeviceWorkerFactory::CreateDeviceWorker(
        trainer_desc.device_worker_name());
    workers_[i]->SetNeedDumpField(need_dump_field_);
    workers_[i]->SetNeedDumpParam(need_dump_param_);
    workers_[i]->SetDumpFieldVector(dump_fields_);
    workers_[i]->SetDumpParamVector(dump_param_);
    workers_[i]->InitRandomDumpConfig(trainer_desc);
    workers_[i]->Initialize(trainer_desc);
    workers_[i]->SetDeviceIndex(i);
    workers_[i]->SetDataFeed(readers[i]);
  }

  // set debug here
  SetDebug(trainer_desc.debug());
}

std::string MultiTrainer::GetDumpPath(int tid) {
  if (user_define_dump_filename_ != "") {
    return string::format_string("%s/part-%s-%05d", dump_fields_path_.c_str(),
                                 user_define_dump_filename_.c_str(), tid);
  }
  return string::format_string("%s/part-%03d-%05d", dump_fields_path_.c_str(),
                               mpi_rank_, tid);
}

void MultiTrainer::InitDumpEnv() {
  queue_ = paddle::framework::MakeChannel<std::string>();
  for (int i = 0; i < thread_num_; ++i) {
    workers_[i]->SetChannelWriter(queue_.get());
  }
  dump_thread_num_ = 1;
  if (dump_file_num_ > mpi_size_) {
    dump_thread_num_ = dump_file_num_ / mpi_size_;
    if (dump_file_num_ % mpi_size_ > mpi_rank_) {
      dump_thread_num_ += 1;
    }
  }
  for (int i = 0; i < dump_thread_num_; i++) {
    dump_thread_.push_back(
        std::thread(std::bind(&TrainerBase::DumpWork, this, i)));
  }
}

// call only after all resources are set in current trainer
void MultiTrainer::InitTrainerEnv(const ProgramDesc& main_program,
                                  const platform::Place& place) {
  for (int i = 0; i < thread_num_; ++i) {
#ifdef PADDLE_WITH_HETERPS
    workers_[i]->SetPlace(places_[i]);
    workers_[i]->SetReaderPlace(places_[i]);
    workers_[i]->SetDeviceContext(
        platform::DeviceContextPool::Instance().Get(places_[i]));
#else
    workers_[i]->SetPlace(place);
    workers_[i]->SetReaderPlace(place);
#endif
    workers_[i]->SetRootScope(root_scope_);
    workers_[i]->CreateDeviceResource(main_program);  // Program
    workers_[i]->BindingDataFeedMemory();
    workers_[i]->CacheProgram(main_program);
  }
#ifdef PADDLE_WITH_HETERPS
  for (int num = 0; num < thread_num_; ++num) {
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
        if (root_var->IsType<SelectedRows>()) {
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
#endif
}

void MultiTrainer::InitOtherEnv(const ProgramDesc& main_program) {
  if (need_dump_field_ || need_dump_param_) {
    InitDumpEnv();
  }
  VLOG(3) << "init other env done.";
}

Scope* MultiTrainer::GetWorkerScope(int thread_id) {
  return workers_[thread_id]->GetThreadScope();
}

void MultiTrainer::Run() {
  VLOG(3) << "Going to run";
  LOG(ERROR) << "multi run " << thread_num_ << "\t" << debug_;
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    if (!debug_) {
      threads_.push_back(
          std::thread(&DeviceWorker::TrainFiles, workers_[thidx].get()));
    } else {
      threads_.push_back(std::thread(&DeviceWorker::TrainFilesWithProfiler,
                                     workers_[thidx].get()));
    }
  }
  for (auto& th : threads_) {
    th.join();
  }
}

#ifdef PADDLE_WITH_HETERPS
void MultiTrainer::MergeDenseParam() {
#ifdef PADDLE_WTIH_PSCORE
  auto communicator = paddle::distributed::Communicator::GetInstance();
  auto& recv_ctx = communicator->GetRecvCtxMap();
  Scope* thread_scope = workers_[0]->GetThreadScope();
  for (auto& iter : recv_ctx) {
    auto& varnames = iter.second;
    for (auto& name : varnames) {
      Variable* root_var = root_scope_->FindVar(name);
      LoDTensor* root_tensor = root_var->GetMutable<LoDTensor>();
      Variable* var = thread_scope->FindVar(name);
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      TensorCopy((*tensor), root_tensor->place(), root_tensor);
    }
  }
#endif
}
#endif

template <typename T>
void MultiTrainer::MergeToRootScope(LoDTensor* root_tensor, LoDTensor* tensor) {
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

void MultiTrainer::Finalize() {
  if (need_dump_field_ || need_dump_param_) {
    FinalizeDumpEnv();
  }
#ifdef PADDLE_WITH_HETERPS
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
  MergeDenseParam();

#endif
  root_scope_->DropKids();
}

}  // end namespace framework
}  // end namespace paddle
