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

#include <atomic>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>         // NOLINT
#include <unordered_map>  // NOLINT
#include <unordered_set>  // NOLINT
#include <utility>        // NOLINT
#include <vector>

#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/platform/timer.h"

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace framework {

std::string PrintLodTensor(Tensor* tensor, int64_t start, int64_t end);
std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index);
bool CheckValidOutput(LoDTensor* tensor, size_t batch_size);

class FleetWrapper;

#define SEC_LOG                                                              \
  VLOG(3) << "[s" << section_id_ << "p" << pipeline_id_ << "t" << thread_id_ \
          << "]: "

class PullDenseWorker {
 public:
  virtual ~PullDenseWorker() {}
  virtual void Initialize(const TrainerDesc& param);
  int Start();
  void Stop();
  void SetRootScope(Scope* scope) { root_scope_ = scope; }
  void IncreaseThreadVersion(int thread_id, uint64_t table_id);
  void ResetThreadVersion(uint64_t table_id);
  void Wait(std::vector<::std::future<int32_t>>* status_vec);
  void PullDense(bool force_update = false);
  int GetThreadIdByScope(const Scope* scope);
  void SetThreadIdByScope(const Scope* scope, int tid);
  static std::shared_ptr<PullDenseWorker> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new paddle::framework::PullDenseWorker());
    }
    return s_instance_;
  }

  static std::shared_ptr<PullDenseWorker> s_instance_;

 private:
  PullDenseWorker() : root_scope_(NULL) {}
  void Run();
  bool CheckUpdateParam(uint64_t table_id);

 private:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  PullDenseWorkerParameter param_;
  DownpourWorkerParameter dwp_param_;
  Scope* root_scope_;
  bool running_;

  static std::map<uint64_t, uint64_t> last_versions_;
  static std::map<uint64_t, uint64_t> current_version_;
  static std::mutex mutex_for_version_;
  static std::map<uint64_t, std::vector<uint64_t>> training_versions_;
  static std::map<uint64_t, std::vector<std::string>> dense_value_names_;

  std::thread t_;
  int thread_num_;
  int sleep_time_ms_;
  int threshold_;

  std::vector<::std::future<int32_t>> pull_dense_status_;
  uint32_t pull_dense_fail_times_ = 0;
  std::vector<float> base_norm_param_;
  std::vector<float> mean_;
  std::vector<float> scale_;
  float squared_sum_epsilon_ = 1e-4;
  std::mutex mutex_for_mean_scale_;
  float total_batch_num_ = 0;
  std::unordered_map<const Scope*, int> scope_to_thread_id_;
};

// should incorporate different type of device
class DeviceWorker {
 public:
  DeviceWorker() {
    no_cvm_ = true;
    use_cvm_ = false;
  }
  virtual ~DeviceWorker() {}
  virtual void Initialize(const TrainerDesc& desc) = 0;
  virtual void SetDeviceIndex(int tid) = 0;
  virtual void TrainFiles() = 0;
  virtual void PrintFetchVars() = 0;
  virtual void TrainFilesWithProfiler() = 0;
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) = 0;
  // will make this zero copy in the future
  virtual void BindingDataFeedMemory() = 0;
  virtual void SetRootScope(Scope* root_scope);
  virtual void SetDataFeed(DataFeed* data_feed);
  virtual void SetNeedDump(bool need_dump_field) {}
  virtual void SetChannelWriter(ChannelObject<std::string>* queue) {}
  virtual void SetPlace(const paddle::platform::Place& place) {
    place_ = place;
  }
  virtual void SetReaderPlace(const paddle::platform::Place& place) {
    device_reader_->SetPlace(place);
  }
  virtual Scope* GetThreadScope() { return thread_scope_; }

 protected:
  Scope* root_scope_ = nullptr;
  Scope* thread_scope_;
  paddle::platform::Place place_;
  DataFeed* device_reader_ = nullptr;
  int64_t batch_num_;
  FetchConfig fetch_config_;
  bool use_cvm_;
  bool no_cvm_;
  std::vector<std::string> all_param_;
};

class CPUWorkerBase : public DeviceWorker {
 public:
  CPUWorkerBase() {}
  virtual ~CPUWorkerBase() {}
  virtual void SetDeviceIndex(int tid) { thread_id_ = tid; }
  virtual void TrainFiles() = 0;
  virtual void TrainFilesWithProfiler() {}
  virtual void PrintFetchVars() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog) {}

 protected:
  int thread_id_;
};

class HogwildWorker : public CPUWorkerBase {
 public:
  HogwildWorker() {}
  virtual ~HogwildWorker() {
    for (OperatorBase* op : ops_) {
      delete op;
    }
    std::vector<OperatorBase*>().swap(ops_);
  }
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void SetNeedDump(bool need_dump_field);
  virtual void SetChannelWriter(ChannelObject<std::string>* queue);
  virtual void PrintFetchVars();
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void BindingDataFeedMemory();
  template <typename T>
  void SetZero(LoDTensor* tensor, LoDTensor* root_tensor, int tensor_dim);

 protected:
  void CreateThreadOperators(const ProgramDesc& program);
  void CreateThreadScope(const ProgramDesc& program);
  virtual void DumpParam(const int batch_id);

  std::vector<std::string> op_names_;
  std::vector<OperatorBase*> ops_;
  bool thread_barrier_;
  // Scope* thread_scope_;
  HogwildWorkerParameter param_;
  std::vector<std::string> skip_ops_;
  std::map<std::string, int> stat_var_name_map_;
  // dump params or grads for debug
  bool need_dump_param_;
  bool need_dump_field_;
  std::vector<std::string> dump_param_;
  std::vector<std::string> dump_fields_;
  ChannelWriter<std::string> writer_;
};

class DownpourWorker : public HogwildWorker {
 public:
  DownpourWorker() {}
  virtual ~DownpourWorker() {}
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();
  virtual void TrainFilesWithProfiler();
  virtual void SetNeedDump(bool need_dump_field);
  virtual void SetChannelWriter(ChannelObject<std::string>* queue);

 protected:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  std::shared_ptr<paddle::framework::PullDenseWorker> pull_dense_worker_;
  void FillSparseValue(size_t table_id);
  void PushGradients();
  void CollectLabelInfo(size_t table_id);
  void AdjustInsWeight();
  // For hit interval ins weight in recall
  void FillInsWeight(size_t table_id);
  void CopySparseTable();
  void CopyDenseTable();
  void CopyDenseVars();
  virtual void DumpParam(const int batch_id);

  DownpourWorkerParameter param_;
  // copy table
  CopyTableConfig copy_table_config_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_sparse_tables_;
  std::unordered_map<uint64_t, std::unordered_set<uint64_t>> feasign_set_;
  // actually pushed feasign of each table
  std::map<uint64_t, std::vector<uint64_t>> sparse_push_keys_;
  std::map<uint64_t, std::vector<std::string>> sparse_key_names_;
  // feasign
  std::map<uint64_t, std::vector<uint64_t>> features_;
  // feasign embedding
  std::map<uint64_t, std::vector<std::vector<float>>> feature_values_;
  std::map<uint64_t, std::vector<std::string>> sparse_value_names_;
  // adjust ins weight
  AdjustInsWeightConfig adjust_ins_weight_config_;
  // check nan and inf during training
  std::vector<std::string> check_nan_var_names_;
  bool need_to_push_sparse_;
  // feasign stats
  std::map<uint64_t, std::vector<float>> feature_labels_;
  std::map<uint64_t, std::vector<std::string>> sparse_grad_names_;
  // feasign embedding gradient
  std::map<uint64_t, std::vector<std::vector<float>>> feature_grads_;
  std::vector<::std::future<int32_t>> push_sparse_status_;
  bool dump_slot_;
  bool need_to_push_dense_;
  std::map<uint64_t, std::vector<std::string>> dense_grad_names_;
  float scale_datanorm_;
  std::vector<::std::future<int32_t>> push_dense_status_;
  // skipped ops
  std::vector<std::string> skip_ops_;
  // just save the value in param_ for easy access
  std::map<uint64_t, std::string> label_var_name_;
  std::map<uint64_t, std::vector<std::string>> dense_value_names_;
  std::map<uint64_t, uint64_t> table_dependency_;
  std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;

 private:
  // std::vector<std::string> dump_param_;
  // just save the value in param_ for easy access
  // std::map<uint64_t, std::string> label_var_name_;
  // std::map<uint64_t, std::vector<std::string>> dense_value_names_;

  std::shared_ptr<PullDenseWorker> _pull_dense_worker;

  std::vector<float> nid_show_;
  // std::map<uint64_t, uint64_t> table_dependency_;
  // std::vector<std::pair<uint64_t, uint64_t>> copy_dense_tables_;

  // recall hit interval ins weight
  RecallInsWeightConfig recall_ins_weight_config_;
  std::vector<std::string> item_sampling_slots_;
  std::vector<float> item_freq_vec_;
  std::vector<float> hit_interval_new_;
  std::vector<int> need_hit_interval_;
};

class DownpourWorkerOpt : public DownpourWorker {
 public:
  DownpourWorkerOpt() {}
  virtual ~DownpourWorkerOpt() {}
  virtual void CreateDeviceResource(const ProgramDesc& main_prog);
  virtual void Initialize(const TrainerDesc& desc);
  virtual void TrainFiles();

 protected:
  void CreateThreadOperatorsWithRerank(const ProgramDesc& program);
  std::vector<std::vector<OperatorBase*>> loss_ops_;
  std::vector<std::vector<std::string>> loss_op_names_;
  std::vector<std::string> loss_names_;
  std::string async_wait_name_;
  int async_index_ = -1;
  uint64_t async_tid_ = 0;
};

#if defined(PADDLE_WITH_NCCL)
using ScopeQueue = operators::reader::BlockingQueue<Scope*>;

class SyncFunctor {
 public:
  SyncFunctor(int rank_id, int rank_num, int sync_steps);
  virtual ~SyncFunctor() {}

  void SetSyncParam(const std::vector<std::string>& sync_param) {
    sync_param_ = &sync_param;
  }
  void SetNcclCtxMap(platform::NCCLContextMap* nccl_ctx_map) {
    nccl_ctx_map_ = nccl_ctx_map;
  }

  int operator()(Scope* scope);
  static std::vector<Scope*> pipeline_scopes_;
  static uint64_t sync_flag_;

 protected:
  const int rank_id_;
  const int rank_num_;
  const std::vector<std::string>* sync_param_ = nullptr;
  platform::NCCLContextMap* nccl_ctx_map_ = nullptr;

  uint64_t sync_signal_;
  const int sync_steps_;
  int counter_;

  void Synchronize();
};

class SectionWorker : public DeviceWorker {
 public:
  SectionWorker() {}
  ~SectionWorker() override {}

  void Initialize(const TrainerDesc& desc) override;

  void BindingDataFeedMemory() override {}
  void CreateDeviceResource(const ProgramDesc& main_prog) override{};

  void TrainFiles() override;
  void TrainFilesWithProfiler() override;

  void PrintFetchVars() override {}

  const platform::Place& place() const { return place_; }

  void SetSectionIndex(int section_id) { section_id_ = section_id; }
  void SetDeviceIndex(int tid) override { pipeline_id_ = tid; }
  void SetThreadIndex(int thread_id) { thread_id_ = thread_id; }
  void SetVarNames(const std::vector<std::string>& in_var_names,
                   const std::vector<std::string>& out_var_names) {
    in_var_names_ = &in_var_names;
    out_var_names_ = &out_var_names;
  }
  void SetScopeQueue(ScopeQueue* in_scope_queue, ScopeQueue* out_scope_queue) {
    in_scope_queue_ = in_scope_queue;
    out_scope_queue_ = out_scope_queue;
  }
  void SetCountMutex(std::mutex* mutex) { worker_count_mutex_ = mutex; }
  void SetWorkerCount(int* worker_count) { worker_count_ = worker_count; }
  void SetSectionNum(int section_num) { section_num_ = section_num; }
  void SetPipelineNum(int pipeline_num) { pipeline_num_ = pipeline_num; }
  void SetNextSectionPlace(const paddle::platform::Place& place) {
    next_section_place_ = place;
  }
  SyncFunctor* sync_func_ = nullptr;
  void SetSyncFunctor(SyncFunctor* sync_func) { sync_func_ = sync_func; }

  static std::atomic<int> cpu_id_;

 protected:
  void AutoSetCPUAffinity(bool reuse);
  int section_id_;
  int pipeline_id_;
  int section_num_;
  int pipeline_num_;
  int thread_id_;
  // This worker will consume scope from in_scope_queue_
  // and produce scope to out_scope_queue_
  ScopeQueue* in_scope_queue_ = nullptr;
  ScopeQueue* out_scope_queue_ = nullptr;
  const std::vector<std::string>* in_var_names_ = nullptr;
  const std::vector<std::string>* out_var_names_ = nullptr;
  std::mutex* worker_count_mutex_ = nullptr;
  int* worker_count_ = nullptr;
  paddle::platform::Place next_section_place_;

  std::vector<std::unique_ptr<OperatorBase>> ops_;

  platform::DeviceContext* dev_ctx_ = nullptr;
};
#endif
}  // namespace framework
}  // namespace paddle
