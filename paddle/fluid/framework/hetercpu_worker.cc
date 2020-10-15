/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/fleet/heter_wrapper.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/string/string_helper.h"

#ifdef PADDLE_WITH_PSLIB

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

void HeterTask::PackTask(Scope* thread_scope, int taskid, DataFeed* reader,
                         int cur_batch, const ProgramDesc& program) {
  // total_time = 0;
  // read_time = 0;
  // pack_time = 0;
  // pull_sparse_local_time = 0;
  taskid_ = taskid;
  auto& block = program.Block(0);
  if (!scope_) {
    scope_ = &(thread_scope->NewScope());
    for (auto& var : block.AllVars()) {
      if (!var->Persistable()) {
        auto* ptr = scope_->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
  state_ = PULL_SPARSE;
  cur_batch_ = cur_batch;
  auto& use_slots = reader->GetUseSlotAlias();
  for (size_t i = 0; i < use_slots.size(); ++i) {
    Variable* thread_var = thread_scope->FindVar(use_slots[i]);
    LoDTensor* thread_tensor = thread_var->GetMutable<LoDTensor>();
    Variable* task_var = scope_->FindVar(use_slots[i]);
    LoDTensor* task_tensor = task_var->GetMutable<LoDTensor>();
    TensorCopy(*thread_tensor, platform::CPUPlace(), task_tensor);
    auto& tensor_lod = thread_tensor->lod()[0];
    LoD thread_lod{tensor_lod};
    task_tensor->set_lod(thread_lod);
  }
}

void HeterCpuWorker::GetXpuOpIndex() {
  xpu_begin_op_index_ = trainer_desc_.xpu_start_idx();
  xpu_end_op_index_ = trainer_desc_.xpu_end_idx();
  VLOG(0) << "xpu begin: " << xpu_begin_op_index_
          << " xpu end: " << xpu_end_op_index_;
  // CHECK(xpu_begin_op_index_ == trainer_desc_.xpu_start_idx());
  // CHECK(xpu_end_op_index_ == trainer_desc_.xpu_end_idx());
  // CHECK(trainer_desc_.op_run_start_idx() == 0);
  // CHECK(trainer_desc_.op_run_end_idx() == xpu_begin_op_index_ - 1);
  // CHECK(trainer_desc_.op_run_end_start_idx() == xpu_end_op_index_ + 1);
  // CHECK(trainer_desc_.op_run_end_end_idx() == ops_.size() - 1);
}

void HeterCpuWorker::Schedule(int taskid) {
  VLOG(3) << "schedule " << taskid;
  auto task = wait_queue_.TryGet(taskid);
  if (task) {
    run_queue_.Put(task->taskid_, task);
  }
}

void HeterCpuWorker::JumpContext(std::shared_ptr<HeterTask> task) {
  VLOG(3) << "jump context " << task->taskid_;
  if (!(wait_queue_.TryPut(task->taskid_, task))) {
    run_queue_.Put(task->taskid_, task);
  }
}

void HeterCpuWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
  mpi_rank_ = desc.mpi_rank();
  trainer_desc_ = desc;
  for (int i = 0; i < param_.sparse_table_size(); ++i) {
    uint64_t table_id =
        static_cast<uint64_t>(param_.sparse_table(i).table_id());
    TableParameter table = param_.sparse_table(i);
    sparse_key_names_[table_id].resize(table.sparse_key_name_size());
    for (int j = 0; j < table.sparse_key_name_size(); ++j) {
      sparse_key_names_[table_id][j] = table.sparse_key_name(j);
    }
    sparse_value_names_[table_id].resize(table.sparse_value_name_size());
    for (int j = 0; j < table.sparse_value_name_size(); ++j) {
      sparse_value_names_[table_id][j] = table.sparse_value_name(j);
    }
    sparse_grad_names_[table_id].resize(table.sparse_grad_name_size());
    for (int j = 0; j < table.sparse_grad_name_size(); ++j) {
      sparse_grad_names_[table_id][j] = table.sparse_grad_name(j);
    }
    label_var_name_[table_id] = table.label_var_name();
    sparse_push_keys_[table_id] = std::vector<uint64_t>();
  }

  for (int i = 0; i < param_.dense_table_size(); ++i) {
    uint64_t table_id = static_cast<uint64_t>(param_.dense_table(i).table_id());
    auto table = param_.dense_table(i);
    dense_value_names_[table_id].resize(table.dense_value_name_size());
    for (int j = 0; j < table.dense_value_name_size(); ++j) {
      dense_value_names_[table_id][j] = table.dense_value_name(j);
    }
    dense_grad_names_[table_id].resize(table.dense_grad_name_size());
    for (int j = 0; j < table.dense_grad_name_size(); ++j) {
      dense_grad_names_[table_id][j] = table.dense_grad_name(j);
    }
  }

  skip_ops_.resize(param_.skip_ops_size());
  for (int i = 0; i < param_.skip_ops_size(); ++i) {
    skip_ops_[i] = param_.skip_ops(i);
  }
  for (int i = 0; i < param_.stat_var_names_size(); ++i) {
    stat_var_name_map_[param_.stat_var_names(i)] = 1;
  }

  need_to_push_sparse_ = param_.push_sparse();
  need_to_push_dense_ = param_.push_dense();

  fleet_ptr_ = FleetWrapper::GetInstance();
  heter_ptr_ = HeterWrapper::GetInstance();
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  dump_fields_.resize(desc.dump_fields_size());
  for (int i = 0; i < desc.dump_fields_size(); ++i) {
    dump_fields_[i] = desc.dump_fields(i);
  }
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
  need_dump_param_ = false;
  dump_param_.resize(desc.dump_param_size());
  for (int i = 0; i < desc.dump_param_size(); ++i) {
    dump_param_[i] = desc.dump_param(i);
  }
  if (desc.dump_param_size() != 0) {
    need_dump_param_ = true;
  }
  for (int i = 0; i < desc.check_nan_var_names_size(); ++i) {
    check_nan_var_names_.push_back(desc.check_nan_var_names(i));
  }
  copy_table_config_ = desc.copy_table_config();
  for (int i = 0; i < copy_table_config_.src_sparse_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_sparse_tables(i);
    uint64_t dest_table = copy_table_config_.dest_sparse_tables(i);
    VLOG(3) << "copy_sparse_tables_ push back " << src_table << "->"
            << dest_table;
    copy_sparse_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (int i = 0; i < copy_table_config_.src_dense_tables_size(); ++i) {
    uint64_t src_table = copy_table_config_.src_dense_tables(i);
    uint64_t dest_table = copy_table_config_.dest_dense_tables(i);
    VLOG(3) << "copy_dense_tables_ push back " << src_table << "->"
            << dest_table;
    copy_dense_tables_.push_back(std::make_pair(src_table, dest_table));
  }
  for (auto& m : copy_table_config_.table_denpendency_map()) {
    if (sparse_key_names_.find(m.key()) != sparse_key_names_.end()) {
      // currently only support one dependency
      for (auto& value : m.values()) {
        table_dependency_[m.key()] = value;
      }
    }
  }
}

void HeterCpuWorker::SetChannelWriter(ChannelObject<std::string>* queue) {
  writer_.Reset(queue);
}

void HeterCpuWorker::SetNeedDump(bool need_dump_field) {
  need_dump_field_ = need_dump_field;
}

// template <typename T>
// std::string PrintLodTensorType(LoDTensor* tensor,
//                                int64_t start, int64_t end) {
//   auto count = tensor->numel();
//   if (start < 0 || end > count) {
//     VLOG(3) << "access violation";
//     return "access violation";
//   }
//   std::ostringstream os;
//   for (int64_t i = start; i < end; i++) {
//     os << ":" << tensor->data<T>()[i];
//   }
//   return os.str();
// }
//
// std::string PrintLodTensorIntType(LoDTensor* tensor, int64_t start,
//                                   int64_t end) {
//   auto count = tensor->numel();
//   if (start < 0 || end > count) {
//     VLOG(3) << "access violation";
//     return "access violation";
//   }
//   std::ostringstream os;
//   for (int64_t i = start; i < end; i++) {
//     os << ":" << static_cast<uint64_t>(tensor->data<int64_t>()[i]);
//   }
//   return os.str();
// }
//
// std::string PrintLodTensor(LoDTensor* tensor, int64_t start, int64_t end) {
//   std::string out_val;
//   if (tensor->type() == proto::VarType::FP32) {
//     out_val = PrintLodTensorType<float>(tensor, start, end);
//   } else if (tensor->type() == proto::VarType::INT64) {
//     out_val = PrintLodTensorIntType(tensor, start, end);
//   } else if (tensor->type() == proto::VarType::FP64) {
//     out_val = PrintLodTensorType<double>(tensor, start, end);
//   } else {
//     out_val = "unsupported type";
//   }
//   return out_val;
// }
//
// std::pair<int64_t, int64_t> GetTensorBound(LoDTensor* tensor, int index) {
//   auto& dims = tensor->dims();
//   if (tensor->lod().size() != 0) {
//     auto& lod = tensor->lod()[0];
//     return {lod[index] * dims[1], lod[index + 1] * dims[1]};
//   } else {
//     return {index * dims[1], (index + 1) * dims[1]};
//   }
// }
//
// bool CheckValidOutput(LoDTensor* tensor, size_t batch_size) {
//   auto& dims = tensor->dims();
//   if (dims.size() != 2) return false;
//   if (tensor->lod().size() != 0) {
//     auto& lod = tensor->lod()[0];
//     if (lod.size() != batch_size + 1) {
//       return false;
//     }
//   } else {
//     if (dims[0] != static_cast<int>(batch_size)) {
//       return false;
//     }
//   }
//   return true;
// }

void HeterCpuWorker::DumpParam() {
  //  std::string os;
  //  for (auto& param : dump_param_) {
  //    os.clear();
  //    os = param;
  //    Variable* var = thread_scope_->FindVar(param);
  //    if (var == nullptr) {
  //      continue;
  //    }
  //    LoDTensor* tensor = var->GetMutable<LoDTensor>();
  //    int64_t len = tensor->numel();
  //    os += PrintLodTensor(tensor, 0, len);
  //    writer_ << os;
  //  }
}

void HeterCpuWorker::CollectLabelInfo(std::shared_ptr<HeterTask> task,
                                      size_t table_idx) {
  if (no_cvm_) {
    return;
  }
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }
  auto& feature = (task->features_)[table_id];
  auto& feature_label = (task->feature_labels_)[table_id];
  Scope* scope = task->scope_;
  feature_label.resize(feature.size());
  Variable* var = scope->FindVar(label_var_name_[table_id]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();

  size_t global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = scope->FindVar(sparse_key_names_[table_id][i]);
    if (fea_var == nullptr) {
      continue;
    }
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var "
                             << sparse_key_names_[table_id][i] << " is null";

    // skip slots which do not have embedding
    Variable* emb_var = scope->FindVar(sparse_value_names_[table_id][i]);
    if (emb_var == nullptr) {
      continue;
    }
    int64_t* ids = tensor->data<int64_t>();
    size_t fea_idx = 0;
    // tensor->lod()[0].size() == batch_size + 1
    for (auto lod_idx = 1u; lod_idx < tensor->lod()[0].size(); ++lod_idx) {
      for (; fea_idx < tensor->lod()[0][lod_idx]; ++fea_idx) {
        // should be skipped feasign defined in protobuf
        if (ids[fea_idx] == 0u) {
          continue;
        }
        feature_label[global_index++] =
            static_cast<float>(label_ptr[lod_idx - 1]);
      }
    }
  }
  CHECK(global_index == feature.size())
      << "expect fea info size:" << feature.size() << " real:" << global_index;
}

void HeterCpuWorker::FillSparseValue(std::shared_ptr<HeterTask> task,
                                     size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }

  auto& fea_value = (task->feature_values_)[table_id];
  Scope* scope = task->scope_;
  auto fea_idx = 0u;

  std::vector<float> init_value(table.fea_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = scope->FindVar(slot_name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << slot_name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = scope->FindVar(emb_slot_name);
    if (var_emb == nullptr) {
      continue;
    }
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr =
        tensor_emb->mutable_data<float>({len, table.emb_dim()}, place_);
    // memset(ptr, 0, sizeof(float) * len * table.emb_dim());
    auto& tensor_lod = tensor->lod()[0];
    LoD data_lod{tensor_lod};
    tensor_emb->set_lod(data_lod);

    bool is_nid = (adjust_ins_weight_config_.need_adjust() &&
                   adjust_ins_weight_config_.nid_slot() == emb_slot_name);
    if (is_nid) {
      nid_show_.clear();
    }
    int nid_ins_index = 0;

    for (int index = 0; index < len; ++index) {
      if (use_cvm_ || no_cvm_) {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data(),
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data(),
               sizeof(float) * table.emb_dim());
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      } else {
        if (ids[index] == 0u) {
          memcpy(ptr + table.emb_dim() * index, init_value.data() + 2,
                 sizeof(float) * table.emb_dim());
          if (is_nid) {
            nid_show_.push_back(-1);
            ++nid_ins_index;
          }
          continue;
        }
        memcpy(ptr + table.emb_dim() * index, fea_value[fea_idx].data() + 2,
               sizeof(float) * table.emb_dim());
        if (is_nid &&
            static_cast<size_t>(index) == tensor->lod()[0][nid_ins_index]) {
          nid_show_.push_back(fea_value[fea_idx][0]);
          ++nid_ins_index;
        }
        fea_idx++;
      }
    }
  }
}

void HeterCpuWorker::AdjustInsWeight(std::shared_ptr<HeterTask> task) {
#ifdef _LINUX
  // check var and tensor not null
  Scope* scope = task->scope_;
  if (!adjust_ins_weight_config_.need_adjust()) {
    VLOG(0) << "need_adjust=false, skip adjust ins weight";
    return;
  }
  Variable* nid_var = scope->FindVar(adjust_ins_weight_config_.nid_slot());
  if (nid_var == nullptr) {
    VLOG(0) << "nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* nid_tensor = nid_var->GetMutable<LoDTensor>();
  if (nid_tensor == nullptr) {
    VLOG(0) << "tensor of nid slot var " << adjust_ins_weight_config_.nid_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  Variable* ins_weight_var =
      scope->FindVar(adjust_ins_weight_config_.ins_weight_slot());
  if (ins_weight_var == nullptr) {
    VLOG(0) << "ins weight var " << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }
  LoDTensor* ins_weight_tensor = ins_weight_var->GetMutable<LoDTensor>();
  if (ins_weight_tensor == nullptr) {
    VLOG(0) << "tensor of ins weight tensor "
            << adjust_ins_weight_config_.ins_weight_slot()
            << " is nullptr, skip adjust ins weight";
    return;
  }

  float* ins_weights = ins_weight_tensor->data<float>();
  size_t len = ins_weight_tensor->numel();  // len = batch size
  // here we assume nid_show slot only has one feasign in each instance
  CHECK(len == nid_show_.size()) << "ins_weight size should be equal to "
                                 << "nid_show size, " << len << " vs "
                                 << nid_show_.size();
  float nid_adjw_threshold = adjust_ins_weight_config_.nid_adjw_threshold();
  float nid_adjw_ratio = adjust_ins_weight_config_.nid_adjw_ratio();
  int64_t nid_adjw_num = 0;
  double nid_adjw_weight = 0.0;
  size_t ins_index = 0;
  for (size_t i = 0; i < len; ++i) {
    float nid_show = nid_show_[i];
    VLOG(3) << "nid_show " << nid_show;
    if (nid_show < 0) {
      VLOG(3) << "nid_show < 0, continue";
      continue;
    }
    float ins_weight = 1.0;
    if (nid_show >= 0 && nid_show < nid_adjw_threshold) {
      ins_weight = log(M_E +
                       (nid_adjw_threshold - nid_show) / nid_adjw_threshold *
                           nid_adjw_ratio);
      // count nid adjw insnum and weight
      ++nid_adjw_num;
      nid_adjw_weight += ins_weight;
      // choose large ins weight
      VLOG(3) << "ins weight new " << ins_weight << ", ins weight origin "
              << ins_weights[ins_index];
      if (ins_weight > ins_weights[ins_index]) {
        VLOG(3) << "ins " << ins_index << " weight changes to " << ins_weight;
        ins_weights[ins_index] = ins_weight;
      }
      ++ins_index;
    }
  }
  VLOG(3) << "nid adjw info: total_adjw_num: " << nid_adjw_num
          << ", avg_adjw_weight: " << nid_adjw_weight;
#endif
}

void HeterCpuWorker::CopySparseTable() {
  for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
    int64_t src_table = copy_sparse_tables_[i].first;
    int64_t dest_table = copy_sparse_tables_[i].second;
    int32_t feanum = 0;
    if (src_table == dest_table) {
      continue;
    } else if (!copy_table_config_.sparse_copy_by_feasign()) {
      if (feasign_set_.find(src_table) == feasign_set_.end()) {
        continue;
      } else if (feasign_set_[src_table].size() == 0) {
        continue;
      }
      feanum = fleet_ptr_->CopyTable(src_table, dest_table);
    } else {
      std::vector<uint64_t> fea_vec(feasign_set_[src_table].begin(),
                                    feasign_set_[src_table].end());
      feanum = fleet_ptr_->CopyTableByFeasign(src_table, dest_table, fea_vec);
      fea_vec.clear();
      std::vector<uint64_t>().swap(fea_vec);
    }
    VLOG(3) << "copy feasign from table " << src_table << " to table "
            << dest_table << ", feasign num=" << feanum;
    feasign_set_[src_table].clear();
    std::unordered_set<uint64_t>().swap(feasign_set_[src_table]);
  }
  feasign_set_.clear();
}

void HeterCpuWorker::CopyDenseTable() {
  if (thread_id_ != 0) {
    return;
  }
  thread_local std::vector<std::future<int32_t>> pull_dense_status;
  for (size_t i = 0; i < copy_dense_tables_.size(); ++i) {
    uint64_t src_table = copy_dense_tables_[i].first;
    uint64_t dest_table = copy_dense_tables_[i].second;
    if (src_table == dest_table) {
      continue;
    }
    int32_t dim = fleet_ptr_->CopyTable(src_table, dest_table);
    VLOG(3) << "copy param from table " << src_table << " to table "
            << dest_table << ", dim=" << dim;
    if (copy_table_config_.dense_pull_after_copy()) {
      VLOG(3) << "dense pull after copy, table=" << dest_table;
      pull_dense_status.resize(0);
      // fleet_ptr_->PullDenseVarsAsync(*root_scope_, dest_table,
      //                                dense_value_names_[dest_table],
      //                                &pull_dense_status);
      for (auto& t : pull_dense_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
          LOG(WARNING) << "pull dense after copy table failed,"
                       << " table=" << dest_table;
        }
      }
    }
  }
}

void HeterCpuWorker::CopyDenseVars() {
  if (thread_id_ != 0) {
    return;
  }
  for (int i = 0; i < copy_table_config_.src_var_list_size(); ++i) {
    auto& src_var_name = copy_table_config_.src_var_list(i);
    auto& dest_var_name = copy_table_config_.dest_var_list(i);
    if (src_var_name == dest_var_name) {
      continue;
    }
    VLOG(3) << "copy dense var from " << src_var_name << " to "
            << dest_var_name;
    Variable* src_var = thread_scope_->FindVar(src_var_name);
    CHECK(src_var != nullptr) << src_var_name << " not found";  // NOLINT
    LoDTensor* src_tensor = src_var->GetMutable<LoDTensor>();
    CHECK(src_tensor != nullptr) << src_var_name
                                 << " tensor is null";  // NOLINT
    float* src_data = src_tensor->data<float>();

    Variable* dest_var = thread_scope_->FindVar(dest_var_name);
    CHECK(dest_var != nullptr) << dest_var_name << " not found";  // NOLINT
    LoDTensor* dest_tensor = dest_var->GetMutable<LoDTensor>();
    CHECK(dest_tensor != nullptr) << dest_var_name
                                  << " tensor is null";  // NOLINT
    float* dest_data = dest_tensor->data<float>();

    CHECK(src_tensor->numel() == dest_tensor->numel())
        << "tensor numel not equal," << src_tensor->numel() << " vs "
        << dest_tensor->numel();
    for (int i = 0; i < src_tensor->numel(); i++) {
      dest_data[i] = src_data[i];
    }
  }
}

void HeterCpuWorker::TrainFilesWithProfiler() {
  VLOG(3) << "Begin to train files with profiler";
  platform::SetNumThreads(1);
  device_reader_->Start();

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    bool need_skip = false;
    for (auto t = 0u; t < skip_ops_.size(); ++t) {
      if (op->Type().find(skip_ops_[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op_name.push_back(op->Type());
    }
  }

  VLOG(3) << "op name size: " << op_name.size();
  op_total_time.resize(op_name.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  double pack_time = 0.0;
  double pull_sparse_local_time = 0.0;
  double op_all_time = 0;
  double xpu_op_time = 0;
  double xpu_wait_time = 0;
  double cpu_op_time = 0;
  double collect_label_time = 0;
  double fill_sparse_time = 0;
  double push_sparse_time = 0;

  int batch_cnt = 0;
  int done_cnt = 0;
  int cur_batch;
  uint64_t total_inst = 0;
  wait_queue_.SetCap(1);
  while (1) {
    std::shared_ptr<HeterTask> task;
    task = run_queue_.Get();
    if (!task) {
      double tmp_read_time;
      timeline.Start();
      cur_batch = device_reader_->Next();
      timeline.Pause();
      tmp_read_time = timeline.ElapsedSec();
      if (cur_batch <= 0) {
        if (batch_cnt == done_cnt) {
          break;
        } else {
          continue;
        }
      }
      batch_cnt += 1;
      int taskid = batch_cnt * worker_num_ + thread_id_;
      timeline.Start();
      task = object_pool_.Get();
      task->Reset();
      task->PackTask(thread_scope_, taskid, device_reader_, cur_batch,
                     program_);
      timeline.Pause();
      task->read_time = tmp_read_time;
      task->pack_time = timeline.ElapsedSec();
      task->total_time = tmp_read_time + task->pack_time;
    }
    for (;;) {
      // pull sparse here
      if (task->state_ == PULL_SPARSE) {
        timeline.Start();
        for (int i = 0;
             i < param_.program_config(0).pull_sparse_table_id_size(); ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).pull_sparse_table_id(i));
          TableParameter table;
          for (auto j : param_.sparse_table()) {
            if (j.table_id() == tid) {
              table = j;
              break;
            }
          }
          fleet_ptr_->HeterPullSparseVars(
              thread_id_, task, tid, sparse_key_names_[tid], table.fea_dim(),
              sparse_value_names_[tid]);
        }
        task->Update();
        // JumpContext(task);
        timeline.Pause();
        task->pull_sparse_local_time += timeline.ElapsedSec();
        task->total_time += timeline.ElapsedSec();
      } else if (task->state_ == OP_RUN) {
        // total_time += task->total_time;
        // read_time += task->read_time;
        // pack_time += task->pack_time;
        // pull_sparse_local_time += task->pull_sparse_local_time;
        for (int i = 0;
             i < param_.program_config(0).pull_sparse_table_id_size(); ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).pull_sparse_table_id(i));
          timeline.Start();
          CollectLabelInfo(task, i);
          timeline.Pause();
          task->collect_label_time += timeline.ElapsedSec();
          task->total_time += timeline.ElapsedSec();
          timeline.Start();
          FillSparseValue(task, i);
          timeline.Pause();
          task->fill_sparse_time += timeline.ElapsedSec();
          task->total_time += timeline.ElapsedSec();

          auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                    sparse_value_names_[tid].end(),
                                    adjust_ins_weight_config_.nid_slot());
          if (nid_iter != sparse_value_names_[tid].end()) {
            AdjustInsWeight(task);
          }
        }

        VLOG(3) << "fill sparse value for all sparse table done.";
        // do computation here
        // int run_op_idx = 0;
        timeline.Start();
        for (int i = 0; i < xpu_begin_op_index_; ++i) {
          auto& op = ops_[i];
          bool need_skip = false;
          for (auto t = 0u; t < skip_ops_.size(); ++t) {
            if (op->Type().find(skip_ops_[t]) != std::string::npos) {
              need_skip = true;
              break;
            }
          }
          if (!need_skip) {
            // timeline.Start();
            op->Run(*(task->scope_), place_);
            // timeline.Pause();
            // op_total_time[run_op_idx++] += timeline.ElapsedSec();
            // total_time += timeline.ElapsedSec();
          }
        }
        task->Update();
        timeline.Pause();
        task->cpu_op_time += timeline.ElapsedSec();
        task->total_time += timeline.ElapsedSec();
      } else if (task->state_ == XPU) {
        timeline.Start();
        VLOG(3) << "call remote xpu taskid = " << task->taskid_;
        std::vector<std::string> send_var_list;
        for (int i = 0; i < trainer_desc_.xpu_recv_list_size(); ++i) {
          send_var_list.push_back(trainer_desc_.xpu_recv_list(i));
        }
        heter_ptr_->CallRemoteXpu(task, this, mpi_rank_, send_var_list);
        timeline.Pause();
        task->xpu_op_time += timeline.ElapsedSec();
        task->total_time += timeline.ElapsedSec();
        task->Update();
        timeline.Start();
        JumpContext(task);
        timeline.Pause();
        task->xpu_wait_time += timeline.ElapsedSec();
        task->total_time += timeline.ElapsedSec();
        break;
      } else if (task->state_ == OP_RUN_END) {
        timeline.Start();
        for (size_t i = xpu_end_op_index_ + 1; i < ops_.size(); ++i) {
          auto& op = ops_[i];
          bool need_skip = false;
          for (auto t = 0u; t < skip_ops_.size(); ++t) {
            if (op->Type().find(skip_ops_[t]) != std::string::npos) {
              need_skip = true;
              break;
            }
          }
          if (!need_skip) {
            op->Run(*(task->scope_), place_);
          }
        }
        // check inf and nan
        for (std::string& var_name : check_nan_var_names_) {
          Variable* var = (task->scope_)->FindVar(var_name);
          if (var == nullptr) {
            continue;
          }
          LoDTensor* tensor = var->GetMutable<LoDTensor>();
          if (tensor == nullptr) {
            continue;
          }
        }
        task->Update();
        timeline.Pause();
        task->cpu_op_time += timeline.ElapsedSec();
        task->total_time += timeline.ElapsedSec();
      } else if (task->state_ == PUSH_GRAD) {
        if (need_to_push_sparse_) {
          // push gradients here
          for (int i = 0;
               i < param_.program_config(0).push_sparse_table_id_size(); ++i) {
            uint64_t tid = static_cast<uint64_t>(
                param_.program_config(0).push_sparse_table_id(i));
            TableParameter table;
            for (auto i : param_.sparse_table()) {
              if (i.table_id() == tid) {
                table = i;
                break;
              }
            }
            timeline.Start();
            fleet_ptr_->HeterPushSparseVars(
                task, tid, sparse_key_names_[tid], sparse_grad_names_[tid],
                table.emb_dim(), &push_sparse_status_, use_cvm_, dump_slot_,
                no_cvm_);
            timeline.Pause();
            task->push_sparse_time += timeline.ElapsedSec();
            task->total_time += timeline.ElapsedSec();
          }
        }

        if (need_to_push_sparse_) {
          VLOG(3) << "push sparse gradient done.";
          int32_t tmp_push_sparse_wait_times = -1;
          static uint32_t push_sparse_wait_times =
              static_cast<uint32_t>(tmp_push_sparse_wait_times);
          if (push_sparse_status_.size() >= push_sparse_wait_times) {
            for (auto& t : push_sparse_status_) {
              t.wait();
            }
            push_sparse_status_.resize(0);
          }

          if (tmp_push_sparse_wait_times == -1) {
            push_sparse_status_.resize(0);
          }
        }

        // thread_scope_->DropKids();
        task->Update();
      } else if (task->state_ == DONE) {
        PrintFetchVars();
        ++done_cnt;
        total_inst += task->cur_batch_;
        object_pool_.Push(task);

        total_time += task->total_time;
        read_time += task->read_time;
        pack_time += task->pack_time;
        pull_sparse_local_time += task->pull_sparse_local_time;
        op_all_time += task->op_all_time;
        xpu_op_time += task->xpu_op_time;
        xpu_wait_time += task->xpu_wait_time;
        cpu_op_time += task->cpu_op_time;
        collect_label_time += task->collect_label_time;
        fill_sparse_time += task->fill_sparse_time;
        push_sparse_time += task->push_sparse_time;
        // ++batch_cnt;
        if (thread_id_ == 0) {
          // should be configured here
          if (done_cnt > 0 && done_cnt % 100 == 0) {
            // double op_sum_time = 0;
            // std::unordered_map<std::string, double> op_to_time;
            // for (size_t i = 0; i < op_total_time.size(); ++i) {
            //   fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
            //           op_name[i].c_str(), op_total_time[i] / done_cnt);
            //   if (op_to_time.find(op_name[i]) == op_to_time.end()) {
            //     op_to_time[op_name[i]] = 0.0;
            //   }
            //   op_to_time[op_name[i]] += op_total_time[i];
            //   op_sum_time += op_total_time[i];
            // }
            // for (auto& i : op_to_time) {
            //   fprintf(stderr, "op [%s] run total time: [%f]ms\n",
            //           i.first.c_str(),
            //           i.second / done_cnt);
            // }
            fprintf(stderr, "cpu op run total time: %fs\n",
                    cpu_op_time / done_cnt);
            fprintf(stderr, "xpu op run total time: %fs\n",
                    xpu_op_time / done_cnt);
            fprintf(stderr, "xpu wait total time: %fs\n",
                    xpu_wait_time / done_cnt);
            fprintf(stderr, "pack task time: %fs\n", pack_time / done_cnt);
            fprintf(stderr, "train total time: %fs\n", total_time / done_cnt);
            fprintf(stderr, "pull sparse local time: %fs\n",
                    pull_sparse_local_time / done_cnt);
            fprintf(stderr, "fill sparse time: %fs\n",
                    fill_sparse_time / done_cnt);
            fprintf(stderr, "push sparse time: %fs\n",
                    push_sparse_time / done_cnt);
            fprintf(stderr, "collect label time: %fs\n",
                    collect_label_time / done_cnt);
            fprintf(stderr, "mean read time: %fs\n", read_time / done_cnt);
            fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
            fprintf(stderr, "cpu op run percent: %f\n",
                    cpu_op_time / total_time * 100);
            fprintf(stderr, "xpu op run percent: %f\n",
                    xpu_op_time / total_time * 100);
            fprintf(stderr, "xpu wait percent: %f\n",
                    xpu_wait_time / total_time * 100);
            fprintf(stderr, "pack task percent: %f\n",
                    pack_time / total_time * 100);
            fprintf(stderr, "pull sparse local time percent: %f\n",
                    pull_sparse_local_time / total_time * 100);
            fprintf(stderr, "collect label time percent: %f\n",
                    collect_label_time / total_time * 100);
            fprintf(stderr, "fill sparse time percent: %f\n",
                    fill_sparse_time / total_time * 100);
            fprintf(stderr, "push sparse time percent: %f\n",
                    push_sparse_time / total_time * 100);
            fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
          }
        }
        break;
      }
    }
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

void HeterCpuWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int done_cnt = 0;
  int cur_batch;
  wait_queue_.SetCap(1);
  need_to_push_dense_ = false;
  while (1) {
    // if (copy_table_config_.need_copy()) {
    //   if (copy_table_config_.sparse_copy_by_feasign()) {
    //     for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
    //       uint64_t tid = copy_sparse_tables_[i].first;
    //       feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
    //                                sparse_push_keys_[tid].end());
    //     }
    //   }
    //   if (batch_cnt % copy_table_config_.batch_num() == 0) {
    //     CopySparseTable();
    //     CopyDenseTable();
    //     CopyDenseVars();
    //   }
    // }

    std::shared_ptr<HeterTask> task;

    task = run_queue_.Get();
    if (!task) {
      cur_batch = device_reader_->Next();
      if (cur_batch <= 0) {
        if (batch_cnt == done_cnt) {
          break;
        } else {
          continue;
        }
      }
      batch_cnt += 1;
      int taskid = batch_cnt * worker_num_ + thread_id_;
      task = object_pool_.Get();
      task->Reset();
      task->PackTask(thread_scope_, taskid, device_reader_, cur_batch,
                     program_);
    }
    for (;;) {
      // pull sparse here
      if (task->state_ == PULL_SPARSE) {
        VLOG(3) << "pull sparse taskid = " << task->taskid_;
        for (int i = 0;
             i < param_.program_config(0).pull_sparse_table_id_size(); ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).pull_sparse_table_id(i));
          TableParameter table;
          for (auto j : param_.sparse_table()) {
            if (j.table_id() == tid) {
              table = j;
              break;
            }
          }
          fleet_ptr_->HeterPullSparseVars(
              thread_id_, task, tid, sparse_key_names_[tid], table.fea_dim(),
              sparse_value_names_[tid]);
        }
        task->Update();
        // JumpContext(task);
        // break;
      } else if (task->state_ == OP_RUN) {
        VLOG(3) << "oprun taskid = " << task->taskid_;
        for (int i = 0;
             i < param_.program_config(0).pull_sparse_table_id_size(); ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).pull_sparse_table_id(i));
          CollectLabelInfo(task, i);
          FillSparseValue(task, i);
          auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                    sparse_value_names_[tid].end(),
                                    adjust_ins_weight_config_.nid_slot());
          if (nid_iter != sparse_value_names_[tid].end()) {
            AdjustInsWeight(task);
          }
        }
        VLOG(3) << "fill sparse value for all sparse table done.";
        // do computation here
        for (int i = 0; i < xpu_begin_op_index_; ++i) {
          auto& op = ops_[i];
          bool need_skip = false;
          for (auto t = 0u; t < skip_ops_.size(); ++t) {
            if (op->Type().find(skip_ops_[t]) != std::string::npos) {
              need_skip = true;
              break;
            }
          }
          if (!need_skip) {
            VLOG(3) << "run op: " << op->Type();
            op->Run(*(task->scope_), place_);
          }
        }
        task->Update();
      } else if (task->state_ == XPU) {
        VLOG(3) << "call remote xpu taskid = " << task->taskid_;
        std::vector<std::string> send_var_list;
        for (int i = 0; i < trainer_desc_.xpu_recv_list_size(); ++i) {
          send_var_list.push_back(trainer_desc_.xpu_recv_list(i));
        }
        heter_ptr_->CallRemoteXpu(task, this, mpi_rank_, send_var_list);
        task->Update();
        JumpContext(task);
        break;
      } else if (task->state_ == OP_RUN_END) {
        for (size_t i = xpu_end_op_index_ + 1; i < ops_.size(); ++i) {
          auto& op = ops_[i];
          bool need_skip = false;
          for (auto t = 0u; t < skip_ops_.size(); ++t) {
            if (op->Type().find(skip_ops_[t]) != std::string::npos) {
              need_skip = true;
              break;
            }
          }
          if (!need_skip) {
            op->Run(*(task->scope_), place_);
          }
        }
        // check inf and nan
        for (std::string& var_name : check_nan_var_names_) {
          Variable* var = (task->scope_)->FindVar(var_name);
          if (var == nullptr) {
            continue;
          }
          LoDTensor* tensor = var->GetMutable<LoDTensor>();
          if (tensor == nullptr) {
            continue;
          }
        }
        task->Update();
      } else if (task->state_ == PUSH_GRAD) {
        VLOG(3) << "push grad taskid = " << task->taskid_;
        if (need_to_push_sparse_) {
          // push gradients here
          for (int i = 0;
               i < param_.program_config(0).push_sparse_table_id_size(); ++i) {
            uint64_t tid = static_cast<uint64_t>(
                param_.program_config(0).push_sparse_table_id(i));
            TableParameter table;
            for (auto i : param_.sparse_table()) {
              if (i.table_id() == tid) {
                table = i;
                break;
              }
            }
            fleet_ptr_->HeterPushSparseVars(
                task, tid, sparse_key_names_[tid], sparse_grad_names_[tid],
                table.emb_dim(), &push_sparse_status_, use_cvm_, dump_slot_,
                no_cvm_);
          }
        }

        if (need_to_push_sparse_) {
          VLOG(3) << "push sparse gradient done.";
          int32_t tmp_push_sparse_wait_times = -1;
          static uint32_t push_sparse_wait_times =
              static_cast<uint32_t>(tmp_push_sparse_wait_times);
          if (push_sparse_status_.size() >= push_sparse_wait_times) {
            for (auto& t : push_sparse_status_) {
              t.wait();
            }
            push_sparse_status_.resize(0);
          }

          if (tmp_push_sparse_wait_times == -1) {
            push_sparse_status_.resize(0);
          }
        }

        // if (need_dump_field_) {
        //   size_t batch_size = device_reader_->GetCurBatchSize();
        //   std::vector<std::string> ars(batch_size);
        //   for (auto& ar : ars) {
        //     ar.clear();
        //   }
        //   auto& ins_id_vec = device_reader_->GetInsIdVec();
        //   auto& ins_content_vec = device_reader_->GetInsContentVec();
        //   for (size_t i = 0; i < ins_id_vec.size(); i++) {
        //     ars[i] += ins_id_vec[i];
        //     ars[i] = ars[i] + "\t" + ins_content_vec[i];
        //   }
        //   for (auto& field : dump_fields_) {
        //     Variable* var = thread_scope_->FindVar(field);
        //     if (var == nullptr) {
        //       continue;
        //     }
        //     LoDTensor* tensor = var->GetMutable<LoDTensor>();
        //     if (!CheckValidOutput(tensor, batch_size)) {
        //       continue;
        //     }
        //     for (size_t i = 0; i < batch_size; ++i) {
        //       auto output_dim = tensor->dims()[1];
        //       std::string output_dimstr =
        //           boost::lexical_cast<std::string>(output_dim);
        //       ars[i] = ars[i] + "\t" + field + ":" + output_dimstr;
        //       auto bound = GetTensorBound(tensor, i);
        //       ars[i] += PrintLodTensor(tensor, bound.first, bound.second);
        //     }
        //   }
        //   // #pragma omp parallel for
        //   for (size_t i = 0; i < ars.size(); i++) {
        //     if (ars[i].length() == 0) {
        //       continue;
        //     }
        //     writer_ << ars[i];
        //   }
        //   if (need_dump_param_ && thread_id_ == 0) {
        //     DumpParam();
        //   }
        // }

        // thread_scope_->DropKids();
        task->Update();
      } else if (task->state_ == DONE) {
        VLOG(3) << "done taskid = " << task->taskid_;
        object_pool_.Push(task);
        PrintFetchVars();
        ++done_cnt;
        // ++batch_cnt;
        break;
      }
    }
  }
  if (need_dump_field_) {
    // writer_.Flush();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

}  // end namespace framework
}  // end namespace paddle
#endif
