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

#include <cstdlib>
#include <ctime>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/platform/cpu_helper.h"

namespace paddle {
namespace framework {
class LoDTensor;
class Variable;
}  // namespace framework
}  // namespace paddle

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

namespace paddle {
namespace framework {

void DownpourWorker::Initialize(const TrainerDesc& desc) {
  param_ = desc.downpour_param();
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

  flag_partial_push_ = false;
  for (auto& m : param_.program_config(0).partial_pushdense_condtable_map()) {
    cond2table_map_[m.key()] = m.value();
    condvalue_set_.insert(m.value());
    flag_partial_push_ = true;
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
  fetch_config_ = desc.fetch_config();
  use_cvm_ = desc.use_cvm();
  // for sparse value accessor, embedding only
  no_cvm_ = desc.no_cvm();
  scale_datanorm_ = desc.scale_datanorm();
  dump_slot_ = desc.dump_slot();
  adjust_ins_weight_config_ = desc.adjust_ins_weight_config();
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

void DownpourWorker::CollectLabelInfo(size_t table_idx) {
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
  auto& feature = features_[table_id];
  auto& feature_label = feature_labels_[table_id];
  feature_label.resize(feature.size());
  Variable* var = thread_scope_->FindVar(label_var_name_[table_id]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label_ptr = tensor->data<int64_t>();

  size_t global_index = 0;
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    VLOG(3) << "sparse_key_names_[" << i
            << "]: " << sparse_key_names_[table_id][i];
    Variable* fea_var = thread_scope_->FindVar(sparse_key_names_[table_id][i]);
    if (fea_var == nullptr) {
      continue;
    }
    LoDTensor* tensor = fea_var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var "
                             << sparse_key_names_[table_id][i] << " is null";

    // skip slots which do not have embedding
    Variable* emb_var =
        thread_scope_->FindVar(sparse_value_names_[table_id][i]);
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

void DownpourWorker::FillSparseValue(size_t table_idx) {
  uint64_t table_id = static_cast<uint64_t>(
      param_.program_config(0).pull_sparse_table_id(table_idx));

  TableParameter table;
  for (auto i : param_.sparse_table()) {
    if (i.table_id() == table_id) {
      table = i;
      break;
    }
  }

  auto& fea_value = feature_values_[table_id];
  auto fea_idx = 0u;

  std::vector<float> init_value(table.fea_dim());
  for (size_t i = 0; i < sparse_key_names_[table_id].size(); ++i) {
    std::string slot_name = sparse_key_names_[table_id][i];
    std::string emb_slot_name = sparse_value_names_[table_id][i];
    Variable* var = thread_scope_->FindVar(slot_name);
    if (var == nullptr) {
      continue;
    }
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    CHECK(tensor != nullptr) << "tensor of var " << slot_name << " is null";
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = thread_scope_->FindVar(emb_slot_name);
    if (var_emb == nullptr) {
      continue;
    }
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr = tensor_emb->mutable_data<float>({len, table.emb_dim()},
                                                 platform::CPUPlace());
    memset(ptr, 0, sizeof(float) * len * table.emb_dim());
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

void DownpourWorker::AdjustInsWeight() {
#ifdef _LINUX
  // check var and tensor not null
  if (!adjust_ins_weight_config_.need_adjust()) {
    VLOG(0) << "need_adjust=false, skip adjust ins weight";
    return;
  }
  Variable* nid_var =
      thread_scope_->FindVar(adjust_ins_weight_config_.nid_slot());
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
      thread_scope_->FindVar(adjust_ins_weight_config_.ins_weight_slot());
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

void DownpourWorker::CopySparseTable() {
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

void DownpourWorker::CopyDenseTable() {
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
      fleet_ptr_->PullDenseVarsAsync(*root_scope_, dest_table,
                                     dense_value_names_[dest_table],
                                     &pull_dense_status, true);
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

void DownpourWorker::CopyDenseVars() {
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

void DownpourWorker::TrainFilesWithProfiler() {
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
  double pull_sparse_time = 0.0;
  double adjust_ins_weight_time = 0.0;
  double collect_label_time = 0.0;
  double fill_sparse_time = 0.0;
  double push_sparse_time = 0.0;
  double push_dense_time = 0.0;
  double copy_table_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  uint64_t total_inst = 0;
  timeline.Start();
  while ((cur_batch = device_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    timeline.Start();
    if (copy_table_config_.need_copy()) {
      VLOG(3) << "copy_sparse_tables_.size " << copy_sparse_tables_.size();
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }
    timeline.Pause();
    copy_table_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();

    VLOG(3) << "program config size: " << param_.program_config_size();
    for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
      timeline.Start();
      fleet_ptr_->PullSparseVarsSync(
          *thread_scope_, tid, sparse_key_names_[tid], &features_[tid],
          &feature_values_[tid], table.fea_dim(), sparse_value_names_[tid]);
      timeline.Pause();
      pull_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      CollectLabelInfo(i);
      timeline.Pause();
      collect_label_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      FillSparseValue(i);
      timeline.Pause();
      fill_sparse_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      timeline.Start();
      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight();
      }
      timeline.Pause();
      adjust_ins_weight_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }
    VLOG(3) << "Fill sparse value for all sparse table done.";

    int run_op_idx = 0;
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
        timeline.Start();
        VLOG(3) << "Going to run op " << op_name[run_op_idx];
        op->Run(*thread_scope_, place_);
        VLOG(3) << "Op " << op_name[run_op_idx] << " Finished";
        timeline.Pause();
        op_total_time[run_op_idx++] += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

    // check inf and nan
    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope_->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains Inf.", var_name));
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains NAN.", var_name));
    }

    if (need_to_push_sparse_) {
      for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
           ++i) {
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
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_, &sparse_push_keys_[tid], no_cvm_);
        timeline.Pause();
        push_sparse_time += timeline.ElapsedSec();
        total_time += timeline.ElapsedSec();
      }
    }

#ifdef PADDLE_WITH_PSLIB
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
          uint64_t tid = copy_sparse_tables_[i].first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
    }
#endif

    if (need_to_push_dense_) {
      timeline.Start();
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        fleet_ptr_->PushDenseVarsAsync(
            *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
            scale_datanorm_, cur_batch);
      }
      timeline.Pause();
      push_dense_time += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
      VLOG(3) << "push sparse and dense gradient done.";
      int32_t tmp_push_dense_wait_times = -1;
      static uint32_t push_dense_wait_times =
          static_cast<uint32_t>(tmp_push_dense_wait_times);
      if (push_dense_status_.size() >= push_dense_wait_times) {
        for (auto& t : push_dense_status_) {
          t.wait();
        }
        push_dense_status_.resize(0);
      }

      if (tmp_push_dense_wait_times == -1) {
        push_dense_status_.resize(0);
      }
    }

    if (need_to_push_sparse_) {
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

      VLOG(3) << "going to increase thread version";
      VLOG(3) << "push dense table id size: "
              << param_.program_config(0).push_dense_table_id_size();
    }

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
      }
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    total_inst += cur_batch;
    ++batch_cnt;

    if (thread_id_ == 0) {
      // should be configured here
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        double op_sum_time = 0;
        std::unordered_map<std::string, double> op_to_time;
        for (size_t i = 0; i < op_total_time.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
          if (op_to_time.find(op_name[i]) == op_to_time.end()) {
            op_to_time[op_name[i]] = 0.0;
          }
          op_to_time[op_name[i]] += op_total_time[i];
          op_sum_time += op_total_time[i];
        }
        for (auto& i : op_to_time) {
          fprintf(stderr, "op [%s] run total time: [%f]ms\n", i.first.c_str(),
                  i.second / batch_cnt);
        }
        fprintf(stderr, "op run total time: %fs\n", op_sum_time / batch_cnt);
        fprintf(stderr, "train total time: %fs\n", total_time / batch_cnt);
        fprintf(stderr, "pull sparse time: %fs\n",
                pull_sparse_time / batch_cnt);
        fprintf(stderr, "fill sparse time: %fs\n",
                fill_sparse_time / batch_cnt);
        fprintf(stderr, "push sparse time: %fs\n",
                push_sparse_time / batch_cnt);
        fprintf(stderr, "push dense time: %fs\n", push_dense_time / batch_cnt);
        fprintf(stderr, "collect label time: %fs\n",
                collect_label_time / batch_cnt);
        fprintf(stderr, "adjust ins weight time: %fs\n",
                adjust_ins_weight_time / batch_cnt);
        fprintf(stderr, "copy table time: %fs\n", copy_table_time / batch_cnt);
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "op run percent: %f\n", op_sum_time / total_time * 100);
        fprintf(stderr, "pull sparse time percent: %f\n",
                pull_sparse_time / total_time * 100);
        fprintf(stderr, "adjust ins weight time percent: %f\n",
                adjust_ins_weight_time / total_time * 100);
        fprintf(stderr, "copy table time percent: %f\n",
                copy_table_time / total_time * 100);
        fprintf(stderr, "collect label time percent: %f\n",
                collect_label_time / total_time * 100);
        fprintf(stderr, "fill sparse time percent: %f\n",
                fill_sparse_time / total_time * 100);
        fprintf(stderr, "push sparse time percent: %f\n",
                push_sparse_time / total_time * 100);
        fprintf(stderr, "push dense time percent: %f\n",
                push_dense_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
    timeline.Start();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

void DownpourWorker::TrainFiles() {
  VLOG(3) << "Begin to train files";
  platform::SetNumThreads(1);
  device_reader_->Start();
  int batch_cnt = 0;
  int cur_batch;
  while ((cur_batch = device_reader_->Next()) > 0) {
    if (copy_table_config_.need_copy()) {
      if (batch_cnt % copy_table_config_.batch_num() == 0) {
        CopySparseTable();
        CopyDenseTable();
        CopyDenseVars();
      }
    }
    // pull sparse here
    for (int i = 0; i < param_.program_config(0).pull_sparse_table_id_size();
         ++i) {
      uint64_t tid = static_cast<uint64_t>(
          param_.program_config(0).pull_sparse_table_id(i));
      TableParameter table;
      for (auto j : param_.sparse_table()) {
        if (j.table_id() == tid) {
          table = j;
          break;
        }
      }
      fleet_ptr_->PullSparseVarsSync(
          *thread_scope_, tid, sparse_key_names_[tid], &features_[tid],
          &feature_values_[tid], table.fea_dim(), sparse_value_names_[tid]);
      CollectLabelInfo(i);
      FillSparseValue(i);
      auto nid_iter = std::find(sparse_value_names_[tid].begin(),
                                sparse_value_names_[tid].end(),
                                adjust_ins_weight_config_.nid_slot());
      if (nid_iter != sparse_value_names_[tid].end()) {
        AdjustInsWeight();
      }
    }
    VLOG(3) << "fill sparse value for all sparse table done.";

    // do computation here
    for (auto& op : ops_) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (op->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      if (!need_skip) {
#ifdef PADDLE_WITH_PSLIB
        try {
          op->Run(*thread_scope_, place_);
        } catch (std::exception& e) {
          fprintf(stderr, "error message: %s\n", e.what());
          auto& ins_id_vec = device_reader_->GetInsIdVec();
          size_t batch_size = device_reader_->GetCurBatchSize();
          std::string s = "";
          for (auto& ins_id : ins_id_vec) {
            if (s != "") s += ",";
            s += ins_id;
          }
          fprintf(stderr, "batch_size: %zu, ins_ids_vec: %s\n", batch_size,
                  s.c_str());
          s = "";
          for (auto& param : all_param_) {
            Variable* var = thread_scope_->FindVar(param);
            if (var == nullptr) {
              continue;
            }
            Tensor* tensor = nullptr;
            int64_t len = 0;
            if (var->IsType<framework::LoDTensor>()) {
              tensor = var->GetMutable<LoDTensor>();
              len = tensor->numel();
            } else if (var->IsType<SelectedRows>()) {
              auto selected_rows = var->GetMutable<SelectedRows>();
              tensor = selected_rows->mutable_value();
              len = tensor->numel();
            }
            if (!tensor->IsInitialized()) {
              continue;
            }
            s += param + ":" + std::to_string(len) + ":";
            s += PrintLodTensor(tensor, 0, len);
            fprintf(stderr, "%s\n", s.c_str());
            fflush(stderr);
            s = "";
          }
          throw e;
        }
#else
        op->Run(*thread_scope_, place_);
#endif
      }
    }

    // check inf and nan
    for (std::string& var_name : check_nan_var_names_) {
      Variable* var = thread_scope_->FindVar(var_name);
      if (var == nullptr) {
        continue;
      }
      LoDTensor* tensor = var->GetMutable<LoDTensor>();
      if (tensor == nullptr) {
        continue;
      }
      PADDLE_ENFORCE_EQ(framework::TensorContainsInf(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains Inf.", var_name));
      PADDLE_ENFORCE_EQ(framework::TensorContainsNAN(*tensor), false,
                        platform::errors::InvalidArgument(
                            "Tensor %s contains NAN.", var_name));
    }

    if (need_to_push_sparse_) {
      // push gradients here
      for (int i = 0; i < param_.program_config(0).push_sparse_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_sparse_table_id(i));
        TableParameter table;
        for (auto i : param_.sparse_table()) {
          if (i.table_id() == tid) {
            table = i;
            break;
          }
        }
        fleet_ptr_->PushSparseVarsWithLabelAsync(
            *thread_scope_, tid, features_[tid], feature_labels_[tid],
            sparse_key_names_[tid], sparse_grad_names_[tid], table.emb_dim(),
            &feature_grads_[tid], &push_sparse_status_, cur_batch, use_cvm_,
            dump_slot_, &sparse_push_keys_[tid], no_cvm_);
      }
    }

#ifdef PADDLE_WITH_PSLIB
    if (copy_table_config_.need_copy()) {
      if (copy_table_config_.sparse_copy_by_feasign()) {
        for (size_t i = 0; i < copy_sparse_tables_.size(); ++i) {
          uint64_t tid = copy_sparse_tables_[i].first;
          feasign_set_[tid].insert(sparse_push_keys_[tid].begin(),
                                   sparse_push_keys_[tid].end());
        }
      }
    }
#endif

    if (need_to_push_dense_) {
      if (flag_partial_push_) {
        Variable* var = (*thread_scope_).FindVar("cond_tag");
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        // check type in python code
        int64_t* cond_value_batch = tensor->data<int64_t>();

        for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
             ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).push_dense_table_id(i));
          if (condvalue_set_.find(tid) != condvalue_set_.end()) {
            // common dense table must push dense
            if (cond2table_map_[cond_value_batch[0]] != tid) {
              // can't push dense
              continue;
            }
          }

          VLOG(3) << "push multitask dense gradient " << tid;
          fleet_ptr_->PushDenseVarsAsync(
              *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
              scale_datanorm_, cur_batch);
        }

      } else {
        for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
             ++i) {
          uint64_t tid = static_cast<uint64_t>(
              param_.program_config(0).push_dense_table_id(i));

          fleet_ptr_->PushDenseVarsAsync(
              *thread_scope_, tid, dense_grad_names_[tid], &push_sparse_status_,
              scale_datanorm_, cur_batch);
        }
      }

      VLOG(3) << "push dense gradient done.";

      // the following code should be more precise and clean
      // TODO(guru4elephant)
      int32_t tmp_push_dense_wait_times = -1;
      static uint32_t push_dense_wait_times =
          static_cast<uint32_t>(tmp_push_dense_wait_times);

      if (push_dense_status_.size() >= push_dense_wait_times) {
        for (auto& t : push_dense_status_) {
          t.wait();
        }
        push_dense_status_.resize(0);
      }

      if (tmp_push_dense_wait_times == -1) {
        push_dense_status_.resize(0);
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

    if (need_to_push_dense_) {
      for (int i = 0; i < param_.program_config(0).push_dense_table_id_size();
           ++i) {
        uint64_t tid = static_cast<uint64_t>(
            param_.program_config(0).push_dense_table_id(i));
        pull_dense_worker_->IncreaseThreadVersion(thread_id_, tid);
      }
    }
    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    PrintFetchVars();
    thread_scope_->DropKids();
    ++batch_cnt;
  }
  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }
  if (copy_table_config_.need_copy()) {
    CopySparseTable();
    CopyDenseTable();
    CopyDenseVars();
  }
}

}  // end namespace framework
}  // end namespace paddle
