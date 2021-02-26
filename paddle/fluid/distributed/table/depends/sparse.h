// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <math.h>  // for sqrt in CPU and CUDA
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

DECLARE_double(eager_delete_tensor_gb);

namespace paddle {
namespace distributed {

using LoDTensor = framework::LoDTensor;
class SparseOptimizer {
 public:
  explicit SparseOptimizer(
      const std::vector<std::string>& value_names,
      const std::vector<int>& value_dims, const std::vector<int>& value_offsets,
      const std::unordered_map<std::string, int>& value_idx)
      : value_names_(value_names),
        value_dims_(value_dims),
        value_offsets_(value_offsets),
        value_idx_(value_idx) {}

  virtual void update(const uint64_t* keys, const float* update_values,
                      size_t num, const std::vector<uint64_t>& offsets,
                      ValueBlock* block) = 0;

  virtual void set_global_lr(float* lr) { global_learning_rate_ = lr; }

  const std::vector<std::string>& value_names_;
  const std::vector<int>& value_dims_;
  const std::vector<int>& value_offsets_;
  const std::unordered_map<std::string, int>& value_idx_;
  int param_offset = 0;
  int update_numel = 0;

 protected:
  float* global_learning_rate_;
};

// sum calc for sparse tensor
class SSUM : public SparseOptimizer {
 public:
  explicit SSUM(const std::vector<std::string>& value_names,
                const std::vector<int>& value_dims,
                const std::vector<int>& value_offsets,
                const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* value = block->Get(id);
      float* param = value + param_offset;
      blas.VADD(update_numel, update_values + x * update_numel, param, param);
    }
  }
};

// sgd optimzer for sparse tensor
class SSGD : public SparseOptimizer {
 public:
  explicit SSGD(const std::vector<std::string>& value_names,
                const std::vector<int>& value_dims,
                const std::vector<int>& value_offsets,
                const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* value = block->Get(id);

      float learning_rate = *(global_learning_rate_) * (value + lr_offset)[0];
      VLOG(4) << "SSGD LearningRate: " << learning_rate;
      float* param = value + param_offset;

      std::vector<float> grads;
      grads.resize(update_numel);
      blas.VCOPY(update_numel, update_values + x * update_numel, grads.data());
      blas.SCAL(update_numel, learning_rate, grads.data());
      blas.VSUB(update_numel, param, grads.data(), param);
    }
  }

  int lr_offset;
};

// adam optimzer for sparse tensor
class SAdam : public SparseOptimizer {
 public:
  explicit SAdam(const std::vector<std::string>& value_names,
                 const std::vector<int>& value_dims,
                 const std::vector<int>& value_offsets,
                 const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment1");
    m1_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment2");
    m2_offset = value_offsets.at(idx);

    idx = value_idx.at("Beta1Pow");
    beta1_pow_offset = value_offsets.at(idx);

    idx = value_idx.at("Beta2Pow");
    beta2_pow_offset = value_offsets.at(idx);

    // add attr later
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1.0e-8;
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* values = block->Get(id);
      float lr_ = *(global_learning_rate_) * (values + lr_offset)[0];
      VLOG(4) << "SAdam LearningRate: " << lr_;
      float* param = values + param_offset;
      float* moment1 = values + m1_offset;
      float* moment2 = values + m2_offset;
      float* beta1_pow = values + beta1_pow_offset;
      float* beta2_pow = values + beta2_pow_offset;

      beta1_pow[0] = beta1_pow[0] * beta1;
      beta2_pow[0] = beta2_pow[0] * beta2;

      lr_ *= sqrt(1 - beta2_pow[0]) / (1 - beta1_pow[0]);

      std::vector<float> grad, grad2, tmp;
      grad.resize(update_numel);
      grad2.resize(update_numel);
      tmp.resize(update_numel);

      blas.VCOPY(update_numel, update_values + x * update_numel, grad.data());
      blas.VCOPY(update_numel, update_values + x * update_numel, grad2.data());

      blas.SCAL(update_numel, 1 - beta1, grad.data());
      blas.VSQUARE(update_numel, grad2.data(), grad2.data());
      blas.SCAL(update_numel, 1 - beta2, grad2.data());

      blas.SCAL(update_numel, beta1, moment1);
      blas.VADD(update_numel, moment1, grad.data(), moment1);
      blas.SCAL(update_numel, beta2, moment2);
      blas.VADD(update_numel, moment2, grad2.data(), moment2);

      float* tmp_ = tmp.data();
      float eps_ = epsilon * sqrt(1 - beta2_pow[0]);

      SQRT<float>(update_numel, moment2, tmp_);
      ADD<float>(update_numel, tmp_, eps_, tmp_);

      blas.VDIV(update_numel, moment1, tmp_, tmp_);
      blas.SCAL(update_numel, lr_, tmp_);
      blas.VSUB(update_numel, param, tmp_, param);
    }
  }

  int lr_offset;
  int m1_offset;
  int m2_offset;
  int beta1_pow_offset;
  int beta2_pow_offset;

  float beta1;
  float beta2;
  float epsilon;
};

class SGeneralOptimizer : public SparseOptimizer {
 public:
  explicit SGeneralOptimizer(
      const std::vector<framework::ProgramDesc>* sub_program,
      const TableParameter& program_config,
      const std::vector<std::string>& value_names,
      const std::vector<int>& value_dims, const std::vector<int>& value_offsets,
      const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    scope_ = new framework::Scope();
    place_ = platform::CPUPlace();
    executor_ = new framework::Executor(place_);
    common_ = program_config.common();

    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    auto& tensor_config = program_config.tensor();

    if (tensor_config.has_main_program_id()) {
      auto main_program_id_ = tensor_config.main_program_id();
      // Run main porgram, if program is used for learning decay
      auto main_program_desc = sub_program->at(main_program_id_);
      auto main_ctx = executor_->Prepare(main_program_desc, 0);
      exec_context_ = std::move(main_ctx);
    }
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    FLAGS_eager_delete_tensor_gb = -1;
    std::unique_ptr<framework::Scope> local_scope = scope_->NewTmpScope();

    auto update_num = offsets.size();
    if (update_num == 0) return;

    auto blas = GetBlas<float>();
    // Grad
    auto* grad_tensor = local_scope->Var("Grad")->GetMutable<LoDTensor>();
    grad_tensor->Resize(
        framework::make_ddim({static_cast<int64_t>(update_num),
                              static_cast<int64_t>(update_numel)}));
    auto* grad_data = grad_tensor->mutable_data<float>(place_);
    blas.VCOPY(update_num * update_numel, update_values, grad_data);

    int size = static_cast<int>(common_.params().size());
    for (int i = 0; i < size; i++) {
      auto& varname = value_names_[i];
      auto* var = local_scope->Var(varname)->GetMutable<LoDTensor>();
      if (varname == "LearningRate") {
        var->Resize(framework::make_ddim({1}));
        float* lr_data = var->mutable_data<float>(place_);
        auto* values = block->Get(keys[offsets[0]]);
        lr_data[0] = *(global_learning_rate_) * (values + value_offsets_[i])[0];
      } else {
        var->Resize(
            framework::make_ddim({static_cast<int64_t>(update_num),
                                  static_cast<int64_t>(value_dims_[i])}));
        auto* var_data = var->mutable_data<float>(place_);
        for (size_t x = 0; x < update_num; x++) {
          auto id = keys[offsets[x]];
          auto* values = block->Get(id);
          blas.VCOPY(value_dims_[i], values + value_offsets_[i],
                     var_data + x * value_dims_[i]);
        }
      }
    }

    executor_->RunPreparedContext(exec_context_.get(), local_scope.get(), false,
                                  false);

    for (int i = 0; i < size; i++) {
      auto& varname = value_names_[i];
      if (varname == "LearningRate") {
        continue;
      }
      const float* var_data =
          local_scope->FindVar(varname)->Get<LoDTensor>().data<float>();
      for (size_t x = 0; x < update_num; x++) {
        auto id = keys[offsets[x]];
        auto* values = block->Get(id);
        blas.VCOPY(value_dims_[i], var_data + x * value_dims_[i],
                   values + value_offsets_[i]);
      }
    }
  }

  framework::Executor* executor_;
  framework::Scope* scope_;
  platform::Place place_;
  std::shared_ptr<framework::ExecutorPrepareContext> exec_context_ = nullptr;
  CommonAccessorParameter common_;
};

}  // namespace distributed
}  // namespace paddle
