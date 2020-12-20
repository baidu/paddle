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

#include "paddle/fluid/framework/trainer_factory.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace framework {

class TrainerBase;

typedef std::shared_ptr<TrainerBase> (*CreatetrainerFunction)();
typedef std::unordered_map<std::string, CreatetrainerFunction> trainerMap;
trainerMap g_trainer_map;

#define REGISTER_TRAINER_CLASS(trainer_class)                   \
  namespace {                                                   \
  std::shared_ptr<TrainerBase> Creator_##trainer_class() {      \
    return std::shared_ptr<TrainerBase>(new trainer_class);     \
  }                                                             \
  class __Registerer_##trainer_class {                          \
   public:                                                      \
    __Registerer_##trainer_class() {                            \
      g_trainer_map[#trainer_class] = &Creator_##trainer_class; \
    }                                                           \
  };                                                            \
  __Registerer_##trainer_class g_registerer_##trainer_class;    \
  }  // namespace

std::string TrainerFactory::TrainerTypeList() {
  std::string trainer_types;
  for (auto iter = g_trainer_map.begin(); iter != g_trainer_map.end(); ++iter) {
    if (iter != g_trainer_map.begin()) {
      trainer_types += ", ";
    }
    trainer_types += iter->first;
  }
  return trainer_types;
}

std::shared_ptr<TrainerBase> TrainerFactory::CreateTrainer(
    std::string trainer_class) {
  if (g_trainer_map.count(trainer_class) < 1) {
    LOG(WARNING) << "Trainer class: " << trainer_class << " not defined";
    LOG(WARNING) << TrainerTypeList();
    exit(-1);
  }
  return g_trainer_map[trainer_class]();
}

REGISTER_TRAINER_CLASS(MultiTrainer);
REGISTER_TRAINER_CLASS(DistMultiTrainer);
#if (defined PADDLE_WITH_CUDA || defined PADDLE_WITH_XPU) && \
    (defined PADDLE_WITH_PSLIB)
REGISTER_TRAINER_CLASS(HeterXpuTrainer);
REGISTER_TRAINER_CLASS(HeterBoxTrainer);
#endif
#ifdef PADDLE_WITH_PSLIB
REGISTER_TRAINER_CLASS(PSGPUTrainer);
#endif
#if defined(PADDLE_WITH_NCCL)
REGISTER_TRAINER_CLASS(PipelineTrainer);
#endif
}  // namespace framework
}  // namespace paddle
