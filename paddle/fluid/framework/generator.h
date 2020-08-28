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

#pragma once

#include <stdint.h>
#include <atomic>
#include <deque>
#include <iostream>  // temp for debug
#include <memory>
#include <mutex>  // NOLINT
#include <random>
#include <typeinfo>
#include <utility>

namespace paddle {
namespace framework {

struct Generator;

const std::shared_ptr<Generator>& getDefaultCUDAGenerator(
    int64_t device_id = -1);

struct GeneratorState {
  int64_t device = -1;
  uint64_t current_seed = 34342423252;
  uint64_t thread_offset;
  std::mt19937_64 cpu_engine;
};

struct Generator {
  Generator() {
    GeneratorState default_gen_state_cpu;
    default_gen_state_cpu.device = -1;
    default_gen_state_cpu.current_seed = 34342423252;
    std::seed_seq seq({34342423252});
    default_gen_state_cpu.cpu_engine = std::mt19937_64(seq);
    this->state_ = std::make_shared<GeneratorState>(default_gen_state_cpu);
  }
  explicit Generator(GeneratorState state_in)
      : state_{std::make_shared<GeneratorState>(state_in)} {}
  Generator(const Generator& other)
      : Generator(other, std::lock_guard<std::mutex>(other.mutex)) {}

  // get random state
  GeneratorState* GetState();
  // set random state
  void SetState(GeneratorState* state_in);
  // get current seed
  uint64_t GetCurrentSeed();
  // random a seed and get
  uint64_t Seed();

  // set seed
  void SetCurrentSeed(uint64_t seed);
  // get cpu engine
  std::mt19937_64& GetCPUEngine();
  // set cpu engine
  void SetCPUEngine(std::mt19937_64 engine);

  uint64_t Random64();

  void SetIsInitPyCUDA(bool);

  bool GetIsInitPyCUDA() const;

  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t total_numel,
                                                uint64_t grid_size,
                                                uint64_t block_size,
                                                uint64_t engine_calls_num);
  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t increament_offset);

  bool is_init_py = false;

  // CPU Generator singleton
  static std::shared_ptr<Generator> GetInstance() {
    if (NULL == gen_instance_) {
      gen_instance_.reset(new paddle::framework::Generator());
    }
    return gen_instance_;
  }

  static std::shared_ptr<Generator> GetInstanceX() {
    if (NULL == gen_instance_) {
      gen_instance_.reset(new paddle::framework::Generator());
    }
    gen_instance_->is_init_py = true;
    return gen_instance_;
  }

 private:
  static std::shared_ptr<Generator> gen_instance_;
  std::shared_ptr<GeneratorState> state_;
  mutable std::mutex mutex;
  bool is_init_py_cuda_ = false;

  Generator(const Generator& other, const std::lock_guard<std::mutex>&)
      : state_(std::make_shared<GeneratorState>(*(other.state_))) {}
};

}  // namespace framework
}  // namespace paddle
