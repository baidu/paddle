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

#include <glog/logging.h>
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

static uint64_t GetRandomSeed() {
  std::random_device rd;
  // double has 53 bit significant, so limit uint64 to 53 bits
  return ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
}

struct GeneratorState {
  int64_t device = -1;
  uint64_t current_seed = 111111111;
  uint64_t thread_offset = 0;
  std::mt19937_64 cpu_engine;
  std::mt19937 cpu_engine_32;
};

struct Generator {
  Generator() {
    auto seed = GetRandomSeed();
    std::seed_seq seq({seed});
    auto engine = std::make_shared<std::mt19937_64>(seq);
    auto engine_32 = std::make_shared<std::mt19937>(seq);
    this->state_.cpu_engine = *engine;
    this->state_.cpu_engine_32 = *engine_32;
    this->state_.device = -1;
    this->state_.current_seed = seed;
    this->state_.thread_offset = 0;
    this->engine_ = engine;
    this->engine_32 = engine_32;
    VLOG(4) << "initial seed: " << this->state_.current_seed
            << ", cpu engine: " << &this->state_.cpu_engine
            << ", cpu engine 32: " << &this->state_.cpu_engine_32;
  }

  explicit Generator(uint64_t seed) {
    std::seed_seq seq({seed});
    auto engine = std::make_shared<std::mt19937_64>(seq);
    auto engine_32 = std::make_shared<std::mt19937>(seq);
    this->state_.cpu_engine = *engine;
    this->state_.cpu_engine_32 = *engine_32;
    this->state_.device = -1;
    this->state_.current_seed = seed;
    this->state_.thread_offset = 0;
    this->engine_ = engine;
    this->engine_32 = engine_32;
    VLOG(4) << "initial seed " << this->state_.current_seed
            << ", cpu engine: " << &this->state_.cpu_engine
            << ", cpu engine 32: " << &this->engine_32;
    this->is_init_py_ = true;  // TODO(zhiqiu): remove it in future
  }

  Generator(uint64_t seed, uint64_t device_id) {
    std::seed_seq seq({seed});
    auto engine = std::make_shared<std::mt19937_64>(seq);
    auto engine_32 = std::make_shared<std::mt19937>(seq);
    this->state_.cpu_engine = *engine;
    this->state_.cpu_engine_32 = *engine_32;
    this->state_.device = device_id;
    this->state_.current_seed = seed;
    this->state_.thread_offset = 0;
    this->engine_ = engine;
    this->engine_32 = engine_32;
    VLOG(4) << "initial seed: " << this->state_.current_seed
            << ", cpu engine: " << &this->state_.cpu_engine
            << ", cpu engine 32: " << &this->state_.cpu_engine_32;
    this->is_init_py_ = false;  // TODO(zhiqiu): remove it in future
  }

  Generator(const Generator& other) = delete;

  // get random state
  GeneratorState GetState();

  // set random state
  void SetState(const GeneratorState&);
  // get current seed
  uint64_t GetCurrentSeed();
  // random a seed and get
  uint64_t Seed();
  // set seed
  void SetCurrentSeed(uint64_t seed);
  // get cpu engine
  std::shared_ptr<std::mt19937_64> GetCPUEngine();
  std::shared_ptr<std::mt19937> GetCPUEngine_32();

  // set cpu engine
  void SetCPUEngine(std::shared_ptr<std::mt19937_64>);

  void SetCPUEngine_32(std::shared_ptr<std::mt19937>);

  uint64_t Random64();

  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t increament_offset);

  void SetIsInitPy(bool);
  bool GetIsInitPy() const;
  uint64_t get_device_id() { return this->state_.device; }

 private:
  GeneratorState state_;
  std::shared_ptr<std::mt19937_64> engine_;
  std::shared_ptr<std::mt19937> engine_32;
  mutable std::mutex mu_;

  // NOTE(zhiqiu): is_init_py_ is used to make generator be compatible with
  // old seed, and it should be removed after all random-related operators
  // and unittests upgrades to use generator.
  bool is_init_py_ = false;
};

// The DefaultCPUGenerator is used in manual_seed()
const std::shared_ptr<Generator>& DefaultCPUGenerator();

const std::shared_ptr<Generator>& DefaultCPUGenerator_32();

// If op seed is set or global is not set, the OpDefaultCPUEngine is used.
std::shared_ptr<std::mt19937_64> OpDefaultCPUEngine();

// std::shared_ptr<std::mt19937> OpDefaultCPUEngine_32();

std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t);

std::shared_ptr<std::mt19937> GetCPURandomEngine_32(uint64_t);

const std::shared_ptr<Generator>& GetDefaultCUDAGenerator(
    int64_t device_id = -1);
}  // namespace framework
}  // namespace paddle
