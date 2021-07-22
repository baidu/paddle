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

#include "paddle/fluid/framework/generator.h"

#include <glog/logging.h>
#include <memory>
#include <utility>

#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace framework {

const std::shared_ptr<Generator>& GetDefaultCUDAGenerator(int64_t device_id) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

  static int64_t num_cuda_devices = -1;
  static std::once_flag num_devices_init_flag;
  static std::deque<std::once_flag> cuda_device_flags;
  static std::vector<std::shared_ptr<Generator>> default_cuda_generators;

  std::call_once(num_devices_init_flag, []() {
    num_cuda_devices = paddle::platform::GetCUDADeviceCount();
    cuda_device_flags.resize(num_cuda_devices);
    default_cuda_generators.resize(num_cuda_devices);
  });
  if (device_id < 0) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "cuda device id shoule be greater than 0"));
  }

  std::call_once(cuda_device_flags[device_id], [device_id]() {
    default_cuda_generators[device_id] =
        std::make_shared<Generator>(GetRandomSeed(), device_id);
    VLOG(4) << "initial seed: "
            << default_cuda_generators[device_id]->GetCurrentSeed();
  });
  return default_cuda_generators[device_id];
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "getDefaultCUDAGenerator only support in CUDA place"));
#endif
}

const std::shared_ptr<Generator>& DefaultCPUGenerator() {
  static auto default_cpu_generator =
      std::make_shared<Generator>(GetRandomSeed());
  VLOG(4) << "initial seed: " << default_cpu_generator->GetCurrentSeed()
          << ", cpu engine: " << default_cpu_generator->GetCPUEngine().get();
  return default_cpu_generator;
}

std::shared_ptr<std::mt19937_64> OpDefaultCPUEngine() {
  static auto op_default_cpu_engine = std::make_shared<std::mt19937_64>();
  return op_default_cpu_engine;
}

// NOTE(zhiqiu): there are 3 conditions:
// (1) op seed is not set and DefaultCPUGenerator is inited, use
// DefaultCPUGenerator
// (2) op seed is not set and DefaultCPUGenerator is not inited, use se
// OpDefaultCPUEngine() and set a radnom seed
// (3) op seed is set, use OpDefaultCPUEngine() and set the seed
std::shared_ptr<std::mt19937_64> GetCPURandomEngine(uint64_t seed) {
  if (DefaultCPUGenerator()->GetIsInitPy() && seed == 0) {
    VLOG(4) << "Use random engine from generator";
    return DefaultCPUGenerator()->GetCPUEngine();
  } else {
    // NOTE(zhiqiu): creating an engine instance everytime instead of using
    // OpDefaultCPUEngine(), this is the legacy behavior of random operators.
    // The benefit is that when runing PE with fixed-seed in multiple thrads,
    // each thread has their own engine, and doesn't affect each other.
    //
    // And we need to measure the determinacy of Generator in PE.
    auto engine = std::make_shared<std::mt19937_64>();
    if (seed == 0) {
      seed = GetRandomSeed();
      VLOG(4) << "Use default random engine with random seed = " << seed;
    } else {
      VLOG(4) << "Use default random engine with fixed random seed = " << seed;
    }
    static std::mutex mu_;
    {
      std::lock_guard<std::mutex> lock(mu_);
      engine->seed(seed);
    }
    return engine;
  }
}

uint64_t GetSeed() { return DefaultCPUGenerator()->GetCurrentSeed(); }

GeneratorState Generator::GetState() {
  std::lock_guard<std::mutex> lock(this->mu_);
  state_.cpu_engine = *engine_;
  return this->state_;
}

void Generator::SetState(const GeneratorState& state) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->state_ = state;
  this->engine_ = std::make_shared<std::mt19937_64>(state.cpu_engine);
}

uint64_t Generator::GetCurrentSeed() {
  std::lock_guard<std::mutex> lock(this->mu_);
  return this->state_.current_seed;
}

uint64_t Generator::Seed() {
  std::lock_guard<std::mutex> lock(this->mu_);
  uint64_t seed;
  std::random_device de;
  seed = ((((uint64_t)de()) << 32) + de()) & 0x1FFFFFFFFFFFFF;
  this->state_.current_seed = seed;
  std::seed_seq seq({seed});
  this->engine_->seed(seq);

  return this->state_.current_seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->state_.current_seed = seed;
  this->state_.thread_offset = 0;
  std::seed_seq seq({seed});
  this->engine_->seed(seq);
}

std::shared_ptr<std::mt19937_64> Generator::GetCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mu_);
  return this->engine_;
}

void Generator::SetCPUEngine(std::shared_ptr<std::mt19937_64> engine) {
  std::lock_guard<std::mutex> lock(this->mu_);
  this->engine_ = engine;
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(this->mu_);
  auto engine = this->engine_;
  return (*engine)();
}

std::pair<uint64_t, uint64_t> Generator::IncrementOffset(
    uint64_t increament_offset) {
  uint64_t cur_offset = this->state_.thread_offset;
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::lock_guard<std::mutex> lock(this->mu_);

  this->state_.thread_offset += increament_offset;

#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Increment Offset only support in CUDA place"));
#endif
  return std::make_pair(this->state_.current_seed, cur_offset);
}

void Generator::SetIsInitPy(bool is_init_py) {
  this->is_init_py_ = is_init_py;
  VLOG(4) << "SetIsInitPy:" << this->is_init_py_;
}
bool Generator::GetIsInitPy() const { return this->is_init_py_; }

}  // namespace framework
}  // namespace paddle
