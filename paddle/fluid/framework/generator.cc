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

#include <deque>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/platform/gpu_info.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {

std::shared_ptr<Generator> Generator::gen_instance_ = NULL;

static int64_t num_cuda_devices = -1;
static std::once_flag num_devices_init_flag;
static std::deque<std::once_flag> cuda_device_flags;

static std::vector<std::shared_ptr<Generator>> default_cuda_generators;

static void InitCUDAGenerators() {
#ifdef PADDLE_WITH_CUDA
  num_cuda_devices = platform::GetCUDADeviceCount();

  cuda_device_flags.resize(num_cuda_devices);
  default_cuda_generators.resize(num_cuda_devices);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "InitCUDAGenerators only support in CUDA place"));
#endif
}

static void initGlobalCUDAGeneratorState(int64_t device = -1) {
#ifdef PADDLE_WITH_CUDA
  GeneratorState default_gen_state_cuda;
  default_gen_state_cuda.thread_offset = 0;
  default_gen_state_cuda.device = device;
  default_gen_state_cuda.current_seed = 34342423252;
  default_cuda_generators[device] =
      std::make_shared<Generator>(default_gen_state_cuda);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "initGlobalCUDAGeneratorState only support in CUDA place"));
#endif
}

const std::shared_ptr<Generator>& getDefaultCUDAGenerator(int64_t device_id) {
#ifdef PADDLE_WITH_CUDA
  std::call_once(num_devices_init_flag, InitCUDAGenerators);
  platform::Place place;
  if (device_id == -1)
    device_id = BOOST_GET_CONST(platform::CUDAPlace, place).GetDeviceId();

  std::call_once(cuda_device_flags[device_id], initGlobalCUDAGeneratorState,
                 device_id);
  return default_cuda_generators[device_id];
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "getDefaultCUDAGenerator only support in CUDA place"));
#endif
}

GeneratorState* Generator::GetState() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_.get();
}

void Generator::SetState(GeneratorState* state_in) {
  std::lock_guard<std::mutex> lock(this->mutex);
  *this->state_ = *state_in;
}

uint64_t Generator::GetCurrentSeed() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->current_seed;
}

uint64_t Generator::Seed() {
  std::lock_guard<std::mutex> lock(this->mutex);
  uint64_t seed;
  std::random_device de;
  seed = ((((uint64_t)de()) << 32) + de()) & 0x1FFFFFFFFFFFFF;
  this->state_->current_seed = seed;
  std::seed_seq seq({seed});
  this->state_->cpu_engine.seed(seq);

  return this->state_->current_seed;
}

void Generator::SetCurrentSeed(uint64_t seed) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->current_seed = uint64_t(seed);
  std::seed_seq seq({seed});
  this->state_->cpu_engine.seed(seq);
}

std::mt19937_64& Generator::GetCPUEngine() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->cpu_engine;
}

void Generator::SetCPUEngine(std::mt19937_64 engine) {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->state_->cpu_engine = std::mt19937_64(engine);
}

uint64_t Generator::Random64() {
  std::lock_guard<std::mutex> lock(this->mutex);
  return this->state_->cpu_engine();
}

std::pair<uint64_t, uint64_t> Generator::IncrementOffset(
    uint64_t total_numel, uint64_t grid_size, uint64_t block_size,
    uint64_t engine_calls_num) {
#ifdef PADDLE_WITH_CUDA
  std::lock_guard<std::mutex> lock(this->mutex);
  uint64_t numel_per_thread =
      (total_numel - 1) / (block_size * grid_size * 4) + 1;
  uint64_t increment = numel_per_thread * engine_calls_num;
  uint64_t cur_offset = this->state_->thread_offset;
  this->state_->thread_offset += increment;
  return std::make_pair(this->state_->current_seed, cur_offset);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Increment Offset only support in CUDA place"));
#endif
}

std::pair<uint64_t, uint64_t> Generator::IncrementOffset(
    uint64_t increament_offset) {
#ifdef PADDLE_WITH_CUDA
  std::lock_guard<std::mutex> lock(this->mutex);
  uint64_t cur_offset = this->state_->thread_offset;
  this->state_->thread_offset += increament_offset;
  return std::make_pair(this->state_->current_seed, cur_offset);
#else
  PADDLE_THROW(platform::errors::PermissionDenied(
      "Increment Offset only support in CUDA place"));
#endif
}

void Generator::SetIsInitPyCUDA(bool is_init_py_cuda) {
  this->is_init_py_cuda_ = is_init_py_cuda;
  VLOG(4) << "SetIsInitPy:" << this->is_init_py_cuda_;
}
bool Generator::GetIsInitPyCUDA() const { return this->is_init_py_cuda_; }

}  // namespace framework
}  // namespace paddle
