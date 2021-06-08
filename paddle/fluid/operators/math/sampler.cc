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

#include "paddle/fluid/operators/math/sampler.h"

#include <glog/logging.h>

#include "paddle/fluid/framework/generator.h"

namespace paddle {
namespace operators {
namespace math {

Sampler::~Sampler() = default;

UniformSampler::UniformSampler(int64_t range, unsigned int seed)
    : Sampler(range, seed), inv_range_(1.0 / (range + 1)) {
  random_engine_ = framework::GetCPURandomEngine(seed_);
  dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);
}

int64_t UniformSampler::Sample() const { return (*dist_)(*random_engine_); }

float UniformSampler::Probability(int64_t value) const { return inv_range_; }

LogUniformSampler::LogUniformSampler(int64_t range, unsigned int seed)
    : Sampler(range, seed), log_range_(log(range + 1)) {
  random_engine_ = framework::GetCPURandomEngine(seed_);
  dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
}

int64_t LogUniformSampler::Sample() const {
  // Got Log Uniform distribution from uniform distribution by
  // inverse_transform_sampling method
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler/
  auto cur_random = (*dist_)(*random_engine_);
  const int64_t value = static_cast<int64_t>(exp(cur_random * log_range_)) - 1;
  // Mathematically, value should be <= range_, but might not be due to some
  // floating point roundoff, so we mod by range_.
  return value % range_;
}

float LogUniformSampler::Probability(int64_t value) const {
  // Given f(x) = 1/[(x+1) * log_range_]
  // The value's  probability  is integral of f(x) from value to (value + 1)
  // More details:
  // https://wanghaoshuang.github.io/2017/11/Log-uniform-distribution-sampler
  return (log((value + 2.0) / (value + 1.0))) / log_range_;
}

CustomSampler::CustomSampler(int64_t range, const float *probabilities,
                             const int *alias, const float *alias_probabilities,
                             unsigned int seed)
    : Sampler(range, seed) {
  random_engine_ = framework::GetCPURandomEngine(seed_);
  real_dist_ = std::make_shared<std::uniform_real_distribution<>>(0, 1);
  int_dist_ = std::make_shared<std::uniform_int_distribution<>>(0, range);

  alias_probs_ = alias_probabilities;
  probs_ = probabilities;
  alias_ = alias;
}

int64_t CustomSampler::Sample() const {
  auto index = (*int_dist_)(*random_engine_);
  auto p = (*real_dist_)(*random_engine_);
  if (p > alias_probs_[index]) {
    int alias = alias_[index];

    if (alias == exceptional_val) {
      LOG(WARNING) << "WARNING: CustomSampler get alias " << exceptional_val;
      return index;
    }

    return alias;
  } else {
    return index;
  }
}

float CustomSampler::Probability(int64_t value) const { return probs_[value]; }

}  // namespace math
}  // namespace operators
}  // namespace paddle
