/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace framework {

// thread-safe.
template <typename TAlgorithm>
class AlgorithmsCache {
 public:
  AlgorithmsCache() : search_times_(0) { hash_.clear(); }

  // Caches the best algorithm for a given combination of tensor dimensions
  //  and compute data type. cudnn_dtype set for different data type.
  TAlgorithm GetAlgorithm(const std::vector<int64_t>& dims1,
                          const std::vector<int64_t>& dims2,
                          const std::vector<int>& strides,
                          const std::vector<int>& paddings,
                          const std::vector<int>& dilations,
                          int algorithm_flags, int64_t cudnn_dtype,
                          std::function<TAlgorithm()> gen_func);

  TAlgorithm GetAlgorithm(int64_t area, int search_times, int algorithm_flags,
                          std::function<TAlgorithm()> gen_func);

 private:
  int64_t Seed(const std::vector<int64_t>& dims1,
               const std::vector<int64_t>& dims2,
               const std::vector<int>& strides,
               const std::vector<int>& paddings,
               const std::vector<int>& dilations, int algorithm_flags,
               int64_t cudnn_dtype);

  std::unordered_map<int64_t, TAlgorithm> hash_;
  int capacity_{65536};
  int search_times_;
  std::mutex cache_mutex;
};

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::GetAlgorithm(
    const std::vector<int64_t>& dims1, const std::vector<int64_t>& dims2,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    const std::vector<int>& dilations, int algorithm_flags, int64_t cudnn_dtype,
    std::function<TAlgorithm()> gen_func) {
  int64_t seed = Seed(dims1, dims2, strides, paddings, dilations,
                      algorithm_flags, cudnn_dtype);
  if (seed == 0) return gen_func();

  TAlgorithm ret;
  auto it = hash_.end();
  bool have_found = false;
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    it = hash_.find(seed);

    if (it != hash_.end()) {
      ret = it->second;
      have_found = true;
    }
  }

  if (!have_found) {
    ret = gen_func();
    std::lock_guard<std::mutex> lock(cache_mutex);
    if (hash_.size() >= capacity_) {
      hash_.erase(hash_.begin());
    }
    hash_[seed] = ret;
  }

  return ret;
}

template <typename TAlgorithm>
TAlgorithm AlgorithmsCache<TAlgorithm>::GetAlgorithm(
    int64_t area, int search_times, int algorithm_flags,
    std::function<TAlgorithm()> gen_func) {
  auto it = hash_.end();
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    it = hash_.find(area);

    if (it != hash_.end()) {
      return it->second;
    }
  }

  bool gene_flag = false;

  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    gene_flag = (search_times_ < search_times);
  }

  TAlgorithm algo{};
  if (gene_flag) {
    algo = gen_func();
    std::lock_guard<std::mutex> lock(cache_mutex);
    hash_[area] = algo;
    ++search_times_;
    return algo;
  }

  int64_t min = static_cast<uint64_t>(INT_MAX);
  {
    std::lock_guard<std::mutex> lock(cache_mutex);
    for (const auto& m : hash_) {
      if (m.first < min) {
        min = m.first;
        algo = m.second;
      }
    }
  }
  return algo;
}

template <typename TAlgorithm>
int64_t AlgorithmsCache<TAlgorithm>::Seed(const std::vector<int64_t>& dims1,
                                          const std::vector<int64_t>& dims2,
                                          const std::vector<int>& strides,
                                          const std::vector<int>& paddings,
                                          const std::vector<int>& dilations,
                                          int algorithm_flags,
                                          int64_t cudnn_dtype) {
  int64_t seed = 0;
  // Hash all of the inputs, use to try and look up a previously
  // discovered algorithm, or fall back to generating a new one.
  std::hash<int64_t> hashFn;
  // do hash like boost
  // https://stackoverflow.com/questions/2590677/how-do-i-combine-hash-values-in-c0x
  for (const auto num : dims1) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }

  for (const auto num : dims2) {
    seed ^= hashFn(num) + 0x9e3779b9 + (seed << 6) + (seed >> 2) + 1;
  }

  for (const auto num : strides) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 2;
  }

  for (const auto num : paddings) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 3;
  }

  for (const auto num : dilations) {
    seed ^= hashFn(static_cast<int64_t>(num)) + 0x9e3779b9 + (seed << 6) +
            (seed >> 2) + 4;
  }

  seed ^= hashFn(static_cast<int64_t>(algorithm_flags)) + 0x9e3779b9 +
          (seed << 6) + (seed >> 2) + 5;

  seed ^= hashFn(static_cast<int64_t>(cudnn_dtype)) + 0x9e3779b9 + (seed << 6) +
          (seed >> 2) + 6;

  VLOG(10) << "seed:" << seed << ", hash_.size:" << hash_.size();
  return seed;
}

}  // namespace framework
}  // namespace paddle
