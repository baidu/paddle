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

#include <x86intrin.h>
#include <cinttypes>
#include <map>
#include <string>
#include <utility>
#include <vector>

/// Convenient macros to use RdtscProfiler class.
/// The macros let user to nest measured code blocks in so called "contexts".

/// Whether or not to sort (by average time) in ascending order the captured
/// code blocks.
#define SORTED_RESULTS(value) rdtsc_profiler::RdtscProfiler::setSorted(value)
/// Mark the beginning of code block with potentially nested blocks inside.
/// The overall execution time of this code block shows as "name.all" in
/// profiler summary.
#define START_CONTEXT(name) rdtsc_profiler::RdtscProfiler::startContext(name)
/// Mark the end of measured code block.
#define END_CONTEXT() rdtsc_profiler::RdtscProfiler::endContext()
/// Mark the beginning of named, measured code block. If it is under "context"
/// then it's name will show in final profiling summary with full context name.
/// (ie. part1.nested_block.foo)
#define BEGIN(name) rdtsc_profiler::RdtscProfiler::begin(name)
/// Mark the end of measured code block.
#define END() rdtsc_profiler::RdtscProfiler::end()
/// Obligatory macro, which mark the beginning of "Overall" measured code block.
/// The results in profiler summary use this code block duration to compute
/// their relative execution time.
#define BEGIN_OVERALL() rdtsc_profiler::RdtscProfiler::begin("Overall")
/// Mark the end of "Overall" measured code block.
#define END_OVERALL() rdtsc_profiler::RdtscProfiler::end()

namespace rdtsc_profiler {

///
/// @brief      This class provides convenient API to profile code using
///             __rdtsc() intrinsic.
///
/// @note     The values shown in final profiler summary are elapsed CPU
///           cycles (of TSC clock) count.
///
class RdtscProfiler {
 public:
  static inline void begin(const std::string& name) {
    instance().beginImpl(name);
  }

  static inline void end() {
    uint64_t end_time = __rdtsc();
    instance().endImpl(end_time);
  }

  static inline void startContext(const std::string& name) {
    instance().startContextImpl(name);
  }

  static inline void endContext() {
    uint64_t end_time = __rdtsc();
    instance().endContextImpl(end_time);
  }

  static void setSorted(bool sorted) { instance().setSortedImpl(sorted); }

  ~RdtscProfiler() { instance().printResults(); }

 private:
  using Measurement = std::pair<std::string, uint64_t>;
  using MeasurementMap = std::map<std::string, std::vector<uint64_t>>;
  using ValueType = MeasurementMap::value_type;

  RdtscProfiler();
  static RdtscProfiler& instance();
  void setSortedImpl(bool sorted);
  void beginImpl(const std::string& name);
  void endImpl(const uint64_t& end_value);
  void addMeasurement(std::string name, uint64_t time);
  void startContextImpl(const std::string& name);
  void endContextImpl(uint64_t value);
  std::string getContextName();
  void printResults();
  int getLongestNameLength();
  double getAvgTime(const std::vector<uint64_t>& values);
  uint64_t getTotalTime(const std::vector<uint64_t>& values);

  bool m_sorted_results{false};
  std::vector<Measurement> m_value_stack;
  std::vector<std::string> m_context_stack;
  MeasurementMap m_measurements;
};

}  // namespace rdtsc_profiler
