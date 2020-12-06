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

#include "tools/rdtsc_profiler/rdtsc_profiler.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <tuple>

namespace rdtsc_profiler {

RdtscProfiler::RdtscProfiler() {}

RdtscProfiler& RdtscProfiler::instance() {
  static RdtscProfiler instance;
  return instance;
}

void RdtscProfiler::setSortedImpl(bool sorted) { m_sorted_results = sorted; }

void RdtscProfiler::beginImpl(const std::string& name) {
  uint64_t begin_time = __rdtsc();
  m_value_stack.emplace_back(name, begin_time);
}

void RdtscProfiler::endImpl(const uint64_t& end_value) {
  std::string context_name = getContextName();
  const auto& begin_value = m_value_stack.back();
  std::string name_with_context = context_name.empty()
                                      ? begin_value.first
                                      : context_name + "." + begin_value.first;
  addMeasurement(name_with_context, end_value - begin_value.second);

  m_value_stack.pop_back();
}

void RdtscProfiler::addMeasurement(std::string name, uint64_t time) {
  if (m_measurements.find(name) != m_measurements.end()) {
    m_measurements[name].emplace_back(time);
  } else {
    m_measurements[name] = {time};
  }
}

void RdtscProfiler::startContextImpl(const std::string& name) {
  m_context_stack.emplace_back(name);
  beginImpl("all");
}

void RdtscProfiler::endContextImpl(uint64_t value) {
  endImpl(value);
  m_context_stack.pop_back();
}

std::string RdtscProfiler::getContextName() {
  std::ostringstream res;
  std::string sep = "";
  for (const auto& ctx : m_context_stack) {
    res << sep << ctx;
    sep = ".";
  }
  return res.str();
}

void RdtscProfiler::printResults() {
  std::cout << "---------------------------------"
            << " Profiling Summary "
            << "---------------------------------" << std::endl;
  auto context_width = std::setw(getLongestNameLength() + 4);
  auto value_width = std::setw(12);

  std::cout << std::left << context_width << "Name" << value_width << "Calls"
            << value_width << "Avg" << value_width << "Min" << value_width
            << "Max" << value_width << "Ratio" << std::endl;

  auto overall_m = m_measurements["Overall"];
  uint64_t overall_total_time = getTotalTime(overall_m);

  // name, count, average, min, max, ratio
  using SummaryElType =
      std::tuple<std::string, uint64_t, double, uint64_t, uint64_t, double>;
  std::vector<SummaryElType> sorted_entries;

  for (auto const& m : m_measurements) {
    double average = getAvgTime(m.second);
    uint64_t total = getTotalTime(m.second);
    auto minmax = std::minmax_element(std::begin(m.second), std::end(m.second));
    sorted_entries.emplace_back(
        m.first,
        m.second.size(),
        average,
        *minmax.first,
        *minmax.second,
        static_cast<double>(total) / overall_total_time);
  }

  if (m_sorted_results) {
    std::sort(std::begin(sorted_entries),
              std::end(sorted_entries),
              [](const SummaryElType& e1, const SummaryElType& e2) {
                return std::get<2>(e1) < std::get<2>(e2);
              });
  }

  for (auto const& m : sorted_entries) {
    std::cout << std::left << context_width << std::get<0>(m)  // name
              << value_width << std::get<1>(m)                 // count
              << value_width << std::get<2>(m)                 // average
              << value_width << std::get<3>(m)                 // min
              << value_width << std::get<4>(m)                 // max
              << value_width << std::get<5>(m)                 // ratio
              << std::endl;
  }
  std::cout << "------------------------" << std::endl;
}

int RdtscProfiler::getLongestNameLength() {
  const auto& me_it =
      std::max_element(std::begin(m_measurements),
                       std::end(m_measurements),
                       [](const ValueType& v1, const ValueType& v2) {
                         return v1.first.size() < v2.first.size();
                       });
  return me_it->first.size();
}

double RdtscProfiler::getAvgTime(const std::vector<uint64_t>& values) {
  return getTotalTime(values) / static_cast<double>(values.size());
}

uint64_t RdtscProfiler::getTotalTime(const std::vector<uint64_t>& values) {
  return std::accumulate(std::begin(values), std::end(values), 0ul);
}

}  // namespace rdtsc_profiler
