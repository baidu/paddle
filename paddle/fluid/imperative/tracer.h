// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <atomic>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ThreadPool.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/jit/program_desc_tracer.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/macros.h"

namespace paddle {
namespace imperative {

class UniqueNameGenerator {
 public:
  explicit UniqueNameGenerator(std::string prefix = "") : prefix_(prefix) {}
  std::string Generate(std::string key = "dygraph_tmp") {
    return prefix_ + key + "_" + std::to_string(id_++);
  }

 private:
  std::atomic<int> id_{0};
  std::string prefix_;
};

class Tracer {
  DISABLE_COPY_AND_ASSIGN(Tracer);

 public:
  Tracer()
      : basic_engine_(new BasicEngine()),
        program_desc_tracer_(new jit::ProgramDescTracer()),
        generator_(new UniqueNameGenerator()) {
    expected_place_ = platform::CPUPlace();
  }

  ~Tracer() = default;

  void TraceOp(const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs,
               const platform::Place& place, bool trace_bacward);

  void TraceOp(const std::string& type, const NameVarBaseMap& ins,
               const NameVarBaseMap& outs, framework::AttributeMap attrs);

  bool ComputeRequiredGrad(const NameVarBaseMap& ins,
                           const NameVarBaseMap& outs, bool trace_backward);

  void SetEnableProgramDescTracing(bool enabled) {
    enable_program_desc_tracing_ = enabled;
  }

  bool IsProgramDescTracingEnabled() const {
    return enable_program_desc_tracing_;
  }

  jit::ProgramDescTracer* GetProgramDescTracer() {
    return program_desc_tracer_.get();
  }

  // Note(Aurelius84): The `tmp` is used as prefix key while naming a temporary
  // intermediate var both in imperative and static mode. But the
  // `UniqueNameGenerator` in C++ and `unique_name.py` in Python doesn't share
  // the same auto-increment id. It will create a variable repeatedly with same
  // name like `tmp_0` in some cases when transform dygraph into static layers.
  // So we modify the default prefix key into `eager_tmp` to distinguish with
  // static graph.
  std::string GenerateUniqueName(std::string key = "dygraph_tmp") {
    return generator_->Generate(key);
  }

  BasicEngine* GetEngine() const { return basic_engine_.get(); }

  platform::Place ExpectedPlace() const { return expected_place_; }

  void SetExpectedPlace(platform::Place place) { expected_place_ = place; }

  bool HasGrad() const { return has_grad_; }

  void SetHasGrad(bool has_grad) { has_grad_ = has_grad; }

  void SetEnableAutoCast(bool enabled) { enable_autocast_ = enabled; }

  bool IsAutoCastEnabled() const { return enable_autocast_; }

 private:
  std::unique_ptr<BasicEngine> basic_engine_;
  std::unique_ptr<jit::ProgramDescTracer> program_desc_tracer_;
  bool enable_program_desc_tracing_{false};
  std::unique_ptr<UniqueNameGenerator> generator_;
  platform::Place expected_place_;
  bool has_grad_{true};
  bool enable_autocast_{false};
};

// To access static variable current_tracer
const std::shared_ptr<Tracer>& GetCurrentTracer();
void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer_);

}  // namespace imperative
}  // namespace paddle
