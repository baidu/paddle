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
#include "paddle/fluid/imperative/tracer.h"
#include <set>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(use_mkldnn);
DECLARE_string(tracer_mkldnn_ops_on);
DECLARE_string(tracer_mkldnn_ops_off);

namespace paddle {
namespace imperative {

static std::shared_ptr<Tracer> g_current_tracer(nullptr);

const std::shared_ptr<Tracer>& GetCurrentTracer() { return g_current_tracer; }

void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer) {
  g_current_tracer = tracer;
  VLOG(6) << "Set current tracer: " << g_current_tracer;
}

static void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& name_pair : outs) {
    for (const auto& vb : name_pair.second) {
      VLOG(6) << "Set output: " << vb->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      vb->InnerSetOverridedStopGradient(generate_grad);
    }
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward) {
  VLOG(1) << "Trace Op: " << type;
  if (FLAGS_use_mkldnn) {
    // if both lists are empty all ops are enabled (default for
    // FLAGS_use_mkldnn=1)
    // if ops_on list is not empty only ops from that list are enabled
    if (!FLAGS_tracer_mkldnn_ops_on.empty()) {
      auto is_on = FLAGS_tracer_mkldnn_ops_on.find(type) != std::string::npos;
      attrs["use_mkldnn"] = is_on;
    } else {
      // if ops_on list is empty all ops are enabled except types from off_list
      auto is_off = FLAGS_tracer_mkldnn_ops_off.find(type) != std::string::npos;
      attrs["use_mkldnn"] = !is_off;
    }
  }
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true);
  }

  NameVarBaseMap new_ins = ins;
  if (enable_autocast_) {
    VLOG(5) << "Auto mixed precision run operator: " << type;
    new_ins = AutoCastInputs(type, ins);
  }

  try {
    OpBase::Run(*op, new_ins, outs, attrs, place);
  } catch (platform::EnforceNotMet& exception) {
    framework::AppendErrorOpHint(type, &exception);
    throw std::move(exception);
  } catch (std::exception& ex) {
    PADDLE_THROW(platform::errors::Fatal(
        "Operator %s raises an %s exception.\n"
        "The exception content is\n:%s.",
        type, platform::demangle(typeid(ex).name()), ex.what()));
  } catch (...) {
    // NOTE: this branch represents a very serious bug with
    // low probability of occurrence, and we can't get its
    // exception content here.
    PADDLE_THROW(platform::errors::Fatal(
        "Operator %s raises an unknown exception.", type));
  }

  if (enable_program_desc_tracing_) {
    VLOG(5) << "Trace op " << type << " into ProgramDesc";
    program_desc_tracer_->InsertOp(type, new_ins, outs, attrs);
  }

  if (ComputeRequiredGrad(new_ins, outs, trace_backward)) {
    CreateGradOpNode(*op, new_ins, outs, attrs, place);
  } else {
    VLOG(3) << "No Grad to track for Op: " << type;
  }
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs,
                     framework::AttributeMap attrs) {
  TraceOp(type, ins, outs, std::move(attrs), expected_place_, has_grad_);
}

bool Tracer::ComputeRequiredGrad(const NameVarBaseMap& ins,
                                 const NameVarBaseMap& outs,
                                 bool trace_backward) {
  if (!trace_backward) return false;

  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverridedStopGradient());
        return true;
      }
    }
  }
  return false;
}

}  // namespace imperative
}  // namespace paddle
