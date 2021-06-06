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
#include <map>
#include <set>
#include <unordered_set>
#include <utility>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/imperative/amp_auto_cast.h"
#include "paddle/fluid/imperative/op_base.h"
#include "paddle/fluid/platform/denormal.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/string/string_helper.h"

DECLARE_bool(use_mkldnn);
DECLARE_string(tracer_mkldnn_ops_on);
DECLARE_string(tracer_mkldnn_ops_off);

uint64_t time_create_op = 0;
uint64_t time_checker = 0;
uint64_t time_amp = 0;
uint64_t time_before_kernel = 0;
uint64_t time_after_kernel = 0;
uint64_t time_kernel = 0;
uint64_t time_create_gradop = 0;
uint64_t python_time = 0;
uint64_t start_time = 0;
uint64_t start_time_python = 0;
uint64_t opbase_run_1 = 0;
// uint64_t opbase_run_2 = 0;
// uint64_t opbase_run_3 = 0;
extern uint64_t opbase_run_2;
extern uint64_t opbase_run_3;
uint64_t opbase_run_4 = 0;

namespace paddle {
namespace imperative {

static std::shared_ptr<Tracer> g_current_tracer(nullptr);

const std::shared_ptr<Tracer>& GetCurrentTracer() { return g_current_tracer; }

void SetCurrentTracer(const std::shared_ptr<Tracer>& tracer) {
  g_current_tracer = tracer;
  VLOG(6) << "Set current tracer: " << g_current_tracer;
}

void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& pair : outs) {
    for (const auto& var : pair.second) {
      // NOTE(zhiqiu): this happends when None output are passed from python
      // side. For example, fake_quantize_dequantize_moving_average_abs_max may
      // pass None OutAccum in eval mode.
      // It can be refined by generate several different pybind interface for
      // one operator with different function signature.
      if (var == nullptr) {
        VLOG(4) << pair.first << " is NULL";
        continue;
      }
      VLOG(6) << "Set output: " << var->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      var->InnerSetOverridedStopGradient(generate_grad);
    }
  }
}

void IncreaseVarbaseReferenceCountUntilCopyComplete(
    const std::shared_ptr<imperative::VarBase>& var,
    const platform::Place& place) {
  // Note(zhiqiu): Follow the logic of TensorCopy to determine the place that we
  // need to add callback, see tensor_utils.cc:245
  auto place_ = platform::is_gpu_place(place) ? place : var->Place();

  auto tracer = imperative::GetCurrentTracer();
  auto gc = tracer->MutableGarbageCollectorIfNotExists(place_);

  // Note(zhiqiu): This is an empty callback, the only way is to "reference"
  // var, so it will not be destructed until the kernels launched at current
  // stream of given place is finished.
  auto callback = [var, place_]() {
    VLOG(4) << "Run callback of var:" << var->Name() << " at place " << place_;
  };

  gc->DirectClearCallback(callback);
}

paddle::framework::GarbageCollector* Tracer::MutableGarbageCollectorIfNotExists(
    const platform::Place& place) {
  // if not exists, create a new GarbageCollector at given place
  if (gcs_.count(place) == 0) {
    std::unique_ptr<framework::GarbageCollector> gc;
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc.reset(new framework::DefaultStreamGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPlace, place), 0));

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use CUDA device since it's not compiled with CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (platform::is_cuda_pinned_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      gc.reset(new framework::CUDAPinnedGarbageCollector(
          BOOST_GET_CONST(platform::CUDAPinnedPlace, place), 0));

      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use CUDAPinned device since it's not compiled with "
          "CUDA,"
          "Please recompile or reinstall Paddle with GPU support."));
#endif
    } else if (platform::is_xpu_place(place)) {
#if defined(PADDLE_WITH_XPU)
      gc.reset(new framework::XPUGarbageCollector(
          BOOST_GET_CONST(platform::XPUPlace, place), 0));
      VLOG(10) << "Created GarbageCollector at " << place;
#else
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Paddle can't use XPU device since it's not compiled with XPU,"
          "Please recompile or reinstall Paddle with XPU support."));
#endif
    } else if (platform::is_cpu_place(place)) {
      gc.reset(new framework::CPUGarbageCollector(
          BOOST_GET_CONST(platform::CPUPlace, place), 0));
      VLOG(10) << "Created GarbageCollector at " << place;
    } else {
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "Unsupported place for garbage collection"));
    }
    gcs_.emplace(place, std::move(gc));
  }

  return gcs_.at(place).get();
}

inline uint64_t GetPosixInUsec() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return (static_cast<uint64_t>(tv.tv_sec) * 1000000 + tv.tv_usec);
}

class tmpclass {
 public:
  tmpclass() {}
  ~tmpclass() {
    // std::cout << "before1: " << before_TraceOp1 << "before2: " <<
    // before_TraceOp2 << "before3: " << before_TraceOp3 << "before4: " <<
    // before_TraceOp4 << ", ing: " << ing_TraceOp << ", python: " <<
    // python_time << std::endl;
    std::cout << "time_create_op " << time_create_op << std::endl;
    std::cout << "time_checker " << time_checker << std::endl;
    std::cout << "time_amp " << time_amp << std::endl;
    std::cout << "time_before_kernel " << time_before_kernel << std::endl;
    std::cout << "time_after_kernel " << time_after_kernel << std::endl;
    std::cout << "time_kernel " << time_kernel << std::endl;
    std::cout << "time_create_gradop " << time_create_gradop << std::endl;
    std::cout << "python_time " << python_time << std::endl;
  }
  int a;
};
tmpclass a;
int run_times = 0;

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const platform::Place& place, bool trace_backward,
                     const std::map<std::string, std::string>& inplace_map) {
  if (run_times < 100) {
    time_create_op = 0;
    time_checker = 0;
    time_amp = 0;
    time_before_kernel = 0;
    time_after_kernel = 0;
    time_kernel = 0;
    time_create_gradop = 0;
    python_time = 0;
    start_time = 0;
    start_time_python = 0;
    run_times++;
  }
  if (start_time_python != 0) {
    python_time = GetPosixInUsec() - start_time_python + python_time;
  }

  platform::RecordEvent op_type_record_event(type);
  platform::ScopedFlushDenormal flush;
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

  start_time = GetPosixInUsec();
  auto op = framework::OpRegistry::CreateOp(type, {}, {}, {}, false);
  time_create_op = GetPosixInUsec() - start_time + time_create_op;
  start_time = GetPosixInUsec();
  const auto& op_info = op->Info();
  auto* attr_checker = op_info.Checker();
  if (attr_checker) {
    attr_checker->Check(&attrs, true);
  }
  time_checker = GetPosixInUsec() - start_time + time_checker;

  start_time = GetPosixInUsec();
  NameVarBaseMap new_ins = ins;
  if (enable_autocast_) {
    VLOG(5) << "Auto mixed precision run operator: " << type;
    new_ins = AutoCastInputs(type, ins);
  }
  time_amp = GetPosixInUsec() - start_time + time_amp;

  try {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      platform::SetDeviceId(BOOST_GET_CONST(platform::CUDAPlace, place).device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with GPU if use CUDAPlace."));
#endif
    } else if (platform::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
      platform::SetXPUDeviceId(
          BOOST_GET_CONST(platform::XPUPlace, place).device);
#else
      PADDLE_THROW(platform::errors::PreconditionNotMet(
          "PaddlePaddle should compile with XPU if use XPUPlace."));
#endif
    }
    opbase_run_1 = GetPosixInUsec();
    OpBase::Run(*op, new_ins, outs, attrs, place);
    opbase_run_4 = GetPosixInUsec();
    time_before_kernel = opbase_run_2 - opbase_run_1 + time_before_kernel;
    time_after_kernel = opbase_run_4 - opbase_run_3 + time_after_kernel;
    time_kernel = opbase_run_3 - opbase_run_2 + time_kernel;
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

  start_time = GetPosixInUsec();
  if (ComputeRequiredGrad(new_ins, outs, trace_backward)) {
    CreateGradOpNode(*op, new_ins, outs, attrs, place, inplace_map);
  } else {
    VLOG(3) << "No Grad to track for Op: " << type;
  }
  time_create_gradop = GetPosixInUsec() - start_time + time_create_gradop;
  start_time_python = GetPosixInUsec();
}

void Tracer::TraceOp(const std::string& type, const NameVarBaseMap& ins,
                     const NameVarBaseMap& outs, framework::AttributeMap attrs,
                     const std::map<std::string, std::string>& inplace_map) {
  TraceOp(type, ins, outs, std::move(attrs), expected_place_, has_grad_,
          inplace_map);
}

void Tracer::SetExpectedPlace(platform::Place place) {
  expected_place_ = place;
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
