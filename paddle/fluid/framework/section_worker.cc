/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_NCCL)
#include <float.h>
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/program_desc.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc& desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.section_param().section_config().program_desc()));
  for (auto& op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void SectionWorker::AutoSetCPUAffinity(bool reuse) {
  unsigned concurrency_cap = std::thread::hardware_concurrency();
  unsigned proc = cpu_id_;

  if (proc >= concurrency_cap) {
    if (reuse) {
      proc %= concurrency_cap;
    } else {
      LOG(INFO) << "All " << concurrency_cap
                << " CPUs have been set affinities. Fail to set " << cpu_id_
                << "th thread.";
      return;
    }
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(proc, &mask);

  if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
    return;
  }

  CPU_ZERO(&mask);
  if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) ||
      (0 == CPU_ISSET(proc, &mask))) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
  }
  VLOG(3) << "Set " << cpu_id_ << "th thread affinity to CPU " << proc;
}

void SectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";
  AutoSetCPUAffinity(true);

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
  if (max_memory_size >= 0) {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      } else {
        gc.reset(new DefaultStreamGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    } else if (platform::is_cpu_place(place_)) {
#endif
      gc.reset(new CPUGarbageCollector(
          BOOST_GET_CONST(platform::CPUPlace, place_), max_memory_size));
#ifdef PADDLE_WITH_CUDA
    }
#endif
  }

  for (int i = 0; i < num_microbatches_; ++i) {
    try {
      for (auto& op : ops_) {
        int op_role = op->Attr<int>(std::string("op_role"));
        // We run op with op_role = kLRSched only for the first microbatch
        // to avoid increasing the @LR_DECAY_STEP@ multiple times.
        bool run_first_mbatch = op_role == static_cast<int>(OpRole::kForward) ||
                                op_role == (static_cast<int>(OpRole::kForward) |
                                            static_cast<int>(OpRole::kLoss)) ||
                                op_role == static_cast<int>(OpRole::kLRSched);
        bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                          op_role == (static_cast<int>(OpRole::kForward) |
                                      static_cast<int>(OpRole::kLoss));
        if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
          VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
                  << i;
          op->Run(*microbatch_scopes_[i], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                                gc.get());
          }
        }
      }
    } catch (platform::EOFException& e) {
      VLOG(3) << "EOF encountered and completed.";
      std::rethrow_exception(std::current_exception());
    }
  }

  // backward pass
  for (int i = 0; i < num_microbatches_; ++i) {
    for (auto& op : ops_) {
      int op_role = op->Attr<int>(std::string("op_role"));
      if (op_role == static_cast<int>(OpRole::kBackward) ||
          op_role == (static_cast<int>(OpRole::kBackward) |
                      static_cast<int>(OpRole::kLoss))) {
        VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
                << i;
        op->Run(*microbatch_scopes_[i], place_);
        if (gc) {
          DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                              gc.get());
        }
      }
    }
  }

  // update pass
  for (auto& op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "Update: running op " << op->Type();
      op->Run(*microbatch_scopes_[0], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[0], op.get(), unused_vars_,
                            gc.get());
      }
    }
  }
  dev_ctx_->Wait();
  ++batch_id_;
}

void SectionWorker::TrainFilesWithProfiler() {
  VLOG(5) << "begin section_worker TrainFiles with profiler";
  AutoSetCPUAffinity(true);

  platform::Timer timeline;

  std::vector<std::string> op_name;
  std::vector<double> op_total_time;
  std::vector<double> op_max_time;
  std::vector<double> op_min_time;
  std::vector<uint64_t> op_count;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  op_max_time.resize(ops_.size());
  op_min_time.resize(ops_.size());
  for (size_t i = 0; i < op_min_time.size(); ++i) {
    op_min_time[i] = DBL_MAX;
    op_max_time[i] = 0.0;
  }
  op_count.resize(ops_.size());

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
  if (max_memory_size >= 0) {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      } else {
        gc.reset(new DefaultStreamGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    } else if (platform::is_cpu_place(place_)) {
#endif
      gc.reset(new CPUGarbageCollector(
          BOOST_GET_CONST(platform::CPUPlace, place_), max_memory_size));
#ifdef PADDLE_WITH_CUDA
    }
#endif
  }

  struct timeval start;
  struct timeval end;
  struct timeval micro_start;
  struct timeval micro_end;

  // Start a minibatch.
  for (int i = 0; i < num_microbatches_; ++i) {
    try {
      int op_idx = 0;
      gettimeofday(&micro_start, NULL);
      for (auto& op : ops_) {
        gettimeofday(&start, NULL);
        int op_role = op->Attr<int>(std::string("op_role"));
        // We run op with op_role = kLRSched only for the first microbatch
        // to avoid increasing the @LR_DECAY_STEP@ multiple times.
        bool run_first_mbatch = op_role == static_cast<int>(OpRole::kForward) ||
                                op_role == (static_cast<int>(OpRole::kForward) |
                                            static_cast<int>(OpRole::kLoss)) ||
                                op_role == static_cast<int>(OpRole::kLRSched);
        bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                          op_role == (static_cast<int>(OpRole::kForward) |
                                      static_cast<int>(OpRole::kLoss));
        if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
          VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
                  << i;
          timeline.Start();
          op->Run(*microbatch_scopes_[i], place_);
          if (gc) {
            DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                                gc.get());
          }
          cudaDeviceSynchronize();
          timeline.Pause();
          gettimeofday(&end, NULL);
          auto time = timeline.ElapsedUS();
          op_total_time[op_idx] += time;
          if (time > op_max_time[op_idx]) {
            op_max_time[op_idx] = time;
          }
          if (time < op_min_time[op_idx]) {
            op_min_time[op_idx] = time;
          }
          op_count[op_idx] += 1;

          std::cout << std::fixed;
          std::cout.precision(0);
          std::cout << "::FWD:B[" << batch_id_ << "]:SCOPE[" << i << "]:OP["
                    << op->Type() << "]:START["
                    << start.tv_sec * 1e6 + start.tv_usec << "]:END["
                    << end.tv_sec * 1e6 + end.tv_usec << "]" << std::endl;
        }
        op_idx++;
      }

      gettimeofday(&micro_end, NULL);
      std::cout << std::fixed;
      std::cout.precision(0);
      std::cout << "!!FWD:B[" << batch_id_ << "]:START["
                << micro_start.tv_sec * 1e6 + micro_start.tv_usec << "]:END["
                << micro_end.tv_sec * 1e6 + micro_end.tv_usec << "]"
                << std::endl;
    } catch (platform::EOFException& e) {
      VLOG(0) << "EOF encountered, and completed";
      VLOG(0) << "============timeline============";
      for (size_t i = 0; i < ops_.size(); ++i) {
        VLOG(0) << "op: " << op_name[i] << ", max_time: " << op_max_time[i]
                << ", min_time: " << op_min_time[i]
                << ", mean_time: " << op_total_time[i] / op_count[i];
      }
      VLOG(0) << "================================";
      std::rethrow_exception(std::current_exception());
    }
  }

  // backward pass
  for (int i = 0; i < num_microbatches_; ++i) {
    int op_idx = 0;
    gettimeofday(&micro_start, NULL);
    for (auto& op : ops_) {
      gettimeofday(&start, NULL);
      int op_role = op->Attr<int>(std::string("op_role"));
      if (op_role == static_cast<int>(OpRole::kBackward) ||
          op_role == (static_cast<int>(OpRole::kBackward) |
                      static_cast<int>(OpRole::kLoss))) {
        VLOG(3) << "Backward: running an op " << op->Type()
                << " for micro-batch " << i;
        timeline.Start();
        op->Run(*microbatch_scopes_[i], place_);
        if (gc) {
          DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                              gc.get());
        }
        cudaDeviceSynchronize();
        gettimeofday(&end, NULL);
        timeline.Pause();
        auto time = timeline.ElapsedUS();
        op_total_time[op_idx] += time;
        if (time > op_max_time[op_idx]) {
          op_max_time[op_idx] = time;
        }
        if (time < op_min_time[op_idx]) {
          op_min_time[op_idx] = time;
        }
        op_count[op_idx] += 1;

        std::cout << std::fixed;
        std::cout.precision(0);
        std::cout << "::BWD:B[" << batch_id_ << "]:SCOPE[" << i << "]:OP["
                  << op->Type() << "]:START["
                  << start.tv_sec * 1e6 + start.tv_usec << "]:END["
                  << end.tv_sec * 1e6 + end.tv_usec << "]" << std::endl;
      }
      op_idx++;
    }

    gettimeofday(&micro_end, NULL);
    std::cout << std::fixed;
    std::cout.precision(0);
    std::cout << "!!BWD:B[" << batch_id_ << "]:START["
              << micro_start.tv_sec * 1e6 + micro_start.tv_usec << "]:END["
              << micro_end.tv_sec * 1e6 + micro_end.tv_usec << "]" << std::endl;
  }

  // update pass
  int op_idx = 0;
  gettimeofday(&micro_start, NULL);
  for (auto& op : ops_) {
    gettimeofday(&start, NULL);
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "Update: running op " << op->Type();
      timeline.Start();
      op->Run(*microbatch_scopes_[0], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[0], op.get(), unused_vars_,
                            gc.get());
      }
      cudaDeviceSynchronize();
      gettimeofday(&end, NULL);
      timeline.Pause();
      auto time = timeline.ElapsedUS();
      op_total_time[op_idx] += time;
      if (time > op_max_time[op_idx]) {
        op_max_time[op_idx] = time;
      }
      if (time < op_min_time[op_idx]) {
        op_min_time[op_idx] = time;
      }
      op_count[op_idx] += 1;

      std::cout << std::fixed;
      std::cout.precision(0);
      std::cout << "::UPD:B[" << batch_id_ << "]:OP[" << op->Type()
                << "]:START[" << start.tv_sec * 1e6 + start.tv_usec << "]:END["
                << end.tv_sec * 1e6 + end.tv_usec << "]" << std::endl;
    }
    op_idx++;
  }
  gettimeofday(&micro_end, NULL);
  std::cout << std::fixed;
  std::cout.precision(0);
  std::cout << "!!UPD:B[" << batch_id_ << "]:START["
            << micro_start.tv_sec * 1e6 + micro_start.tv_usec << "]:END["
            << micro_end.tv_sec * 1e6 + micro_end.tv_usec << "]" << std::endl;
  dev_ctx_->Wait();
  ++batch_id_;
}

}  // namespace framework
}  // namespace paddle
#endif
