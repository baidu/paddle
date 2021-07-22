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

#pragma once

#include <time.h>
#include <map>
#include <memory>
#include <mutex>   // NOLINT
#include <random>  // local_random_engine
#include <set>
#include <string>
#include <thread>  // NOLINT
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"

namespace paddle {
namespace framework {

class FleetWrapper;

inline double current_realtime() {
#if !defined(_WIN32)
  struct timespec tp;
  clock_gettime(CLOCK_REALTIME, &tp);
  return tp.tv_sec + tp.tv_nsec * 1e-9;
#else
  return 0.0;
#endif
}

inline std::default_random_engine& local_random_engine() {
  struct engine_wrapper_t {
    std::default_random_engine engine;
    engine_wrapper_t() {
      //static std::atomic<uint64_t> x(0);
      std::seed_seq sseq = {x++, x++, x++,
                            static_cast<uint64_t>(current_realtime() * 1000)};
      engine.seed(sseq);
    }
  };
  thread_local engine_wrapper_t r;
  return r.engine;
}

class AsyncExecutor {
 public:
  AsyncExecutor(Scope* scope, const platform::Place& place);
  virtual ~AsyncExecutor() {}
  void RunFromFile(const ProgramDesc& main_program,
                   const std::string& data_feed_desc_str,
                   const std::vector<std::string>& filelist,
                   const int thread_num,
                   const std::vector<std::string>& fetch_var_names,
                   const std::string& mode, const bool debug);

  // TODO(guru4elephant): make init server decoupled from executor
  void InitServer(const std::string& dist_desc, int index);
  void InitWorker(const std::string& dist_desc,
                  const std::vector<uint64_t>& host_sign_list, int node_num,
                  int index);
  uint64_t StartServer();
  void StopServer();
  void GatherServers(const std::vector<uint64_t>& host_sign_list, int node_num);
  void InitModel();
  void SaveModel(const std::string& path);

 public:
  std::shared_ptr<paddle::framework::FleetWrapper> fleet_ptr_;
  Scope* root_scope_;
  platform::Place place_;

 private:
  int actual_thread_num_;
};

}  // namespace framework
}  // namespace paddle
