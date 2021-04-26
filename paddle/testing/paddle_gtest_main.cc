/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/npu_info.h"
#include "paddle/fluid/platform/npu_instance.h"

int main(int argc, char** argv) {
  paddle::memory::allocation::UseAllocatorStrategyGFlag();
  testing::InitGoogleTest(&argc, argv);
  std::vector<char*> new_argv;
  std::string gflags_env;
  for (int i = 0; i < argc; ++i) {
    new_argv.push_back(argv[i]);
  }

  std::vector<std::string> envs;
  std::vector<std::string> undefok;
#if defined(PADDLE_WITH_DISTRIBUTE) && !defined(PADDLE_WITH_GRPC) && \
    !defined(PADDLE_WITH_PSLIB)
  std::string str_max_body_size;
  if (::GFLAGS_NAMESPACE::GetCommandLineOption("max_body_size",
                                               &str_max_body_size)) {
    setenv("FLAGS_max_body_size", "2147483647", 1);
    envs.push_back("max_body_size");
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_ASCEND_CL)
  envs.push_back("fraction_of_gpu_memory_to_use");
  envs.push_back("initial_gpu_memory_in_mb");
  envs.push_back("reallocate_gpu_memory_in_mb");
  envs.push_back("allocator_strategy");
  envs.push_back("selected_gpus");
#elif __clang__
  envs.push_back("use_mkldnn");
  envs.push_back("initial_cpu_memory_in_mb");
  envs.push_back("allocator_strategy");

  undefok.push_back("use_mkldnn");
  undefok.push_back("initial_cpu_memory_in_mb");
#else
  envs.push_back("use_pinned_memory");
  envs.push_back("use_mkldnn");
  envs.push_back("initial_cpu_memory_in_mb");
  envs.push_back("allocator_strategy");

  undefok.push_back("use_pinned_memory");
  undefok.push_back("use_mkldnn");
  undefok.push_back("initial_cpu_memory_in_mb");
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
  envs.push_back("selected_npus");
#endif

  char* env_str = nullptr;
  if (envs.size() > 0) {
    std::string env_string = "--tryfromenv=";
    for (auto t : envs) {
      env_string += t + ",";
    }
    env_string = env_string.substr(0, env_string.length() - 1);
    env_str = strdup(env_string.c_str());
    new_argv.push_back(env_str);
    VLOG(1) << "gtest env_string:" << env_string;
  }

  char* undefok_str = nullptr;
  if (undefok.size() > 0) {
    std::string undefok_string = "--undefok=";
    for (auto t : undefok) {
      undefok_string += t + ",";
    }
    undefok_string = undefok_string.substr(0, undefok_string.length() - 1);
    undefok_str = strdup(undefok_string.c_str());
    new_argv.push_back(undefok_str);
    VLOG(1) << "gtest undefok_string:" << undefok_string;
  }

  int new_argc = static_cast<int>(new_argv.size());
  char** new_argv_address = new_argv.data();
  ::GFLAGS_NAMESPACE::ParseCommandLineFlags(
      &new_argc, &new_argv_address, false);
  paddle::framework::InitDevices();

  int ret = RUN_ALL_TESTS();

#ifdef PADDLE_WITH_ASCEND_CL
  paddle::platform::AclInstance::Instance().Finalize();
#endif

  if (env_str) free(env_str);
  if (undefok_str) free(undefok_str);

  return ret;
}
