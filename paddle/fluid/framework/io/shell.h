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

#include <fcntl.h>
#include <sys/stat.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>
#else
#include <sys/syscall.h>
#endif
#include <sys/types.h>
#ifndef _WIN32
#include <sys/wait.h>
#endif
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "paddle/fluid/platform/port.h"
#include "paddle/fluid/string/string_helper.h"

#if defined(__arm__) || defined(__aarch64__) || defined(__ARM_NEON) || \
    defined(__ARM_NEON__)
#define PADDLE_ARM
#endif

namespace paddle {
namespace framework {

inline bool& shell_verbose_internal() {
  static bool x = false;
  return x;
}

inline bool shell_verbose() { return shell_verbose_internal(); }

inline void shell_set_verbose(bool x) { shell_verbose_internal() = x; }

extern std::shared_ptr<FILE> shell_fopen(const std::string& path,
                                         const std::string& mode);

std::shared_ptr<FILE> shell_popen(const std::string& cmd,
                                  const std::string& mode, int* err_no,
                                  int* status = NULL,
                                  bool redirect_stderr = false);

extern std::pair<std::shared_ptr<FILE>, std::shared_ptr<FILE>> shell_p2open(
    const std::string& cmd);

inline void shell_execute(const std::string& cmd) {
  int err_no = 0;
  do {
    err_no = 0;
    shell_popen(cmd, "w", &err_no);
  } while (err_no == -1);
}

// time_out:ms, default value:-1 means forever.
// sleep_inter:ms, default -1 means not sleep.
extern std::string shell_get_command_output(const std::string& cmd,
                                            int time_out = 10 * 60 * 1000,
                                            int sleep_inter = 1000);
// time_out:ms, default -1 means forever.
// sleep_inter:ms, default -1 means not sleep.
extern std::vector<std::string> shell_execute_cmd(const std::string& cmd,
                                                  int time_out = 0,
                                                  int sleep_inter = 0,
                                                  bool redirect_stderr = false);

}  // namespace framework
}  // namespace paddle
