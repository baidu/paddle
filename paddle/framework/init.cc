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
#include <algorithm>
#include <string>

#include "paddle/framework/executor.h"
#include "paddle/framework/init.h"
#include "paddle/platform/place.h"
#include "paddle/string/piece.h"

namespace paddle {
namespace framework {

std::once_flag gflags_init_flag;

// TODO(qijun) move init gflags to init.cc
void InitGflags(std::vector<std::string> &argv) {
  std::call_once(gflags_init_flag, [&]() {
    int argc = argv.size();
    char **arr = new char *[argv.size()];
    std::string line;
    for (size_t i = 0; i < argv.size(); i++) {
      arr[i] = &argv[i][0];
      line += argv[i];
      line += ' ';
    }
    google::ParseCommandLineFlags(&argc, &arr, true);
    VLOG(1) << "Init commandline: " << line;
  });
}

bool InitDevices(const std::vector<std::string> &devices) {
  // device format
  // CPU
  // GPU:1
  // TODO:(dzhwinter) add device format annotation for users.
  std::vector<platform::Place> places;
  for (auto &device : devices) {
    auto p = string::Piece(device);
    if (string::Find(p, ':', 0) == string::Piece::npos) {
      places.emplace_back(platform::CPUPlace());
    } else if (string::HasPrefix(p, "GPU")) {
#ifdef PADDLE_WITH_CUDA
      auto pos = string::RFind(p, ':', string::Piece::npos);
      auto number = device.substr(pos + 1);
      places.emplace_back(platform::GPUPlace(std::stoi(number)));
#else
      LOG(WARNING)
          << "'GPU' is not supported, Please re-compile with WITH_GPU option";
#endif
    } else {
      return false;
    }
  }

  if (std::find(places.begin(), places.end(), platform::CPUPlace()) ==
      places.end()) {
    places.emplace_back(platform::CPUPlace());
    LOG(WARNING) << "Not specified any device, use CPU by Default.";
  }
  DeviceContextPool::Create(places);
  return true;
  return true;
}

}  // namespace framework
}  // namespace paddle
