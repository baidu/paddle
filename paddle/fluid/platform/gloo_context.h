//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"

namespace paddle {
namespace platform {

#if defined(PADDLE_WITH_GLOO)
struct GlooParallelStrategy {
  int rank{0};
  int rank_num{1};
  std::string iface;
  int init_seconds{9999999};
  int run_seconds{9999999};
  std::string ip_address;
  int ip_port;
  std::string scope{"worker"};
};

class GlooParallelContext {
 public:
  explicit GlooParallelContext(const GlooParallelStrategy& strategy)
      : strategy_(strategy) {}

  virtual ~GlooParallelContext() {}

  virtual void Init();

 protected:
  GlooParallelStrategy strategy_;
};
#endif

}  //  namespace platform
}  //  namespace paddle
