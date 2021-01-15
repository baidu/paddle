// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_ASCEND
#include "paddle/fluid/framework/fleet/ascend_wrapper.h"
namespace paddle {
namespace framework {
std::shared_ptr<AscendInstance> AscendInstance::ascend_instance_ = nullptr;
}  // end namespace framework
}  // end namespace paddle
#endif
