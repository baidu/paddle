/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/npu_op_runner.h"
#include "paddle/fluid/operators/unsqueeze_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    unsqueeze, ops::UnsqueezeKernel<plat::NPUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    unsqueeze2, ops::UnsqueezeKernel<plat::NPUDeviceContext, float>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, double>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::UnsqueezeKernel<plat::NPUDeviceContext, int64_t>);
#endif
