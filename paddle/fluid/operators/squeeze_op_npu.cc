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
#include "paddle/fluid/operators/squeeze_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_NPU_KERNEL(
    squeeze, ops::SqueezeKernel<plat::NPUDeviceContext, float>,
    ops::SqueezeKernel<plat::NPUDeviceContext, double>,
    ops::SqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int>,
    ops::SqueezeKernel<plat::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze2, ops::SqueezeKernel<plat::NPUDeviceContext, float>,
    ops::SqueezeKernel<plat::NPUDeviceContext, double>,
    ops::SqueezeKernel<plat::NPUDeviceContext, plat::float16>,
    ops::SqueezeKernel<plat::NPUDeviceContext, bool>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int>,
    ops::SqueezeKernel<plat::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<plat::NPUDeviceContext, int64_t>);
#endif
