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

#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    squeeze,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int64_t>);
REGISTER_OP_NPU_KERNEL(
    squeeze2,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, bool>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, uint8_t>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int8_t>,
    ops::SqueezeKernel<paddle::platform::NPUDeviceContext, int64_t>);
#endif
