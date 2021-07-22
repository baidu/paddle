/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU
#include "paddle/fluid/operators/controlflow/logical_op_xpu.h"
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    logical_and,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, bool>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, int8_t>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, int16_t>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, int>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, int64_t>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, float>,
    ops::BinaryLogicalOpXPUKernel<ops::XpuLogicalType::XPU_AND, double>);
#endif
