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

#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    concat, ops::ConcatKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ConcatKernel<paddle::platform::CUDADeviceContext, uint8_t>);
REGISTER_OP_CUDA_KERNEL(
    concat_grad,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, plat::float16>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::ConcatGradKernel<paddle::platform::CUDADeviceContext, uint8_t>);
