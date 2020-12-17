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

#include "paddle/fluid/operators/scale_op.h"
#include "paddle/fluid/platform/float16.h"
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    scale,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, float>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, double>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   uint8_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, int8_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   int16_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext, int>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   int64_t>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   plat::float16>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   plat::complex64>,
    paddle::operators::ScaleKernel<paddle::platform::CUDADeviceContext,
                                   plat::complex128>);
