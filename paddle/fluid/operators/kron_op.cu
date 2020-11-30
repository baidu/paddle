/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/kron_op.h"
#include "paddle/fluid/platform/complex128.h"
#include "paddle/fluid/platform/complex64.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    kron, ops::KronKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KronKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KronKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::float16>,
    ops::KronKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::complex64>,
    ops::KronKernel<paddle::platform::CUDADeviceContext,
                    paddle::platform::complex128>,
    ops::KronKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KronKernel<paddle::platform::CUDADeviceContext, int64_t>);

REGISTER_OP_CUDA_KERNEL(
    kron_grad, ops::KronGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::float16>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::complex64>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext,
                        paddle::platform::complex128>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KronGradKernel<paddle::platform::CUDADeviceContext, int64_t>);
