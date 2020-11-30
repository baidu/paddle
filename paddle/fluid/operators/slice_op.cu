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

#include "paddle/fluid/operators/slice_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    slice, ops::SliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::complex128>,
    ops::SliceKernel<paddle::platform::CUDADeviceContext, plat::float16>);

REGISTER_OP_CUDA_KERNEL(
    slice_grad,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::complex64>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::complex128>,
    ops::SliceGradKernel<paddle::platform::CUDADeviceContext, plat::float16>);
