/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/tril_triu_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    tril_triu, ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TrilTriuOpKernel<paddle::platform::CUDADeviceContext, plat::float16>);
REGISTER_OP_CUDA_KERNEL(
    tril_triu_grad,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TrilTriuGradOpKernel<paddle::platform::CUDADeviceContext,
                              plat::float16>);
