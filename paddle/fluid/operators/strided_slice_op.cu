/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/strided_slice_op.h"
#include "paddle/fluid/platform/complex.h"

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    strided_slice,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext, int>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext, float>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext, double>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<float>>,
    ops::StridedSliceKernel<paddle::platform::CUDADeviceContext,
                            paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    strided_slice_grad,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext, bool>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext, int>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext, double>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<float>>,
    ops::StridedSliceGradKernel<paddle::platform::CUDADeviceContext,
                                paddle::platform::complex<double>>);
