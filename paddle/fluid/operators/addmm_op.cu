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

#include "paddle/fluid/operators/addmm_op.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;
#ifdef PADDLE_WITH_HIP
REGISTER_OP_CUDA_KERNEL(addmm,
                        ops::AddMMKernel<plat::CUDADeviceContext, float>);
REGISTER_OP_CUDA_KERNEL(addmm_grad,
                        ops::AddMMGradKernel<plat::CUDADeviceContext, float>);
#else
REGISTER_OP_CUDA_KERNEL(addmm, ops::AddMMKernel<plat::CUDADeviceContext, float>,
                        ops::AddMMKernel<plat::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(addmm_grad,
                        ops::AddMMGradKernel<plat::CUDADeviceContext, float>,
                        ops::AddMMGradKernel<plat::CUDADeviceContext, double>);
#endif
