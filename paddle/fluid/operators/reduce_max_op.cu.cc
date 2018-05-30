// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reduce_min_max_op.h"

REGISTER_OP_CUDA_KERNEL(reduce_max,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          float, ops::MaxFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          double, ops::MaxFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          int, ops::MaxFunctor>,
                        ops::ReduceKernel<paddle::platform::CUDADeviceContext,
                                          int64_t, ops::MaxFunctor>);
REGISTER_OP_CUDA_KERNEL(
    reduce_max_grad, ops::ReduceGradKernel<paddle::platform::CUDADeviceContext,
                                           float, ops::MaxGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, double,
                          ops::MaxGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int,
                          ops::MaxGradFunctor>,
    ops::ReduceGradKernel<paddle::platform::CUDADeviceContext, int64_t,
                          ops::MaxGradFunctor>);
