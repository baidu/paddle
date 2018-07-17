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

#define EIGEN_USE_GPU
#include "paddle/fluid/operators/overflow_op.h"
#include "paddle/fluid/platform/float16.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

#define REGISTER_OVERFLOW_CUDA_KERNEL(op_type, functor)                       \
  REGISTER_OP_CUDA_KERNEL(                                                    \
      op_type, ops::OverflowKernel<paddle::platform::CUDADeviceContext, int,  \
                                   ops::functor<int>>,                        \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, float,         \
                          ops::functor<float>>,                               \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, double,        \
                          ops::functor<double>>,                              \
      ops::OverflowKernel<paddle::platform::CUDADeviceContext, plat::float16, \
                          ops::functor<plat::float16>>);

FOR_EACH_KERNEL_FUNCTOR(REGISTER_OVERFLOW_CUDA_KERNEL);
