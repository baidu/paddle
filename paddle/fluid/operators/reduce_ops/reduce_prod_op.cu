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

#include "paddle/fluid/operators/reduce_ops/reduce_functor_op.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_prod_op.h"

// reduce_prod
#ifdef __HIPCC__
// Eigen3/unsupported/Eigen/CXX11/src/Tensor/TensorReductionGpu.h:922
// do not support double in HIPCC platform (Eigen3 to be fixed)
REGISTER_OP_CUDA_KERNEL(
    reduce_prod, ops::ReduceCudaKernel<float, paddle::operators::CustomMul>,
    ops::ReduceCudaKernel<int, paddle::operators::CustomMul>,
    ops::ReduceCudaKernel<int64_t, paddle::operators::CustomMul>);
#else
REGISTER_OP_CUDA_KERNEL(
    reduce_prod, ops::ReduceCudaKernel<float, paddle::operators::CustomMul>,
    ops::ReduceCudaKernel<int, paddle::operators::CustomMul>,
    ops::ReduceCudaKernel<double, paddle::operators::CustomMul>,
    ops::ReduceCudaKernel<int64_t, paddle::operators::CustomMul>);
#endif
