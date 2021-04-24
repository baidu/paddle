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

#include "paddle/fluid/operators/controlflow/compare_op.h"

REGISTER_COMPARE_KERNEL(less_than, CUDA, paddle::operators::LessThanFunctor,
                        paddle::operators::GreaterEqualFunctor);
REGISTER_COMPARE_KERNEL(less_equal, CUDA, paddle::operators::LessEqualFunctor,
                        paddle::operators::GreaterThanFunctor);
REGISTER_COMPARE_KERNEL(greater_than, CUDA,
                        paddle::operators::GreaterThanFunctor,
                        paddle::operators::LessEqualFunctor);
REGISTER_COMPARE_KERNEL(greater_equal, CUDA,
                        paddle::operators::GreaterEqualFunctor,
                        paddle::operators::LessThanFunctor);
REGISTER_COMPARE_KERNEL(equal, CUDA, paddle::operators::EqualFunctor,
                        paddle::operators::EqualFunctor);
REGISTER_COMPARE_KERNEL(not_equal, CUDA, paddle::operators::NotEqualFunctor,
                        paddle::operators::NotEqualFunctor);
