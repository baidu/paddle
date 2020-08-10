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

#pragma once

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

// This file almostly contains all the infershape functions that are used in
// operators.

namespace paddle {
namespace operators {

// shape input(0) -> output(0) without change.
void UnaryOpUnchangedInferShape(framework::InferShapeContext* ctx);
// shape input(0) -> output(0) without change, check if axis in range [-Rank(x),
// Rank(x)-1]
void UnaryOpUnchangedInferShapeCheckAxis(framework::InferShapeContext* ctx);
// broadcast input(0) and input(1) -> output(0)
void BinaryOpBroadcastInferShape(framework::InferShapeContext* ctx);

}  // namespace operators
}  // namespace paddle
