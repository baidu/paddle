//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

void BindDistFleetWrapper(py::module* m);
void BindPSHost(py::module* m);
void BindCommunicatorContext(py::module* m);
void BindDistCommunicator(py::module* m);
void BindHeterClient(py::module* m);
void BindGraphNode(py::module* m);
void BindGraphPyService(py::module* m);
void BindGraphPyFeatureNode(py::module* m);
void BindGraphPyServer(py::module* m);
void BindGraphPyClient(py::module* m);
void BindIndexNode(py::module* m);
void BindTreeIndex(py::module* m);
void BindIndexWrapper(py::module* m);
void BindIndexSampler(py::module* m);
void BindGraphItem(py::module* m);
void BindGraphIndex(py::module* m);
void BindSparseShardingTools(py::module* m);
}  // namespace pybind
}  // namespace paddle
