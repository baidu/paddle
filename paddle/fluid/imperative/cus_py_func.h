// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <atomic>
#include <future>  // NOLINT
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "ThreadPool.h"
#include "paddle/fluid/imperative/basic_engine.h"
#include "paddle/fluid/imperative/jit/program_desc_tracer.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/platform/macros.h"

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace imperative {
namespace py = ::pybind11;

class CusPyFunc_Context {
  DISABLE_COPY_AND_ASSIGN(CusPyFunc_Context);

 public:
  CusPyFunc_Context() {}
};  // _CusPyFuncContext

py::object CusPyFunc_apply(const py::object &cls, py::args args,
                           py::kwargs kwargs);
size_t _AppendPythonContext2Op(const py::object &py_contex);

}  // namespace imperative
}  // namespace paddle
