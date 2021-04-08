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

#include <utility>
#include <vector>

#include "paddle/fluid/framework/python_headers.h"

namespace paddle {
namespace imperative {

namespace py = ::pybind11;

class PyLayerContext {
 public:
  explicit PyLayerContext(const py::handle& handle) : context(handle.ptr()) {
    Py_INCREF(context);
  }
  ~PyLayerContext() { Py_DECREF(context); }
  PyLayerContext() = delete;

  PyObject* GetMatableCtx() { return context; }

 private:
  // TODO(weixin): support save_for_backward

  PyObject* context;
};

}  // namespace imperative
}  // namespace paddle
