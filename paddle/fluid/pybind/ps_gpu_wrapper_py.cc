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
#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/fleet/ps_gpu_wrapper.h"
#include "paddle/fluid/pybind/ps_gpu_wrapper_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
#ifdef PADDLE_WITH_HETERPS
void BindPSGPUWrapper(py::module* m) {
  py::class_<framework::PSGPUWrapper, std::shared_ptr<framework::PSGPUWrapper>>(
      *m, "PSGPU")
      .def(py::init([]() { return framework::PSGPUWrapper::GetInstance(); }))
      .def("set_slot_vector", &framework::PSGPUWrapper::SetSlotVector,
           py::call_guard<py::gil_scoped_release>())
      .def("init_GPU_server", &framework::PSGPUWrapper::InitializeGPUServer,
           py::call_guard<py::gil_scoped_release>())
      .def("set_dataset", &framework::PSGPUWrapper::SetDataset,
           py::call_guard<py::gil_scoped_release>())
      .def("init_gpu_ps", &framework::PSGPUWrapper::InitializeGPU,
           py::call_guard<py::gil_scoped_release>())
      .def("end_pass", &framework::PSGPUWrapper::EndPass,
           py::call_guard<py::gil_scoped_release>())
      .def("build_gpu_ps", &framework::PSGPUWrapper::BuildGPUPS,
           py::call_guard<py::gil_scoped_release>())
      .def("finalize", &framework::PSGPUWrapper::Finalize,
           py::call_guard<py::gil_scoped_release>());
}  // end PSGPUWrapper
#endif
}  // end namespace pybind
}  // end namespace paddle
