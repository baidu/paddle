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

#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/pybind/generator_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindGenerator(py::module* m_ptr) {
  auto& m = *m_ptr;
  py::class_<framework::GeneratorState>(m, "GeneratorState", "");
  py::class_<std::mt19937_64>(m, "mt19937_64", "");
  py::class_<framework::Generator, std::shared_ptr<framework::Generator>>(
      m, "Generator")
      .def(py::init([]() { return framework::Generator::GetInstanceX(); }),
           py::return_value_policy::reference)
      .def("get_state", &framework::Generator::GetState,
           py::return_value_policy::move)
      .def("set_state", &framework::Generator::SetState)
      .def("manual_seed", &framework::Generator::SetCurrentSeed)
      .def("seed", &framework::Generator::Seed)
      .def("initial_seed", &framework::Generator::GetCurrentSeed)
      .def("random", &framework::Generator::Random64)
      .def("get_cpu_engine", &framework::Generator::GetCPUEngine,
           py::return_value_policy::move)
      .def("set_cpu_engine", &framework::Generator::SetCPUEngine)
      .def_property("is_init_py_cuda", &framework::Generator::GetIsInitPyCUDA,
                    &framework::Generator::SetIsInitPyCUDA);
  m.def("default_cuda_generator", &framework::getDefaultCUDAGenerator);
}  // end Generator
}  // end namespace pybind
}  // end namespace paddle
