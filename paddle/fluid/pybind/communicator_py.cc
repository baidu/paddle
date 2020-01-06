/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/communicator_py.h"

#include <Python.h>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "paddle/fluid/framework/program_desc.h"
#include "pybind11/pybind11.h"

#include "paddle/fluid/operators/distributed/communicator.h"

namespace py = pybind11;

using paddle::framework::ProgramDesc;
using paddle::framework::Scope;
using paddle::operators::distributed::AsyncCommunicator;
using paddle::operators::distributed::Communicator;
using paddle::operators::distributed::GeoSgdCommunicator;
using paddle::operators::distributed::HalfAsyncCommunicator;

namespace paddle {
namespace pybind {

void BindCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const ProgramDesc& program,
                       Scope* param_scope,
                       std::map<std::string, int>& env_flags) {
        if (mode == "HALF_ASYNC") {
          Communicator::InitInstance<HalfAsyncCommunicator>(
              program, param_scope, env_flags);
        } else if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(program, param_scope,
                                                        env_flags);
        } else {
          VLOG(0) << "unknown MODE for communicator";
        }
      }))

      .def(py::init([](
          const ProgramDesc& program, Scope* training_scope,
          std::map<std::string,
                   std::map<std::string, std::vector<std::string>>>& vars_info,
          int& trainers, int& geo_need_push_nums,
          std::map<std::string, int>& env_flags) {
        VLOG(0) << "using geo sgd communicator";
        Communicator::InitInstance<GeoSgdCommunicator>(
            program, training_scope, vars_info, trainers, geo_need_push_nums,
            env_flags);
        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("is_running", &Communicator::IsRunning);
}
}  // namespace pybind
}  // namespace paddle
