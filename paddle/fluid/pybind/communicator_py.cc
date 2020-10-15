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
#include "paddle/fluid/operators/distributed/communicator_common.h"
#include "paddle/fluid/operators/distributed/large_scale_kv.h"

namespace py = pybind11;

using paddle::framework::ProgramDesc;
using paddle::framework::Scope;
using paddle::operators::distributed::AsyncCommunicator;
using paddle::operators::distributed::Communicator;
using paddle::operators::distributed::GeoCommunicator;
using paddle::operators::distributed::HalfAsyncCommunicator;
using paddle::operators::distributed::SyncCommunicator;

using paddle::operators::distributed::CommContext;
using paddle::operators::distributed::RpcCtxMap;

using paddle::operators::distributed::LargeScaleKV;

namespace paddle {
namespace pybind {

void BindCommunicatorContext(py::module* m) {
  py::class_<CommContext>(*m, "CommContext")
      .def(
          py::init<const std::string&, const std::vector<std::string>&,
                   const std::vector<std::string>&, const std::vector<int64_t>&,
                   const std::vector<std::string>&, int, bool, bool, bool>())
      .def("var_name", [](const CommContext& self) { return self.var_name; })
      .def("trainer_id",
           [](const CommContext& self) { return self.trainer_id; })
      .def("split_varnames",
           [](const CommContext& self) { return self.splited_varnames; })
      .def("split_endpoints",
           [](const CommContext& self) { return self.epmap; })
      .def("sections",
           [](const CommContext& self) { return self.height_sections; })
      .def("aggregate", [](const CommContext& self) { return self.merge_add; })
      .def("is_sparse", [](const CommContext& self) { return self.is_sparse; })
      .def("is_distributed",
           [](const CommContext& self) { return self.is_distributed; })
      .def("origin_varnames",
           [](const CommContext& self) { return self.origin_varnames; })
      .def("__str__", [](const CommContext& self) { return self.print(); });
}

void BindCommunicator(py::module* m) {
  // Communicator is already used by nccl, change to DistCommunicator
  py::class_<Communicator, std::shared_ptr<Communicator>>(*m,
                                                          "DistCommunicator")
      .def(py::init([](const std::string& mode, const RpcCtxMap& send_ctx,
                       const RpcCtxMap& recv_ctx, Scope* param_scope,
                       std::map<std::string, std::string>& envs) {
        if (mode == "HALF_ASYNC") {
          Communicator::InitInstance<HalfAsyncCommunicator>(send_ctx, recv_ctx,
                                                            param_scope, envs);
        } else if (mode == "ASYNC") {
          Communicator::InitInstance<AsyncCommunicator>(send_ctx, recv_ctx,
                                                        param_scope, envs);
        } else if (mode == "SYNC") {
          Communicator::InitInstance<SyncCommunicator>(send_ctx, recv_ctx,
                                                       param_scope, envs);
        } else if (mode == "GEO") {
          Communicator::InitInstance<GeoCommunicator>(send_ctx, recv_ctx,
                                                      param_scope, envs);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "unsuported communicator MODE"));
        }

        return Communicator::GetInstantcePtr();
      }))
      .def("stop", &Communicator::Stop)
      .def("start", &Communicator::Start)
      .def("is_running", &Communicator::IsRunning)
      .def("recv", &Communicator::RecvNoBarrier);
}

void BindLargeScaleKV(py::module* m) {
  py::class_<LargeScaleKV, std::shared_ptr<LargeScaleKV>>(*m, "LargeScaleKV")
      .def(py::init([]() { return LargeScaleKV::GetInstantcePtr(); }))
      .def("load",
           [](LargeScaleKV& self, const std::string& table_name,
              const std::string& dir) {
             auto* sparse_variable = self.Get(table_name);
             sparse_variable->Load(dir);
           })
      .def("save",
           [](LargeScaleKV& self, const std::string& table_name,
              const std::string& dir) {
             auto* sparse_variable = self.Get(table_name);
             sparse_variable->Save(dir);
           })
      .def("size", [](LargeScaleKV& self, const std::string& table_name) {
        auto* sparse_variable = self.Get(table_name);
        return sparse_variable->Size();
      });
}
}  // namespace pybind
}  // namespace paddle
