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

#include "paddle/fluid/pybind/gloo_context_py.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/memory/allocation/mmap_allocator.h"
#include "paddle/fluid/platform/gloo_context.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

// Bind Methods
void BindGlooContext(py::module *m) {
// define parallel context for gloo
#if defined(PADDLE_WITH_GLOO)
  py::class_<platform::GlooParallelStrategy> gloo_parallel_strategy(
      *m, "GlooParallelStrategy", "");
  gloo_parallel_strategy.def(py::init())
      .def_property("rank_num",
                    [](const platform::GlooParallelStrategy &self) {
                      return self.rank_num;
                    },
                    [](platform::GlooParallelStrategy &self, int nranks) {
                      self.rank_num = nranks;
                    })
      .def_property(
          "rank",
          [](const platform::GlooParallelStrategy &self) { return self.rank; },
          [](platform::GlooParallelStrategy &self, int rank) {
            self.rank = rank;
          })
      .def_property(
          "iface",
          [](const platform::GlooParallelStrategy &self) { return self.iface; },
          [](platform::GlooParallelStrategy &self, const std::string &iface) {
            self.iface = iface;
          })
      .def_property("init_seconds",
                    [](const platform::GlooParallelStrategy &self) {
                      return self.init_seconds;
                    },
                    [](platform::GlooParallelStrategy &self, int init_seconds) {
                      self.init_seconds = init_seconds;
                    })
      .def_property("run_seconds",
                    [](const platform::GlooParallelStrategy &self) {
                      return self.run_seconds;
                    },
                    [](platform::GlooParallelStrategy &self, int run_seconds) {
                      self.run_seconds = run_seconds;
                    })
      .def_property(
          "ip_address",
          [](const platform::GlooParallelStrategy &self) {
            return self.ip_address;
          },
          [](platform::GlooParallelStrategy &self,
             const std::string &ip_address) { self.ip_address = ip_address; })
      .def_property("ip_port",
                    [](const platform::GlooParallelStrategy &self) {
                      return self.ip_port;
                    },
                    [](platform::GlooParallelStrategy &self, int ip_port) {
                      self.ip_port = ip_port;
                    });

  py::class_<platform::GlooParallelContext> gloo_ctx(*m, "GlooParallelContext");
  gloo_ctx.def(py::init<const platform::GlooParallelStrategy &>())
      .def("init", [](platform::GlooParallelContext &self) { self.Init(); });
#endif
}

}  // namespace pybind
}  // namespace paddle
