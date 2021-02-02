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
#include "paddle/fluid/imperative/cus_py_func.h"
#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/cus_py_func_op.h"

namespace paddle {
namespace imperative {
namespace py = ::pybind11;
size_t _AppendPythonContext2Op(const py::object& py_contex) {
  return operators::CusPyFunc_AppendPythonContext(py_contex);
}

// copy from tracer.cc
void PassStopGradient(const NameVarBaseMap& outs, bool generate_grad) {
  for (const auto& pair : outs) {
    for (const auto& var : pair.second) {
      // NOTE(zhiqiu): this happends when None output are passed from python
      // side. For example, fake_quantize_dequantize_moving_average_abs_max may
      // pass None OutAccum in eval mode.
      // It can be refined by generate several different pybind interface for
      // one operator with different function signature.
      if (var == nullptr) {
        VLOG(4) << pair.first << " is NULL";
        continue;
      }
      VLOG(6) << "Set output: " << var->Name() << "'s OverridedStopGradient as "
              << generate_grad;
      var->InnerSetOverridedStopGradient(generate_grad);
    }
  }
}
// copy from tracer.cc
bool RequiredGrad(const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        VLOG(6) << "Find out input: " << var_base->Name()
                << "'s GeneratedGrad is True";
        PassStopGradient(outs, var_base->OverridedStopGradient());
        return true;
      }
    }
  }
  return false;
}

py::object CusPyFunc_apply(const py::object& cls, const py::args args,
                           const py::kwargs kwargs) {
  auto bk_function = cls.attr("_backward_function");
  auto py_contex = bk_function();
  auto forward = cls.attr("forward");
  auto result_forward = forward(*args, **kwargs);

  // make inputs to varbase
  std::vector<std::shared_ptr<imperative::VarBase>> input_vars;
  // process args,`input_vars` only collect `imperative::VarBase`
  if (!args.empty()) {
    for (auto ptr = args.begin(); ptr != args.end(); ptr++) {
      try {
        auto a = ptr->cast<std::shared_ptr<VarBase>>();
        // only push varbase
        input_vars.push_back(a);
      } catch (py::cast_error&) {
      }
    }
  }
  // process kwargs, only collect `imperative::VarBase`
  if (!kwargs.empty()) {
    for (auto ptr = kwargs.begin(); ptr != kwargs.end(); ptr++) {
      try {
        auto a = ptr->second.cast<std::shared_ptr<VarBase>>();
        // only push varbase
        input_vars.push_back(a);
      } catch (py::cast_error&) {
      }
    }
  }
  NameVarBaseMap ins = {{"X", input_vars}};

  // process outputs:result_forward
  std::vector<std::shared_ptr<imperative::VarBase>> output_vars;
  // if tuple/list
  if (PyTuple_Check(result_forward.ptr()) ||
      PyList_Check(result_forward.ptr())) {
    auto tuple_result = result_forward.cast<py::tuple>();
    for (size_t i = 0; i < tuple_result.size(); i++) {
      output_vars.push_back(
          tuple_result[i].cast<std::shared_ptr<imperative::VarBase>>());
    }
  } else {  // varbase
    try {
      auto temp_out =
          result_forward.cast<std::shared_ptr<imperative::VarBase>>();
      output_vars.push_back(temp_out);
    } catch (py::cast_error&) {
      std::cout << "wrong type of python return-----\n";
    }
  }

  NameVarBaseMap outs = {{"Out", output_vars}};
  // checko number of output/input

  // context_id
  int context_id = cls.attr("_backward_id").cast<int>();
  framework::AttributeMap attrs = {{"context_id", context_id}};

  // place
  auto place = platform::CPUPlace();
  auto op = framework::OpRegistry::CreateOp("cus_py_func", {}, {}, {}, false);

  if (RequiredGrad(ins, outs)) {
    CreateGradOpNode(*op, ins, outs, attrs, place);
  } else {
    VLOG(3) << "No Grad to track for Op: " << op->Type();
  }

  return result_forward;
}

}  // namespace imperative
}  // namespace paddle
