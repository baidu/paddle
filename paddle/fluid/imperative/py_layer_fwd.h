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

#include <string>
#include <vector>
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/prepared_operator.h"
#include "paddle/fluid/imperative/tracer.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/operators/py_layer_op.h"

namespace paddle {
namespace imperative {

namespace py = ::pybind11;

bool RequiredGrad(const NameVarBaseMap& ins, const NameVarBaseMap& outs) {
  for (const auto& name_pair : ins) {
    for (const auto& var_base : name_pair.second) {
      if (!var_base->OverridedStopGradient()) {
        for (const auto& pair : outs) {
          for (const auto& var : pair.second) {
            if (var) {
              var->SetOverridedStopGradient(false);
              SetForwardDataTypeOfGradVar(var);
              VLOG(3) << "Set output: " << var->Name()
                      << "'s OverridedStopGradient as "
                      << var->OverridedStopGradient();
            }
          }
        }
        return true;
      }
    }
  }
  return false;
}

std::shared_ptr<GradOpNode> CreateGradOpNode(
    const std::string& type, const NameVarBaseMap& ins,
    const NameVarBaseMap& outs, const framework::AttributeMap& attrs,
    const platform::Place& place,
    const std::map<std::string, std::string>& inplace_map,
    const std::shared_ptr<operators::PyLayerContext>& py_context) {
  operators::PyLayerGradOpMaker<paddle::imperative::OpBase> maker(
      type, ins, outs, attrs, inplace_map);

  maker.SetPyLayerContext(py_context);
  auto grad_node = maker();
  if (grad_node && !grad_node->empty()) {
    for (auto& grad_op : *grad_node) {
      grad_op.SetId(OpBase::GenerateUniqueId());
      grad_op.SetPlace(place);
      ClearNoNeedBufferInputs(&grad_op);
    }
    return grad_node;
  } else {
    return nullptr;
  }
}

py::object PyLayerApply(const platform::Place& place, const py::handle& cls,
                        const py::args args, const py::kwargs kwargs) {
  py::gil_scoped_acquire guard;
  auto bk_function = cls.attr("_backward_function");
  auto context = bk_function();
  auto forward = cls.attr("forward");

  auto result_forward = forward(context, *args, **kwargs);
  std::shared_ptr<operators::PyLayerContext> py_layer_ctx =
      std::make_shared<operators::PyLayerContext>(context.ptr());
  // make inputs to varbase
  std::vector<std::shared_ptr<imperative::VarBase>> input_vars;
  // process args,`input_vars` only collect `imperative::VarBase`
  if (!args.empty()) {
    for (auto ptr = args.begin(); ptr != args.end(); ptr++) {
      // Only collect Tensor type in 'args' and pass them to backward. Ignore
      // other types of input temporarily.
      if (py::isinstance<imperative::VarBase>(*ptr)) {
        try {
          auto a = ptr->cast<std::shared_ptr<VarBase>>();
          input_vars.push_back(a);
        } catch (py::cast_error& err) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The `PyLayer.forward` function contains invalid argument, the "
              "`%s` type argument can not be cast into `Tensor`.",
              ptr->ptr()->ob_type->tp_name));
        }
      }
    }
  }
  // process kwargs, only collect `imperative::VarBase`
  if (!kwargs.empty()) {
    for (auto ptr = kwargs.begin(); ptr != kwargs.end(); ptr++) {
      // Only collect Tensor type in 'kwargs' and pass them to backward.
      // Ignore other types of input temporarily.
      if (py::isinstance<imperative::VarBase>(*ptr->second)) {
        try {
          auto a = ptr->second.cast<std::shared_ptr<VarBase>>();
          input_vars.push_back(a);
        } catch (py::cast_error&) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The `PyLayer.forward` function contains invalid argument, the "
              "`%s` type argument can not be cast into `Tensor`.",
              ptr->second.ptr()->ob_type->tp_name));
        }
      }
    }
  }
  NameVarBaseMap ins = {{"X", input_vars}};

  std::vector<std::shared_ptr<imperative::VarBase>> output_vars;
  if (PyTuple_Check(result_forward.ptr()) ||
      PyList_Check(result_forward.ptr())) {
    auto tuple_result = result_forward.cast<py::tuple>();
    for (size_t i = 0; i < tuple_result.size(); i++) {
      // Only collect Tensor type of output and pass them to backward.
      // Ignore other types of input temporarily.
      if (py::isinstance<imperative::VarBase>(tuple_result[i])) {
        try {
          auto temp_out =
              tuple_result[i].cast<std::shared_ptr<imperative::VarBase>>();
          output_vars.push_back(temp_out);
        } catch (py::cast_error&) {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "The `PyLayer.forward` function returns invalid argument, the "
              "`%s` type argument can not be cast into `Tensor`.",
              tuple_result[i].ptr()->ob_type->tp_name));
        }
      }
    }
  } else {
    // Only collect Tensor type of output and pass them to backward.
    // Ignore other types of input temporarily.
    if (py::isinstance<imperative::VarBase>(result_forward)) {
      try {
        auto temp_out =
            result_forward.cast<std::shared_ptr<imperative::VarBase>>();
        output_vars.push_back(temp_out);
      } catch (py::cast_error&) {
        PADDLE_THROW(platform::errors::InvalidArgument(
            "The `PyLayer.forward` function returns invalid argument, the `%s` "
            "type argument can not be cast into `Tensor`.",
            result_forward.ptr()->ob_type->tp_name));
      }
    }
  }
  if (output_vars.size() == 0) {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "At least one output of `PyLayer.forward` is a `Tensor`."));
  }

  NameVarBaseMap outs = {{"Out", output_vars}};

  if (RequiredGrad(ins, outs)) {
    std::map<std::string, std::string> inplace_map{};
    bool if_inplace = false;
    for (auto temp_ins : input_vars) {
      if (if_inplace) {
        break;
      }
      for (auto temp_outs : output_vars) {
        if (temp_ins->Name() == temp_outs->Name()) {
          if_inplace = true;
          break;
        }
      }
    }
    if (if_inplace) {
      inplace_map["X"] = "Out";
    }

    CreateGradOpNode("py_layer", ins, outs, {{}}, place, inplace_map,
                     py_layer_ctx);
  } else {
    VLOG(3) << "No Grad to track for Op: py_layer_op";
  }

  return result_forward;
}

}  // namespace imperative
}  // namespace paddle
