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

#include <vector>

#include "paddle/fluid/operators/py_layer_op.h"

namespace paddle {
namespace operators {

namespace py = ::pybind11;

void RunPyObject(py::object *py_object,
                 const std::vector<framework::Variable *> &ins,
                 std::vector<framework::Variable *> *outs) {
  py::gil_scoped_acquire guard;

  auto py_function = py_object->attr("backward");

  py::tuple inputs(ins.size());
  for (size_t i = 0; i < ins.size(); i++) {
    auto in_var = ins[i];

    if (in_var != nullptr) {
      char name[50] = {};
      snprintf(name, sizeof(name), "generator_custom_py_layer%d@@grad \n",
               static_cast<int>(i));

      std::shared_ptr<imperative::VariableWrapper> temp_wrap =
          std::make_shared<imperative::VariableWrapper>(name, *in_var);
      temp_wrap->InnerSetOverridedStopGradient(true);
      std::shared_ptr<imperative::VarBase> temp_varbase =
          std::make_shared<imperative::VarBase>(temp_wrap);
      try {
        inputs[i] = py::cast(temp_varbase).ptr();
      } catch (py::cast_error &) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The output of backward should be `Tensor`."));
      }
    }
  }

  auto py_result = py_function(*py_object, *inputs);

  if (PyTuple_Check(py_result.ptr()) || PyList_Check(py_result.ptr())) {
    auto result_tuple = py_result.cast<py::tuple>();
    if (result_tuple.size() != outs->size()) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The number of outputs of backward should be %d, but received %d.",
          outs->size(), result_tuple.size()));
    }
    for (size_t i = 0; i < result_tuple.size(); i++) {
      if (Py_None != result_tuple[i].ptr()) {
        try {
          auto result_var = result_tuple[i].cast<imperative::VarBase *>();
          *(*outs)[i] = result_var->Var();
        } catch (py::cast_error &) {
          PADDLE_THROW(platform::errors::Unimplemented(
              "The output of backward should be `Tensor`."));
        }
      } else {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The output of backward can not be `None`."));
      }
    }
  } else {
    if (Py_None != py_result.ptr()) {
      try {
        auto result_var = py_result.cast<imperative::VarBase *>();
        *((*outs)[0]) = result_var->Var();
      } catch (py::cast_error &) {
        PADDLE_THROW(platform::errors::Unimplemented(
            "The output of backward should be `Tensor`."));
      }
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "The output of backward can not be `None`."));
    }
  }
}

void PyLayerGradOpMaker<paddle::imperative::OpBase>::Apply(
    GradOpPtr<paddle::imperative::OpBase> grad_op) const {
  grad_op->SetType("py_layer");
  auto &inner_op = grad_op->InnerOp();
  auto py_layer_op_const = dynamic_cast<const PyLayerOp *>(&inner_op);

  if (py_layer_op_const) {
    auto py_layer_op = const_cast<PyLayerOp *>(py_layer_op_const);
    py_layer_op->GetMutablePyLayerContext() = py_context;

  } else {
    PADDLE_THROW(platform::errors::Fatal(
        "PyLayerGradOpMaker can't cast %s to PyLayerOp*.",
        typeid(&inner_op).name()));
  }

  auto fwd_ins = this->Input("X");
  auto fwd_outs = this->Output("Out");

  auto fwd_out_grads = this->OutputGrad("Out");
  using return_type = decltype(fwd_out_grads);
  return_type bwd_ins;
  bwd_ins.reserve(fwd_ins.size() + fwd_outs.size());
  for (auto var : fwd_ins) {
    bwd_ins.emplace_back(var);
  }
  for (auto var : fwd_outs) {
    bwd_ins.emplace_back(var);
  }

  bwd_ins.reserve(bwd_ins.size() + fwd_out_grads.size());
  bwd_ins.insert(bwd_ins.end(), fwd_out_grads.begin(), fwd_out_grads.end());

  auto bwd_outs = this->InputGrad("X", false);

  grad_op->SetInput("X", bwd_ins);
  grad_op->SetOutput("Out", bwd_outs);
}

class PyLayerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Inputs of PyLayer op.").AsDuplicable();
    AddOutput("Out", "Outputs of PyLayer op").AsDuplicable();
    AddComment(R"DOC("PyLayer Op")DOC");
  }
};

template <typename DeviceContext, typename T>
class PyLayerOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &op_ = ctx.GetOp();
    auto pylayer_op = dynamic_cast<const PyLayerOp *>(&op_);
    if (pylayer_op) {
      auto py_layer_context = pylayer_op->GetMutablePyLayerContext();
      py::object bk_ctx(py::handle(py_layer_context->GetMatableCtx()), true);
      auto &input_vars = ctx.MultiInputVar("X");
      auto output_vars = ctx.MultiOutputVar("Out");
      RunPyObject(&bk_ctx, input_vars, &output_vars);

    } else {
      PADDLE_THROW(platform::errors::Fatal(
          "PyLayerOpKernel can't cast %s to PyLayer*.", typeid(&op_).name()));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(py_layer, ops::PyLayerOp, ops::PyLayerOpMaker,
                  ops::PyLayerGradOpMaker<paddle::imperative::OpBase>,
                  ops::PyLayerGradOpMaker<paddle::framework::OpDesc>);

REGISTER_OP_CPU_KERNEL(
    py_layer, ops::PyLayerOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::PyLayerOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::PyLayerOpKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex64>,
    ops::PyLayerOpKernel<paddle::platform::CPUDeviceContext,
                         paddle::platform::complex128>);
#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    py_layer, ops::PyLayerOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::PyLayerOpKernel<paddle::platform::CUDADeviceContext, double>,
    ops::PyLayerOpKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex64>,
    ops::PyLayerOpKernel<paddle::platform::CUDADeviceContext,
                         paddle::platform::complex128>);
#endif  // PADDLE_WITH_CUDA
