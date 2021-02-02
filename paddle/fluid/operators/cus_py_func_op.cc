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

#include "paddle/fluid/operators/cus_py_func_op.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
namespace paddle {
namespace operators {

namespace py = ::pybind11;

static std::vector<py::object> g_py_callables;

const char kContextId[] = "context_id";

size_t CusPyFunc_AppendPythonContext(const py::object &py_obj) {
  g_py_callables.emplace_back(py_obj);
  return g_py_callables.size() - 1;
}

void RunPyFunc(py::object *py_function,
               const std::vector<framework::Variable *> &ins,
               std::vector<framework::Variable *> *outs) {
  py::gil_scoped_acquire guard;
  if (PyCallable_Check(py_function->ptr())) {
    // throw error mmsg,  'py_function' is not Callable
  }

  py::tuple inputs(ins.size());
  for (size_t i = 0; i < ins.size(); i++) {
    auto in_var = ins[i];
    if (in_var == nullptr) {
      continue;
    }
    // auto
    // name=imperative::GetCurrentTracer()->GenerateUniqueName("generated_tensor");
    char name[50] = {};
    // can be same name?
    snprintf(name, sizeof(name), "generator_custom_py_func%d@@grad \n",
             static_cast<int>(i));
    imperative::VarBase temp_varbase(false, name);

    if (!temp_varbase.MutableVar()->IsInitialized()) {
      // temp_varbase.MutableVar()->SharePlaceholderWith(*in_var);
      framework::CopyVariable(*in_var, temp_varbase.MutableVar());
    }
    inputs[i] = py::cast(temp_varbase);
  }

  auto py_result = (*py_function)(*inputs);

  if (PyTuple_Check(py_result.ptr()) || PyList_Check(py_result.ptr())) {
    auto result_tuple = py_result.cast<py::tuple>();
    for (size_t i = 0; i < result_tuple.size(); i++) {
      // !!! :Commented out the copy constructor of
      // `paddle::imperative::VarBase`
      auto result_var = result_tuple[i].cast<imperative::VarBase *>();
      // (*outs)[i]->SharePlaceholderWith(*(result_var->MutableVar()));
      framework::CopyVariable(*(result_var->MutableVar()), (*outs)[i]);
    }
  } else {
    try {
      auto result_var = py_result.cast<imperative::VarBase *>();
      // (*outs)[0]->SharePlaceholderWith(*(result_var->MutableVar()));
      framework::CopyVariable(*(result_var->MutableVar()), (*outs)[0]);
    } catch (py::cast_error &) {
    }
  }
}

class CusPyFuncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Inputs of CusPyFunc op.").AsDuplicable();
    AddOutput("Out", "Outputs of CusPyFunc op").AsDuplicable();

    AddAttr<int>(kContextId, "Index of registered backward Python context.")
        .SetDefault(-1);

    AddComment(R"DOC("CusPyFunc Op")DOC");
  }
};

template <typename T>
class CusPyFuncGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetAttrMap(this->Attrs());

    grad_op->SetType("cus_py_func");

    // All forward inputs
    auto fwd_ins = this->Input("X");
    // All forward outputs
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
};

class CusPyFuncOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    // todo:check
  }

 protected:
  /* see [Why use single type kernel] */
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    // todo:float32 ?
    return framework::OpKernelType(framework::proto::VarType::FP32,
                                   ctx.GetPlace());
  }
};

template <typename DeviceContext, typename T>
class CusPyFuncOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &input_vars = ctx.MultiInputVar("X");

    auto output_vars = ctx.MultiOutputVar("Out");

    auto callable_id = static_cast<size_t>(ctx.Attr<int>(kContextId));
    auto *py_function = &g_py_callables[callable_id];
    RunPyFunc(py_function, input_vars, &output_vars);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    cus_py_func, ops::CusPyFuncOp, ops::CusPyFuncOpMaker,
    // ops::PyFuncOpVarTypeInference, //ops::PyFuncOpShapeInference,
    ops::CusPyFuncGradOpMaker<paddle::framework::OpDesc>,
    ops::CusPyFuncGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    cus_py_func,
    ops::CusPyFuncOpKernel<paddle::platform::CPUDeviceContext, float>);
#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    cus_py_func,
    ops::CusPyFuncOpKernel<paddle::platform::CUDADeviceContext, float>);
#endif  // PADDLE_WITH_CUDA
