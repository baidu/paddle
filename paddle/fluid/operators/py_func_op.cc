// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/py_func_op.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

namespace py = ::pybind11;

static std::vector<py::object> g_py_callables;

const char kForwardPythonCallableId[] = "forward_callable_id";
const char kBackwardPythonCallableId[] = "backward_callable_id";
const char kPyFuncBackwardSkipVars[] = "backward_skip_vars";

size_t AppendPythonCallableObjectAndReturnId(const py::object &py_obj) {
  g_py_callables.emplace_back(py_obj);
  return g_py_callables.size() - 1;
}

// Return py::object* instead of py::object
// Returning py::object would cause reference count increasing
// but without GIL, reference count in Python may not be safe
static py::object *GetPythonCallableObject(size_t i) {
  PADDLE_ENFORCE_LT(
      i, g_py_callables.size(),
      platform::errors::InvalidArgument(
          "Invalid python callable id %d, which should be less than %d.", i,
          g_py_callables.size()));
  return &g_py_callables[i];
}

static std::string PythonFuncDebugString(const py::object &py_callable) {
  py::gil_scoped_acquire guard;
  std::string wrapper_func_str = py::str(py_callable);
  auto inner_func = py_callable.attr("_func");
  std::string inner_func_str = py::str(inner_func);
  return inner_func_str + " wrapped by " + wrapper_func_str;
}

static void CallPythonFunc(py::object *callable,
                           const std::vector<framework::LoDTensor> &ins,
                           std::vector<framework::LoDTensor *> *outs) {
  py::gil_scoped_acquire guard;
  py::tuple in_args(ins.size());
  for (size_t i = 0; i < ins.size(); ++i) {
    in_args[i] = ins[i].IsInitialized() ? py::cast(ins[i]) : py::cast(nullptr);
  }

  auto ret = (*callable)(*in_args);
  auto ret_tuple = py::cast<py::tuple>(ret);
  size_t ret_num = py::len(ret_tuple);
  size_t out_num = outs->size();
  if (UNLIKELY(ret_num != out_num)) {
    // Python function has no return values or returns None
    // In this case, ret_num = 1 && ret[0] == None && out_num should be 0
    // Otherwise, ret_num must be equal to out_num
    PADDLE_ENFORCE_EQ(ret_num == 1, true,
                      platform::errors::InvalidArgument(
                          "Python function has no return values or returns "
                          "None. In this case, ret_num = 1 && ret[0] == None "
                          "&& out_num should be 0. But ret_num is %d",
                          ret_num));

    PADDLE_ENFORCE_EQ(
        out_num == 0, true,
        platform::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But out_num is %d",
            out_num));

    PADDLE_ENFORCE_EQ(
        py::cast<framework::LoDTensor *>(ret_tuple[0]) == nullptr, true,
        platform::errors::InvalidArgument(
            "Python function has no return values or returns None. In "
            "this case, ret_num = 1 && ret[0] == None && out_num should "
            "be 0. But ret[0] is not None"));
  }

  for (size_t i = 0; i < out_num; ++i) {
    auto *out = (*outs)[i];
    if (out == nullptr) {
      continue;
    }
    try {
      auto *py_out_tensor = py::cast<framework::LoDTensor *>(ret_tuple[i]);
      PADDLE_ENFORCE_NOT_NULL(py_out_tensor,
                              platform::errors::InvalidArgument(
                                  "Output tensor %d should not be nullptr", i));
      out->set_lod(py_out_tensor->lod());
      out->ShareDataWith(*py_out_tensor);
    } catch (py::cast_error &) {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "The %d-th output must be LoDTensor.", i));
    }
  }
}

class PyFuncOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Inputs of py_func op.").AsDuplicable();
    AddOutput("Out", "Outputs of py_func op").AsDuplicable();
    AddAttr<int>(kForwardPythonCallableId,
                 "Index of registered forward Python function.")
        .SetDefault(0);
    AddAttr<int>(kBackwardPythonCallableId,
                 "Index of registered backward Python function.")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>(kPyFuncBackwardSkipVars,
                                      "Unused forward in/out in backward op")
        .SetDefault(std::vector<std::string>());
    AddComment(R"DOC("PyFunc Op")DOC");
  }
};

/**
 * There are several benefits when backward op of py_func op is
 * still py_func op.
 *
 *  - Less codes are needed, since codes of backward is almost
 *    the same as forward.
 *
 *  - To support high order derivative, so that py_func is
 *    infinite-order differentiable
 */
// class PyFuncOpGradDescMaker : public framework::GradOpDescMakerBase {
//  private:
//   static std::string DebugString(const std::vector<std::string> &strs) {
//     if (strs.empty()) return "";
//     std::string ret = strs[0];
//     for (size_t i = 1; i < strs.size(); ++i) {
//       ret += " ";
//       ret += strs[i];
//     }
//     return ret;
//   }
template <typename T>
class SkipVar;

// inline void set_skip_var(IOT dist_vector,IOT source_vector,SKIPT skip_set);
template <>
class SkipVar<paddle::framework::OpDesc> {
 public:
  template <typename TD, typename TS, typename TSKIP>
  static void set_skip_var(TD *dist_vector, const TS &source_vector,
                           const TSKIP &skip_set) {
    for (auto &vector : source_vector) {
      if (skip_set.count(vector) == 0) {
        dist_vector->emplace_back(vector);
      }
    }
  }
};

template <>
class SkipVar<paddle::imperative::OpBase> {
 public:
  template <typename TD, typename TS, typename TSKIP>
  static void set_skip_var(TD *dist_vector, const TS &source_vector,
                           const TSKIP &skip_set) {
    for (auto &vector : source_vector) {
      // if (skip_set.count(vector) == 0) {
      dist_vector->emplace_back(vector);
      // }
    }
  }
};
template <typename T>
class PyFuncGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    auto &fwd_attrs = this->Attrs();
    // no backward op when backward_id is less than 0
    if (BOOST_GET_CONST(int, fwd_attrs.at(kBackwardPythonCallableId)) < 0) {
      // return {};
      return;
    }
    grad_op->SetType("py_func");
    framework::AttributeMap bwd_attrs;
    bwd_attrs[kForwardPythonCallableId] =
        fwd_attrs.at(kBackwardPythonCallableId);
    bwd_attrs[kBackwardPythonCallableId] = -1;
    grad_op->SetAttrMap(bwd_attrs);

    // All forward inputs
    auto fwd_ins = this->Input("X");
    // All forward outputs
    auto fwd_outs = this->Output("Out");

    // For memory reused, some inputs/output in forward part may be not needed
    // in backward part. Skipping these vars helps to save memory
    auto &backward_skip_var_list = BOOST_GET_CONST(
        std::vector<std::string>, fwd_attrs.at(kPyFuncBackwardSkipVars));
    std::unordered_set<std::string> backward_skip_var_set(
        backward_skip_var_list.begin(), backward_skip_var_list.end());
    // Backward OG cannot be skipped
    // But in Python side, if OG is kEmptyVarName, input tensor would be None
    auto fwd_out_grads = this->OutputGrad("Out");
    using return_type = decltype(fwd_out_grads);
    return_type bwd_ins;
    bwd_ins.reserve(fwd_ins.size() + fwd_outs.size());

    SkipVar<T>::set_skip_var(&bwd_ins, fwd_ins, backward_skip_var_set);
    SkipVar<T>::set_skip_var(&bwd_ins, fwd_outs, backward_skip_var_set);

    bwd_ins.reserve(bwd_ins.size() + fwd_out_grads.size());
    bwd_ins.insert(bwd_ins.end(), fwd_out_grads.begin(), fwd_out_grads.end());

    // Backward IG cannot be skipped
    // But in Python side, if IG is not needed, users can just return None
    auto bwd_outs = this->InputGrad("X", false);

    grad_op->SetInput("X", bwd_ins);
    grad_op->SetOutput("Out", bwd_outs);
  }
};

class PyFuncOp : public framework::OperatorWithKernel {
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
class PyFuncOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &input_vars = ctx.MultiInputVar("X");

    auto output_vars = ctx.MultiOutputVar("Out");

    auto input_var_names = ctx.InputNames("X");
    auto output_var_names = ctx.OutputNames("Out");
    // auto &in_arg_names = Inputs("X");
    // auto &out_arg_names = Outputs("Out");

    std::vector<framework::LoDTensor> inputs(input_var_names.size());
    for (size_t i = 0; i < input_var_names.size(); ++i) {
      auto in_var = input_vars[i];
      // When py_func op is called in backward, in_var may be null
      if (in_var == nullptr) {
        continue;
      }
      auto &in_tensor = in_var->Get<framework::LoDTensor>();
      if (!in_tensor.IsInitialized()) {
        continue;
      }
      if (platform::is_gpu_place(in_tensor.place())) {
        framework::TensorCopySync(in_tensor, platform::CPUPlace(), &inputs[i]);
      } else {
        inputs[i].ShareDataWith(in_tensor);
      }
      inputs[i].set_lod(in_tensor.lod());
    }

    std::vector<framework::LoDTensor *> outputs(output_var_names.size());
    for (size_t i = 0; i < output_var_names.size(); ++i) {
      auto *out_var = output_vars[i];
      outputs[i] =
          out_var ? out_var->GetMutable<framework::LoDTensor>() : nullptr;
    }

    auto callable_id =
        static_cast<size_t>(ctx.Attr<int>(kForwardPythonCallableId));
    auto *py_callable = GetPythonCallableObject(callable_id);
    VLOG(10) << "Call Python function with id " << callable_id << ": "
             << PythonFuncDebugString(*py_callable);
    CallPythonFunc(py_callable, inputs, &outputs);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    py_func, ops::PyFuncOp, ops::PyFuncOpMaker,
    // ops::PyFuncOpVarTypeInference, //ops::PyFuncOpShapeInference,
    ops::PyFuncGradOpMaker<paddle::framework::OpDesc>,
    ops::PyFuncGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    py_func, ops::PyFuncOpKernel<paddle::platform::CPUDeviceContext, float>);
#ifdef PADDLE_WITH_CUDA
REGISTER_OP_CUDA_KERNEL(
    py_func, ops::PyFuncOpKernel<paddle::platform::CUDADeviceContext, float>);
#endif  // PADDLE_WITH_CUDA
