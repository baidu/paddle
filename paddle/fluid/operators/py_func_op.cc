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
          "py::cast to LoDTensor error. The %d-th output expection is "
          "LoDTensor",
          i));
    }
  }
}

class PyFuncOpVarTypeInference : public framework::StaticGraphVarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    bool has_out = ctx->HasOutput("Out");
    bool has_in = ctx->HasInput("X");

    /**
     * X or Out can be empty, so that py_func can be more flexible
     * to support Python functions with no input or no output
     */
    PADDLE_ENFORCE_EQ(
        has_in || has_out, true,
        platform::errors::InvalidArgument("Input(X) or Output(Out) must exist, "
                                          "but has_in is %d, has_out is %d.",
                                          has_in, has_out));

    PADDLE_ENFORCE_GE(
        BOOST_GET_CONST(int, ctx->GetAttr(kForwardPythonCallableId)), 0,
        platform::errors::InvalidArgument(
            "Function id cannot be less than 0, but received value is %d.",
            BOOST_GET_CONST(int, ctx->GetAttr(kForwardPythonCallableId))));

    if (!has_out) return;

    /**
     * Traverse all outputs, check if name of any output ends with @GRAD.
     * If found, set its shape, dtype, lod_level, type to be the same as
     * the corresponding forward variable
     */
    const std::string kGradVarSuffix = framework::kGradVarSuffix;
    auto &out_var_names = Output(ctx, "Out");
    for (auto &out_var_name : out_var_names) {
      if (out_var_name == framework::kEmptyVarName ||
          out_var_name.size() < kGradVarSuffix.size()) {
        continue;
      }

      size_t len = out_var_name.size() - kGradVarSuffix.size();
      if (out_var_name.substr(len) == kGradVarSuffix) {
        auto fwd_var_name = out_var_name.substr(0, len);
        OP_INOUT_CHECK(HasVar(ctx, out_var_name), "Var", out_var_name,
                       "py_func");
        OP_INOUT_CHECK(HasVar(ctx, fwd_var_name), "Var", fwd_var_name,
                       "py_func");
        VLOG(10) << "Infer var_desc of Output(" << out_var_name << ") as Input("
                 << fwd_var_name << ")";

        SetShape(ctx, out_var_name, GetShape(ctx, fwd_var_name));
        SetDataType(ctx, out_var_name, GetDataType(ctx, fwd_var_name));
        SetLoDLevel(ctx, out_var_name, GetLoDLevel(ctx, fwd_var_name));
        SetType(ctx, out_var_name, GetType(ctx, fwd_var_name));
      }
    }
  }
};

class PyFuncOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE_EQ(!ctx->IsRuntime(), true,
                      platform::errors::InvalidArgument(
                          "Infer shape cannot be called in runtime."));
  }
};

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
class PyFuncOpGradDescMaker : public framework::GradOpDescMakerBase {
 private:
  static std::string DebugString(const std::vector<std::string> &strs) {
    if (strs.empty()) return "";
    std::string ret = strs[0];
    for (size_t i = 1; i < strs.size(); ++i) {
      ret += " ";
      ret += strs[i];
    }
    return ret;
  }

 public:
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override {
    auto &fwd_attrs = Attrs();
    // no backward op when backward_id is less than 0
    if (BOOST_GET_CONST(int, fwd_attrs.at(kBackwardPythonCallableId)) < 0) {
      return {};
    }

    std::unique_ptr<framework::OpDesc> grad_op(new framework::OpDesc());
    grad_op->SetType("py_func");

    framework::AttributeMap bwd_attrs;
    bwd_attrs[kForwardPythonCallableId] =
        fwd_attrs.at(kBackwardPythonCallableId);
    bwd_attrs[kBackwardPythonCallableId] = -1;
    grad_op->SetAttrMap(bwd_attrs);

    // All forward inputs
    auto fwd_ins = Input("X");
    // All forward outputs
    auto fwd_outs = Output("Out");

    // For memory reused, some inputs/output in forward part may be not needed
    // in backward part. Skipping these vars helps to save memory
    auto &backward_skip_var_list = BOOST_GET_CONST(
        std::vector<std::string>, fwd_attrs.at(kPyFuncBackwardSkipVars));
    std::unordered_set<std::string> backward_skip_var_set(
        backward_skip_var_list.begin(), backward_skip_var_list.end());
    std::vector<std::string> bwd_ins;
    bwd_ins.reserve(fwd_ins.size() + fwd_outs.size());
    for (auto &fwd_in : fwd_ins) {
      if (backward_skip_var_set.count(fwd_in) == 0) {
        bwd_ins.emplace_back(fwd_in);
      }
    }

    for (auto &fwd_out : fwd_outs) {
      if (backward_skip_var_set.count(fwd_out) == 0) {
        bwd_ins.emplace_back(fwd_out);
      }
    }

    // Backward OG cannot be skipped
    // But in Python side, if OG is kEmptyVarName, input tensor would be None
    auto fwd_out_grads = OutputGrad("Out");
    bwd_ins.reserve(bwd_ins.size() + fwd_out_grads.size());
    bwd_ins.insert(bwd_ins.end(), fwd_out_grads.begin(), fwd_out_grads.end());

    // Backward IG cannot be skipped
    // But in Python side, if IG is not needed, users can just return None
    auto bwd_outs = InputGrad("X", false);

    VLOG(10) << "PyFunc Grad Input: " << DebugString(bwd_ins);
    VLOG(10) << "PyFunc Grad Output: " << DebugString(bwd_outs);

    grad_op->SetInput("X", bwd_ins);
    grad_op->SetOutput("Out", bwd_outs);

    std::vector<std::unique_ptr<framework::OpDesc>> ret(1);
    ret[0] = std::move(grad_op);
    return ret;
  }
};

class PyFuncOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &in_arg_names = Inputs("X");
    auto &out_arg_names = Outputs("Out");

    std::vector<framework::LoDTensor> inputs(in_arg_names.size());
    for (size_t i = 0; i < in_arg_names.size(); ++i) {
      auto in_var = scope.FindVar(in_arg_names[i]);
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

    std::vector<framework::LoDTensor *> outputs(out_arg_names.size());
    for (size_t i = 0; i < out_arg_names.size(); ++i) {
      auto *out_var = scope.FindVar(out_arg_names[i]);
      outputs[i] =
          out_var ? out_var->GetMutable<framework::LoDTensor>() : nullptr;
    }

    auto callable_id = static_cast<size_t>(Attr<int>(kForwardPythonCallableId));
    auto *py_callable = GetPythonCallableObject(callable_id);
    VLOG(10) << "Call Python function with id " << callable_id << ": "
             << PythonFuncDebugString(*py_callable);
    CallPythonFunc(py_callable, inputs, &outputs);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(py_func, ops::PyFuncOp, ops::PyFuncOpMaker,
                  ops::PyFuncOpVarTypeInference, ops::PyFuncOpShapeInference,
                  ops::PyFuncOpGradDescMaker);
