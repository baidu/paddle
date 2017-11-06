/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/op_registry.h"

namespace paddle {
namespace operators {
class ArrayOpBase : public framework::OperatorBase {
 public:
  ArrayOpBase(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {}

 protected:
  size_t GetOffset(const framework::Scope &scope,
                   const platform::DeviceContext &dev_ctx) const {
    auto *i = scope.FindVar(Input("I"));
    PADDLE_ENFORCE(i != nullptr, "I must be set");
    auto &i_tensor = i->Get<framework::LoDTensor>();
    PADDLE_ENFORCE_EQ(i_tensor.numel(), 1);
    size_t offset;
    if (platform::is_gpu_place(i_tensor.place())) {
      // FIXME: Avoid copy from GPU to CPU
      framework::Tensor t;
      t.CopyFrom(i_tensor, platform::CPUPlace(), dev_ctx);
      dev_ctx.Wait();
      offset = static_cast<size_t>(*t.data<int64_t>());
    } else {
      offset = static_cast<size_t>(*i_tensor.data<int64_t>());
    }
    return offset;
  }
};

class WriteToArrayOp : public ArrayOpBase {
 public:
  WriteToArrayOp(const std::string &type,
                 const framework::VariableNameMap &inputs,
                 const framework::VariableNameMap &outputs,
                 const framework::AttributeMap &attrs)
      : ArrayOpBase(type, inputs, outputs, attrs) {}

  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto *x = scope.FindVar(Input("X"));
    PADDLE_ENFORCE(x != nullptr, "X must be set");
    auto &x_tensor = x->Get<framework::LoDTensor>();
    size_t offset = GetOffset(scope, dev_ctx);
    auto *out =
        scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensorArray>();
    if (offset >= out->size()) {
      out->resize(offset + 1);
    }
    auto *out_tensor = &out->at(offset);
    out_tensor->CopyFrom(x_tensor, dev_ctx.GetPlace(), dev_ctx);
    out_tensor->set_lod(x_tensor.lod());
  }
};

class WriteToArrayOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  WriteToArrayOpProtoMaker(framework::OpProto *proto,
                           framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor) the tensor will be written to tensor array");
    AddInput(
        "I",
        "(Tensor) the subscript index in tensor array. The number of element "
        "should be 1");
    AddOutput("Out", "(TensorArray) the tensor array will be written");
    AddComment(R"DOC(Write a LoDTensor to a LoDTensor array.

Assume T is LoDTensor, i is the subscript of the array, and A is the array. The
equation is

A[i] = T
)DOC");
  }
};

class WriteToArrayInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    PADDLE_ENFORCE(context->HasInput("I"), "Must set the subscript index");
    PADDLE_ENFORCE_EQ(framework::product(context->GetInputDim("I")), 1,
                      "The number of element of subscript index must be 1");
    PADDLE_ENFORCE(context->HasInput("X"), NotHasXError());
    PADDLE_ENFORCE(context->HasOutput("Out"), NotHasOutError());
    context->SetOutputDim("Out", context->GetInputDim("X"));
  }

 protected:
  virtual const char *NotHasXError() const { return "Must set the lod tensor"; }

  virtual const char *NotHasOutError() const {
    return "Must set the lod tensor array";
  }
};

class WriteToArrayInferVarType : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDescBind &op_desc,
                  framework::BlockDescBind *block) const override {
    VLOG(10) << "I am here?";
    for (auto &out_var : op_desc.OutputArgumentNames()) {
      VLOG(10) << "Set Variable " << out_var << " as LOD_TENSOR_ARRAY";
      block->Var(out_var)->SetType(framework::VarDesc::LOD_TENSOR_ARRAY);
    }
  }
};

class ReadFromArrayOp : public ArrayOpBase {
 public:
  ReadFromArrayOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : ArrayOpBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    auto *x = scope.FindVar(Input("X"));
    PADDLE_ENFORCE(x != nullptr, "X must be set");
    auto &x_array = x->Get<framework::LoDTensorArray>();
    auto *out = scope.FindVar(Output("Out"));
    PADDLE_ENFORCE(out != nullptr, "Out must be set");
    auto *out_tesnor = out->GetMutable<framework::LoDTensor>();
    size_t offset = GetOffset(scope, dev_ctx);
    PADDLE_ENFORCE_LT(offset, x_array.size());
    out_tesnor->CopyFrom(x_array[offset], dev_ctx.GetPlace(), dev_ctx);
    out_tesnor->set_lod(x_array[offset].lod());
  }
};

class ReadFromArrayProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ReadFromArrayProtoMaker(framework::OpProto *proto,
                          framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(TensorArray) the array will be read from.");
    AddInput("I",
             "(Tensor) the subscript index in tensor array. The number of "
             "element should be 1");
    AddOutput("Out", "(LoDTensor) the tensor will be read from.");
    AddComment(R"DOC(Read a LoDTensor from a LoDTensor Array

Assume T is LoDTensor, i is th e subscript of the array, and A is the array. The
equation is

T = A[i]
)DOC");
  }
};

class ReadFromArrayInferShape : public WriteToArrayInferShape {
 protected:
  const char *NotHasXError() const override {
    return "The input array X must be set";
  }
  const char *NotHasOutError() const override {
    return "The output tensor out must be set";
  }
};

class WriteToArrayGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *grad_op = new framework::OpDescBind();
    grad_op->SetType("read_from_array");
    grad_op->SetInput("I", Input("I"));
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDescBind>(grad_op);
  }
};

class ReadFromArrayGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDescBind> Apply() const override {
    auto *grad_op = new framework::OpDescBind();
    grad_op->SetType("write_to_array");
    grad_op->SetInput("I", Input("I"));
    grad_op->SetInput("X", OutputGrad("Out"));
    grad_op->SetOutput("Out", InputGrad("X"));
    grad_op->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDescBind>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(write_to_array, ops::WriteToArrayOp,
                  ops::WriteToArrayInferShape, ops::WriteToArrayOpProtoMaker,
                  ops::WriteToArrayGradMaker, ops::WriteToArrayInferVarType);
REGISTER_OPERATOR(read_from_array, ops::ReadFromArrayOp,
                  ops::ReadFromArrayInferShape, ops::ReadFromArrayProtoMaker,
                  ops::ReadFromArrayGradMaker);
