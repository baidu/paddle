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

#include <string>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type_traits.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace framework {
class OpDesc;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

using LoDTensor = paddle::framework::LoDTensor;
using LoDTensorBlockingQueueHolder =
    paddle::operators::reader::LoDTensorBlockingQueueHolder;

namespace paddle {
namespace operators {

class EnqueueOp : public framework::OperatorBase {
 public:
  EnqueueOp(const std::string& type, const framework::VariableNameMap& inputs,
            const framework::VariableNameMap& outputs,
            const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const auto& queue_name = Attr<std::string>("queue_name");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE_NOT_NULL(
        queue_holder_var,
        platform::errors::NotFound(
            "No LoDTensorBlockingQueueHolder variable with name %s found.",
            queue_name));
    const std::string& var_name = Input("X");
    auto* in_var = scope.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        in_var, platform::errors::NotFound("No variable with name %s found.",
                                           var_name));
    auto* in_tensor = in_var->GetMutable<LoDTensor>();
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();

    std::vector<LoDTensor> lod_tensor_vec;
    lod_tensor_vec.emplace_back(*in_tensor);
    queue_holder->GetQueue()->Push(lod_tensor_vec);
  }
};

class EnqueueOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "`lod_tensor` to enqueue");
    AddAttr<std::string>("queue_name",
                         "Name of the `LoDTensorBlockingQueueHolder` variable");
    AddComment(R"DOC(
			Enqueue operator.
      )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = ::paddle::operators;

REGISTER_OP_WITHOUT_GRADIENT(enqueue, ops::EnqueueOp, ops::EnqueueOpMaker);
