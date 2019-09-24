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

#include "paddle/fluid/operators/elementwise/elementwise_max_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

class ElementwiseMaxOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Max"; }
  std::string GetEquation() const override { return "Out = max(X, Y)"; }

  void AddInputX() override {
    AddInput(
        "X",
        "(Variable), The first tensor holding the elements to be compared.");
  }

  void AddInputY() override {
    AddInput(
        "Y",
        "(Variable), The second tensor holding the elements to be compared.");
  }

  std::string GetOpFuntionality() const override {
    return "Compare two tensors and returns a new tensor containing the "
           "element-wise maxima.";
  }

  void AddOpComment() override {
    std::string doc = string::Sprintf(R"DOC(
%s

Examples:

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        def gen_data():
            return {
                "x": np.array([2, 3, 4]),
                "y": np.array([1, 5, 2])
            }
        x = fluid.layers.data(name="x", shape=[3], dtype='float32')
        y = fluid.layers.data(name="y", shape=[3], dtype='float32')
        z = fluid.layers.elementwise_max(x, y)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value) #[2, 5, 4]

    .. code-block:: python

        import paddle.fluid as fluid
        import numpy as np
        def gen_data():
            return {
                "x": np.ones((2, 3, 4, 5)).astype('float32'),
                "y": np.zeros((3, 4)).astype('float32')
            }
        x = fluid.layers.data(name="x", shape=[2,3,4,5], dtype='float32')
        y = fluid.layers.data(name="y", shape=[3,4], dtype='float32')
        z = fluid.layers.elementwise_max(x, y, axis=1)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        z_value = exe.run(feed=gen_data(),
                            fetch_list=[z.name])
        print(z_value)#[[[[1., 1., 1., 1., 1.] .... [1., 1., 1., 1., 1.]]]]

)DOC",
                                      GetCommentExamples());

    AddComment(doc);
  };
};

class ElementwiseMaxGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_max_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_max, ops::ElementwiseOp,
                  ops::ElementwiseMaxOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMaxGradOpDescMaker);

REGISTER_OPERATOR(elementwise_max_grad, ops::ElementwiseOpGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_max,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMaxKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_max_grad,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseMaxGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
