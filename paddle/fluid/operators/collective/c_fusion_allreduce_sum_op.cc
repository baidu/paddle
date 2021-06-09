/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/collective/c_fusion_allreduce_op.h"

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

template <typename T>
class CFusionAllReduceSumOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    bool use_mp = BOOST_GET_CONST(bool, this->GetAttr("use_model_parallel"));
    if (use_mp) {
      retv->SetType("c_identity");
    } else {
      retv->SetType("c_allreduce_sum");
    }
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

class CFusionAllReduceSumOpMaker : public CFusionAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "Sum"; }
};

DECLARE_INPLACE_OP_INFERER(FusionAllreduceSumInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OPERATOR(c_fusion_allreduce_sum, ops::CFusionAllReduceOp,
                  ops::CFusionAllReduceSumOpGradMaker<paddle::framework::OpDesc>,
                  ops::CFusionAllReduceSumOpGradMaker<paddle::imperative::OpBase>,
                  ops::CFusionAllReduceSumOpMaker, ops::FusionAllreduceSumInplaceInferer);

REGISTER_OP_CPU_KERNEL(c_fusion_allreduce_sum,
                       ops::CFusionAllReduceOpCPUKernel<ops::kRedSum, float>,
                       ops::CFusionAllReduceOpCPUKernel<ops::kRedSum, double>,
                       ops::CFusionAllReduceOpCPUKernel<ops::kRedSum, int>,
                       ops::CFusionAllReduceOpCPUKernel<ops::kRedSum, int64_t>,
                       ops::CFusionAllReduceOpCPUKernel<ops::kRedSum, plat::float16>)
