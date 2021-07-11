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

#include "paddle/fluid/operators/collective/c_fusion_allreduce.h"

namespace ops = paddle::operators;
namespace plat = paddle::platform;

namespace paddle {
namespace operators {

class CFusionAllReduceSumOpMaker : public CFusionAllReduceOpMaker {
 protected:
  std::string GetName() const override { return "c_fusion_allreduce_sum"; }
};

DECLARE_INPLACE_OP_INFERER(FusionAllreduceSumInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(c_fusion_allreduce_sum, paddle::operators::CFusionAllReduceOp,
    paddle::operators::CFusionAllReduceSumOpMaker, ops::FusionAllreduceSumInplaceInferer);

#if defined(PADDLE_WITH_ASCEND_CL)
REGISTER_OP_NPU_KERNEL(
    c_fusion_allreduce_sum, ops::CFusionAllReduceOpASCENDKernel<ops::kRedSum, int>,
    ops::CFusionAllReduceOpASCENDKernel<ops::kRedSum, int8_t>,
    ops::CFusionAllReduceOpASCENDKernel<ops::kRedSum, float>,
    ops::CFusionAllReduceOpASCENDKernel<ops::kRedSum, plat::float16>)
#endif



