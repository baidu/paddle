/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#include "paddle/fluid/operators/requantize_op.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace operators {

framework::OpKernelType ReQuantOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  framework::LibraryType library_ = framework::LibraryType::kMKLDNN;
  framework::DataLayout layout_ = framework::DataLayout::kMKLDNN;

  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "Input"), ctx.GetPlace(),
      layout_, library_);
}

void ReQuantOpMaker::Make() {
  AddInput("Input", "Input data");
  AddOutput("Output", "Output data");
  AddAttr<float>("Scale_in", "Scale in data").SetDefault({1.0f});
  AddAttr<float>("Scale_out", "Scale out data").SetDefault({1.0f});
  AddAttr<float>("Shift_in", "Shift in data").SetDefault({1.0f});
  AddAttr<float>("Shift_out", "Shift out data").SetDefault({1.0f});
  AddComment(
      R"DOC(This op will re-quantize data from INT8 with scale_in to INT8 with scale_out)DOC");
}

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(requantize, ops::ReQuantOp, ops::ReQuantOpMaker);
