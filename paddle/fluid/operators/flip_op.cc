/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/flip_op.h"

#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/op_version_registry.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class FlipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("X"), true,
        platform::errors::NotFound("Input(X) of FlipOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of FlipOp should not be null."));
    auto x_dims = ctx->GetInputDim("X");
    auto flip_dims = ctx->Attrs().Get<std::vector<int>>("axis");
    size_t flip_dims_size = flip_dims.size();

    if (flip_dims_size > 0) {
      // check if dims axis within range
      auto min_max_d = std::minmax_element(flip_dims.begin(), flip_dims.end());
      PADDLE_ENFORCE_LT(
          *min_max_d.first, x_dims.size(),
          platform::errors::InvalidArgument(
              "min(axes) should be less than the input tensor X's "
              "axes of FlipOp. But received min(axes) = %d,  "
              "X's axes = %d, X's shape = [%s]",
              *min_max_d.first, x_dims.size(), x_dims));
      PADDLE_ENFORCE_GE(*min_max_d.first, x_dims.size() * -1,
                        platform::errors::InvalidArgument(
                            "min(axes) should be greater than or equal to the "
                            "input tensor X's "
                            "axes of FlipOp times -1. But received "
                            "min(axes) = %d,  X's "
                            "axes = %d, X's shape = [%s]",
                            *min_max_d.first, x_dims.size() * -1, x_dims));
      PADDLE_ENFORCE_LT(
          *min_max_d.second, x_dims.size(),
          platform::errors::InvalidArgument(
              "max(axes) should be less than the input tensor X's "
              "axes of FlipOp. But received max(axes) = %d,  "
              "X's axes = %d, X's shape = [%s]",
              *min_max_d.second, x_dims.size(), x_dims));
      PADDLE_ENFORCE_GE(*min_max_d.second, x_dims.size() * -1,
                        platform::errors::InvalidArgument(
                            "max(axes) should be greater than or equal to the "
                            "input tensor X's "
                            "axes of FlipOp times -1. But received "
                            "max(axes) = %d,  X's "
                            "axes = %d, X's shape = [%s]",
                            *min_max_d.second, x_dims.size() * -1, x_dims));

      // check duplicates in dims
      flip_dims.erase(std::unique(flip_dims.begin(), flip_dims.end()),
                      flip_dims.end());
      PADDLE_ENFORCE_EQ(flip_dims.size(), flip_dims_size,
                        platform::errors::InvalidArgument(
                            "axes has duplicates, original flip axes size=%d, "
                            "but unique flip axes size=%d.)",
                            flip_dims_size, flip_dims.size()));
    }

    VLOG(3) << "flip operator x.shape=" << x_dims;

    std::vector<int64_t> output_dims(x_dims.size());
    for (int i = 0; i < x_dims.size(); ++i) {
      output_dims[i] = x_dims[i];
    }
    ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
    ctx->ShareLoD("X", "Out");
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    framework::LibraryType library = framework::LibraryType::kPlain;
    framework::DataLayout layout = framework::DataLayout::kAnyLayout;
    int customized_type_value =
        framework::OpKernelType::kDefaultCustomizedTypeValue;
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(input_data_type, ctx.GetPlace(), layout,
                                   library, customized_type_value);
  }
};

class FlipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of flip op.");
    AddOutput("Out", "(Tensor), The output tensor of flip op.");
    AddAttr<std::vector<int>>("axis", "The axes to flip on.");
    AddComment(R"DOC(
          Flip Operator.
          Reverse the order of a n-D tensor along given axis in axes.
      )DOC");
  }
};

class FlipOpInferVarType : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string>& GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Out"}};
    return m;
  }
};

template <typename T>
class FlipOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("flip");
    retv->SetInput("X", this->OutputGrad("Out"));
    retv->SetOutput("Out", this->InputGrad("X"));
    retv->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(flip, ops::FlipOp, ops::FlipOpMaker, ops::FlipOpInferVarType,
                  ops::FlipOpGradMaker<paddle::framework::OpDesc>,
                  ops::FlipOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(
    flip, ops::FlipKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FlipKernel<paddle::platform::CPUDeviceContext, double>,
    ops::FlipKernel<paddle::platform::CPUDeviceContext, int32_t>,
    ops::FlipKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::FlipKernel<paddle::platform::CPUDeviceContext, bool>);

/* ==========================  register checkpoint ===========================*/
REGISTER_OP_VERSION(flip).AddCheckpoint(
    R"ROC(Upgrade flip, add new attr [axis] and delete attr [dims].)ROC",
    paddle::framework::compatible::OpVersionDesc()
        .NewAttr("axis", "The added attr 'axis' doesn't set default value.",
                 boost::none)
        .DeleteAttr("dims", "The attr 'dims' is deleted."));
