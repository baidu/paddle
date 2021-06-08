// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <sstream>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_memory_aligment.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class CoalesceTensorOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_var_names = context.InputNames("Input");
    auto out_var_names = context.OutputNames("Output");
    auto &in_vars = context.MultiInputVar("Input");
    auto out_vars = context.MultiOutputVar("Output");

    PADDLE_ENFORCE_GT(in_var_names.size(), static_cast<size_t>(0),
                      platform::errors::InvalidArgument(
                          "The CoalesceTensor operator has no input."));
    PADDLE_ENFORCE_EQ(in_var_names.size(), out_var_names.size(),
                      platform::errors::InvalidArgument(
                          "The number of CoalesceTensor operator's input and "
                          "output is not match, "
                          "input number is %u, output number is %u.",
                          in_var_names.size(), out_var_names.size()));

    // Input & Output check: only support LoDTensor
    for (size_t i = 0; i < in_var_names.size(); ++i) {
      PADDLE_ENFORCE_NOT_NULL(
          in_vars[i],
          platform::errors::NotFound("The input variable %s of CoalesceTensor "
                                     "operator does not exist.",
                                     in_var_names[i]));
      PADDLE_ENFORCE_NOT_NULL(
          out_vars[i],
          platform::errors::NotFound("The output variable %s of CoalesceTensor "
                                     "operator does not exist.",
                                     out_var_names[i]));
      PADDLE_ENFORCE_EQ(in_vars[i]->IsType<framework::LoDTensor>(), true,
                        platform::errors::InvalidArgument(
                            "The input variable %s of CoalesceTensor operator "
                            "is not LoDTensor.",
                            in_var_names[i]));
      PADDLE_ENFORCE_EQ(out_vars[i]->IsType<framework::LoDTensor>(), true,
                        platform::errors::InvalidArgument(
                            "The output variable %s of CoalesceTensor operator "
                            "is not LoDTensor.",
                            out_var_names[i]));
    }

    auto in_tensors = context.MultiInput<framework::LoDTensor>("Input");
    bool use_align = context.Attr<bool>("use_align");

    if (context.Attr<bool>("check_name")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            in_var_names[i], out_var_names[i],
            platform::errors::InvalidArgument(
                "The input and output variable of CoalesceTensor operator is "
                "different, %dth input is %s, %dth output is %s.",
                i, in_var_names[i], i, out_var_names[i]));
      }
    } else {
      // Init the output as input
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        out_vars[i]->GetMutable<framework::LoDTensor>()->Resize(
            in_tensors[i]->dims());
      }
    }

    auto &dev_ctx = context.template device_context<DeviceContext>();

    // Get numel and dtype
    size_t numel = 0;
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    size_t size_of_dtype = framework::SizeOfType(dtype);
    GetMemSizeAndDtype(in_tensors, in_var_names, &numel, size_of_dtype,
                       context.GetPlace(), use_align);

    // Alloc the continuous space
    auto fused_tensor = context.Output<framework::LoDTensor>("FusedOutput");
    fused_tensor->Resize(framework::make_ddim({static_cast<int64_t>(numel)}))
        .mutable_data(context.GetPlace(), dtype);

    // Init the continuous space
    auto out_tensors = context.MultiOutput<framework::LoDTensor>("Output");
    size_t offset = 0;
    if (context.Attr<bool>("copy_data")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        auto len = static_cast<size_t>(in_tensors[i]->numel());
        auto sub_tensor = fused_tensor->Slice(
            static_cast<int64_t>(offset), static_cast<int64_t>(offset + len));
        framework::TensorCopy(*in_tensors[i], context.GetPlace(), dev_ctx,
                              &sub_tensor);

        offset += use_align ? platform::Alignment(len * size_of_dtype,
                                                  context.GetPlace()) /
                                  size_of_dtype
                            : len;
      }
    } else if (context.Attr<bool>("set_constant")) {
      math::SetConstant<DeviceContext, T> set_constant;
      set_constant(dev_ctx, fused_tensor,
                   static_cast<T>(context.Attr<float>("constant")));
    } else if (context.Attr<bool>("persist_output")) {
      for (size_t i = 0; i < out_var_names.size(); ++i) {
        auto len = static_cast<size_t>(out_tensors[i]->numel());
        auto sub_tensor = fused_tensor->Slice(
            static_cast<int64_t>(offset), static_cast<int64_t>(offset + len));
        // some var may not persistable, or persistable var may not init
        if (out_tensors[i]->IsInitialized()) {
          framework::TensorCopy(*out_tensors[i], context.GetPlace(), dev_ctx,
                                &sub_tensor);
        }
        offset += use_align ? platform::Alignment(len * size_of_dtype,
                                                  context.GetPlace()) /
                                  size_of_dtype
                            : len;
      }
    }

    // Make the outputs point to the continuous space.
    offset = 0;
    std::stringstream ss;
    ss << "alloc_space_for_vars: ";
    for (size_t i = 0; i < out_tensors.size(); ++i) {
      auto len = static_cast<size_t>(out_tensors[i]->numel());
      auto dim = out_tensors[i]->dims();
      out_tensors[i]
          ->ShareDataWith(fused_tensor->Slice(
              static_cast<int64_t>(offset), static_cast<int64_t>(offset + len)))
          .Resize(dim);
      len = use_align
                ? platform::Alignment(len * size_of_dtype, context.GetPlace()) /
                      size_of_dtype
                : len;
      offset += len;
      ss << "output(" << out_var_names[i] << ")  dim:(" << dim << ")"
         << " address: " << out_tensors[i]->data<void>() << ", ";
    }
    VLOG(10) << ss.str();
  }

 private:
  void GetMemSizeAndDtype(
      const std::vector<const framework::LoDTensor *> &lod_tensors,
      const std::vector<std::string> var_names, size_t *numel,
      const size_t &size_of_dtype, const platform::Place &place,
      const bool use_align = true) const {
    PADDLE_ENFORCE_EQ(
        lod_tensors.size(), var_names.size(),
        platform::errors::InvalidArgument(
            "The number of input tensor and variable does not match, the "
            "number of input tensor is %u, the number of input variable is %u.",
            lod_tensors.size(), var_names.size()));
    *numel = 0;
    std::stringstream ss;
    ss << "alloc_space_for_vars: ";
    for (size_t i = 0; i < var_names.size(); ++i) {
      PADDLE_ENFORCE_EQ(lod_tensors[i]->IsInitialized(), true,
                        platform::errors::InvalidArgument(
                            "Tensor `%s` is not initialized.", var_names[i]));

      auto size = lod_tensors[i]->numel();
      PADDLE_ENFORCE_GT(
          size, 0,
          platform::errors::InvalidArgument(
              "The number of tensor `%s`'s elements is 0.", var_names[i]));
      ss << "input(" << var_names[i] << ") dim:(" << lod_tensors[i]->dims()
         << ") "
         << " addres:" << lod_tensors[i]->data<void>() << ", ";
      *numel += use_align
                    ? platform::Alignment(
                          static_cast<size_t>(size) * size_of_dtype, place) /
                          size_of_dtype
                    : static_cast<size_t>(size);
    }

    VLOG(10) << ss.str();
  }
};

class CoalesceTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }
};

class CoalesceTensorOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(vector<LoDTensor>) The input tensors of"
             " coalesce_tensor operator.")
        .AsDuplicable();
    AddOutput("Output",
              "(vector<LoDTensor>) The output "
              "tensors of coalesce_tensor operator. And the address "
              "of output tensors are continuous, they are sliced from the "
              "tensor of FusedOutput.")
        .AsDuplicable();
    AddOutput("FusedOutput",
              "(LoDTensor) The output tensor "
              "of coalesce_tensor operator. And the tensors of"
              " Output is sliced from the tensor of FusedOutput.");
    AddAttr<int>("dtype", "The output data type.");
    AddAttr<bool>("copy_data", "Whether to copy the Input value to Output.")
        .SetDefault(false);
    AddAttr<bool>("set_constant",
                  "Whether to set the Output with a constant value.")
        .SetDefault(false);
    AddAttr<bool>("persist_output",
                  "Whether to persist the original Output value.")
        .SetDefault(false);
    AddAttr<float>("constant",
                   "If set_constant is true, the constant value will be used "
                   "to set the Output.")
        .SetDefault(0.0);
    AddAttr<bool>("check_name",
                  "Whether to check the name of Input and Output to ensure "
                  "they are the same separately.")
        .SetDefault(false);
    AddAttr<bool>("use_align",
                  "Whether to consider memory chunk and take alignment into "
                  "account for inputs and outputs.")
        .SetDefault(true);
    AddComment(R"DOC(
CoalesceTensor Operator.

coalesce_tensor is used to make the address of Output
continuous according to the Input. This Op will alloc a big tensor
according to the tensors of Input, the dtype is the same with those input tensors,
the size is the sum of those input tensors' numel, and the dim of the big
tensor is {sum(numel)}. And the big tensor is stored in FusedOutput.
The tensors of Output are sliced from the tensor of FusedOutput.
Note that, the dtype of Input should be the same, and the dim of Input
and Output should equal.
The tensors of Input and Output could be the same or different. And
coalesce_tensor allows copying the value of Input to Output, or
setting the Output with a constant value, or persist the original Output
value.

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(coalesce_tensor, paddle::operators::CoalesceTensorOp,
                  paddle::operators::CoalesceTensorOpMaker);
namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_CPU_KERNEL(
    coalesce_tensor,
    ops::CoalesceTensorOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::CoalesceTensorOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::CoalesceTensorOpKernel<paddle::platform::CPUDeviceContext, double>);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
REGISTER_OP_CUDA_KERNEL(
    coalesce_tensor,
    ops::CoalesceTensorOpKernel<paddle::platform::CUDADeviceContext,
                                plat::float16>,
    ops::CoalesceTensorOpKernel<paddle::platform::CUDADeviceContext, int>,
    ops::CoalesceTensorOpKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CoalesceTensorOpKernel<paddle::platform::CUDADeviceContext, double>);
#endif

#ifdef PADDLE_WITH_XPU
REGISTER_OP_XPU_KERNEL(
    coalesce_tensor,
    ops::CoalesceTensorOpKernel<paddle::platform::XPUDeviceContext,
                                plat::float16>,
    ops::CoalesceTensorOpKernel<paddle::platform::XPUDeviceContext, int>,
    ops::CoalesceTensorOpKernel<paddle::platform::XPUDeviceContext, float>,
    ops::CoalesceTensorOpKernel<paddle::platform::XPUDeviceContext, double>);
#endif

REGISTER_OP_VERSION(coalesce_tensor)
    .AddCheckpoint(
        R"ROC(
              Upgrade coalesce_tensor: add a new attribute [use_align].)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "use_align",
            "In order to optionally take memory alignment into account when "
            "coalescing tensors. The default value is true to be compatible "
            "with before.",
            true));
