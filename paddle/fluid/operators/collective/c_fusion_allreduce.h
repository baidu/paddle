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

#pragma once

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/collective/c_allreduce_op.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_ASCEND_CL) || defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/bkcl_helper.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include <gloo/allreduce.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {
class CFusionAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    //fused_var
    OP_INOUT_CHECK(ctx->HasInputs("X"), "Input", "X",
                   "c_fusion_allreduce");
    OP_INOUT_CHECK(ctx->HasInputs("SplittedInput"), "Input", "SplittedInput",
                   "c_fusion_allreduce");
    //fused_var
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out",
                   "c_fusion_allreduce");
    OP_INOUT_CHECK(ctx->HasOutputs("SplittedOutput"), "Output", "SplittedOutput",
                   "c_fusion_allreduce");

    PADDLE_ENFORCE_EQ(
        ctx->Inputs("SplittedInput").size(), ctx->Outputs("SplittedOutput").size(),
        platform::errors::InvalidArgument(
            "The input(X) and output(Out) should have same size in "
            "Operator(c_fusion_allreduce), size of input(X) is %d "
            "and size of output(Out) is %d.",
            ctx->Inputs("SplittedInput").size(), ctx->Outputs("SplittedOutput").size()));
    auto x_dims = ctx->GetInputsDim("X");
    ctx->SetOutputsDim("Out", x_dims);
  }

 protected:
  /*
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
  */
};
template <ReduceType red_type, typename T>
class CFusionAllReduceOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(platform::errors::Unavailable(
        "CFusionAllReduceOpCPUKernel not suppored"));
  }
};

template <ReduceType red_type, typename T>
class CFusionAllReduceOpASCENDKernel : public framework::OpKernel<T> {
 public:
  CFusionAllReduceOpASCENDKernel():_allreduce_op_kernel(new CAllReduceOpASCENDKernel<red_type,T>()){};
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    //const auto x = ctx.InputVar("X");
    //auto out = ctx.OuputVar("Out");
    auto splits = ctx.MultiInputVar("SplittedInput");
    auto splits_size = splits.size();
    auto offset = ctx.Attr<int>("align_size");
    VLOG(4) << "The NPU fusion allreduce sum input:"<< splits_size;

    // Check whether the address space is contiguous.
    // std::sort(splits.begin(),splits.end());

    //size_t size_of_dtype = framework::SizeOfType(T);
    for (size_t k = 1; k < splits.size(); ++k) {
      //pre
      auto pre_var = splits.at(k - 1);
      auto per_tensor = &pre_var->Get<framework::LoDTensor>();
      const void *pre_address = per_tensor->data<void>();
      //int64_t pre_len = per_tensor->numel();
      //auto offset = platform::Alignment(pre_len * size_of_dtype, places_[0]);
      void * infer_address = reinterpret_cast<void *>(
          reinterpret_cast<uintptr_t>(pre_address) + offset);

      //next
      auto cur_var = splits.at(k);
      auto cur_tensor = &cur_var->Get<framework::LoDTensor>();
      const void *cur_address = cur_tensor->data<void>();

      VLOG(10) << string::Sprintf(
          "Input[%d]: pre_address:0X%02x infer_address:0X%02x cur_address:0X%02x offset:%d",
          k - 1, pre_address, infer_address, cur_address, offset);
      PADDLE_ENFORCE_EQ(
          infer_address, cur_address,
          platform::errors::InvalidArgument(
              "The infered address of the next tensor should be equal to the "
              "real address of the next tensor. But got infered address is %p "
              "and real address is %p.",
              infer_address, cur_address));
    }

    _allreduce_op_kernel->Compute(ctx);
    return; 
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
 private:
  std::unique_ptr<CAllReduceOpASCENDKernel<red_type,T>> _allreduce_op_kernel;
};

class CFusionAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the allreduced result.");
    AddInput("SplittedInput", "(Tensor), tensor to be allreduced.").AsDuplicable();
    AddOutput("SplittedOutput", "(Tensor), tensor to be allreduced.").AsDuplicable();
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for all reduce.")
        .SetDefault("tag");
    AddAttr<int>("align_size", "align size for npu memory")
        .SetDefault(256);
#endif
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddAttr<bool>(
        "use_model_parallel",
        "(bool default false) use this op with model parallel mode. In model "
        "parallel mode, the backward is c_identity which returns itself for "
        "c_allreduce_sum.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CFusionAllReduce %s Operator
)DOC", GetName(), GetName()));
  }
 protected:
  virtual std::string GetName() const = 0;
};


}  // namespace operators
}  // namespace paddle
