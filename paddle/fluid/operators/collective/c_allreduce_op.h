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

#if defined(PADDLE_WITH_NCCL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include <gloo/allreduce.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

enum ReduceType { kRedSum, kRedMax, kRedMin, kRedProd };

class CAllReduceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    int64_t send_numel = in->numel();
    const T* send_buff = in->data<T>();
    T* recv_buff = out->mutable_data<T>(in->dims(), place);
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(), true,
        platform::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));
    gloo::AllreduceOptions opts(gloo->GetContext());
    opts.setInput(const_cast<T*>(send_buff), send_numel);
    opts.setOutput(recv_buff, send_numel);
    switch (red_type) {
      case kRedSum:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::sum<T>));
        break;
      case kRedMax:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::max<T>));
        break;
      case kRedMin:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::min<T>));
        break;
      case kRedProd:
        opts.setReduceFunction(
            static_cast<void (*)(void*, const void*, const void*, size_t)>(
                &gloo::product<T>));
        break;
      default:
        PADDLE_ENFORCE_EQ(true, false,
                          platform::errors::InvalidArgument(
                              "Invalid reduce type: %d.", red_type));
    }
    gloo::allreduce(opts);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");

    auto place = ctx.GetPlace();
    HcclDataType dtype = platform::ToHCCLDataType(in->type());

    int64_t numel = in->numel();
    void* sendbuff = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));

    // void* sendbuff = in->data<void>();

    out->Resize(in->dims());
    void* recvbuff = reinterpret_cast<void*>(const_cast<T*>(out->data<T>()));
    // void* recvbuff = out->mutable_data<T>(place);


    //int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::HCCLCommContext::Instance().Get();

    aclrtStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    HcclReduceOp hccl_red_type = HCCL_REDUCE_SUM;
    switch (red_type) {
      case kRedSum:
        hccl_red_type = HCCL_REDUCE_SUM;
        break;

      case kRedMax:
        hccl_red_type = HCCL_REDUCE_MAX;
        break;

      case kRedMin:
        hccl_red_type = HCCL_REDUCE_MIN;
        break;

      case kRedProd:
        hccl_red_type = HCCL_REDUCE_PROD;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }
    platform::dynload::HcclAllReduce(
        sendbuff, recvbuff, numel, dtype, hccl_red_type, comm->comm(), stream);

    // PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::HcclAllReduce(
    //     sendbuff, recvbuff, numel, dtype, hccl_red_type, comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype = platform::ToHCCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<void>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    cudaStream_t stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::CUDADeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    ncclRedOp_t nccl_red_type = ncclSum;
    switch (red_type) {
      case kRedSum:
        nccl_red_type = ncclSum;
        break;

      case kRedMax:
        nccl_red_type = ncclMax;
        break;

      case kRedMin:
        nccl_red_type = ncclMin;
        break;

      case kRedProd:
        nccl_red_type = ncclProd;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclAllReduce(
        sendbuff, recvbuff, numel, dtype, nccl_red_type, comm->comm(), stream));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with GPU."));
#endif
  }
};

class CAllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the allreduced result.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CAllReduce %s Operator

Call collective AllReduce with reduce type %s. If input and output are
the same variable, in-place allreduce will be used.
Reference: https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/docs/usage/operations.html#allreduce
)DOC",
                               GetName(), GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
};

}  // namespace operators
}  // namespace paddle
