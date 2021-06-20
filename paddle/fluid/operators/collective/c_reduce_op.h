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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL) || \
    defined(PADDLE_WITH_XPU_BKCL) || defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#endif

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include "paddle/fluid/platform/nccl_helper.h"
#endif

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/platform/bkcl_helper.h"
#endif

#if defined(PADDLE_WITH_GLOO)
#include <gloo/reduce.h>
#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#endif

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/hccl_helper.h"
#endif

namespace paddle {
namespace operators {

enum ReduceType { kRedSum, kRedMax, kRedMin, kRedProd };

class CReduceOp : public framework::OperatorWithKernel {
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
class CReduceOpCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_GLOO)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    auto root_id = ctx.Attr<int>("root_id");

    auto place = ctx.GetPlace();
    int64_t send_numel = in->numel();
    const T* send_buff = in->data<T>();
    T* recv_buff = out->mutable_data<T>(in->dims(), place);
    auto gloo = paddle::framework::GlooWrapper::GetInstance();
    PADDLE_ENFORCE_EQ(
        gloo->IsInitialized(), true,
        platform::errors::PreconditionNotMet(
            "You must initialize the gloo environment first to use it."));
    gloo::ReduceOptions opts(gloo->GetContext());
    opts.setInput(const_cast<T*>(send_buff), send_numel);
    opts.setOutput(recv_buff, send_numel);
    opts.setRoot(root_id);
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
    gloo::reduce(opts);
#else
    PADDLE_THROW(platform::errors::Unavailable(
        "PaddlePaddle should compile with GLOO by setting WITH_GLOO=ON"));
#endif
  }
};

template <ReduceType red_type, typename T>
class CReduceOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::LoDTensor>("X");
    auto out = ctx.Output<framework::LoDTensor>("Out");
    auto place = ctx.GetPlace();
    EcclDataType dtype = platform::ToHCCLDataType(in->type());
    int64_t numel = in->numel();

    void* sendbuff = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));
    void* recvbuff = reinterpret_cast<void*>(out->data<T>());

    int ring_id = ctx.Attr<int>("ring_id");
    int root_id = ctx.Attr<int>("root_id");
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto comm =
        paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = static_cast<platform::NPUDeviceContext*>(dev_ctx)->stream();
    } else {
      stream = comm->stream();
    }

    int rank_id = comm->rank();

    EcclReductionOp hccl_red_type = SUM;
    switch (red_type) {
      case kRedSum:
        hccl_red_type = SUM;
        break;

      case kRedMax:
        hccl_red_type = MAX;
        break;

      case kRedMin:
        hccl_red_type = MIN;
        break;

      case kRedProd:
        hccl_red_type = PROD;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    VLOG(3) << "begin hccl reduce, parameter is: "
            << "input num: " << numel << "root_id: " << root_id
            << "dtype: " << dtype << "hccl_red_type: " << hccl_red_type
            << ", group is: " << group;

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::eccl_all_reduce(
        sendbuff, recvbuff, numel, dtype, hccl_red_type, comm->comm().c_str(),
        reinterpret_cast<void*>(stream), AUTO));

    if (rank_id != root_id) {
      auto npu_place = BOOST_GET_CONST(platform::NPUPlace, place);
      memory::Copy(npu_place, reinterpret_cast<void*>(out->data<T>()),
                   npu_place,
                   reinterpret_cast<void*>(const_cast<T*>(in->data<T>())),
                   numel * sizeof(T), stream);
    }

    out->Resize(in->dims());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CReduceOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    BKCLDataType dtype = platform::ToBKCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<void>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    int root = ctx.Attr<int>("root_id");
    auto comm = platform::BKCLCommContext::Instance().Get(rid, place);

    XPUStream stream = nullptr;
    if (ctx.Attr<bool>("use_calc_stream")) {
      auto dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      stream = static_cast<platform::XPUDeviceContext*>(dev_ctx)
                   ->x_context()
                   ->xpu_stream;
    } else {
      stream = comm->stream();
    }

    BKCLOp bkcl_red_type = BKCL_ADD;
    switch (red_type) {
      case kRedSum:
        bkcl_red_type = BKCL_ADD;
        break;

      case kRedMax:
        bkcl_red_type = BKCL_MAX;
        break;

      case kRedMin:
        bkcl_red_type = BKCL_MIN;
        break;

      case kRedProd:
        bkcl_red_type = BKCL_PRODUCT;
        break;

      default:
        PADDLE_THROW(platform::errors::InvalidArgument(
            "Invalid reduce type: %d", red_type));
    }

    PADDLE_ENFORCE_EQ(bkcl_reduce(comm->comm(), sendbuff, recvbuff, numel,
                                  dtype, bkcl_red_type, root, stream),
                      BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                        "BKCL all reduce failed"));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CReduceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype = platform::ToNCCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<void>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
    int root = ctx.Attr<int>("root_id");
    auto comm = platform::NCCLCommContext::Instance().Get(rid, place);

    gpuStream_t stream = nullptr;
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
        PADDLE_ENFORCE_EQ(true, false, platform::errors::InvalidArgument(
                                           "red_type must be one of kRedSum, "
                                           "kRedMax, kRedMin, kRedProd."));
    }

    PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclReduce(
        sendbuff, recvbuff, numel, dtype, nccl_red_type, root, comm->comm(),
        stream));
#else
    PADDLE_ENFORCE_EQ(true, false,
                      platform::errors::Unavailable(
                          "PaddlePaddle should compile with GPU.."));
#endif
  }
};

class CReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be reduced.");
    AddOutput("Out", "(Tensor) the reduced result.");
    AddAttr<int>("ring_id", "(int default 0) communication ring id.")
        .SetDefault(0);
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for reduce.")
        .SetDefault("tag");
#endif
    AddAttr<int>("root_id", "(int default 0) root id.").SetDefault(0);
    AddAttr<bool>(
        "use_calc_stream",
        "(bool default false) eject CUDA operations to calculation stream.")
        .SetDefault(false);
    AddComment(string::Sprintf(R"DOC(
CReduce %s Operator

Call collective Reduce with reduce type %s. If input and output are
the same variable, in-place reduce will be used.
)DOC",
                               GetName(), GetName()));
  }

 protected:
  virtual std::string GetName() const = 0;
};

}  // namespace operators
}  // namespace paddle
