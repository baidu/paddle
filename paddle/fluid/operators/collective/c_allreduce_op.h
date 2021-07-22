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
#include "paddle/fluid/operators/npu_op_runner.h"

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

inline bool FoundNanOrInf(const paddle::platform::NPUDeviceContext& ctx, aclrtStream stream,
        const paddle::framework::Tensor* float_status, paddle::framework::Tensor* tmp){
    //using float16 = paddle::platform::float16;
    const auto& runner_float_status =
        NpuOpRunner("NPUGetFloatStatus", {*float_status}, {*tmp},
                    {{"message", std::string("check_nan_and_inf")}});
    runner_float_status.Run(stream);

    paddle::framework::Tensor sum;
    sum.mutable_data<float>({1}, ctx.GetPlace());
    const auto& runner_reduce_sum =
        NpuOpRunner("ReduceSumD", {*float_status}, {sum},
                    {{"axes", std::vector<int>{0}}, {"keep_dims", true}});
    runner_reduce_sum.Run(stream);

    std::vector<float> sum_vec;
    TensorToVector(sum, ctx, &sum_vec);
    bool found_inf_data = (sum_vec[0] > 1);

    VLOG(4) << "found_inf_data:" << found_inf_data;
    return found_inf_data;
}

// return true if found_inf_or_nan or return false;
template<typename T>
bool CheckNumerics(const framework::ExecutionContext& exe_ctx, 
        aclrtStream stream, const paddle::framework::Tensor* in){
    auto& dev_ctx = exe_ctx.template device_context<paddle::platform::NPUDeviceContext>();
    using Tensor = paddle::framework::Tensor;
    Tensor out(in->type());
    out.Resize(in->dims());
    out.mutable_data(dev_ctx.GetPlace());

    bool found_inf_data=false;

    try {
        const auto& runner =
            NpuOpRunner("CheckNumerics", {*in}, {out},
                        {{"message", std::string("check_numberics")}});
        runner.Run(stream);
        dev_ctx.Wait();
    } catch (platform::EnforceNotMet& exception) {
        LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
        found_inf_data = true;
    } catch (...) {
        LOG(WARNING) << "[check_nan_and_inf] detected contains NaN or INF!!!";
        found_inf_data = true;
    }
    
    //std::vector<T> out_cpu;
    //TensorToVector(out, exe_ctx.device_context(), &out_cpu);
    //VLOG(4) << "out_cpu size:" << out_cpu.size() << ", poistion 0:" << out_cpu[0];

    return found_inf_data;
}

template <ReduceType red_type, typename T>
class CAllReduceOpASCENDKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_ASCEND_CL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");
    const auto* float_status = ctx.Input<framework::Tensor>("FloatStatus");
    auto place = ctx.GetPlace();
    HcclDataType dtype = platform::ToHCCLDataType(in->type());
    int64_t numel = in->numel();

    void* sendbuff = reinterpret_cast<void*>(const_cast<T*>(in->data<T>()));
    out->mutable_data<T>(in->dims(), ctx.GetPlace());
    void* recvbuff = reinterpret_cast<void*>(out->data<T>());

    int ring_id = ctx.Attr<int>("ring_id");
    std::string group =
        std::string(HCOM_GROUP_PREFIX) + std::to_string(ring_id);
    auto comm =
        paddle::platform::HCCLCommContext::Instance().Get(ring_id, place);

    aclrtStream stream = nullptr;
    auto dev_ctx = static_cast<platform::NPUDeviceContext*>(platform::DeviceContextPool::Instance().Get(place));
    if (ctx.Attr<bool>("use_calc_stream")) {
      stream = dev_ctx->stream();
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

    VLOG(3) << "after hccl allreduce, parameter is: "
            << "input num: " << numel << "dtype: " << dtype
            << "hccl_red_type: " << hccl_red_type << ", group is: " << group
            << ", sendbuff:" << sendbuff << ", recvbuff:" << recvbuff
            << ", out_size:" << out->memory_size()
            << ", use_calc_stream:" << ctx.Attr<bool>("use_calc_stream")
            << ", stream:" << stream;

    VLOG(4) << "prepare to FoundNanInf";
    framework::Tensor tmp;
    tmp.mutable_data<float>({8}, ctx.GetPlace());

    bool nan_or_inf=false;
    bool check_numerics=false;
    if (float_status){
        nan_or_inf = FoundNanOrInf(ctx.template device_context<paddle::platform::NPUDeviceContext>(),
                dev_ctx->stream(), float_status, &tmp);
    }else{
        auto d_type = in->type();
        switch (d_type) {
            case framework::proto::VarType::FP16:
            case framework::proto::VarType::FP32:{
                check_numerics=CheckNumerics<T>(ctx, dev_ctx->stream(), in);
                VLOG(4) << "check_numerics:" << check_numerics;
                break;
            }
            default:
              break;
        }
    }

    if(nan_or_inf || check_numerics){
        T inf = static_cast<T>(std::numeric_limits<float>::infinity());
        VLOG(4) << "fill input data constant inf";
        auto dims=in->dims();
        auto mutable_in=const_cast<framework::Tensor*>(in);
        FillNpuTensorWithConstant<T>(mutable_in, inf);
        mutable_in->Resize(dims);
    }

    VLOG(3) << "begin hccl allreduce, parameter is: "
            << "input num: " << numel << "dtype: " << dtype
            << "hccl_red_type: " << hccl_red_type << ", group is: " << group
            << ", sendbuff:" << sendbuff
            << ", recvbuff:" <<  recvbuff
            << ", out_size:" << out->memory_size();

    PADDLE_ENFORCE_NPU_SUCCESS(platform::dynload::HcclAllReduce(
        sendbuff, recvbuff, numel, dtype, hccl_red_type, comm->comm(),
        reinterpret_cast<void*>(stream)));

    out->Resize(in->dims());
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should compile with NPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_XPU_BKCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    BKCLDataType dtype = platform::ToBKCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
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

    PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(), sendbuff, recvbuff, numel,
                                      dtype, bkcl_red_type, stream),
                      BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                        "BKCL all reduce failed"));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "PaddlePaddle should be compiled with XPU."));
#endif
  }
};

template <ReduceType red_type, typename T>
class CAllReduceOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
    auto in = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    auto place = ctx.GetPlace();
    ncclDataType_t dtype = platform::ToNCCLDataType(in->type());
    int64_t numel = in->numel();
    const void* sendbuff = in->data<T>();
    out->Resize(in->dims());
    void* recvbuff = out->mutable_data<T>(place);

    int rid = ctx.Attr<int>("ring_id");
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
#if defined(PADDLE_WITH_ASCEND_CL)
    AddAttr<std::string>("tag", "(string default tag) tag for all reduce.")
        .SetDefault("tag");
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
#if defined(PADDLE_WITH_ASCEND_CL)
     AddInput("FloatStatus",
             "(Tensor) 1-dim tensor of shape [8], allocated by "
             "alloc_float_status op").AsDispensable();
#endif
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
