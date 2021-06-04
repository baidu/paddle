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

#include "paddle/fluid/operators/fused/fusion_seqexpand_concat_fc_op.h"

#include <string>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/platform/cpu_info.h"

namespace paddle {
namespace operators {

void FusionSeqExpandConcatFCOp::InferShape(
    framework::InferShapeContext* ctx) const {
  PADDLE_ENFORCE_GT(ctx->Inputs("X").size(), 1UL,
                    platform::errors::InvalidArgument(
                        "Inputs(X) of FusionSeqExpandConcatFCOp should larger "
                        "than 1, but received value is: %d.",
                        ctx->Inputs("X").size()));
  OP_INOUT_CHECK(ctx->HasInput("FCWeight"), "Input", "FCWeight",
                 "fusion_seqexpand_concat_fc");
  OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out",
                 "fusion_seqexpand_concat_fc");
  OP_INOUT_CHECK(ctx->HasOutput("FCOut"), "Output", "FCOut",
                 "fusion_seqexpand_concat_fc");

  auto ins_dims = ctx->GetInputsDim("X");
  auto w_dims = ctx->GetInputDim("FCWeight");  // (M0+M1+M2+..) x D
  PADDLE_ENFORCE_EQ(
      w_dims.size(), 2,
      platform::errors::InvalidArgument(
          "Input(FCWeight)'s rank must be 2, but received value is: %d.",
          w_dims.size()));
  const int D = w_dims[1];
  int sum = ins_dims[0][1];
  for (size_t i = 1; i < ins_dims.size(); ++i) {
    sum += ins_dims[i][1];
  }
  PADDLE_ENFORCE_EQ(
      sum, w_dims[0],
      platform::errors::InvalidArgument("FC height should be sum of all inputs "
                                        "width, but received FC height is: %d, "
                                        "sum of all inputs width is: %d.",
                                        w_dims[0], sum));
  if (ctx->HasInput("FCBias")) {
    auto b_dims = ctx->GetInputDim("FCBias");
    PADDLE_ENFORCE_EQ(
        b_dims.size() == 1 || b_dims.size() == 2, true,
        platform::errors::InvalidArgument(
            "FCBias dim should be 1 or 2, but received value is: %d.",
            b_dims.size()));
    if (b_dims.size() == 1) {
      PADDLE_ENFORCE_EQ(b_dims[0], D,
                        platform::errors::InvalidArgument(
                            "FCBias shapes must be %d when FCBias dim = 1, but "
                            "received value is: %d.",
                            D, b_dims[0]));
    } else {
      PADDLE_ENFORCE_EQ(b_dims[0], 1,
                        platform::errors::InvalidArgument(
                            "FCBias shapes must be 1x%d, when FCBias dim = 2, "
                            "but received dim[0] is: %d.",
                            D, b_dims[0]));
      PADDLE_ENFORCE_EQ(b_dims[1], D,
                        platform::errors::InvalidArgument(
                            "FCBias shapes must be 1x%d, when FCBias dim = 2, "
                            "but received dim[1] is: %d.",
                            D, b_dims[1]));
    }
  }

  ctx->SetOutputDim("Out", {ins_dims[0][0], D});
  // fcout should be reshape when run since can not get lod in infershape
  // explicit share the ref lod
  ctx->ShareLoD("X", "Out", 0);
}

framework::OpKernelType FusionSeqExpandConcatFCOp::GetExpectedKernelType(
    const framework::ExecutionContext& ctx) const {
  return framework::OpKernelType(
      OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.device_context());
}

void FusionSeqExpandConcatFCOpMaker::Make() {
  AddInput("X",
           "(LoDTensor) input LodDTensors, the first one must be have ref lod "
           "for sequence expand, and the rest input should have same lod.")
      .AsDuplicable();
  AddInput("FCWeight", "(Tensor) the weights of fc.");
  AddInput("FCBias", "(Tensor, optional) the bias of fc.").AsDispensable();
  AddOutput("Out", "(LoDTensor) Output LodTensor.");
  AddOutput(
      "FCOut",
      "(Tensor) the intermediate tensor to keep the result of fc."
      "Shape is (N x D), where N is the batch size, D is the output dim of fc")
      .AsIntermediate();
  AddAttr<std::string>("fc_activation",
                       "(string, default: identity)"
                       "The activation for the result of fc."
                       "`identity` by default.")
      .SetDefault("identity")
      .InEnum({"sigmoid", "tanh", "relu", "identity"});
  AddComment(R"DOC(
Fusion Sequence expand + concat + fc Operator.

All below conditions should be meet:

The ref_level of seq_expand should be 0.

The ref lod of seq_expand level is the first input of concat.

The other inputs should have same lod and same batch size of ref lod.

The seq len of other inputs should be 1.

The concat axis should be 1.

)DOC");
}

template <typename T>
class FusionSeqExpandConcatFCOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using DeviceContext = paddle::platform::CPUDeviceContext;
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto* w = ctx.Input<Tensor>("FCWeight");
    auto* b = ctx.Input<Tensor>("FCBias");
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* fc_out = ctx.Output<Tensor>("FCOut");

    auto* ref_in = ins[0];
    auto ref_lod = ref_in->lod();
    auto in1_lod = ins[1]->lod();
    auto ref_dims = ref_in->dims();  // T x M0
    auto in1_dims = ins[1]->dims();  // N x M1
    auto w_dims = w->dims();
    const int N = ref_lod[0].size() - 1;
    const int total_T = ref_dims[0];
    const int M0 = ref_dims[1];
    const int M1 = in1_dims[1];
    const int D = w_dims[1];

    // some check and fcout should be reshape here
    // since infershape can not get lod info
    PADDLE_ENFORCE_EQ(
        ref_lod.size(), 1UL,
        platform::errors::InvalidArgument(
            "Only support input lod size is 1, but received value is: %d.",
            ref_lod.size()));
    PADDLE_ENFORCE_EQ(
        in1_lod.size(), 1UL,
        platform::errors::InvalidArgument(
            "Only support input lod size is 1, but received value is: %d.",
            in1_lod.size()));
    PADDLE_ENFORCE_EQ(static_cast<int>(in1_lod[0].size() - 1), N,
                      platform::errors::InvalidArgument(
                          "Batch size of all inputs should be equal to %d, but "
                          "received value is: %d.",
                          N, static_cast<int>(in1_lod[0].size() - 1)));
    PADDLE_ENFORCE_EQ(
        static_cast<int>(in1_lod[0][N]), N,
        platform::errors::InvalidArgument("Seq_length of other inputs should "
                                          "be %d, but received value is: %d.",
                                          N, static_cast<int>(in1_lod[0][N])));
    PADDLE_ENFORCE_EQ(
        in1_dims[0], N,
        platform::errors::InvalidArgument(
            "input height should be batch size: %d, but received value is %d.",
            N, in1_dims[0]));
    for (size_t i = 2; i < ins.size(); ++i) {
      PADDLE_ENFORCE_EQ(ins[i]->dims()[0], N,
                        platform::errors::InvalidArgument(
                            "All other inputs height should be equal to %d, "
                            "but received value is: %d.",
                            N, ins[i]->dims()[0]));
      PADDLE_ENFORCE_EQ(ins[i]->lod(), in1_lod,
                        platform::errors::InvalidArgument(
                            "All other inputs should have same lod: %d, but "
                            "received value is: %d.",
                            in1_lod, ins[i]->lod()));
    }
    fc_out->Resize({N, D});

    std::function<void(const int, const T*, T*)> fc_act;
    auto& fc_act_str = ctx.Attr<std::string>("fc_activation");
    if (platform::MayIUse(platform::avx)) {
      math::VecActivations<T, platform::avx> act_functor;
      fc_act = act_functor(fc_act_str);
    } else {
      math::VecActivations<T, platform::isa_any> act_functor;
      fc_act = act_functor(fc_act_str);
    }

    const T* ref_in_data = ref_in->data<T>();
    const T* in1_data = ins[1]->data<T>();
    const T* w_data = w->data<T>();
    T* out_data = out->mutable_data<T>(ctx.GetPlace());
    T* fc_out_data = fc_out->mutable_data<T>(ctx.GetPlace());

    auto blas = math::GetBlas<DeviceContext, T>(ctx);

    auto& dev_ctx = ctx.template device_context<platform::CPUDeviceContext>();
    math::FCFunctor<DeviceContext, T> fc;
    fc(dev_ctx, total_T, D, M0, ref_in_data, w_data, out_data,
       b ? b->data<T>() : nullptr);
    w_data = w_data + M0 * D;
    // first write on
    blas.MatMul(N, D, M1, in1_data, w_data, fc_out_data);
    w_data = w_data + M1 * D;
    for (size_t i = 2; i < ins.size(); ++i) {
      // add on
      const T* in_data = ins[i]->data<T>();
      const int K = ins[i]->dims()[1];
      blas.GEMM(CblasNoTrans, CblasNoTrans, N, D, K, static_cast<T>(1), in_data,
                K, w_data, D, static_cast<T>(1), fc_out_data, D);
      w_data = w_data + K * D;
    }
    T* cur_out_data = out_data;
    for (int i = 0; i < N; ++i) {
      int seq_len = ref_lod[0][i + 1] - ref_lod[0][i];
      T* src = fc_out_data + i * D;
      for (int step = 0; step < seq_len; ++step) {
        blas.VADD(D, cur_out_data, src, cur_out_data);
        cur_out_data = cur_out_data + D;
      }
    }
    fc_act(total_T * D, out_data, out_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_seqexpand_concat_fc, ops::FusionSeqExpandConcatFCOp,
                  ops::FusionSeqExpandConcatFCOpMaker);

REGISTER_OP_CPU_KERNEL(fusion_seqexpand_concat_fc,
                       ops::FusionSeqExpandConcatFCOpKernel<float>,
                       ops::FusionSeqExpandConcatFCOpKernel<double>);
