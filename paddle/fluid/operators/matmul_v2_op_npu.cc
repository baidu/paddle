/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_ASCEND_CL
#include <memory>
#include <string>

#include "paddle/fluid/operators/matmul_v2_op.h"
#include "paddle/fluid/operators/npu_op_runner.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class MatMulV2NPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* y = ctx.Input<framework::Tensor>("Y");
    auto* out = ctx.Output<framework::Tensor>("Out");
    bool transpose_x = ctx.Attr<bool>("transpose_X");  
    bool transpose_y = ctx.Attr<bool>("transpose_Y");  
    PADDLE_ENFORCE_EQ(transpose_x, false, platform::errors::InvalidArgument(
                "matmul npu not support transpose_x is true"));
    //PADDLE_ENFORCE_EQ(transpose_y, false, platform::errors::InvalidArgument(
    //            "matmul npu not support transpose_y is true"));

    auto stream =
            ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();

    if (x->dims().size() == 2){
      if (transpose_y){
        // transpose
        std::vector<int> perm_vec; 
        perm_vec.push_back(1);
        perm_vec.push_back(0);
        framework::AttributeMap attr_input_tmp= {{"perm", perm_vec}};
        framework::Tensor tmp_y(y->type());
        tmp_y.Resize(framework::make_ddim({y->dims()[1], y->dims()[0]}));
        tmp_y.mutable_data<T>(ctx.GetPlace());  
        auto runner_tmp = NpuOpRunner("TransposeD", {*y}, {tmp_y}, attr_input_tmp);
        runner_tmp.Run(stream);

        // matmul
        framework::AttributeMap attr_input= {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}};
        out->mutable_data<T>(ctx.GetPlace());

        auto runner = NpuOpRunner("MatMul", {*x, tmp_y}, {*out}, attr_input);

        runner.Run(stream);

      } else{

        framework::AttributeMap attr_input= {{"transpose_x1", transpose_x}, {"transpose_x2", transpose_y}};
        out->mutable_data<T>(ctx.GetPlace());

        auto runner = NpuOpRunner("MatMul", {*x, *y}, {*out}, attr_input);

        auto stream =
            ctx.template device_context<paddle::platform::NPUDeviceContext>()
                .stream();
        runner.Run(stream);
      }
    
    } else if (x->dims().size() > 2){
       
      framework::AttributeMap attr_input= {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}};
      out->mutable_data<T>(ctx.GetPlace());

      auto runner = NpuOpRunner("BatchMatMul", {*x, *y}, {*out}, attr_input);

      auto stream =
          ctx.template device_context<paddle::platform::NPUDeviceContext>()
              .stream();
      runner.Run(stream);

    }
  }
};

//template <typename DeviceContext, typename T>
//class MatMulV2GradNPUKernel : public framework::OpKernel<T> {
// public:
//  void Compute(const framework::ExecutionContext& ctx) const override {
//    auto* x = ctx.Input<framework::Tensor>("X");
//    auto* y = ctx.Input<framework::Tensor>("Y");
//    auto* out = ctx.Output<framework::Tensor>(framework::GradVarName("Out"));
//    auto* dx = ctx.Input<framework::Tensor>(framework::GradVarName("X"));
//    auto* dy = ctx.Input<framework::Tensor>(framework::GradVarName("Y"));
//    bool transpose_x = ctx.Attr<bool>("transpose_X");  
//    bool transpose_y = ctx.Attr<bool>("transpose_Y");  
//
//    if (x->dims().size() == 2){
//      if (transpose_y){
//
//        framework::AttributeMap attr_input= {{"transpose_x1", false}, {"transpose_x2", false}};
//        dx->mutable_data<T>(ctx.GetPlace());
//        dy->mutable_data<T>(ctx.GetPlace());
//        auto runner = NpuOpRunner("MatMulV2", {*x, *y}, {*out}, attr_input);
//        
//      }
//
//
//      auto runner = NpuOpRunner("MatMulV2", {*x, *y}, {*out}, attr_input);
//
//      auto stream =
//          ctx.template device_context<paddle::platform::NPUDeviceContext>()
//              .stream();
//      runner.Run(stream);
//    
//    } else if (x->dims().size() > 2){
//       
//      framework::AttributeMap attr_input= {{"adj_x1", transpose_x}, {"adj_x2", transpose_y}};
//      out->mutable_data<T>(ctx.GetPlace());
//
//      auto runner = NpuOpRunner("BatchMatMul", {*x, *y}, {*out}, attr_input);
//
//      auto stream =
//          ctx.template device_context<paddle::platform::NPUDeviceContext>()
//              .stream();
//      runner.Run(stream);
//
//    }
//  }
//};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_NPU_KERNEL(
    matmul_v2,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, float>,
    ops::MatMulV2NPUKernel<paddle::platform::NPUDeviceContext, paddle::platform::float16>);
#endif
