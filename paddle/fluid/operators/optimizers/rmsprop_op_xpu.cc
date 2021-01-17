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


#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/operators/optimizers/rmsprop_op.h"
#include <gflags/gflags.h>
#include <iostream>

using namespace std;

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

	template <typename DeviceContext, typename T>
	class RmspropOpXPUKernel : public framework::OpKernel<T> {

	public:
		void Compute(const framework::ExecutionContext& ctx) const override {


			using paddle::framework::LoDTensor;

			//check Param & Grad tensor type
			const auto* param_var = ctx.InputVar("Param");
			PADDLE_ENFORCE_EQ(param_var->IsType<LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                    	  framework::ToTypeName(param_var->Type())));

			
			const auto* grad_var = ctx.InputVar("Grad");
			PADDLE_ENFORCE_EQ(grad_var->IsType<LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "Tensor holds the wrong type，Expected Var(%s)'s "
                          "type is LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(grad_var->Type())));


			//inputs
			auto& param = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Param"), "Input", "Param", "Rmsprop");
			auto& meanSquare = GET_DATA_SAFELY(ctx.Input<LoDTensor>("MeanSquare"), "Input", "MeanSquare", "Rmsprop");
			auto& grad = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Grad"), "Input", "Grad", "Rmsprop");
			auto& mom = GET_DATA_SAFELY(ctx.Input<LoDTensor>("Moment"), "Input", "Moment", "Rmsprop");

			auto* learning_rate = ctx.Input<Tensor>("LearningRate");
			
			PADDLE_ENFORCE_EQ(learning_rate->dims().size(), 1,
                      platform::errors::InvalidArgument(
                          "learining rate should have dimension = 1."
                          " But received learning rate dim [%s] ",
                          learning_rate->dims().size()));

			T lr = static_cast<T>(GetAttrFromTensor(learning_rate)); 

			//constants
			T epsilon = static_cast<T>(ctx.Attr<float>("epsilon"));
			T decay = static_cast<T>(ctx.Attr<float>("decay"));
			T momentum = static_cast<T>(ctx.Attr<float>("momentum"));
			//T centered = static_cast<T>(ctx.Attr<bool>("centered"));

			/*PADDLE_ENFORCE_EQ(centered, false,
                      platform::errors::InvalidArgument(
                          "Centered option is not yet supported"));*/



			//outputs
			auto& param_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("ParamOut"), "Output", "ParamOut", "Rmsprop");
			auto& mom_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("MomentOut"), "Output", "MomentOut", "Rmsprop");
			auto& mom_sqrt_out = GET_DATA_SAFELY(ctx.Output<LoDTensor>("MeanSquareOut"), "Output", "MeanSquareOut", "Rmsprop");
			auto& dev_ctx = ctx.template device_context<DeviceContext>();

			///// rmsprop优化算法
			///
			/// ms_out[i] = rho * ms[i] + (1 - rho) * (g[i] * g[i]);
			///
			/// mom_out[i] = momentum * mom[i] + lr * (g[i] / ((float)sqrt(ms_out[i] + epsilon)));
			///
			/// p_out[i] = p[i] - mom_out[i];
			///DLL_EXPORT int rmsprop(Context* ctx, const float* p, const float* ms, const float* g, const float* mom,
        	///float epsilon, float rho, float momentum, float lr,
        	///float *ms_out, float *mom_out, float *p_out, int n)
			
			
			int res = xpu::rmsprop(dev_ctx.x_context(), param.template data<T>(), meanSquare.template data<T>(), grad.template data<T>(), 
				mom.template data<T>(), epsilon, decay, momentum, lr, mom_sqrt_out.template mutable_data<T>(ctx.GetPlace()), 
				mom_out.template mutable_data<T>(ctx.GetPlace()), param_out.template mutable_data<T>(ctx.GetPlace()), param.numel());


			PADDLE_ENFORCE_EQ(res == xpu::Error_t::SUCCESS, true,
                        platform::errors::External(
                            "XPU API return wrong value[%d], please check "
                            "where Baidu Kunlun Card is properly installed.",
                            res));

		}

	};

} // namespace operators
} //namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    rmsprop, ops::RmspropOpXPUKernel<paddle::platform::XPUDeviceContext, float>);


#endif