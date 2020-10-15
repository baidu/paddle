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
#include "paddle/fluid/operators/dropout_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/platform/xpu_header.h"
namespace paddle {
namespace operators {

#ifdef PADDLE_WITH_XPU
static std::map<int, float*> mask_data_tables;
static const int max_data_size = 32 * 1024 * 1024;
static std::mutex s_mask_data_table_lock;
template <typename DeviceContext, typename T>
class DropoutXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");
    auto dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    float* mask_data_table = nullptr;
    PADDLE_ENFORCE_EQ(!context.HasInput("Seed"), true,
                      platform::errors::InvalidArgument(
                          ("Input(Seed) not supported on XPU")));
    if (!context.Attr<bool>("is_test")) {
      int dev_id =
          BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()).GetDeviceId();
      int prop = static_cast<int>(dropout_prob * 100);
      int is_upscale = (dropout_implementation == "upscale_in_train");
      /* mask_data_tables key contains 3 part:
       *  | 31-16  | 15-8 | 7-0        |
       *  | dev_id | prob | is_upscale |
       */
      int index = (dev_id << 16) + (prop << 8) + is_upscale;
      std::lock_guard<std::mutex> lock(s_mask_data_table_lock);
      if (mask_data_tables.find(index) == mask_data_tables.end()) {
        float* mask_data_host = new float[max_data_size];
        std::random_device rnd;
        std::minstd_rand engine;
        int seed =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
        engine.seed(seed);
        std::uniform_real_distribution<float> dist(0, 1);
        for (size_t i = 0; i < max_data_size; ++i) {
          if (dist(engine) < dropout_prob) {
            mask_data_host[i] = 0.0f;
          } else {
            if (is_upscale) {
              mask_data_host[i] = 1.0f / static_cast<T>(1.0f - dropout_prob);
            } else {
              mask_data_host[i] = 1.0;
            }
          }
        }
        PADDLE_ENFORCE(
            xpu_malloc(reinterpret_cast<void**>(&mask_data_table),
                       max_data_size * sizeof(float)) == xpu::Error_t::SUCCESS,
            "XPU no enough memory");
        memory::Copy(BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()),
                     mask_data_table, platform::CPUPlace(), mask_data_host,
                     max_data_size * sizeof(float));
        mask_data_tables[index] = mask_data_table;
        free(mask_data_host);
      } else {
        mask_data_table = mask_data_tables[index];
      }
    }
    if (!context.Attr<bool>("is_test")) {  // Train
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<T>(context.GetPlace());
      size_t size = framework::product(mask->dims());
      auto& dev_ctx = context.template device_context<DeviceContext>();
      int r = xpu::dropout(dev_ctx.x_context(), mask_data_table, x_data,
                           mask_data, y_data, max_data_size, size);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    } else {  // Infer
      float scale = 0.0f;
      if (dropout_implementation == "upscale_in_train") {
        scale = 1.0f;
      } else {
        scale = static_cast<T>(1.0f - dropout_prob);
      }
      auto& dev_ctx = context.template device_context<DeviceContext>();
      int r = xpu::scale(dev_ctx.x_context(), x->numel(), scale, 0.0f, 0,
                         x_data, y_data);
      PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                        platform::errors::InvalidArgument("XPU kernel error!"));
    }
  }
};
template <typename DeviceContext, typename T>
class DropoutGradXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");
    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto& dev_ctx = context.template device_context<DeviceContext>();
    int r = xpu::elementwise_mul(dev_ctx.x_context(), grad_y->data<T>(),
                                 mask->data<T>(), grad_x->data<T>(),
                                 grad_y->numel());
    PADDLE_ENFORCE_EQ(r == xpu::Error_t::SUCCESS, true,
                      platform::errors::InvalidArgument("XPU kernel error!"));
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    dropout, ops::DropoutXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    dropout_grad,
    ops::DropoutGradXPUKernel<paddle::platform::XPUDeviceContext, float>);
#endif
