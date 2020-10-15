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

#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DpsgdOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *param_var = ctx.InputVar("Param");
    PADDLE_ENFORCE_EQ(param_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Param").front(),
                          framework::ToTypeName(param_var->Type())));

    const auto *grad_var = ctx.InputVar("Grad");
    PADDLE_ENFORCE_EQ(grad_var->IsType<framework::LoDTensor>(), true,
                      platform::errors::InvalidArgument(
                          "The Var(%s)'s type should be LoDTensor, "
                          "but the received is %s",
                          ctx.InputNames("Grad").front(),
                          framework::ToTypeName(grad_var->Type())));

    const auto *learning_rate = ctx.Input<framework::Tensor>("LearningRate");

    const auto *param = ctx.Input<framework::Tensor>("Param");
    const auto *grad = ctx.Input<framework::Tensor>("Grad");

    auto *param_out = ctx.Output<framework::Tensor>("ParamOut");

    auto sz = param_out->numel();
    PADDLE_ENFORCE_EQ(param->numel(), sz,
                      platform::errors::InvalidArgument(
                          "Input parameter's number of elements is error, "
                          "expected %zu, but received %zu."));
    PADDLE_ENFORCE_EQ(grad->numel(), sz,
                      platform::errors::InvalidArgument(
                          "Input gradient's number of elements is error, "
                          "expected %zu, but received %zu."));

    const T *lr = learning_rate->data<T>();
    const T *param_data = param->data<T>();
    const T *grad_data = grad->data<T>();

    T *out_data = param_out->mutable_data<T>(ctx.GetPlace());

    T clip = static_cast<T>(ctx.Attr<float>("clip"));
    T batch_size = static_cast<T>(ctx.Attr<float>("batch_size"));
    T sigma = static_cast<T>(ctx.Attr<float>("sigma"));

    // compute clipping
    float l2_norm = 0.0;
    for (int64_t i = 0; i < grad->numel(); ++i) {
      l2_norm = l2_norm + grad_data[i] * grad_data[i];
    }
    l2_norm = std::sqrt(l2_norm);

    float scale = 1.0;
    if (l2_norm > clip) {
      scale = l2_norm / clip;
    }

    // generate gaussian noise.
    // [https://en.wikipedia.org/wiki/Box-Muller_transform]
    float V1, V2, S;
    float X;
    float mu = 0.0;
    float U1, U2;
    unsigned seed = static_cast<unsigned int>(ctx.Attr<int>("seed"));
    if (seed == 0) {
      seed = (unsigned)(time(NULL));
    }
    std::minstd_rand engine;
    engine.seed(seed);
    std::uniform_real_distribution<T> dist(0.0, 1.0);
    do {
      U1 = dist(engine);
      U2 = dist(engine);
      V1 = 2 * U1 - 1;
      V2 = 2 * U2 - 1;
      S = V1 * V1 + V2 * V2;
    } while (S >= 1 || S == 0);

    X = V1 * sqrt(-2 * log(S) / S);

    float gaussian_noise = mu + X * sigma;

    // update parameters
    for (int64_t i = 0; i < grad->numel(); ++i) {
      out_data[i] =
          param_data[i] -
          lr[0] * (grad_data[i] / scale + gaussian_noise / batch_size);
    }
    // CCS16 - Deep Learning with Differential Privacy.
    // [https://arxiv.org/abs/1607.00133]
  }  // Compute
};   // class
}  // namespace operators
}  // namespace paddle
