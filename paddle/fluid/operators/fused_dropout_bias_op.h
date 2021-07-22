// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include <algorithm>
#include <cstring>
#include <random>
#include <string>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];
};

template <typename T>
inline int VectorizedSize(const T* pointer) {
  uint64_t address = reinterpret_cast<uint64_t>(pointer);
  constexpr int vec4 = std::alignment_of<AlignedVector<T, 4>>::value;  // NOLINT
  if (address % vec4 == 0) {
    return 4;
  }
  return 1;
}

using Tensor = framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPUDropoutBiasFuseKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto x_dim = x->dims();
    auto x_width = x_dim[x_dim.size() - 1];
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* bias = context.Input<Tensor>("Bias");
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    const auto* bias_data = bias->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      size_t size = framework::product(mask->dims());

      if (dropout_prob == 1.0f) {
        std::memset(y_data, 0, size * sizeof(*y_data));        // NOLINT
        std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
        return;
      }

      int seed_data = 0;
      if (seed) {
        seed_data = *(seed->data<int>());
      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }
      auto engine = framework::GetCPURandomEngine(seed_data);

      std::uniform_real_distribution<float> dist(0, 1);

      for (size_t i = 0; i < size; ++i) {
        size_t bias_id;
        if ((x_width & (x_width - 1)) == 0) {
          bias_id = i & (x_width - 1);
        } else {
          bias_id = i % x_width;
        }
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (upscale_in_train) {
            y_data[i] = (x_data[i] + bias_data[bias_id]) /
                        static_cast<T>(1.0f - dropout_prob);
          } else {
            y_data[i] = x_data[i] + bias_data[bias_id];
          }
        }
      }
    } else {
      const auto* X_data = x->data<T>();
      const auto* Bias_data = bias->data<T>();
      auto* Y_data = y->mutable_data<T>(context.GetPlace());
      T fraction = static_cast<T>(1.0f - dropout_prob);
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < x->numel(); i++) {
        size_t bias_id;
        if ((x_width & (x_width - 1)) == 0) {
          bias_id = i & (x_width - 1);
        } else {
          bias_id = i % x_width;
        }
        if (upscale_in_train) {
          Y_data[i] = X_data[i] + Bias_data[bias_id];
        } else {
          Y_data[i] = (X_data[i] + Bias_data[bias_id]) * fraction;
        }
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutBiasFuseGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(!context.Attr<bool>("is_test"), true,
                      platform::errors::PreconditionNotMet(
                          "GradOp is only callable when is_test is false"));

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_b = context.Output<Tensor>(framework::GradVarName("Bias"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto x_dim = grad_y->dims();  // using y's dim to compute x's dim
    auto x_width = x_dim[x_dim.size() - 1];
    auto x_height = framework::product(x_dim) / x_width;
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    grad_b->mutable_data<T>(context.GetPlace());

    auto M = EigenVector<uint8_t>::Flatten(*mask);
    auto dX = EigenVector<T>::Flatten(*grad_x);
    auto db = EigenVector<T>::Flatten(*grad_b);
    auto dY = EigenVector<T>::Flatten(*grad_y);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    if (dropout_implementation == "upscale_in_train") {
      float dropout_prob = context.Attr<float>("dropout_prob");
      if (dropout_prob == 1.0f) {
        dX.device(place) = static_cast<T>(0) * dY;
        Tensor x_grad_2d;
        x_grad_2d.ShareDataWith(*grad_x).Resize({x_height, x_width});
        auto x_grad_matrix = EigenMatrix<T>::From(x_grad_2d);
        Eigen::DSizes<int, 1> along_axis(0);
        db.device(place) = x_grad_matrix.sum(along_axis);
      } else {
        dX.device(place) =
            dY * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
        Tensor x_grad_2d;
        x_grad_2d.ShareDataWith(*grad_x).Resize({x_height, x_width});
        auto x_grad_matrix = EigenMatrix<T>::From(x_grad_2d);
        Eigen::DSizes<int, 1> along_axis(0);
        db.device(place) = x_grad_matrix.sum(along_axis);
      }
    } else {
      dX.device(place) = dY * M.cast<T>();
      Tensor x_grad_2d;
      x_grad_2d.ShareDataWith(*grad_x).Resize({x_height, x_width});
      auto x_grad_matrix = EigenMatrix<T>::From(x_grad_2d);
      Eigen::DSizes<int, 1> along_axis(0);
      db.device(place) = x_grad_matrix.sum(along_axis);
    }
  }
};
}  // namespace operators
}  // namespace paddle
