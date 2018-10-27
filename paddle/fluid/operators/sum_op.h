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
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class SumKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_vars = context.MultiInputVar("X");
    size_t in_num = in_vars.size();
    auto out_var = context.OutputVar("Out");

    bool in_place = out_var == in_vars[0];

    if (out_var->IsType<framework::LoDTensor>()) {
      auto *out = context.Output<LoDTensor>("Out");
      if (!in_place) {
        out->mutable_data<T>(context.GetPlace());
      }
      auto result = EigenVector<T>::Flatten(*out);
      auto &place =
          *context.template device_context<DeviceContext>().eigen_device();
      int start = in_place ? 1 : 0;
      if (!in_place) {
        if ((in_num >= 2) && in_vars[0]->IsType<framework::LoDTensor>() &&
            in_vars[1]->IsType<framework::LoDTensor>()) {
          auto &in_0 = in_vars[0]->Get<framework::LoDTensor>();
          auto &in_1 = in_vars[1]->Get<framework::LoDTensor>();
          if (in_0.numel() && in_1.numel()) {
            auto in_0_e = EigenVector<T>::Flatten(in_0);
            auto in_1_e = EigenVector<T>::Flatten(in_1);
            result.device(place) = in_0_e + in_1_e;
            start = 2;
          }
        }
        if (start != 2) {
          math::SetConstant<DeviceContext, T> constant_functor;
          constant_functor(context.template device_context<DeviceContext>(),
                           out, 0.0);
        }
      }

      math::SelectedRowsAddToTensor<DeviceContext, T> functor;
      // If in_place, just skip the first tensor
      for (size_t i = start; i < in_num; i++) {
        if (in_vars[i]->IsType<framework::LoDTensor>()) {
          auto &in_t = in_vars[i]->Get<framework::LoDTensor>();
          if (in_t.numel() == 0) {
            continue;
          }
          auto in = EigenVector<T>::Flatten(in_t);
          result.device(place) = result + in;
        } else if (in_vars[i]->IsType<framework::SelectedRows>()) {
          auto &in_t = in_vars[i]->Get<framework::SelectedRows>();
          functor(context.template device_context<DeviceContext>(), in_t, out);
        } else {
          PADDLE_THROW("Variable type must be LoDTensor/SelectedRows.");
        }
      }
    } else if (out_var->IsType<framework::SelectedRows>()) {
      if (in_place && in_vars.size() < 2) {
        return;
      }

      std::vector<const paddle::framework::SelectedRows *> inputs;
      SelectedRows temp_in0;

      if (in_place) {
        auto &in0 = in_vars[0]->Get<SelectedRows>();
        temp_in0.set_height(in0.height());
        temp_in0.set_rows(in0.rows());
        framework::TensorCopy(in0.value(), in0.place(),
                              context.device_context(),
                              temp_in0.mutable_value());
        inputs.push_back(&temp_in0);
        for (size_t i = 1; i < in_vars.size(); ++i) {
          inputs.push_back(&in_vars[i]->Get<SelectedRows>());
        }
      } else {
        for (auto &in_var : in_vars) {
          inputs.push_back(&in_var->Get<SelectedRows>());
        }
      }

      auto *out = context.Output<SelectedRows>("Out");
      out->mutable_rows()->clear();

      math::scatter::MergeAdd<DeviceContext, T> merge_add;
      merge_add(context.template device_context<DeviceContext>(), inputs, out);
    } else if (out_var->IsType<framework::LoDTensorArray>()) {
      auto &out_array = *out_var->GetMutable<framework::LoDTensorArray>();
      for (size_t i = in_place ? 1 : 0; i < in_vars.size(); ++i) {
        PADDLE_ENFORCE(in_vars[i]->IsType<framework::LoDTensorArray>(),
                       "Only support all inputs are TensorArray");
        auto &in_array = in_vars[i]->Get<framework::LoDTensorArray>();

        for (size_t i = 0; i < in_array.size(); ++i) {
          if (in_array[i].numel() != 0) {
            if (i >= out_array.size()) {
              out_array.resize(i + 1);
            }
            if (out_array[i].numel() == 0) {
              framework::TensorCopy(in_array[i], in_array[i].place(),
                                    context.device_context(), &out_array[i]);
              out_array[i].set_lod(in_array[i].lod());
            } else {
              PADDLE_ENFORCE(out_array[i].lod() == in_array[i].lod());
              auto in = EigenVector<T>::Flatten(in_array[i]);
              auto result = EigenVector<T>::Flatten(out_array[i]);
              result.device(*context.template device_context<DeviceContext>()
                                 .eigen_device()) = result + in;
            }
          }
        }
      }
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   out_var->Type().name());
    }
  }
};
}  // namespace operators
}  // namespace paddle
