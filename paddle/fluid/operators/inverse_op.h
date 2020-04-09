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

#pragma once

#include "Eigen/Core"
#include "Eigen/LU"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class InverseKernel : public framework::OpKernel<T> {
  using Matrix =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using EigenMatrixMap = Eigen::Map<Matrix>;
  using ConstEigenMatrixMap = Eigen::Map<const Matrix>;

 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* output = context.Output<framework::Tensor>("Output");
    output->mutable_data<T>(context.GetPlace());

    if (platform::is_cpu_place(context.GetPlace())) {
#ifndef PADDLE_WITH_MKLML
      const auto& input_dims = input->dims();
      const int rank = input_dims.size();
      int N = input_dims[rank - 1];
      int batch_size = rank > 2 ? input->numel() / (N * N) : 1;

      const T* input_ptr = input->data<T>();
      T* output_ptr = output->mutable_data<T>(context.GetPlace());
      for (int i = 0; i < batch_size; ++i) {
        ConstEigenMatrixMap mat(input_ptr + i * N * N, N, N);
        EigenMatrixMap mat_inv(output_ptr + i * N * N, N, N);
        Eigen::PartialPivLU<Matrix> lu;
        lu.compute(mat);

        const T min_abs_pivot = lu.matrixLU().diagonal().cwiseAbs().minCoeff();
        PADDLE_ENFORCE_GT(
            min_abs_pivot, static_cast<T>(0),
            platform::errors::InvalidArgument("Input is not invertible."));
        mat_inv.noalias() = lu.inverse();
      }
      return;
#endif
    }

    auto blas = math::GetBlas<DeviceContext, T>(context);
    blas.MatInv(*input, output);
  }
};

}  // namespace operators
}  // namespace paddle
