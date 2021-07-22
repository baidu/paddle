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

#include <cstring>
#include <random>
#include <string>

#include <algorithm>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/jit/registry.h"
#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/gpu_launch_config.h"

namespace paddle {
namespace operators {

// aligned vector generates vectorized load/store on CUDA
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

#if defined(__NVCC__) || defined(__HIPCC__)
template <typename T, typename MaskType, int VecSize>
__global__ void DropoutGradCUDAKernel(const T* dout, const MaskType* mask,
                                      const T factor, const int64_t size,
                                      T* dx) {
  int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = AlignedVector<T, VecSize>;
  using MaskLoadT = AlignedVector<MaskType, VecSize>;

  for (int i = idx * VecSize; i < size; i += blockDim.x * gridDim.x * VecSize) {
    T dout_vec[VecSize];
    LoadT* dout_value = reinterpret_cast<LoadT*>(&dout_vec);
    *dout_value = *reinterpret_cast<const LoadT*>(&dout[i]);

    MaskType mask_vec[VecSize];
    MaskLoadT* mask_value = reinterpret_cast<MaskLoadT*>(&mask_vec);
    *mask_value = *reinterpret_cast<const MaskLoadT*>(&mask[i]);

    T dx_vec[VecSize];

#pragma unroll
    for (int ii = 0; ii < VecSize; ii++) {
      dx_vec[ii] = dout_vec[ii] * static_cast<T>(mask_vec[ii]) * factor;
    }

    *(reinterpret_cast<LoadT*>(&dx[i])) = *reinterpret_cast<LoadT*>(&dx_vec[0]);
  }
}
#endif

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;

template <typename DeviceContext, typename T>
class CPUDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* seed =
        context.HasInput("Seed") ? context.Input<Tensor>("Seed") : nullptr;
    auto* y = context.Output<Tensor>("Out");
    const auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());
    const float dropout_prob = context.Attr<float>("dropout_prob");

    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    bool upscale_in_train = (dropout_implementation == "upscale_in_train");
    if (!context.Attr<bool>("is_test")) {
      auto* mask = context.Output<Tensor>("Mask");
      auto* mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      int size = framework::product(mask->dims());

      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        std::memset(y_data, 0, size * sizeof(*y_data));        // NOLINT
        std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
        return;
      }

      // std::minstd_rand engine;
      // NOTE: fixed seed should only be used in unittest or for debug.
      // Guarantee to use random seed in training.
      int seed_data = 0;
      if (seed) {
        seed_data = *(seed->data<int>());
      } else {
        seed_data =
            context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : 0;
      }
      auto engine = framework::GetCPURandomEngine_32(seed_data);
      float factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
      std::uniform_real_distribution<float> dist(0, 1);

#ifdef __AVX__
      float* mask_temp = new float[size];
      constexpr unsigned int block = YMM_FLOAT_BLOCK;
      int end = size & ~(block - 1);
      int i = 0;
      __m256 one = _mm256_set1_ps(1.0f);
      __m256 _factor = _mm256_set1_ps(factor);
      __m256 _dropout = _mm256_set1_ps(dropout_prob);
      for (i = 0; i < end; i += block) {
        __m256 _rand = _mm256_set_ps(
            dist(*engine), dist(*engine), dist(*engine), dist(*engine),
            dist(*engine), dist(*engine), dist(*engine), dist(*engine));
        __m256 _mask =
            _mm256_and_ps(_mm256_cmp_ps(_rand, _dropout, _CMP_GE_OS), one);
        _mm256_storeu_ps(reinterpret_cast<float*>(mask_temp + i), _mask);
        if (upscale_in_train) {
          __m256 _temp = _mm256_mul_ps(
              _mm256_mul_ps(_mm256_loadu_ps((const float*)x_data + i), _mask),
              _factor);
          _mm256_storeu_ps(reinterpret_cast<float*>(y_data) + i, _temp);
        } else {
          _mm256_storeu_ps(
              reinterpret_cast<float*>(y_data) + i,
              _mm256_mul_ps(_mm256_loadu_ps((const float*)x_data + i), _mask));
        }
      }

      for (; i < size; ++i) {
        float rand = dist(*engine);
        if (rand < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (upscale_in_train) {
            y_data[i] = x_data[i] * factor;
          } else {
            y_data[i] = x_data[i];
          }
        }
      }
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
      for (int i = 0; i < end; i++) {
        mask_data[i] = static_cast<uint8_t>(mask_temp[i]);
      }
#else
      int i = 0;
      float rand_number = 0.0f;
      for (i = 0; i < size; ++i) {
        rand_number = dist(*engine);
        if (rand_number < dropout_prob) {
          mask_data[i] = 0;
          y_data[i] = 0;
        } else {
          mask_data[i] = 1;
          if (upscale_in_train) {
            y_data[i] = x_data[i] * factor;
          } else {
            y_data[i] = x_data[i];
          }
        }
      }
#endif
    } else {
      if (upscale_in_train) {
        const auto* X_data = x->data<T>();
        auto* Y_data = y->mutable_data<T>(context.GetPlace());
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
        for (int i = 0; i < x->numel(); i++) {
          Y_data[i] = X_data[i];
        }
      } else {
        auto X = EigenMatrix<T>::Reshape(*x, 1);
        auto Y = EigenMatrix<T>::Reshape(*y, 1);
        auto& place =
            *context.template device_context<DeviceContext>().eigen_device();
        Y.device(place) = X * static_cast<T>(1.0f - dropout_prob);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class DropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(!context.Attr<bool>("is_test"), true,
                      platform::errors::PreconditionNotMet(
                          "GradOp is only callable when is_test is false"));

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* mask = context.Input<Tensor>("Mask");
    grad_x->mutable_data<T>(context.GetPlace());
    auto size = grad_x->numel();

    auto M = EigenVector<uint8_t>::Flatten(*mask);
    auto dX = EigenVector<T>::Flatten(*grad_x);
    auto dY = EigenVector<T>::Flatten(*grad_y);

    auto& place =
        *context.template device_context<DeviceContext>().eigen_device();
    auto& dropout_implementation =
        context.Attr<std::string>("dropout_implementation");
    if (dropout_implementation == "upscale_in_train") {
      float dropout_prob = context.Attr<float>("dropout_prob");
      if (dropout_prob == 1.0f) {
        dX.device(place) = static_cast<T>(0) * dY;
      } else {
        int vec_size = VectorizedSize<T>(grad_y->data<T>());
        if (platform::is_gpu_place(context.GetPlace()) && vec_size == 4 &&
            size % 4 == 0) {
#if defined(__NVCC__) || defined(__HIPCC__)
          auto factor = static_cast<T>(1.0f / (1.0f - dropout_prob));
          auto stream = context.cuda_device_context().stream();
          platform::GpuLaunchConfig config = platform::GetGpuLaunchConfig1D(
              context.cuda_device_context(), size);
          DropoutGradCUDAKernel<
              T, uint8_t,
              4><<<config.block_per_grid, config.thread_per_block, 0, stream>>>(
              grad_y->data<T>(), mask->data<uint8_t>(), factor, size,
              grad_x->data<T>());
#endif
        } else {
          dX.device(place) =
              dY * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
        }
      }
    } else {
      dX.device(place) = dY * M.cast<T>();
    }
  }
};

}  // namespace operators
}  // namespace paddle
