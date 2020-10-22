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

#include <unordered_set>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/sample_prob.h"
#include "paddle/fluid/operators/math/softmax.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct TolerableValue {
  HOSTDEVICE T operator()(const T& x) const {
#if defined(__CUDA_ARCH__)
    PADDLE_ENFORCE(std::is_floating_point<T>::value,
                   "TolerableValue should be float in sample_logits_op.");
#else
    PADDLE_ENFORCE(std::is_floating_point<T>::value,
                   platform::errors::InvalidArgument(
                       "TolerableValue should be float in sample_logits_op."));
#endif
    const T kApproInf = 1e20;
    if (x == INFINITY) return kApproInf;
    if (x == -INFINITY) return -kApproInf;
    return x;
  }
};

// UNDERSTAND: something like take_along_axis in numpy.
template <typename T>
static void CPUTakeAlongD1(const platform::DeviceContext& ctx,
                           const framework::Tensor& array,
                           const framework::Tensor& index,
                           framework::Tensor* value) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.GetPlace()), true,
      platform::errors::InvalidArgument("This kernel only runs on CPU."));
  // UNDERSTAND: check shape src(B, C), index(B, K), out should also be (B, K)
  const auto batch_size = index.dims()[0];
  const auto num_take = index.dims()[1];
  const auto array_dims = array.dims();
  const auto idx_dims = index.dims();
  PADDLE_ENFORCE_EQ(idx_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "index of CPUTakeAlongD1 should be 2D. "
                        "But received shape = [%s] and dimension is %d.",
                        idx_dims, idx_dims.size()));
  PADDLE_ENFORCE_EQ(array_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "array of CPUTakeAlongD1 should be 2D. "
                        "But received shape = [%s] and dimension is %d.",
                        array_dims, array_dims.size()));
  PADDLE_ENFORCE_EQ(idx_dims[0], array_dims[0],
                    platform::errors::InvalidArgument(
                        "The first dimension of index and array of "
                        "CPUTakeAlongD1 should be equal. "
                        "But received index shape = [%s], array shape = [%s], "
                        "and the first dimensions are %d and %d.",
                        idx_dims, array_dims, idx_dims[0], array_dims[0]));
  PADDLE_ENFORCE_EQ(
      idx_dims, value->dims(),
      platform::errors::InvalidArgument(
          "index and array of CPUTakeAlongD1 should have the same shape. "
          "But received index shape = [%s], array shape = [%s].",
          idx_dims, value->dims()));

  // UNDERSTAND: no allocations here
  const T* p_array = array.data<T>();
  const int64_t* p_index = index.data<int64_t>();
  T* p_value = value->data<T>();

  // src slice size
  const auto array_slice_size = array_dims[1];

  // index slice size
  const auto idx_slice_size = idx_dims[1];
  const auto value_slice_size = idx_slice_size;

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_take; ++j) {
      auto array_index = p_index[i * idx_slice_size + j];
      p_value[i * value_slice_size + j] =
          p_array[i * array_slice_size + array_index];
    }
  }
}

// UNDERSTAND: something like put_along_axis in numpy but if there is duplicate
// indices, scatter is done in += way.
template <typename T>
static void CPUPutAlongD1(const platform::DeviceContext& ctx,
                          framework::Tensor* array,
                          const framework::Tensor& index,
                          const framework::Tensor& value) {
  PADDLE_ENFORCE_EQ(
      platform::is_cpu_place(ctx.GetPlace()), true,
      platform::errors::InvalidArgument("This kernel only runs on CPU."));
  // UNDERSTAND: check shape src(B, C), index(B, K), out should also be (B, K)
  const auto batch_size = index.dims()[0];
  const auto num_put = index.dims()[1];
  auto array_dims = array->dims();
  auto idx_dims = index.dims();
  PADDLE_ENFORCE_EQ(idx_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "index of CPUPutAlongD1 should be 2D. "
                        "But received shape = [%s] and dimension is %d.",
                        idx_dims, idx_dims.size()));
  PADDLE_ENFORCE_EQ(array_dims.size(), 2,
                    platform::errors::InvalidArgument(
                        "array of CPUPutAlongD1 should be 2D. "
                        "But received shape = [%s] and dimension is %d.",
                        array_dims, array_dims.size()));
  PADDLE_ENFORCE_EQ(idx_dims[0], array_dims[0],
                    platform::errors::InvalidArgument(
                        "The first dimension of index and array of "
                        "CPUPutAlongD1 should be equal. "
                        "But received index shape = [%s], array shape = [%s], "
                        "and the first dimensions are %d and %d.",
                        idx_dims, array_dims, idx_dims[0], array_dims[0]));
  PADDLE_ENFORCE_EQ(
      idx_dims, value.dims(),
      platform::errors::InvalidArgument(
          "index and array of CPUPutAlongD1 should have the same shape. "
          "But received index shape = [%s], array shape = [%s].",
          idx_dims, value.dims()));

  // UNDERSTAND: no allocations here
  T* p_array = array->data<T>();
  const int64_t* p_index = index.data<int64_t>();
  const T* p_value = value.data<T>();

  // slice sizes
  const auto array_slice_size = array_dims[1];
  const auto idx_slice_size = idx_dims[1];
  const auto value_slice_size = idx_slice_size;

  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < num_put; ++j) {
      auto array_index = p_index[i * idx_slice_size + j];
      p_array[i * array_slice_size + array_index] +=
          p_value[i * value_slice_size + j];
    }
  }
}

// UNDERSTAND: compute accidentdal hits from samples and minus corresponding
// logits by a float max, here 1e20
template <typename T>
static void compute_remove_accidental_hits(const platform::DeviceContext& ctx,
                                           framework::Tensor* sampled_logits,
                                           const framework::Tensor& samples,
                                           const int num_true) {
  const auto batch_size = sampled_logits->dims()[0];
  const auto num_sampled_classes = sampled_logits->dims()[1];
  T* sampled_logits_data = sampled_logits->data<T>();
  const auto samples_data = samples.data<int64_t>();

  std::unordered_set<int64_t> tmp_true_labels;
  for (int i = 0; i < batch_size; ++i) {
    tmp_true_labels.clear();
    tmp_true_labels.insert(samples_data + i * num_sampled_classes,
                           samples_data + i * num_sampled_classes + num_true);
    for (int j = num_true; j < num_sampled_classes; ++j) {
      const auto idx = i * num_sampled_classes + j;
      if (tmp_true_labels.find(samples_data[idx]) != tmp_true_labels.end())
        sampled_logits_data[idx] -= 1e20;
    }
  }
}

template <typename T>
class SampleLogitsKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_cpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument("this kernel only runs on cpu."));
    VLOG(3) << "Enter SampleLogitsKernel";
    // get necessary inputs
    const Tensor* logits = context.Input<Tensor>("Logits");
    const Tensor* labels = context.Input<Tensor>("Labels");

    // get necessary outputs
    Tensor* samples = context.Output<Tensor>("Samples");
    Tensor* probabilities = context.Output<Tensor>("Probabilities");
    Tensor* sampled_logits = context.Output<Tensor>("SampledLogits");
    Tensor* sampled_labels = context.Output<Tensor>("SampledLabels");

    // shapes
    const auto batch_size = logits->dims()[0];
    const auto num_classes = logits->dims()[1];
    const auto labels_dim = labels->dims();
    const auto num_true = labels_dim[1];
    const auto samples_dim = samples->dims();

    // attrs
    const auto num_samples = context.Attr<int>("num_samples");
    const bool use_customized_samples =
        context.Attr<bool>("use_customized_samples");
    const bool remove_accidental_hits =
        context.Attr<bool>("remove_accidental_hits");

    // device contexts
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();

    // UNDERSTAND: allocate memories for temporaries
    sampled_logits->mutable_data<T>(samples_dim, context.GetPlace());
    auto sampled_labels_data =
        sampled_labels->mutable_data<int64_t>(labels_dim, context.GetPlace());
    for (int i = 0; i < batch_size; ++i) {
      for (int j = 0; j < num_true; ++j) {
        sampled_labels_data[i * num_true + j] = j;
      }
    }

    if (use_customized_samples) {
      const Tensor* customized_samples =
          context.Input<Tensor>("CustomizedSamples");
      const Tensor* customized_probabilities =
          context.Input<Tensor>("CustomizedProbabilities");
      PADDLE_ENFORCE_EQ(customized_samples, samples,
                        platform::errors::InvalidArgument(
                            "CustomizedSamples must be the same Tensor with "
                            "Samples when use_customized_samples = True"));
      PADDLE_ENFORCE_EQ(
          customized_probabilities, probabilities,
          platform::errors::InvalidArgument(
              "CustomizedProbabilities must be the same Tensor with "
              "Probabilities when use_customized_samples = True"));
    } else {
      samples->mutable_data<int64_t>(context.GetPlace());
      probabilities->mutable_data<T>(samples_dim, context.GetPlace());
      // UNDERSTAND: sampling
      const auto seed = context.Attr<int>("seed");
      auto sampler_with_prob =
          math::SampleWithProb<platform::CPUDeviceContext, T>();
      sampler_with_prob(dev_ctx, math::LogUniformSampler(num_classes, seed),
                        num_samples, labels, samples, probabilities);
    }

    // UNDERSTAND: gather sampled logits and remove accidental hits if needed
    CPUTakeAlongD1<T>(dev_ctx, *logits, *samples, sampled_logits);
    if (remove_accidental_hits) {
      compute_remove_accidental_hits<T>(dev_ctx, sampled_logits, *samples,
                                        num_true);
    }

    // subtracted sampled logits with logQ(y|x)
    auto probs = EigenMatrix<T>::From(*probabilities);
    auto smp_logits = EigenMatrix<T>::From(*sampled_logits);
    smp_logits.device(*dev_ctx.eigen_device()) =
        (smp_logits - probs.log().unaryExpr(TolerableValue<T>()))
            .unaryExpr(TolerableValue<T>());
  }
};

template <typename T>
class SampleLogitsGradKernel : public framework::OpKernel<T> {
 public:
  using Tensor = framework::Tensor;
  void Compute(const framework::ExecutionContext& context) const override {
    auto logits_grad = context.Output<Tensor>(framework::GradVarName("Logits"));
    const Tensor* samples = context.Input<Tensor>("Samples");
    const Tensor* sampled_logits_grad =
        context.Input<Tensor>(framework::GradVarName("SampledLogits"));
    logits_grad->mutable_data<T>(context.GetPlace());

    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, logits_grad, static_cast<T>(0));

    // UNDERSTAND: scatter it back to logit_grad
    CPUPutAlongD1<T>(dev_ctx, logits_grad, *samples, *sampled_logits_grad);
  }
};

}  // namespace operators
}  // namespace paddle
