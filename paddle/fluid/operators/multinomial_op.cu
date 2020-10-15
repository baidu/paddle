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

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/multinomial_op.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void NormalizeProbability(T* norm_probs, const T* in_data,
                                     T* sum_rows) {
  int id = threadIdx.x + blockIdx.x * blockDim.x +
           blockIdx.y * gridDim.x * blockDim.x;
  norm_probs[id] = in_data[id] / sum_rows[blockIdx.y];
}

template <typename T>
__global__ void GetCumulativeProbs(T* norm_probs_data,
                                   int64_t num_distributions,
                                   int64_t num_categories,
                                   T* cumulative_probs) {
  for (int id = blockIdx.x; id < num_distributions; id += gridDim.x) {
    thrust::inclusive_scan(thrust::device,
                           norm_probs_data + id * num_categories,
                           norm_probs_data + (id + 1) * num_categories,
                           cumulative_probs + id * num_categories);
  }
}

template <typename T>
struct RandomGeneratorCudaFunctor {
  unsigned int seed_;
  __host__ __device__ RandomGeneratorCudaFunctor(int seed) : seed_(seed) {}

  __host__ __device__ T operator()(const unsigned int n) const {
    thrust::minstd_rand rng;
    rng.seed(seed_);
    thrust::uniform_real_distribution<T> dist(0.0, 1.0);
    rng.discard(n);
    return dist(rng);
  }
};

template <typename T>
__device__ int binarySearchFunctor(T* cumulative_probs, T* norm_probs_data,
                                   int num_categories, T rng_number) {
  int left = 0;
  int right = num_categories;

  while (right - left > 0) {
    int mid = left + (right - left) / 2;

    T temp_prob = cumulative_probs[mid];
    if (temp_prob < rng_number) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }

  if (left == num_categories) {
    left = num_categories - 1;
  }

  while (left >= 1 && norm_probs_data[left] == 0) left--;

  return left;
}

template <typename T>
__global__ void sampleMultinomialWithReplacement(
    T* rng_data, const int64_t num_samples, int64_t* out_data,
    const int64_t num_distributions, const int64_t num_categories,
    T* cumulative_probs, T* norm_probs_data) {
  // use binary search to get the selected category sample id.
  // let cumulative_probs[id-1] < rng_data < cumulative_probs[id].

  int idx = threadIdx.x + blockIdx.x * blockDim.x +
            blockIdx.y * gridDim.x * blockDim.x;

  // for every distribution
  for (int dist = blockIdx.y; dist < num_distributions; dist += gridDim.y) {
    // for every sample
    for (int sample = blockIdx.x * blockDim.x + threadIdx.x;
         sample < num_samples; sample += blockDim.x * gridDim.x) {
      T rng_number = rng_data[sample + dist * num_samples];

      // Find the bucket that a uniform random number lies in
      int selected_category = binarySearchFunctor<T>(
          cumulative_probs + dist * num_categories,
          norm_probs_data + dist * num_categories, num_categories, rng_number);

      out_data[sample + dist * num_samples] = selected_category;
    }
  }
}

template <typename T>
class MultinomialOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const auto x = ctx.Input<framework::Tensor>("X");
    auto out = ctx.Output<framework::Tensor>("Out");

    const int64_t num_samples = ctx.Attr<int>("num_samples");
    const bool replacement = ctx.Attr<bool>("replacement");

    auto* in_data = x->data<T>();
    int64_t* out_data = out->mutable_data<int64_t>(ctx.GetPlace());

    auto in_dims = x->dims();
    int64_t in_rank = in_dims.size();
    const int64_t num_categories = in_dims[in_rank - 1];
    const int64_t num_distributions = in_rank > 1 ? in_dims[in_rank - 2] : 1;

    // If replacement is False, it's not a replaceable sample. Every category
    // can
    // be used only once. So after every sample, probability of the distribution
    // will change. The implementation can't be parallelizable. Thus, call CPU
    // implementation ``MultinomialFunctor`` to sample the distribution.
    if (!replacement) {
      int64_t in_data_numel = x->numel();
      int64_t out_data_numel = out->numel();

      T* cpu_in_data = new T[in_data_numel];
      int64_t* cpu_out_data = new int64_t[out_data_numel];

      cudaMemcpy(cpu_in_data, in_data, in_data_numel * sizeof(T),
                 cudaMemcpyDeviceToHost);

      MultinomialFunctor<T>(cpu_out_data, cpu_in_data, num_samples, replacement,
                            num_categories, num_distributions);
      cudaMemcpy(out_data, cpu_out_data, out_data_numel * sizeof(int64_t),
                 cudaMemcpyHostToDevice);

      delete[] cpu_in_data;
      delete[] cpu_out_data;
      return;
    }

    // Sum of input may not be 1. To get probability in range [0, 1], calculate
    // sum of each row of input, and then use the sum to normalize the input.
    // sum_row_data: sum of each row
    framework::Tensor sum_rows_tensor;
    auto* sum_rows_data =
        sum_rows_tensor.mutable_data<T>({num_distributions}, ctx.GetPlace());

    auto& place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();

    if (num_distributions == 1) {
      auto eigen_input = framework::EigenVector<T>::Flatten(*x);
      auto eigen_sum_rows = framework::EigenVector<T>::Flatten(sum_rows_tensor);
      eigen_sum_rows.device(place) =
          eigen_input.sum(Eigen::DSizes<int, 1>(1))
              .eval()
              .reshape(Eigen::DSizes<int, 1>(sum_rows_tensor.dims()[0]));
    } else {
      auto eigen_input = framework::EigenMatrix<T>::From(*x);
      auto eigen_sum_rows = framework::EigenVector<T>::Flatten(sum_rows_tensor);
      eigen_sum_rows.device(place) = eigen_input.sum(Eigen::DSizes<int, 1>(1));
    }

    // Normalize row of each distribution to get the probability in range [0,
    // 1].
    // norm_probs_data: probability of the distribution
    framework::Tensor norm_probs_tensor;
    auto* norm_probs_data = norm_probs_tensor.mutable_data<T>(
        {num_distributions, num_categories}, ctx.GetPlace());

    // number of threads in a block is min(num_categories, 512)
    dim3 block_norm(num_categories < 512 ? num_categories : 512);
    dim3 grid_norm((num_categories - 1) / block_norm.x + 1, num_distributions);
    NormalizeProbability<
        T><<<grid_norm, block_norm, 0, ctx.cuda_device_context().stream()>>>(
        norm_probs_data, in_data, sum_rows_data);

    // Get cumulative probability of each distribution. It's the same function
    // of
    // ``cumsum`` op.
    framework::Tensor cumulative_probs_tensor;
    auto* cumulative_probs = cumulative_probs_tensor.mutable_data<T>(
        {num_distributions, num_categories}, ctx.GetPlace());
    dim3 block_cumsum(1);
    dim3 grid_cumsum(num_distributions);
    GetCumulativeProbs<T><<<grid_cumsum, block_cumsum, 0,
                            ctx.cuda_device_context().stream()>>>(
        norm_probs_data, num_distributions, num_categories, cumulative_probs);

    // Generate random number for each sample.
    std::random_device rd;
    auto seed = rd();

    framework::Tensor rng_data_tensor;
    auto* rng_data = rng_data_tensor.mutable_data<T>(
        {num_distributions, num_samples}, ctx.GetPlace());

    thrust::counting_iterator<unsigned int> index_sequence_begin(0);
    platform::Transform<platform::CUDADeviceContext> trans;
    auto* context =
        static_cast<const platform::CUDADeviceContext*>(&ctx.device_context());
    trans(*context, index_sequence_begin,
          index_sequence_begin + num_distributions * num_samples, rng_data,
          RandomGeneratorCudaFunctor<T>(seed));

    // Sample the multinomial distributions.
    dim3 block_sample(128);
    dim3 grid_sample((num_samples - 1) / block_sample.x + 1, num_distributions);
    sampleMultinomialWithReplacement<T><<<grid_sample, block_sample, 0,
                                          ctx.cuda_device_context().stream()>>>(
        rng_data, num_samples, out_data, num_distributions, num_categories,
        cumulative_probs, norm_probs_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    multinomial, ops::MultinomialOpKernel<plat::CUDADeviceContext, float>,
    ops::MultinomialOpKernel<plat::CUDADeviceContext, double>);
