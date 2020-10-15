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
#include <vector>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/hostdevice.h"

namespace paddle {
namespace operators {

/**
 * Samples a multinomial distribution given a probability input
 */

template <typename T>
void MultinomialFunctor(int64_t* out_data, const T* in_data,
                        const int64_t num_samples, const bool replacement,
                        const int64_t num_categories,
                        const int64_t num_distributions) {
  std::vector<T> cumulative_probs(num_categories);

  std::uniform_real_distribution<T> dist(0, 1);
  auto gen_ptr = framework::DefaultCPUGenerator();
  auto engine = gen_ptr->GetCPUEngine();

  for (int64_t i = 0; i < num_distributions; i++) {
    T probs_sum = 0;
    T prob_value;
    int64_t num_zeros = 0;
    for (int64_t j = 0; j < num_categories; j++) {
      prob_value = in_data[i * num_categories + j];
      PADDLE_ENFORCE_GE(
          prob_value, 0.0,
          platform::errors::OutOfRange(
              "The input of multinomial distribution should be >= 0"));
      PADDLE_ENFORCE_EQ((std::isinf(static_cast<double>(prob_value)) ||
                         std::isnan(static_cast<double>(prob_value))),
                        false, platform::errors::OutOfRange(
                                   "The input of multinomial distribution "
                                   "shoud not be infinity or NaN"));
      probs_sum += prob_value;
      if (prob_value == 0) {
        num_zeros += 1;
      }
      cumulative_probs[j] = probs_sum;
    }
    PADDLE_ENFORCE_GT(probs_sum, 0.0, platform::errors::OutOfRange(
                                          "The sum of input should not be 0"));
    PADDLE_ENFORCE_EQ(
        (replacement || (num_categories - num_zeros >= num_samples)), true,
        platform::errors::OutOfRange("When replacement is False, number of "
                                     "samples should be less than non-zero "
                                     "categories"));

    for (int64_t j = 0; j < num_categories; j++) {
      cumulative_probs[j] /= probs_sum;
    }

    for (int64_t s = 0; s < num_samples; s++) {
      T uniform_rand = dist(*engine);
      // use binary search to get the selected category sample id.
      // let cumulative_probs[id-1] < uniform_rand < cumulative_probs[id].
      int64_t left = 0;
      int64_t right = num_categories;
      int64_t mid;
      int64_t sample_id;
      T temp_prob;
      cumulative_probs[(num_categories - 1)] = 1;

      while (right > left) {
        mid = left + (right - left) / 2;
        temp_prob = cumulative_probs[mid];
        if (temp_prob < uniform_rand) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }
      sample_id = left;

      out_data[i * num_samples + s] = sample_id;

      // if replacement is false, the selected category should be removed.
      if (!replacement && s < num_samples - 1) {
        T sample_prob;
        T new_prob = 0;
        T new_sum;

        if (sample_id != 0) {
          new_prob = cumulative_probs[sample_id - 1];
        }
        sample_prob = cumulative_probs[sample_id] - new_prob;
        new_sum = 1.0 - sample_prob;

        for (int64_t j = 0; j < num_categories; j++) {
          new_prob = cumulative_probs[j];
          if (j >= sample_id) {
            new_prob -= sample_prob;
          }
          new_prob /= new_sum;
          cumulative_probs[j] = new_prob;
        }
      }
    }
  }
}

template <typename DeviceContext, typename T>
class MultinomialOpKernel;

}  // namespace operators
}  // namespace paddle
