//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <map>
#include <set>
#include <vector>
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
template <typename T>
class ClassCenterSampleCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* label = ctx.Input<Tensor>("Label");
    auto* remapped_label = ctx.Output<Tensor>("RemappedLabel");
    auto* sampled_local_class_center =
        ctx.Output<Tensor>("SampledLocalClassCenter");
    int num_classes = ctx.Attr<int>("num_classes");
    int num_samples = ctx.Attr<int>("num_samples");

    int seed = ctx.Attr<int>("seed");
    bool fix_seed = ctx.Attr<bool>("fix_seed");
    PADDLE_ENFORCE_GT(num_classes, 0,
                      platform::errors::InvalidArgument(
                          "The value 'num_classes' for Op(class_center_sample) "
                          "must be greater than 0, "
                          "but the value given is %d.",
                          num_classes));

    PADDLE_ENFORCE_GT(num_samples, 0,
                      platform::errors::InvalidArgument(
                          "The value 'num_samples' for Op(class_center_sample) "
                          "must be greater than 0, "
                          "but the value given is %d.",
                          num_samples));

    PADDLE_ENFORCE_LE(num_samples, num_classes,
                      platform::errors::InvalidArgument(
                          "The value 'num_samples' for Op(class_center_sample) "
                          "must be less than or equal to %d, "
                          "but the value given is %d.",
                          num_classes, num_samples));

    int64_t numel = label->numel();
    auto* label_ptr = label->data<T>();

    // get unique positive class center by ascending
    std::set<T, std::less<T>> unique_label;
    for (int64_t i = 0; i < numel; ++i) {
      unique_label.insert(label_ptr[i]);
    }

    // constrcut a lookup table and get sampled_local_class_center
    int actual_num_samples = unique_label.size();
    T* sampled_local_class_center_ptr =
        sampled_local_class_center->mutable_data<T>({actual_num_samples},
                                                    ctx.GetPlace());
    std::map<T, T> new_class_dict;
    T idx = 0;
    for (auto& t : unique_label) {
      new_class_dict[t] = idx;
      sampled_local_class_center_ptr[idx] = t;
      idx++;
    }

    if (!fix_seed) {
      std::random_device rnd;
      seed = rnd();
    }
    std::uniform_int_distribution<T> dist(0, num_classes - 1);
    auto engine = framework::GetCPURandomEngine(seed);
    // sample negative class center randomly
    while (unique_label.size() < static_cast<size_t>(num_samples)) {
      unique_label.insert(dist(*engine));
    }

    // remap the input label to sampled class
    auto* remmaped_label_ptr = remapped_label->mutable_data<T>(ctx.GetPlace());
    for (int64_t i = 0; i < numel; ++i) {
      remmaped_label_ptr[i] = new_class_dict[label_ptr[i]];
    }
  }
};

}  // namespace operators
}  // namespace paddle
