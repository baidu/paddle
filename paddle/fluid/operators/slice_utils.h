/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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
#include <paddle/fluid/framework/operator.h>
#include <string>
#include <vector>

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T = int64_t>
inline void CheckAndUpdateSliceAttrs(const framework::DDim in_dims,
                                     const std::vector<T>& axes,
                                     std::vector<T>* starts,
                                     std::vector<T>* ends,
                                     std::vector<int64_t>* steps = nullptr,
                                     std::vector<T>* infer_flags = nullptr) {
  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    T dim_value = in_dims[axis];

    if (dim_value > 0) {
      if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
        continue;
      }
      T start = (*starts)[i] < 0 ? ((*starts)[i] + dim_value) : (*starts)[i];
      start = std::max(start, static_cast<T>(0));

      T end = (*ends)[i] < 0 ? ((*ends)[i] + dim_value) : (*ends)[i];
      end = std::min(end, dim_value);

      T step = steps == nullptr ? 1 : (*steps)[i];
      PADDLE_ENFORCE_NE(
          step, 0, platform::errors::InvalidArgument(
                       "Step should not be 0, but received step = %d.", step));

      if (step > 0) {
        start = std::min(start, dim_value);
        end = std::max(end, static_cast<T>(0));
        PADDLE_ENFORCE_GT(
            end, start,
            platform::errors::InvalidArgument(
                "When step > 0, end should be greater than start, but "
                "received end = %d, start = %d.",
                end, start));
      } else {
        // NOTE(liym27): When step < 0, start should less and equal to
        // dim_value-1
        // "end is -1" means contain the 0-th element of this axis.
        start = std::min(start, dim_value - 1);
        end = std::max(end, static_cast<T>(-1));
        PADDLE_ENFORCE_GT(
            start, end,
            platform::errors::InvalidArgument(
                "When step < 0, start should be greater than end, but "
                "received start = %d, end = %d.",
                start, end));
      }

      (*starts)[i] = start;
      (*ends)[i] = end;
    }
  }
}

template <typename T = int64_t>
inline framework::DDim GetSliceDims(const framework::DDim in_dims,
                                    const std::vector<T>& axes,
                                    const std::vector<T>& starts,
                                    const std::vector<T>& ends,
                                    std::vector<T>* steps = nullptr,
                                    std::vector<T>* infer_flags = nullptr) {
  framework::DDim slice_dims(in_dims);

  for (size_t i = 0; i < axes.size(); ++i) {
    T axis = axes[i];
    if (infer_flags != nullptr && (*infer_flags)[i] == -1) {
      slice_dims[axis] = -1;
      continue;
    }

    T start = starts[i];
    T end = ends[i];
    T step = steps == nullptr ? 1 : (*steps)[i];

    if (step > 0) {
      slice_dims[axis] = (end - start + step - 1) / step;
    } else {
      slice_dims[axis] = (end - start + step + 1) / step;
    }
  }
  return slice_dims;
}

template <typename T = int64_t>
inline framework::DDim GetDecreasedDims(const framework::DDim slice_dims,
                                        const std::vector<T>& decrease_axes,
                                        std::vector<T>* infer_flags = nullptr) {
  framework::DDim decreased_dims(slice_dims);
  if (decrease_axes.size() > 0) {
    for (size_t i = 0; i < decrease_axes.size(); ++i) {
      T axis = decrease_axes[i];
      if (infer_flags && (*infer_flags)[i] != -1) {
        PADDLE_ENFORCE_EQ(
            decreased_dims[axis], 1,
            platform::errors::InvalidArgument("decrease dim should be 1"));
      }
      decreased_dims[axis] = 0;
    }

    std::vector<T> new_shape;
    for (int i = 0; i < decreased_dims.size(); ++i) {
      if (decreased_dims[i] != 0) {
        new_shape.push_back(decreased_dims[i]);
      }
    }

    // NOTE(liym27): Paddle does not support that the rank of Tensor is 0, and
    // uses [1] instead.
    if (new_shape.size() == 0) {
      new_shape.push_back(1);
    }

    decreased_dims = framework::make_ddim(new_shape);
  }
  return decreased_dims;
}

}  // namespace operators
}  // namespace paddle
