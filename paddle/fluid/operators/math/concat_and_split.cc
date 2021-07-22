/* Copyright (c) 2018 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/math/concat_and_split.h"

namespace paddle {
namespace framework {
class Tensor;
}  // namespace framework
namespace platform {
class CPUDeviceContext;
struct bfloat16;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {
namespace math {

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    // TODO(zcd): Add input data validity checking
    size_t num = input.size();

    int64_t rows = 1;
    auto dim_0 = input[0].dims();
    for (int i = 0; i < axis; ++i) {
      rows *= dim_0[i];
    }
    int64_t out_rows = rows, out_cols = 0;

    std::vector<int64_t> input_cols(input.size());
    for (size_t i = 0; i < num; ++i) {
      int64_t t_cols = input[i].numel() / rows;
      out_cols += t_cols;
      input_cols[i] = t_cols;
    }
    auto cpu_place = BOOST_GET_CONST(platform::CPUPlace, context.GetPlace());

    // computation
    auto output_data = output->data<T>();
    int64_t col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int64_t col_len = input_cols[j];
      auto input_data = input[j].data<T>();
      for (int64_t k = 0; k < out_rows; ++k) {
        memory::Copy(cpu_place, output_data + k * out_cols + col_idx, cpu_place,
                     input_data + k * col_len, sizeof(T) * col_len);
      }
      col_idx += col_len;
    }
  }
};

/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class SplitFunctor<platform::CPUDeviceContext, T> {
 public:
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    // TODO(zcd): Add input data validity checking
    size_t num = outputs->size();

    int input_rows = 1;
    auto dim_0 = ref_inputs[0]->dims();
    for (int i = 0; i < axis; ++i) {
      input_rows *= dim_0[i];
    }

    int input_cols = 0;

    std::vector<int64_t> output_cols(outputs->size());
    for (size_t i = 0; i < num; ++i) {
      int t_cols = ref_inputs[i]->numel() / input_rows;
      input_cols += t_cols;
      output_cols[i] = t_cols;
    }
    auto cpu_place = BOOST_GET_CONST(platform::CPUPlace, context.GetPlace());

    // computation
    for (int k = 0; k < input_rows; ++k) {
      const T* src_ptr = input.data<T>() + k * input_cols;
      int col_idx = 0;
      for (size_t j = 0; j < num; ++j) {
        int col_len = output_cols[j];
        auto* out_tensor = outputs->at(j);
        if (out_tensor != nullptr) {
          T* dst_ptr = out_tensor->data<T>() + k * col_len;
          memory::Copy(cpu_place, dst_ptr, cpu_place, src_ptr + col_idx,
                       sizeof(T) * col_len);
        }
        col_idx += col_len;
      }
    }
  }
};

#ifdef PADDLE_WITH_XPU
/*
 * All tensors' dimension should be the same and the values of
 * each dimension must be the same, except the axis dimension.
 */
template <typename T>
class ConcatFunctor<platform::XPUDeviceContext, T> {
 public:
  void operator()(const platform::XPUDeviceContext& context,
                  const std::vector<framework::Tensor>& input, int axis,
                  framework::Tensor* output) {
    int dev_id =
        BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()).GetDeviceId();
    platform::XPUDeviceGuard guard(dev_id);

    int num = input.size();
    auto input_dims = input[0].dims();

    std::vector<std::vector<int>> xdims_list(num);
    for (int i = 0; i < num; ++i) {
      std::vector<int> tmp_dims(input_dims.size());
      for (int j = 0; j < input_dims.size(); ++j) {
        tmp_dims[j] = input[i].dims()[j];
      }
      xdims_list[i] = tmp_dims;
    }

    std::vector<const T*> ptrs;
    for (int i = 0; i < num; ++i) {
      ptrs.push_back(input[i].data<T>());
    }

    auto r = xpu::concat<T>(context.x_context(), ptrs, output->data<T>(),
                            xdims_list, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r, XPUAPIErrorMsg[r]));
  }
};

template <typename T>
class SplitFunctor<platform::XPUDeviceContext, T> {
 public:
  void operator()(const platform::XPUDeviceContext& context,
                  const framework::Tensor& input,
                  const std::vector<const framework::Tensor*>& ref_inputs,
                  const int axis, std::vector<framework::Tensor*>* outputs) {
    int dev_id =
        BOOST_GET_CONST(platform::XPUPlace, context.GetPlace()).GetDeviceId();
    platform::XPUDeviceGuard guard(dev_id);

    auto& ins = ref_inputs;

    int num = ins.size();
    auto input_dims = ins[0]->dims();
    std::vector<int> split_list(num);
    std::vector<int> xdims_list(input_dims.size());
    int total_length = 0;
    for (int i = 0; i < num; ++i) {
      split_list[i] = ins[i]->dims()[axis];
      total_length += ins[i]->dims()[axis];
    }

    for (int i = 0; i < input_dims.size(); ++i) {
      if (i == axis) continue;
      xdims_list[i] = input_dims[i];
    }
    xdims_list[axis] = total_length;

    std::vector<T*> ptrs(num);
    for (int i = 0; i < num; ++i) {
      ptrs[i] = outputs->at(i)->data<T>();
    }

    auto r = xpu::split<T>(context.x_context(), input.data<T>(), ptrs,
                           xdims_list, split_list, axis);
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External(
            "XPU API return wrong value[%d %s], please check whether "
            "Baidu Kunlun Card is properly installed.",
            r, XPUAPIErrorMsg[r]));
  }
};
#endif

#define DEFINE_FUNCTOR(type)                                      \
  template class ConcatFunctor<platform::CPUDeviceContext, type>; \
  template class SplitFunctor<platform::CPUDeviceContext, type>;

FOR_ALL_TYPES(DEFINE_FUNCTOR);

#ifdef PADDLE_WITH_XPU
#define DEFINE_XPU_FUNCTOR(type)                                  \
  template class ConcatFunctor<platform::XPUDeviceContext, type>; \
  template class SplitFunctor<platform::XPUDeviceContext, type>;

DEFINE_XPU_FUNCTOR(float)
#endif

}  // namespace math
}  // namespace operators
}  // namespace paddle
