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

#include <set>
#include <unordered_map>

#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"

namespace paddle {
namespace operators {
namespace math {
template <typename T>
struct SelectedRowsAdd<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::SelectedRows& input2,
                  framework::SelectedRows* output) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(in1_height, input2.height());
    output->set_height(in1_height);

    auto& in1_rows = input1.rows();
    auto& in2_rows = input2.rows();
    std::vector<int64_t> out_rows;
    out_rows.reserve(in1_rows.size() + in2_rows.size());

    // concat rows
    out_rows.insert(out_rows.end(), in1_rows.begin(), in1_rows.end());
    out_rows.insert(out_rows.end(), in2_rows.begin(), in2_rows.end());
    output->set_rows(out_rows);

    auto* out_value = output->mutable_value();
    auto& in1_value = input1.value();
    auto& in2_value = input2.value();

    auto in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, in2_value.numel() / in2_rows.size());
    PADDLE_ENFORCE_EQ(in1_row_numel, out_value->numel() / out_rows.size());

    auto in1_place = input1.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in1_place));
    auto in2_place = input2.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in2_place));
    auto out_place = context.GetPlace();
    PADDLE_ENFORCE(platform::is_cpu_place(out_place));

    auto* out_data = out_value->data<T>();
    auto* in1_data = in1_value.data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(out_place), out_data,
                 boost::get<platform::CPUPlace>(in1_place), in1_data,
                 in1_value.numel() * sizeof(T));

    auto* in2_data = in2_value.data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(out_place),
                 out_data + in1_value.numel(),
                 boost::get<platform::CPUPlace>(in2_place), in2_data,
                 in2_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAdd<platform::CPUDeviceContext, float>;
template struct SelectedRowsAdd<platform::CPUDeviceContext, double>;

template <typename T>
struct SelectedRowsAddTensor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input1,
                  const framework::Tensor& input2, framework::Tensor* output) {
    auto in1_height = input1.height();
    auto in2_dims = input2.dims();
    auto out_dims = output->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);
    PADDLE_ENFORCE_EQ(in1_height, out_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2.numel() / in1_height);
    PADDLE_ENFORCE_EQ(in1_row_numel, output->numel() / in1_height);

    SetConstant<platform::CPUDeviceContext, T> functor;
    functor(context, output, 0.0);

    auto* in1_data = in1_value.data<T>();
    auto* out_data = output->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        out_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }

    auto out_eigen = framework::EigenVector<T>::Flatten(*output);
    auto in2_eigen = framework::EigenVector<T>::Flatten(input2);
    out_eigen.device(*context.eigen_device()) = out_eigen + in2_eigen;
  }
};

template struct SelectedRowsAddTensor<platform::CPUDeviceContext, float>;
template struct SelectedRowsAddTensor<platform::CPUDeviceContext, double>;

template <typename T>
struct SelectedRowsAddTo<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input1,
                  const int64_t input2_offset,
                  framework::SelectedRows* input2) {
    auto in1_height = input1.height();
    PADDLE_ENFORCE_EQ(in1_height, input2->height());

    auto& in1_rows = input1.rows();
    auto& in2_rows = *(input2->mutable_rows());

    auto& in1_value = input1.value();
    auto* in2_value = input2->mutable_value();

    // concat rows
    in2_rows.Extend(in1_rows.begin(), in1_rows.end());

    auto in1_place = input1.place();
    PADDLE_ENFORCE(platform::is_cpu_place(in1_place));
    auto in2_place = input2->place();
    PADDLE_ENFORCE(platform::is_cpu_place(in2_place));

    auto* in1_data = in1_value.data<T>();
    auto* in2_data = in2_value->data<T>();
    memory::Copy(boost::get<platform::CPUPlace>(in2_place),
                 in2_data + input2_offset,
                 boost::get<platform::CPUPlace>(in1_place), in1_data,
                 in1_value.numel() * sizeof(T));
  }
};

template struct SelectedRowsAddTo<platform::CPUDeviceContext, float>;
template struct SelectedRowsAddTo<platform::CPUDeviceContext, double>;
template struct SelectedRowsAddTo<platform::CPUDeviceContext, int>;
template struct SelectedRowsAddTo<platform::CPUDeviceContext, int64_t>;

template <typename T>
struct SelectedRowsSumTo<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<framework::SelectedRows*>& input1,
                  const std::vector<int64_t>& input2_offsets,
                  framework::SelectedRows* input2) {
    // Ensure all selected rows have the same height
    size_t size = 0u;
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      auto& in_rows = (*iter)->rows();
      size += in_rows.end() - in_rows.begin();
      auto in1_height = (*iter)->height();
      PADDLE_ENFORCE_EQ(in1_height, input2->height());
    }
    // concat rows
    std::vector<int64_t> in2_rows;
    in2_rows.reserve(in2_rows.size() + size);
    for (auto iter = input1.begin(); iter != input1.end(); ++iter) {
      const framework::Vector<int64_t>& in_rows = (*iter)->rows();
      in2_rows.insert(in2_rows.end(), in_rows.begin(), in_rows.end());
    }
    input2->set_rows(in2_rows);

    auto* in2_value = input2->mutable_value();
    auto* in2_data = in2_value->data<T>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    size_t offset = 0u;
    for (size_t i = 0u; i != input1.size(); ++i) {
      auto& in_value = input1[i]->value();
      const auto* in_data = in_value.data<T>();
      offset += input2_offsets[i];
      blas.VCOPY(in_value.numel(), in_data, in2_data + offset);
    }
  }
};

template struct SelectedRowsSumTo<platform::CPUDeviceContext, float>;
template struct SelectedRowsSumTo<platform::CPUDeviceContext, double>;

template <typename T>
struct SelectedRowsAddToTensor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2->numel() / in1_height);

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->data<T>();

    for (size_t i = 0; i < in1_rows.size(); i++) {
      for (int64_t j = 0; j < in1_row_numel; j++) {
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
      }
    }
  }
};

template struct SelectedRowsAddToTensor<platform::CPUDeviceContext, float>;
template struct SelectedRowsAddToTensor<platform::CPUDeviceContext, double>;
template struct SelectedRowsAddToTensor<platform::CPUDeviceContext, int>;
template struct SelectedRowsAddToTensor<platform::CPUDeviceContext, int64_t>;

// This is a separated namespace for manipulate SelectedRows typed
// data. Like merge duplicated rows, adding two SelectedRows etc.
//
// Another group of functors is called "scatter updates", which means
// use SelectedRows to update a dense tensor with different Ops, like
// add or mul.
namespace scatter {

template <typename DeviceContext, typename T>
typename std::enable_if<
    std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_to(const DeviceContext& ctx, BlasT<DeviceContext, T>* blas,
                   size_t data_len, const T* in, T* out) {
  blas->AXPY(data_len, 1., in, out);
}

template <typename DeviceContext, typename T>
typename std::enable_if<
    !std::is_floating_point<T>::value &&
    std::is_same<DeviceContext, platform::CPUDeviceContext>::value>::type
elementwise_add_to(const DeviceContext& ctx, BlasT<DeviceContext, T>* blas,
                   size_t data_len, const T* in, T* out) {
  for (int64_t i = 0; i < data_len; i++) {
    out[i] += in[i];
  }
}

template <typename T>
struct MergeAdd<platform::CPUDeviceContext, T> {
  framework::SelectedRows operator()(const platform::CPUDeviceContext& context,
                                     const framework::SelectedRows& input) {
    framework::SelectedRows out;
    (*this)(context, input, &out);
    return out;
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const framework::SelectedRows& input,
                  framework::SelectedRows* output) {
    std::vector<const framework::SelectedRows*> inputs;
    inputs.push_back(&input);
    (*this)(context, inputs, output);
  }

  void operator()(const platform::CPUDeviceContext& context,
                  const std::vector<const framework::SelectedRows*>& inputs,
                  framework::SelectedRows* output) {
    PADDLE_ENFORCE_GT(inputs.size(), 0, "should have at least one input");
    auto input_width = inputs[0]->value().dims()[1];
    auto input_height = inputs[0]->height();
    framework::SelectedRows& out = *output;
    std::set<int64_t> merged_row_set;
    for (auto* input : inputs) {
      PADDLE_ENFORCE_EQ(input_width, input->value().dims()[1],
                        "all input should have same "
                        "dimension except for the first one");
      PADDLE_ENFORCE_EQ(input_height, input->height(),
                        "all input should have same height");
      merged_row_set.insert(input->rows().begin(), input->rows().end());
    }
    std::vector<int64_t> merge_rows(merged_row_set.begin(),
                                    merged_row_set.end());
    std::unordered_map<int64_t, size_t> rows_to_id;
    for (size_t i = 0; i < merge_rows.size(); ++i) {
      rows_to_id[merge_rows[i]] = i;
    }

    out.set_rows(merge_rows);
    out.set_height(input_height);
    out.mutable_value()->mutable_data<T>(
        framework::make_ddim(
            {static_cast<int64_t>(merge_rows.size()), input_width}),
        context.GetPlace());

    math::SetConstant<platform::CPUDeviceContext, T> constant_functor;
    constant_functor(context, out.mutable_value(), 0.0);

    auto* out_data = out.mutable_value()->data<T>();

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (auto* input : inputs) {
      auto* input_data = input->value().data<T>();
      auto& input_rows = input->rows();

      for (size_t i = 0; i < input_rows.size(); i++) {
        size_t out_i = rows_to_id[input_rows[i]];
        elementwise_add_to<platform::CPUDeviceContext, T>(
            context, &blas, static_cast<size_t>(input_width),
            &input_data[i * input_width], &out_data[out_i * input_width]);
      }
    }
  }
};

template struct MergeAdd<platform::CPUDeviceContext, int>;
template struct MergeAdd<platform::CPUDeviceContext, int64_t>;
template struct MergeAdd<platform::CPUDeviceContext, float>;
template struct MergeAdd<platform::CPUDeviceContext, double>;

template <typename T>
struct UpdateToTensor<platform::CPUDeviceContext, T> {
  void operator()(const platform::CPUDeviceContext& context,
                  const ScatterOps& op, const framework::SelectedRows& input1,
                  framework::Tensor* input2) {
    auto in1_height = input1.height();
    auto in2_dims = input2->dims();
    PADDLE_ENFORCE_EQ(in1_height, in2_dims[0]);

    auto& in1_value = input1.value();
    auto& in1_rows = input1.rows();

    int64_t in1_row_numel = in1_value.numel() / in1_rows.size();
    PADDLE_ENFORCE_EQ(in1_row_numel, input2->numel() / in1_height);

    auto* in1_data = in1_value.data<T>();
    auto* input2_data = input2->data<T>();

    // FIXME(typhoonzero): use macro fix the below messy code.
    switch (op) {
      case ScatterOps::ASSIGN:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::ADD:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] +=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUB:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] -=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::SUBBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] -
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
      case ScatterOps::MUL:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] *=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIV:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] /=
            in1_data[i * in1_row_numel + j];
        break;
      case ScatterOps::DIVBY:
        INLINE_FOR2(in1_rows.size(), in1_row_numel)
        input2_data[in1_rows[i] * in1_row_numel + j] =
            in1_data[i * in1_row_numel + j] /
            input2_data[in1_rows[i] * in1_row_numel + j];
        break;
    }
  }
};

}  // namespace scatter
}  // namespace math
}  // namespace operators
}  // namespace paddle
