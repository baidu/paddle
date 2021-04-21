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

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memory.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/xpu_header.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;

using TensorList = std::vector<framework::Tensor>;

template <typename TensorType, typename T>
void reset_parameter_vector(const std::vector<TensorType>& raw_params_vec,
                            const int& num_layers, const bool& is_bidirec,
                            std::vector<std::vector<T*>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    params_vec->at(i).resize(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      using remove_cv_t = typename std::remove_cv<T>::type;
      params_vec->at(i)[j] =
          raw_params_vec[tensor_idx]->template data<remove_cv_t>();
    }
  }
}

template <typename DeviceContext, typename T>
class RnnXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    auto state = ctx.MultiOutput<Tensor>("State");
    auto* output = ctx.Output<Tensor>("Out");
    auto* reserve_data = ctx.Output<Tensor>("Reserve");
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const std::string& mode = ctx.Attr<std::string>("mode");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    PADDLE_ENFORCE_EQ(
        mode, "LSTM",
        platform::errors::InvalidArgument(
            "XPU only support LSTM mode now, current mode is %s", mode));

    PADDLE_ENFORCE_EQ(is_bidirec, false,
                      platform::errors::InvalidArgument(
                          "XPU only support unidirectional LSTM now"));

    auto init_h = pre_state[0];
    auto init_c = pre_state[1];
    auto last_h = state[0];
    auto last_c = state[1];

    // check shape
    int seq_len = input->dims()[0];
    int batch_size = input->dims()[1];
    int input_dim = input->dims()[2];

    PADDLE_ENFORCE_EQ(
        init_h->dims()[0], num_layers,
        platform::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers, init_c->dims()[0]));

    std::vector<std::vector<const T*>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);

    // init the output and allocate the memory
    int block_size = seq_len * batch_size * hidden_size;
    output->mutable_data<T>(ctx.GetPlace());
    last_h->mutable_data<T>(ctx.GetPlace());
    last_c->mutable_data<T>(ctx.GetPlace());
    reserve_data->Resize({num_layers * block_size * 6});
    reserve_data->mutable_data<T>(ctx.GetPlace());
    Tensor i_f_g_o_holder = reserve_data->Slice(0, num_layers * block_size * 5);
    Tensor output_holder = reserve_data->Slice(num_layers * block_size * 5,
                                               num_layers * block_size * 6);

    TensorList init_h_unbind = Unbind(*init_h);
    TensorList init_c_unbind = Unbind(*init_c);

    TensorList last_h_unbind = Unbind(*last_h);
    TensorList last_c_unbind = Unbind(*last_c);

    for (int i = 0; i < num_layers; i++) {
      // get ptr from tensor
      auto h_0 = init_h_unbind[i].data<T>();
      auto c_0 = init_c_unbind[i].data<T>();
      auto w_x = parameter_lists[i][0];
      auto w_h = parameter_lists[i][1];
      auto b_x = parameter_lists[i][2];
      auto b_h = parameter_lists[i][3];
      auto last_h_ptr = last_h_unbind[i].data<T>();
      auto last_c_ptr = last_c_unbind[i].data<T>();

      auto i_f_g_o =
          i_f_g_o_holder.Slice(i * block_size * 5, (i + 1) * block_size * 5)
              .data<T>();
      auto c = i_f_g_o + seq_len * batch_size * hidden_size * 4;

      auto y = output->data<T>();
      std::vector<int> seq_len_tensor(batch_size, seq_len);
      if (has_seq_length) {
        seq_len_tensor = operators::GetDataFromTensor(sequence_length);
      }
      // run kernel
      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      if (i == 0) {
        auto x = input->data<T>();
        int r = xpu::lstm_train<T, T, int16_t>(
            dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
            (const T*)w_x, (const T*)w_h, (const T*)b_x, (const T*)b_h,
            reinterpret_cast<T*>(y), reinterpret_cast<T*>(last_h_ptr),
            reinterpret_cast<T*>(last_c_ptr), batch_size, input_dim,
            hidden_size, seq_len, seq_len_tensor, nullptr, nullptr, nullptr,
            nullptr, reinterpret_cast<T*>(i_f_g_o), reinterpret_cast<T*>(c));
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::External("XPU API rnn op return wrong "
                                       "value[%d %s]",
                                       r, XPUAPIErrorMsg[r]));
      } else {
        framework::Tensor input_holder;
        input_holder.mutable_data<T>(ctx.GetPlace());
        TensorCopySync(*output, ctx.GetPlace(), &input_holder);
        auto x = input_holder.data<T>();
        int r = xpu::lstm_train<T, T, int16_t>(
            dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
            (const T*)w_x, (const T*)w_h, (const T*)b_x, (const T*)b_h,
            reinterpret_cast<T*>(y), reinterpret_cast<T*>(last_h_ptr),
            reinterpret_cast<T*>(last_c_ptr), batch_size, hidden_size,
            hidden_size, seq_len, seq_len_tensor, nullptr, nullptr, nullptr,
            nullptr, reinterpret_cast<T*>(i_f_g_o), reinterpret_cast<T*>(c));
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::External("XPU API rnn op return wrong "
                                       "value[%d %s]",
                                       r, XPUAPIErrorMsg[r]));
      }

      Tensor y_holder =
          output_holder.Slice(i * block_size, (i + 1) * block_size);
      TensorCopySync(*output, ctx.GetPlace(), &y_holder);
    }
  }
};

template <typename DeviceContext, typename T>
class RnnXPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // get the tensor pointer for the input
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    auto* output = ctx.Input<Tensor>("Out");
    auto* reserve_data = ctx.Input<Tensor>("Reserve");
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const std::string& mode = ctx.Attr<std::string>("mode");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    PADDLE_ENFORCE_EQ(
        mode, "LSTM",
        platform::errors::InvalidArgument(
            "XPU only support LSTM mode now, current mode is %s", mode));

    PADDLE_ENFORCE_EQ(is_bidirec, false,
                      platform::errors::InvalidArgument(
                          "XPU only support unidirectional LSTM now"));

    auto init_h = pre_state[0];
    auto init_c = pre_state[1];

    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto state_grad = ctx.MultiInput<Tensor>(framework::GradVarName("State"));
    auto last_h_grad = state_grad[0];
    auto last_c_grad = state_grad[1];

    // get the tensor pointer for the output
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto weight_grad_list = ctx.MultiOutput<framework::Tensor>(
        framework::GradVarName("WeightList"));
    auto pre_state_grad =
        ctx.MultiOutput<Tensor>(framework::GradVarName("PreState"));
    Tensor* init_h_grad = nullptr;
    Tensor* init_c_grad = nullptr;
    if (pre_state_grad.size() > 0) {  // has gradient
      init_h_grad = pre_state_grad[0];
      init_c_grad = pre_state_grad[1];
    }

    auto init_h_unbind = Unbind(*init_h);
    auto init_c_unbind = Unbind(*init_c);
    auto last_h_grad_unbind = Unbind(*last_h_grad);
    auto last_c_grad_unbind = Unbind(*last_c_grad);

    // check shape
    int seq_len = input->dims()[0];
    int batch_size = input->dims()[1];
    int input_dim = input->dims()[2];

    PADDLE_ENFORCE_EQ(
        init_h->dims()[0], num_layers,
        platform::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers, init_c->dims()[0]));

    int block_size = seq_len * batch_size * hidden_size;
    std::vector<std::vector<const T*>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);
    for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
      weight_grad_list[i]->mutable_data<T>(ctx.GetPlace());
    }
    std::vector<std::vector<T*>> parameter_lists_grad;
    parameter_lists_grad.resize(num_layers);
    reset_parameter_vector(weight_grad_list, num_layers, is_bidirec,
                           &parameter_lists_grad);
    Tensor i_f_g_o_holder = reserve_data->Slice(0, num_layers * block_size * 5);
    Tensor output_holder = reserve_data->Slice(num_layers * block_size * 5,
                                               num_layers * block_size * 6);

    // allocate the memory and initization the input_grad
    input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
    TensorList init_h_grad_unbind;
    TensorList init_c_grad_unbind;
    if (init_h_grad) {
      init_h_grad->mutable_data<T>(init_h->dims(), ctx.GetPlace());
      init_h_grad_unbind = Unbind(*init_h_grad);
    }
    if (init_c_grad) {
      init_c_grad->mutable_data<T>(init_c->dims(), ctx.GetPlace());
      init_c_grad_unbind = Unbind(*init_c_grad);
    }

    framework::Tensor output_grad_holder;
    output_grad_holder.mutable_data<T>(ctx.GetPlace());
    TensorCopySync(*output_grad, ctx.GetPlace(), &output_grad_holder);

    for (int i = num_layers - 1; i >= 0; i--) {
      // get ptr from tensor
      auto h_0 = init_h_unbind[i].data<T>();
      auto c_0 = init_c_unbind[i].data<T>();
      auto w_x = parameter_lists[i][0];
      auto w_h = parameter_lists[i][1];
      Tensor y_holder;
      if (i == num_layers - 1) {
        TensorCopySync(*output, ctx.GetPlace(), &y_holder);
      } else {
        TensorCopySync(
            output_holder.Slice(i * block_size, (i + 1) * block_size),
            ctx.GetPlace(), &y_holder);
      }
      auto y = y_holder.data<T>();
      auto y_grad = output_grad_holder.data<T>();
      auto last_h_grad_ptr = last_h_grad_unbind[i].data<T>();
      auto last_c_grad_ptr = last_c_grad_unbind[i].data<T>();
      auto h_0_grad = init_h_grad ? init_h_grad_unbind[i].data<T>() : nullptr;
      auto c_0_grad = init_c_grad ? init_c_grad_unbind[i].data<T>() : nullptr;
      auto w_x_grad = parameter_lists_grad[i][0];
      auto w_h_grad = parameter_lists_grad[i][1];
      auto b_x_grad = parameter_lists_grad[i][2];
      auto b_h_grad = parameter_lists_grad[i][3];
      auto i_f_g_o =
          i_f_g_o_holder.Slice(i * block_size * 5, (i + 1) * block_size * 5)
              .data<T>();
      auto c = i_f_g_o + seq_len * batch_size * hidden_size * 4;

      std::vector<int> seq_len_tensor(batch_size, seq_len);
      if (has_seq_length) {
        seq_len_tensor = operators::GetDataFromTensor(sequence_length);
      }

      auto& dev_ctx = ctx.template device_context<DeviceContext>();
      if (i == 0) {
        auto x = input->data<T>();
        auto x_grad = input_grad->data<T>();
        int r = xpu::lstm_grad<T, T, int16_t>(
            dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
            (const T*)w_x, (const T*)w_h, (const T*)y, (const T*)y_grad,
            (const T*)last_h_grad_ptr, (const T*)last_c_grad_ptr,
            reinterpret_cast<T*>(x_grad), reinterpret_cast<T*>(h_0_grad),
            reinterpret_cast<T*>(c_0_grad), w_x_grad, w_h_grad, b_x_grad,
            b_h_grad, batch_size, input_dim, hidden_size, seq_len,
            seq_len_tensor, nullptr, nullptr, nullptr, nullptr, i_f_g_o, c);
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::External("XPU API rnn op return wrong "
                                       "value[%d %s]",
                                       r, XPUAPIErrorMsg[r]));
      } else {
        auto x =
            output_holder.Slice((i - 1) * block_size, i * block_size).data<T>();
        Tensor input_grad_holder;
        TensorCopySync(output_grad_holder, ctx.GetPlace(), &input_grad_holder);
        auto x_grad = input_grad_holder.data<T>();
        int r = xpu::lstm_grad<T, T, int16_t>(
            dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
            (const T*)w_x, (const T*)w_h, (const T*)y, (const T*)y_grad,
            (const T*)last_h_grad_ptr, (const T*)last_c_grad_ptr,
            reinterpret_cast<T*>(x_grad), reinterpret_cast<T*>(h_0_grad),
            reinterpret_cast<T*>(c_0_grad), w_x_grad, w_h_grad, b_x_grad,
            b_h_grad, batch_size, hidden_size, hidden_size, seq_len,
            seq_len_tensor, nullptr, nullptr, nullptr, nullptr, i_f_g_o, c);
        PADDLE_ENFORCE_EQ(
            r, xpu::Error_t::SUCCESS,
            platform::errors::External("XPU API rnn op return wrong "
                                       "value[%d %s]",
                                       r, XPUAPIErrorMsg[r]));
        TensorCopySync(input_grad_holder, ctx.GetPlace(), &output_grad_holder);
      }
    }
    Tensor input_grad_cpu;
    TensorCopySync(*input_grad, platform::CPUPlace(), &input_grad_cpu);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    rnn, ops::RnnXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    rnn_grad, ops::RnnXPUGradKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
