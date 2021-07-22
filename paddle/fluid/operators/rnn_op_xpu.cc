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

    PADDLE_ENFORCE_EQ(
        num_layers, 1,
        platform::errors::InvalidArgument(
            "XPU only support 1 layer LSTM now, current layer num is %s",
            num_layers));

    auto init_h = pre_state[0];
    auto init_c = pre_state[1];
    auto last_h = state[0];
    auto last_c = state[1];

    // check shape
    const int& direction_num = is_bidirec ? 2 : 1;
    int seq_len = input->dims()[0];
    int batch_size = input->dims()[1];
    int input_dim = input->dims()[2];

    PADDLE_ENFORCE_EQ(init_h->dims()[0], num_layers * direction_num,
                      platform::errors::InvalidArgument(
                          "The num_layers of in RNN layer must"
                          " be the same as first dim of init "
                          "hidden, but received num_layers:%d,"
                          " dim:%d",
                          num_layers * direction_num, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers * direction_num,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers * direction_num, init_c->dims()[0]));

    std::vector<std::vector<const T*>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    last_h->mutable_data<T>(ctx.GetPlace());
    last_c->mutable_data<T>(ctx.GetPlace());
    reserve_data->Resize(
        {seq_len * batch_size * hidden_size * 5 * direction_num});
    reserve_data->mutable_data<T>(ctx.GetPlace());

    // get ptr from tensor
    const T* x = input->data<T>();
    const T* h_0 = init_h->data<T>();
    const T* c_0 = init_c->data<T>();
    const T* forward_w_x = parameter_lists[0][0];
    const T* forward_w_h = parameter_lists[0][1];
    const T* forward_b_x = parameter_lists[0][2];
    const T* forward_b_h = parameter_lists[0][3];
    const T* backward_w_x = is_bidirec ? parameter_lists[0][4] : nullptr;
    const T* backward_w_h = is_bidirec ? parameter_lists[0][5] : nullptr;
    const T* backward_b_x = is_bidirec ? parameter_lists[0][6] : nullptr;
    const T* backward_b_h = is_bidirec ? parameter_lists[0][7] : nullptr;
    T* y = output->data<T>();
    T* last_h_ptr = last_h->data<T>();
    T* last_c_ptr = last_c->data<T>();
    T* i_f_g_o = reserve_data->data<T>();
    T* c = i_f_g_o + seq_len * batch_size * hidden_size * 4 * direction_num;

    std::vector<int> seq_len_tensor(batch_size, seq_len);
    if (has_seq_length) {
      seq_len_tensor = operators::GetDataFromTensor(sequence_length);
    }

    // run kernel
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (!is_bidirec) {
      r = xpu::lstm_train<T, T, int16_t>(
          dev_ctx.x_context(), x, h_0, c_0, forward_w_x, forward_w_h,
          forward_b_x, forward_b_h, y, last_h_ptr, last_c_ptr, batch_size,
          input_dim, hidden_size, seq_len, seq_len_tensor, nullptr, nullptr,
          nullptr, nullptr, i_f_g_o, c);
    } else {
      r = xpu::bilstm_train<T, T, int16_t>(
          dev_ctx.x_context(), x, h_0, c_0, forward_w_x, forward_w_h,
          forward_b_x, forward_b_h, backward_w_x, backward_w_h, backward_b_x,
          backward_b_h, y, last_h_ptr, last_c_ptr, batch_size, input_dim,
          hidden_size, seq_len, seq_len_tensor, nullptr, nullptr, nullptr,
          nullptr, nullptr, nullptr, i_f_g_o, c);
    }
    PADDLE_ENFORCE_EQ(r, xpu::Error_t::SUCCESS,
                      platform::errors::External("RnnXPU(lstm) return wrong "
                                                 "value[%d %s]",
                                                 r, XPUAPIErrorMsg[r]));
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

    PADDLE_ENFORCE_EQ(
        num_layers, 1,
        platform::errors::InvalidArgument(
            "XPU only support 1 layer LSTM now, current layer num is %s",
            num_layers));

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

    // check shape
    const int& direction_num = is_bidirec ? 2 : 1;
    int seq_len = input->dims()[0];
    int batch_size = input->dims()[1];
    int input_dim = input->dims()[2];

    PADDLE_ENFORCE_EQ(init_h->dims()[0], num_layers * direction_num,
                      platform::errors::InvalidArgument(
                          "The num_layers of in RNN layer must"
                          " be the same as first dim of init "
                          "hidden, but received num_layers:%d,"
                          " dim:%d",
                          num_layers * direction_num, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers * direction_num,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers * direction_num, init_c->dims()[0]));

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

    // allocate the memory and initization the input_grad
    input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
    if (init_h_grad) {
      init_h_grad->mutable_data<T>(init_h->dims(), ctx.GetPlace());
    }
    if (init_c_grad) {
      init_c_grad->mutable_data<T>(init_c->dims(), ctx.GetPlace());
    }

    // get ptr from tensor
    const T* x = input->data<T>();
    const T* h_0 = init_h->data<T>();
    const T* c_0 = init_c->data<T>();
    const T* forward_w_x = parameter_lists[0][0];
    const T* forward_w_h = parameter_lists[0][1];
    const T* backward_w_x = parameter_lists[0][4];
    const T* backward_w_h = parameter_lists[0][5];
    const T* y = output->data<T>();
    const T* y_grad = output_grad->data<T>();
    const T* last_h_grad_ptr = last_h_grad->data<T>();
    const T* last_c_grad_ptr = last_c_grad->data<T>();
    T* x_grad = input_grad->data<T>();
    T* h_0_grad = init_h_grad ? init_h_grad->data<T>() : nullptr;
    T* c_0_grad = init_c_grad ? init_c_grad->data<T>() : nullptr;
    T* forward_w_x_grad = parameter_lists_grad[0][0];
    T* forward_w_h_grad = parameter_lists_grad[0][1];
    T* forward_b_x_grad = parameter_lists_grad[0][2];
    T* forward_b_h_grad = parameter_lists_grad[0][3];
    T* backward_w_x_grad = is_bidirec ? parameter_lists_grad[0][4] : nullptr;
    T* backward_w_h_grad = is_bidirec ? parameter_lists_grad[0][5] : nullptr;
    T* backward_b_x_grad = is_bidirec ? parameter_lists_grad[0][6] : nullptr;
    T* backward_b_h_grad = is_bidirec ? parameter_lists_grad[0][7] : nullptr;
    const T* i_f_g_o = reserve_data->data<T>();
    const T* c =
        i_f_g_o + seq_len * batch_size * hidden_size * 4 * direction_num;

    std::vector<int> seq_len_tensor(batch_size, seq_len);
    if (has_seq_length) {
      seq_len_tensor = operators::GetDataFromTensor(sequence_length);
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::Error_t::SUCCESS;
    if (!is_bidirec) {
      r = xpu::lstm_grad<T, T, int16_t>(
          dev_ctx.x_context(), x, h_0, c_0, forward_w_x, forward_w_h, y, y_grad,
          last_h_grad_ptr, last_c_grad_ptr, x_grad, h_0_grad, c_0_grad,
          forward_w_x_grad, forward_w_h_grad, forward_b_x_grad,
          forward_b_h_grad, batch_size, input_dim, hidden_size, seq_len,
          seq_len_tensor, nullptr, nullptr, nullptr, nullptr, i_f_g_o, c);
    } else {
      r = xpu::bilstm_grad<T, T, int16_t>(
          dev_ctx.x_context(), x, h_0, c_0, forward_w_x, forward_w_h,
          backward_w_x, backward_w_h, y, y_grad, last_h_grad_ptr,
          last_c_grad_ptr, x_grad, h_0_grad, c_0_grad, forward_w_x_grad,
          forward_w_h_grad, forward_b_x_grad, forward_b_h_grad,
          backward_w_x_grad, backward_w_h_grad, backward_b_x_grad,
          backward_b_h_grad, batch_size, input_dim, hidden_size, seq_len,
          seq_len_tensor, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          i_f_g_o, c);
    }

    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("RnnXPUGrad(lstm) return wrong "
                                   "value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
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
