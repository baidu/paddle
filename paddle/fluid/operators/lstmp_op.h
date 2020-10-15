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
#include <string>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/lstm_compute.h"
#include "paddle/fluid/operators/math/sequence2batch.h"
#include "paddle/fluid/platform/transform.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
using platform::Transform;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
class _ClipFunctor {
 public:
  explicit _ClipFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T& x) const {
    if (x < min_)
      return min_;
    else if (x > max_)
      return max_;
    else
      return x;
  }

 private:
  T min_;
  T max_;
};

template <typename T>
class _ClipGradFunctor {
 public:
  explicit _ClipGradFunctor(const T min, const T max) : min_(min), max_(max) {}
  HOSTDEVICE T operator()(const T& x, const T& y) const {
    return (y > min_ && y < max_) ? x : 0;
  }

 private:
  T min_;
  T max_;
};

template <typename DeviceContext, typename T>
inline void ReorderInitState(const DeviceContext& ctx,
                             const framework::Tensor& src,
                             framework::Vector<size_t> index,
                             framework::Tensor* dst, bool indexed_src) {
  math::CopyMatrixRowsFunctor<DeviceContext, T> row_shuffle;
  dst->mutable_data<T>(src.dims(), ctx.GetPlace());
  row_shuffle(ctx, src, index, dst, indexed_src);
}

template <typename DeviceContext, typename T>
class LSTMPKernel : public framework::OpKernel<T> {
 public:
  template <typename Device, typename X, typename Y>
  void ActCompute(const math::detail::ActivationType act_type, const Device& d,
                  X x, Y y) const {
    if (act_type == math::detail::ActivationType::kIdentity)
      y.device(d) = x;
    else if (act_type == math::detail::ActivationType::kSigmoid)
      SigmoidFunctor<T>()(d, x, y);
    else if (act_type == math::detail::ActivationType::kTanh)
      TanhFunctor<T>()(d, x, y);
    else if (act_type == math::detail::ActivationType::kReLU)
      ReluFunctor<T>()(d, x, y);
    else
      PADDLE_THROW(
          platform::errors::InvalidArgument("unsupported activation type"));
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<LoDTensor>("Input");
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* proj_weight = ctx.Input<Tensor>("ProjWeight");
    auto* bias = ctx.Input<Tensor>("Bias");

    auto* hidden_t0 = ctx.Input<Tensor>("H0");
    auto* cell_t0 = ctx.Input<Tensor>("C0");

    auto proj_clip = static_cast<T>(ctx.Attr<float>("proj_clip"));
    auto cell_clip = static_cast<T>(ctx.Attr<float>("cell_clip"));

    auto* batch_gate = ctx.Output<LoDTensor>("BatchGate");
    batch_gate->mutable_data<T>(ctx.GetPlace());
    auto* proj_out = ctx.Output<LoDTensor>("Projection");
    proj_out->mutable_data<T>(ctx.GetPlace());
    auto* cell_out = ctx.Output<LoDTensor>("Cell");
    cell_out->mutable_data<T>(ctx.GetPlace());

    bool is_reverse = ctx.Attr<bool>("is_reverse");
    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;
    auto& device_ctx = ctx.template device_context<DeviceContext>();
    to_batch(device_ctx, *input, batch_gate, true, is_reverse);

    auto in_dims = input->dims();
    int frame_size = static_cast<int>(in_dims[1] / 4);
    framework::DDim dims({in_dims[0], frame_size});
    framework::DDim proj_dims({in_dims[0], proj_weight->dims()[1]});

    if (bias) {
      Tensor b = *bias;
      b.Resize({bias->numel(), 1});
      Tensor gate_bias = b.Slice(0, 4 * frame_size);
      math::RowwiseAdd<DeviceContext, T> add_bias;
      add_bias(device_ctx, *batch_gate, gate_bias, batch_gate);
    }

    math::LstmMetaValue<T> lstmp_value;
    if (bias && ctx.Attr<bool>("use_peepholes")) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      // the code style in LstmpMetaValue will be updated later.

      lstmp_value.check_ig = bias_data + 4 * frame_size;
      lstmp_value.check_fg = lstmp_value.check_ig + frame_size;
      lstmp_value.check_og = lstmp_value.check_fg + frame_size;
    } else {
      lstmp_value.check_ig = nullptr;
      lstmp_value.check_fg = nullptr;
      lstmp_value.check_og = nullptr;
    }
    lstmp_value.prev_state_value = nullptr;
    Tensor ordered_c0;
    Tensor ordered_h0;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (cell_t0) {
      // Since the batch computing for LSTMP reorders the input sequence
      // according to their length. The initialized cell state also needs
      // to reorder.
      ReorderInitState<DeviceContext, T>(device_ctx, *cell_t0, order,
                                         &ordered_c0, true);
      lstmp_value.prev_state_value = ordered_c0.data<T>();
    }

    // Use the local variable as here.
    LoDTensor batch_proj, batch_cell;
    auto* batch_cell_pre_act = ctx.Output<LoDTensor>("BatchCellPreAct");
    batch_cell_pre_act->mutable_data<T>(dims, ctx.GetPlace());
    auto* batch_hidden = ctx.Output<LoDTensor>("BatchHidden");
    batch_hidden->mutable_data<T>(dims, ctx.GetPlace());    // T x D
    batch_proj.mutable_data<T>(proj_dims, ctx.GetPlace());  // T x P
    batch_cell.mutable_data<T>(dims, ctx.GetPlace());       // T x D

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto gate_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));
    auto cell_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("cell_activation"));
    auto cand_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("candidate_activation"));
    auto proj_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("proj_activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto blas = math::GetBlas<DeviceContext, T>(device_ctx);
    for (size_t n = 0; n < num_batch; n++) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor gate_t = batch_gate->Slice(bstart, bend);
      Tensor hidden_t = batch_hidden->Slice(bstart, bend);
      Tensor proj_t = batch_proj.Slice(bstart, bend);
      Tensor cell_t = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act_t = batch_cell_pre_act->Slice(bstart, bend);

      int cur_batch_size = bend - bstart;

      if (n > 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_proj_t = batch_proj.Slice(pre_h_start, pre_h_end);
        blas.MatMul(pre_proj_t, false, *weight, false, static_cast<T>(1.0),
                    &gate_t, static_cast<T>(1.0));
      } else if (hidden_t0) {
        // If n == 0 and there is no initialized hidden state, that is to say
        // the H0 is zeros, the calculation W_h * H0 will be skiped.
        // If n == 0 and there is initialized hidden state, calculate W_h * H0.

        // Since the batch computing for LSTMP reorders the input sequence
        // according to their length. The initialized hidden state also needs
        // to reorder.
        ReorderInitState<DeviceContext, T>(device_ctx, *hidden_t0, order,
                                           &ordered_h0, true);
        blas.MatMul(ordered_h0, false, *weight, false, static_cast<T>(1.0),
                    &gate_t, static_cast<T>(1.0));
      }

      lstmp_value.gate_value = gate_t.data<T>();
      lstmp_value.output_value = hidden_t.data<T>();
      lstmp_value.state_value = cell_t.data<T>();
      lstmp_value.state_active_value = cell_pre_act_t.data<T>();
      math::LstmUnitFunctor<DeviceContext, T>::compute(
          device_ctx, lstmp_value, frame_size, cur_batch_size, cell_clip,
          gate_act, cell_act, cand_act);
      lstmp_value.prev_state_value = lstmp_value.state_value;
      blas.MatMul(hidden_t, false, *proj_weight, false, static_cast<T>(1.0),
                  &proj_t, static_cast<T>(0.0));
      if (proj_act != math::detail::ActivationType::kIdentity) {
        auto proj_t_dev = EigenMatrix<T>::From(proj_t);
        ActCompute(cell_act, place, proj_t_dev, proj_t_dev);
      }
      if (proj_clip && proj_clip > 0.0) {
        T* x_data = proj_t.data<T>();
        int64_t numel = proj_t.numel();
        Transform<DeviceContext> trans;
        trans(ctx.template device_context<DeviceContext>(), x_data,
              x_data + numel, x_data,
              _ClipFunctor<T>(-1.0 * proj_clip, proj_clip));
      }
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    batch_proj.set_lod(batch_gate->lod());
    // restore the output hidden in LoDTensor from the batch hidden
    to_seq(device_ctx, batch_proj, proj_out);

    batch_cell.set_lod(batch_gate->lod());
    // restore the output cell state in LoDTensor from the batch cell
    to_seq(device_ctx, batch_cell, cell_out);
  }
};

template <typename DeviceContext, typename T>
class LSTMPGradKernel : public framework::OpKernel<T> {
 public:
  template <typename Device, typename X, typename Y, typename DX, typename DY>
  void ActGradCompute(const math::detail::ActivationType act_type,
                      const Device& d, X x, Y y, DX dx, DY dy) const {
    // x is dummy and won't be used even in Relu(use y instead)
    if (act_type == math::detail::ActivationType::kIdentity)
      dx.device(d) = dy;
    else if (act_type == math::detail::ActivationType::kSigmoid)
      SigmoidGradFunctor<T>()(d, x, y, dy, dx);
    else if (act_type == math::detail::ActivationType::kTanh)
      TanhGradFunctor<T>()(d, x, y, dy, dx);
    else if (act_type == math::detail::ActivationType::kReLU)
      ReluGradFunctor<T>()(d, x, y, dy, dx);
    else
      PADDLE_THROW(
          platform::errors::InvalidArgument("unsupported activation type"));
  }

  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* weight = ctx.Input<Tensor>("Weight");
    auto* proj_weight = ctx.Input<Tensor>("ProjWeight");
    auto* bias = ctx.Input<Tensor>("Bias");

    auto* proj_out = ctx.Input<LoDTensor>("Projection");
    auto* cell_out = ctx.Input<LoDTensor>("Cell");

    auto proj_clip = static_cast<T>(ctx.Attr<float>("proj_clip"));
    auto cell_clip = static_cast<T>(ctx.Attr<float>("cell_clip"));

    auto* batch_gate = ctx.Input<LoDTensor>("BatchGate");
    auto* batch_cell_pre_act = ctx.Input<LoDTensor>("BatchCellPreAct");
    auto* batch_hidden = ctx.Input<LoDTensor>("BatchHidden");

    auto* projection_g =
        ctx.Input<LoDTensor>(framework::GradVarName("Projection"));

    auto* in_g = ctx.Output<LoDTensor>(framework::GradVarName("Input"));
    auto* weight_g = ctx.Output<Tensor>(framework::GradVarName("Weight"));
    auto* proj_weight_g =
        ctx.Output<Tensor>(framework::GradVarName("ProjWeight"));
    auto* bias_g = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    auto* h0 = ctx.Input<Tensor>("H0");
    auto* c0 = ctx.Input<Tensor>("C0");

    auto* h0_g = ctx.Output<Tensor>(framework::GradVarName("H0"));
    auto* c0_g = ctx.Output<Tensor>(framework::GradVarName("C0"));

    auto& device_ctx = ctx.template device_context<DeviceContext>();
    math::SetConstant<DeviceContext, T> zero;
    if (weight_g) {
      weight_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, weight_g, static_cast<T>(0.0));
    }
    if (proj_weight_g) {
      proj_weight_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, proj_weight_g, static_cast<T>(0.0));
    }

    // ordered_h0/c0 is the reordered hidden/cell initialization.
    // ordered_h0_g/c0_g is the reordered gradient of hidden/cell
    // initialization.
    Tensor ordered_h0, ordered_c0, ordered_h0_g, ordered_c0_g;

    framework::Vector<size_t> order(batch_gate->lod()[2]);

    if (c0) {
      ReorderInitState<DeviceContext, T>(device_ctx, *c0, order, &ordered_c0,
                                         true);
    }
    if (c0 && c0_g) {
      ordered_c0_g.mutable_data<T>(c0_g->dims(), ctx.GetPlace());
    }

    // batch_gate dims equal to input dims
    auto in_dims = batch_gate->dims();
    auto out_dims = cell_out->dims();
    framework::DDim proj_dims({in_dims[0], proj_weight->dims()[1]});
    int frame_size = static_cast<int>(in_dims[1] / 4);
    PADDLE_ENFORCE_EQ(frame_size, out_dims[1],
                      platform::errors::InvalidArgument(
                          "The second dimension of Input(Cell) should be %d, "
                          "but received %d in LSTMP@Grad operator.",
                          frame_size, out_dims[1]));

    math::LstmMetaValue<T> lstmp_value;
    if (bias && ctx.Attr<bool>("use_peepholes")) {
      T* bias_data = const_cast<T*>(bias->data<T>());
      lstmp_value.check_ig = bias_data + 4 * frame_size;
      lstmp_value.check_fg = lstmp_value.check_ig + frame_size;
      lstmp_value.check_og = lstmp_value.check_fg + frame_size;
    } else {
      lstmp_value.check_ig = nullptr;
      lstmp_value.check_fg = nullptr;
      lstmp_value.check_og = nullptr;
    }

    math::LstmMetaGrad<T> lstmp_grad;

    if (bias && bias_g) {
      bias_g->mutable_data<T>(ctx.GetPlace());
      zero(device_ctx, bias_g, static_cast<T>(0.0));
    }
    if (bias && bias_g && ctx.Attr<bool>("use_peepholes")) {
      T* bias_g_data = bias_g->data<T>();
      lstmp_grad.check_ig_grad = bias_g_data + 4 * frame_size;
      lstmp_grad.check_fg_grad = lstmp_grad.check_ig_grad + frame_size;
      lstmp_grad.check_og_grad = lstmp_grad.check_fg_grad + frame_size;
    } else {
      lstmp_grad.check_ig_grad = nullptr;
      lstmp_grad.check_fg_grad = nullptr;
      lstmp_grad.check_og_grad = nullptr;
    }

    math::LoDTensor2BatchFunctor<DeviceContext, T> to_batch;

    auto ToBatch = [&batch_gate, &to_batch](
        const DeviceContext& ctx, const framework::LoDTensor& src,
        const framework::DDim& dims, framework::LoDTensor& dst) {
      dst.mutable_data<T>(dims, ctx.GetPlace());
      dst.set_lod(batch_gate->lod());
      to_batch(ctx, src, &dst, false);
    };

    LoDTensor batch_hidden_g, batch_proj, batch_proj_g, batch_cell;
    batch_hidden_g.mutable_data<T>(out_dims, ctx.GetPlace());
    ToBatch(device_ctx, *proj_out, proj_dims, batch_proj);        // T x P
    ToBatch(device_ctx, *projection_g, proj_dims, batch_proj_g);  // T x P
    ToBatch(device_ctx, *cell_out, out_dims, batch_cell);         // T x D

    LoDTensor batch_cell_g, batch_gate_g;
    batch_cell_g.mutable_data<T>(out_dims, ctx.GetPlace());
    // TODO(qingqing) support the case output cell has gradient.
    // to_batch(device_ctx, *cell_g, batch_cell_g, false);
    zero(device_ctx, &batch_cell_g, static_cast<T>(0.0));
    batch_gate_g.mutable_data<T>(batch_gate->dims(), ctx.GetPlace());
    batch_gate_g.set_lod(batch_gate->lod());

    auto gate_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("gate_activation"));
    auto cell_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("cell_activation"));
    auto cand_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("candidate_activation"));
    auto proj_act = math::detail::GetActivationType(
        ctx.Attr<std::string>("proj_activation"));
    auto& place = *ctx.template device_context<DeviceContext>().eigen_device();

    auto batch_starts = batch_gate->lod()[0];
    size_t num_batch = batch_starts.size() - 1;
    auto blas = math::GetBlas<DeviceContext, T>(device_ctx);
    for (int n = static_cast<int>(num_batch) - 1; n >= 0; n--) {
      int bstart = static_cast<int>(batch_starts[n]);
      int bend = static_cast<int>(batch_starts[n + 1]);

      Tensor cur_proj = batch_proj.Slice(bstart, bend);
      Tensor proj_g = batch_proj_g.Slice(bstart, bend);

      if (proj_clip && proj_clip > 0.0) {
        T* dx_data = proj_g.data<T>();
        T* x_data = cur_proj.data<T>();
        int64_t numel = proj_g.numel();
        Transform<DeviceContext> trans;
        trans(ctx.template device_context<DeviceContext>(), dx_data,
              dx_data + numel, x_data, dx_data,
              _ClipGradFunctor<T>(-1.0 * proj_clip, proj_clip));
      }

      if (proj_act != math::detail::ActivationType::kIdentity) {
        auto cur_proj_dev = EigenMatrix<T>::From(cur_proj);
        auto proj_g_dev = EigenMatrix<T>::From(proj_g);
        ActGradCompute(cell_act, place, cur_proj_dev, cur_proj_dev, proj_g_dev,
                       proj_g_dev);
      }
      /* hidden state backwarad */
      Tensor out_g = batch_hidden_g.Slice(bstart, bend);
      blas.MatMul(proj_g, false, *proj_weight, true, static_cast<T>(1.0),
                  &out_g, static_cast<T>(0.0));
      /* projection weight backward*/
      if (proj_weight_g) {
        Tensor hidden_t = batch_hidden->Slice(bstart, bend);
        blas.MatMul(hidden_t, true, proj_g, false, static_cast<T>(1.0),
                    proj_weight_g, static_cast<T>(1.0));
      }

      Tensor gate = batch_gate->Slice(bstart, bend);
      Tensor cell = batch_cell.Slice(bstart, bend);
      Tensor cell_pre_act = batch_cell_pre_act->Slice(bstart, bend);
      lstmp_value.gate_value = gate.data<T>();
      lstmp_value.state_value = cell.data<T>();
      lstmp_value.state_active_value = cell_pre_act.data<T>();

      Tensor gate_g = batch_gate_g.Slice(bstart, bend);
      Tensor cell_g = batch_cell_g.Slice(bstart, bend);
      lstmp_grad.state_grad = cell_g.data<T>();
      lstmp_grad.gate_grad = gate_g.data<T>();
      lstmp_grad.output_grad = out_g.data<T>();

      if (n > 0) {
        int bstart_pre = static_cast<int>(batch_starts[n - 1]);
        Tensor cell_pre = batch_cell.Slice(bstart_pre, bstart);
        Tensor cell_pre_g = batch_cell_g.Slice(bstart_pre, bstart);
        lstmp_value.prev_state_value = cell_pre.data<T>();
        lstmp_grad.prev_state_grad = cell_pre_g.data<T>();
      } else {
        lstmp_value.prev_state_value = c0 ? ordered_c0.data<T>() : nullptr;
        lstmp_grad.prev_state_grad = c0_g ? ordered_c0_g.data<T>() : nullptr;
      }

      int cur_batch_size = bend - bstart;
      // lstmp_value.output_value not used in bp, set to null
      // lstmp_grad.state_active_grad not used in bp, set to null
      lstmp_value.output_value = nullptr;
      lstmp_grad.state_active_grad = nullptr;

      math::LstmUnitGradFunctor<DeviceContext, T>::compute(
          device_ctx, lstmp_value, lstmp_grad, frame_size, cur_batch_size,
          cell_clip, gate_act, cell_act, cand_act);

      if (n > 0) {
        int pre_h_start = static_cast<int>(batch_starts[n - 1]);
        int pre_h_end = pre_h_start + cur_batch_size;
        auto pre_proj_g = batch_proj_g.Slice(pre_h_start, pre_h_end);
        blas.MatMul(gate_g, false, *weight, true, static_cast<T>(1.0),
                    &pre_proj_g, static_cast<T>(1.0));
        if (weight_g) {
          /* weight backward*/
          auto pre_proj = batch_proj.Slice(pre_h_start, pre_h_end);
          blas.MatMul(pre_proj, true, gate_g, false, static_cast<T>(1.0),
                      weight_g, static_cast<T>(1.0));
        }
      } else {
        if (h0 && weight_g) {
          ReorderInitState<DeviceContext, T>(device_ctx, *h0, order,
                                             &ordered_h0, true);
          if (weight_g) {
            blas.MatMul(ordered_h0, true, gate_g, false, static_cast<T>(1.0),
                        weight_g, static_cast<T>(1.0));
          }
        }
        if (h0 && (h0_g || proj_weight_g)) {
          ordered_h0_g.mutable_data<T>(h0_g->dims(), ctx.GetPlace());
          blas.MatMul(gate_g, false, *weight, true, static_cast<T>(1.0),
                      &ordered_h0_g, static_cast<T>(0.0));
        }
      }
    }

    math::Batch2LoDTensorFunctor<DeviceContext, T> to_seq;
    if (in_g) {
      /* backward data */
      in_g->mutable_data<T>(ctx.GetPlace());
      to_seq(device_ctx, batch_gate_g, in_g);
    }
    if (bias && bias_g) {
      /* backward bias */
      Tensor b_g = *bias_g;
      b_g.Resize({bias_g->numel(), 1});
      Tensor gate_bias_g = b_g.Slice(0, 4 * frame_size);
      math::ColwiseSum<DeviceContext, T> col_sum;
      col_sum(device_ctx, batch_gate_g, &gate_bias_g);
    }

    if (h0 && h0_g) {
      ReorderInitState<DeviceContext, T>(device_ctx, ordered_h0_g, order, h0_g,
                                         false);
    }
    if (c0 && c0_g) {
      ReorderInitState<DeviceContext, T>(device_ctx, ordered_c0_g, order, c0_g,
                                         false);
    }
  }
};

}  // namespace operators
}  // namespace paddle
