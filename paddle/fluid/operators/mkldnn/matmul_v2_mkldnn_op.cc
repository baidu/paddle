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

#include "paddle/fluid/operators/mkldnn/matmul_mkldnn_op.h"

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using framework::DataLayout;
using framework::ExecutionContext;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::MKLDNNGetDataType;
using platform::to_void_cast;
using Tensor = framework::Tensor;

template <typename T>
class MatMulV2MKLDNNHandler : public platform::MKLDNNHandlerT<T, dnnl::matmul> {
 public:
  MatMulV2MKLDNNHandler(const MKLDNNDeviceContext& dev_ctx,
                        const mkldnn::engine engine, platform::Place cpu_place,
                        const std::vector<int64_t>& x_org_dims, bool trans_x,
                        const std::vector<int64_t>& y_org_dims, bool trans_y,
                        const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, dnnl::matmul>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(dev_ctx, x_org_dims, uniq_name)) {
    if (!this->isCached()) {
      // M X K * K X N
      std::vector<int64_t> x_dims(x_org_dims);
      std::vector<int64_t> y_dims(y_org_dims);

      const int MB_idx = x_dims.size() - 3;
      const int H_idx = x_dims.size() - 2;
      const int W_idx = x_dims.size() - 1;

      if (trans_x) std::swap(x_dims[H_idx], x_dims[W_idx]);
      if (trans_y) std::swap(y_dims[H_idx], y_dims[W_idx]);

      const memory::dim M = x_dims[H_idx];
      const memory::dim K = x_dims[W_idx];
      const memory::dim N = y_dims[W_idx];

      std::vector<int64_t> x_strides(x_dims.size() - 3, 1);
      std::vector<int64_t> y_strides(x_dims.size() - 3, 1);
      std::vector<int64_t> out_strides(x_dims.size() - 3, 1);
      std::vector<int64_t> out_ddims(x_dims.size() - 3, 1);

      x_strides.reserve(x_dims.size());
      y_strides.reserve(x_dims.size());
      out_strides.reserve(x_dims.size());

      if (!trans_x) {
        x_strides.insert(x_strides.end(), {M * K, K, 1});
      } else {
        x_strides.insert(x_strides.end(), {M * K, 1, M});
      }

      if (!trans_y) {
        y_strides.insert(y_strides.end(), {N * K, N, 1});
      } else {
        y_strides.insert(y_strides.end(), {N * K, 1, K});
      }

      out_strides.insert(out_strides.end(), {M * N, N, 1});
      out_ddims.insert(out_ddims.end(),
                       {std::max(x_dims[MB_idx], y_dims[MB_idx]), M, N});

      for (int i = x_dims.size() - 4; i >= 0; --i) {
        out_ddims[i] = std::max(x_dims[i], y_dims[i]);
        x_strides[i] = x_dims[i + 1] * x_strides[i + 1];
        y_strides[i] = y_dims[i + 1] * y_strides[i + 1];
        out_strides[i] = out_ddims[i + 1] * out_strides[i + 1];
      }

      auto x_md = memory::desc(x_dims, MKLDNNGetDataType<T>(), x_strides);
      auto y_md = memory::desc(y_dims, MKLDNNGetDataType<T>(), y_strides);
      auto out_md =
          memory::desc(out_ddims, MKLDNNGetDataType<T>(), out_strides);

      this->AcquireForwardPrimitiveDescriptor(x_md, y_md, out_md);
    }
  }

  std::shared_ptr<memory> AcquireWeightsMemory(const Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->weights_desc(),
                                            to_void_cast<T>(input_data),
                                            "@weights_mem_p");
  }
};

template <typename T>
class MatMulV2MKLDNNKernel : public MatMulGradMKLDNNKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 protected:
  void ExecuteMatMul(const ExecutionContext& ctx,
                     const MKLDNNDeviceContext& dev_ctx,
                     const mkldnn::engine onednn_engine,
                     platform::Place cpu_place, const Tensor* x,
                     std::vector<int64_t>& x_dims, bool trans_x,
                     const Tensor* y, std::vector<int64_t>& y_dims,
                     bool trans_y, Tensor* out, std::vector<int64_t>& out_dims,
                     int execution_number = 0) const {
    MatMulV2MKLDNNHandler<T> handler(
        dev_ctx, onednn_engine, ctx.GetPlace(), x_dims, trans_x, y_dims,
        trans_y, ctx.InputName("X") + std::to_string(execution_number));

    const auto src_memory_p = handler.AcquireSrcMemory(x);
    const auto weights_memory_p = handler.AcquireWeightsMemory(y);
    const auto dst_memory_p = handler.AcquireDstMemory(out);

    auto matmul_p = handler.AcquireForwardPrimitive();

    std::unordered_map<int, memory> matmul_args = {
        {DNNL_ARG_SRC, *src_memory_p},
        {DNNL_ARG_WEIGHTS, *weights_memory_p},
        {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = MKLDNNDeviceContext::tls().get_stream();
    matmul_p->execute(astream, matmul_args);
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(
        GetMKLDNNFormat(dst_memory_p->get_desc().reshape(out_dims)));
  }

 private:
  void CalculateMatrixDims(const ExecutionContext& ctx,
                           const std::vector<int64_t>& x_dims,
                           const std::vector<int64_t>& y_dims,
                           std::vector<int64_t>& x_bd_dims,
                           std::vector<int64_t>& y_bd_dims,
                           std::vector<int64_t>& out_dims, Tensor* out) const {
    if (x_dims.size() == 1) {
      x_bd_dims[x_bd_dims.size() - 1] = x_dims[0];
    } else if (x_dims.size() == 2) {
      x_bd_dims[2] = x_dims[1];
      x_bd_dims[1] = x_dims[0];
    } else {
      for (size_t i = 0; i < x_dims.size(); ++i) {
        x_bd_dims[i] = x_dims[i];
      }
    }
    if (y_dims.size() == 1) {
      y_bd_dims[x_bd_dims.size() - 2] = y_dims[0];
    } else if (y_dims.size() == 2) {
      y_bd_dims[2] = y_dims[1];
      y_bd_dims[1] = y_dims[0];
    } else {
      for (size_t i = 0; i < y_dims.size(); ++i) {
        y_bd_dims[i] = y_dims[i];
      }
    }

    if ((y_dims.size() == x_dims.size()) && y_dims.size() > 2) {
      for (size_t i = 0; i < x_dims.size() - 2; ++i) {
        PADDLE_ENFORCE_EQ(
            x_dims[i] == y_dims[i] || x_dims[i] == 1 || y_dims[i] == 1, true,
            platform::errors::InvalidArgument(
                "Tensor dimensions are incorrect for broadcasting."
                "Dimensions in X and Y must be same or equal to 1, but "
                "received x_dim[%d]=%d and y_dims[%d]= %d",
                i, x_dims[i], i, y_dims[i]));
        out_dims[i] = std::max(x_dims[i], y_dims[i]);
      }
      out->Resize(framework::make_ddim(out_dims));
    }
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");
    auto* out = ctx.Output<Tensor>("Out");
    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");

    auto x_dims = framework::vectorize(x->dims());
    auto y_dims = framework::vectorize(y->dims());
    auto out_dims = framework::vectorize(out->dims());

    int ndims = std::max(x->dims().size(), y->dims().size());
    ndims = std::max(ndims, 3);

    std::vector<int64_t> x_bd_dims(ndims, 1);
    std::vector<int64_t> y_bd_dims(ndims, 1);

    CalculateMatrixDims(ctx, x_dims, y_dims, x_bd_dims, y_bd_dims, out_dims,
                        out);

    ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x, x_bd_dims,
                  trans_x, y, y_bd_dims, trans_y, out, out_dims);
  }
};

template <typename T>
class MatMulV2GradMKLDNNKernel : public MatMulV2MKLDNNKernel<T> {
 public:
  void Compute(const ExecutionContext& ctx) const override { RunKernel(ctx); }

 private:
  void CalculateGradMatrixDims(const ExecutionContext& ctx, Tensor* dx_tmp,
                               Tensor* dy_tmp,
                               const std::vector<int64_t>& dx_dims,
                               const std::vector<int64_t>& dy_dims,
                               std::vector<int64_t>& dx_bd_dims,
                               std::vector<int64_t>& dy_bd_dims) const {
    for (size_t i = 0; i < dx_dims.size() - 2; ++i) {
      if (dx_dims[i] != dy_dims[i]) {
        if (dx_dims[i] == 1) {
          dx_bd_dims[i] = dy_dims[i];
        } else {
          dy_bd_dims[i] = dx_dims[i];
        }
      }
    }

    dx_tmp->Resize(framework::make_ddim(dx_bd_dims));
    dx_tmp->mutable_data<T>(ctx.GetPlace());
    dy_tmp->Resize(framework::make_ddim(dy_bd_dims));
    dy_tmp->mutable_data<T>(ctx.GetPlace());
  }

  void ReduceSumForMatmulGradOutput(const ExecutionContext& ctx,
                                    const MKLDNNDeviceContext& dev_ctx,
                                    const mkldnn::engine onednn_engine,
                                    const Tensor* dx_tmp, Tensor* dx,
                                    std::vector<int64_t> dx_dims) const {
    platform::ReductionMKLDNNHandler<T> handler(
        dnnl::algorithm::reduction_sum, 0.0f, 0.0f, dev_ctx, onednn_engine,
        ctx.GetPlace(), dx_tmp, dx, ctx.InputName("X"), dx_dims);

    auto src_memory_p = handler.AcquireSrcMemory(dx_tmp);
    auto dst_memory_p = handler.AcquireDstMemory(dx);

    std::unordered_map<int, dnnl::memory> reduction_args = {
        {DNNL_ARG_SRC, *src_memory_p}, {DNNL_ARG_DST, *dst_memory_p}};

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    auto reduction_p = handler.AcquireForwardPrimitive();

    reduction_p->execute(astream, reduction_args);
    astream.wait();
  }

  void RunKernel(const ExecutionContext& ctx) const {
    const auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Input<Tensor>("Y");

    auto x_dims = framework::vectorize(x->dims());
    auto y_dims = framework::vectorize(y->dims());

    bool is_broadcast = true;
    if (x_dims.size() <= 2 || y_dims.size() <= 2) {
      is_broadcast = false;
    } else if (x_dims.size() != y_dims.size()) {
      is_broadcast = true;
    } else {
      is_broadcast =
          !std::equal(x_dims.cbegin(), x_dims.cbegin() + x_dims.size() - 2,
                      y_dims.cbegin());
    }

    // if no broadcasting is needed, we can simply use matmul's grad and avoid
    // using reduce_sum
    if (!is_broadcast) {
      MatMulGradMKLDNNKernel<T>::Compute(ctx);
      return;
    }

    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = ctx.Output<Tensor>(framework::GradVarName("Y"));

    bool trans_x = ctx.Attr<bool>("trans_x");
    bool trans_y = ctx.Attr<bool>("trans_y");
    auto dout_dims = framework::vectorize(dout->dims());

    int ndims = std::max(x->dims().size(), y->dims().size());
    ndims = std::max(ndims, 3);

    // in broadcasting scenario new memory is required because
    // reduce sum must be calculated upon broadcasted dims
    Tensor dx_tmp, dy_tmp;

    std::vector<int64_t> dx_bd_dims(x_dims);
    std::vector<int64_t> dy_bd_dims(y_dims);

    CalculateGradMatrixDims(ctx, &dx_tmp, &dy_tmp, x_dims, y_dims, dx_bd_dims,
                            dy_bd_dims);

    if (trans_x && trans_y) {
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), y,
                          y_dims, true, dout, dout_dims, true, &dx_tmp,
                          dx_bd_dims, 1);
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                          dout_dims, true, x, x_dims, true, &dy_tmp, dy_bd_dims,
                          2);
    } else if (trans_x) {
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), y,
                          y_dims, false, dout, dout_dims, true, &dx_tmp,
                          dx_bd_dims, 1);
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x,
                          x_dims, false, dout, dout_dims, false, &dy_tmp,
                          dy_bd_dims, 2);
    } else if (trans_y) {
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                          dout_dims, false, y, y_dims, false, &dx_tmp,
                          dx_bd_dims, 1);
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                          dout_dims, true, x, x_dims, false, &dy_tmp,
                          dy_bd_dims, 2);
    } else {
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), dout,
                          dout_dims, false, y, y_dims, true, &dx_tmp,
                          dx_bd_dims, 1);
      this->ExecuteMatMul(ctx, dev_ctx, onednn_engine, ctx.GetPlace(), x,
                          x_dims, true, dout, dout_dims, false, &dy_tmp,
                          dy_bd_dims, 2);
    }

    if (x_dims != dx_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx, dev_ctx, onednn_engine, &dx_tmp, dx,
                                   x_dims);
    } else {
      *dx = std::move(dx_tmp);
    }
    if (y_dims != dy_bd_dims) {
      ReduceSumForMatmulGradOutput(ctx, dev_ctx, onednn_engine, &dy_tmp, dy,
                                   y_dims);
    } else {
      *dy = std::move(dy_tmp);
    }

    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(x->format());
    dy->set_layout(framework::DataLayout::kMKLDNN);
    dy->set_format(y->format());
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP_KERNEL(matmul_v2, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MatMulV2MKLDNNKernel<float>,
                   ops::MatMulV2MKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(matmul_v2_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::MatMulV2GradMKLDNNKernel<float>,
                   ops::MatMulV2GradMKLDNNKernel<paddle::platform::bfloat16>);
