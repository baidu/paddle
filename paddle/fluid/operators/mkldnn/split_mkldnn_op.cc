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

#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;

static inline std::vector<std::vector<int64_t>> CalculateOutsDims(
    const framework::DDim& in_dims, const size_t num,
    const std::vector<int>& sections, const size_t axis,
    const int outs_number) {
  std::vector<std::vector<int64_t>> outs_dims(outs_number,
                                              framework::vectorize(in_dims));

  if (num > 0) {
    PADDLE_ENFORCE_EQ(in_dims[axis] % num, 0,
                      platform::errors::InvalidArgument(
                          "The input's size along the split dimension "
                          "must be evenly divisible by Attr(num_or_sections). "
                          "But received Attr(num_or_sections) "
                          "= %d, input(X)'s shape = [%s], Attr(dim) = %d.",
                          num, in_dims, axis));

    const size_t out_axis_dim = in_dims[axis] / num;

    for (auto& out_dim : outs_dims) out_dim[axis] = out_axis_dim;
  } else {
    for (size_t i = 0; i < outs_dims.size(); ++i)
      outs_dims[i][axis] = sections[i];
  }
  return outs_dims;
}

template <typename T>
class SplitMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    const auto* x = ctx.Input<Tensor>("X");
    auto outs = ctx.MultiOutput<Tensor>("Out");

    int num = ctx.Attr<int>("num");
    auto sections = ctx.Attr<std::vector<int>>("sections");
    int axis = ctx.Attr<int>("axis");
    auto outs_number = outs.size();
    const auto x_dims = x->dims();

    bool need_resize = false;
    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<Tensor>("AxisTensor");
      axis = GetDataFromTensor(axis_tensor)[0];
      need_resize = true;
    }

    auto sections_tensor_list = ctx.MultiInput<Tensor>("SectionsTensorList");
    if (sections_tensor_list.size() > 0) {
      sections = GetDataFromTensorList(sections_tensor_list);
      need_resize = true;
    }

    if (need_resize) {
      const auto outs_dims =
          CalculateOutsDims(x->dims(), num, sections, axis, outs_number);
      for (size_t i = 0; i < outs.size(); ++i) {
        outs[i]->Resize(framework::make_ddim(outs_dims[i]));
      }
    }

    auto x_vec_dims = framework::vectorize(x_dims);

    mkldnn::memory::data_type x_type = framework::ToMKLDNNDataType(x->type());
    auto key = platform::CreateKey(dev_ctx, x_vec_dims, axis, num, sections,
                                   x->format(), x_type);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    std::vector<int64_t> offset(x_vec_dims.size(), 0);

    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, x->type(), x_type, dev_ctx, onednn_engine, key);
    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->format(), platform::to_void_cast(x->data<T>()));

    for (size_t i = 0; i < outs_number; ++i) {
      auto out_vec_dims = framework::vectorize(outs[i]->dims());
      auto slice_mem_p = reorder_handler.AcquireSrcSubmemory(
          out_vec_dims, offset, reorder_src_memory_p, i);

      auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
          outs[i], out_vec_dims, i, x->format(), ctx.GetPlace());
      auto reorder_p =
          reorder_handler.AcquireReorder(reorder_dst_memory_p, slice_mem_p, i);

      reorder_p->execute(astream, *slice_mem_p, *reorder_dst_memory_p);

      offset[axis] += num > 0 ? x->dims()[axis] / num : sections[i];

      outs[i]->set_layout(framework::DataLayout::kMKLDNN);
      outs[i]->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
    }
    astream.wait();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(split, MKLDNN, paddle::platform::CPUPlace,
                   ops::SplitMKLDNNKernel<float>,
                   ops::SplitMKLDNNKernel<paddle::platform::bfloat16>);
