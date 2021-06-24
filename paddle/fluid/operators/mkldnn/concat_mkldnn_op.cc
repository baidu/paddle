/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <memory>
#include "paddle/fluid/operators/concat_op.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::concat;
using mkldnn::stream;
using platform::to_void_cast;

static void EnforceLayouts(const std::vector<const Tensor*> inputs) {
  for (auto* input : inputs) {
    PADDLE_ENFORCE_EQ(
        input->layout(), DataLayout::kMKLDNN,
        platform::errors::InvalidArgument("Wrong layout set for Input tensor"));
    PADDLE_ENFORCE_NE(
        input->format(), MKLDNNMemoryFormat::undef,
        platform::errors::InvalidArgument("Wrong format set for Input tensor"));
  }
}

static memory::desc CreateMemDesc(const Tensor& input,
                                  const memory::data_type& dt) {
  const auto dims = paddle::framework::vectorize<int64_t>(input.dims());
  const auto format = input.format();
  auto mem_desc = memory::desc(dims, dt, format);
  return mem_desc;
}

static platform::CPUPlace GetCpuPlace(
    const paddle::framework::ExecutionContext& ctx) {
  auto place = ctx.GetPlace();
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(place),
                 platform::errors::InvalidArgument("It must use CPUPlace."));
  return BOOST_GET_CONST(platform::CPUPlace, place);
}

static const mkldnn::engine& GetMKLDNNEngine(
    const paddle::framework::ExecutionContext& ctx) {
  auto& dev_ctx = ctx.template device_context<platform::MKLDNNDeviceContext>();
  return dev_ctx.GetEngine();
}

// From a multi-input, gather only nonempty inputs
static const std::vector<const Tensor*> ReduceMultiInput(
    const std::vector<const Tensor*>& inputs) {
  std::vector<const Tensor*> reduced(inputs.size());
  auto end_it = std::copy_if(inputs.begin(), inputs.end(), reduced.begin(),
                             [](const Tensor* t) { return t->numel() > 0; });
  reduced.resize(std::distance(reduced.begin(), end_it));
  return reduced;
}

static const std::vector<int> GetDimsForKey(
    const std::vector<const Tensor*>& inputs) {
  auto dims_key = paddle::framework::vectorize<int>(inputs[0]->dims());
  for (auto it = std::next(inputs.begin()); it != inputs.end(); ++it) {
    dims_key.push_back((*it)->dims()[0]);
  }
  return dims_key;
}

template <typename T>
class ConcatPrimitiveFactory {
 public:
  concat::primitive_desc CreateConcatPrimDescriptor(
      const std::vector<const Tensor*> multi_input, Tensor* output,
      int concat_axis, const mkldnn::engine& mkldnn_engine,
      const memory::data_type& dt = memory::data_type::f32) {
    CreateSourcesDescriptors(multi_input, mkldnn_engine, dt);
    auto dst_desc = CreateDstMemDescriptor(output, dt);
    return concat::primitive_desc(dst_desc, concat_axis, srcs_d, mkldnn_engine);
  }

  concat CreateConcatPrimitive(const concat::primitive_desc& concat_pd,
                               Tensor* output, platform::CPUPlace place,
                               const mkldnn::engine& mkldnn_engine) {
    dst_mem = mkldnn::memory(
        concat_pd.dst_desc(), mkldnn_engine,
        output->mutable_data<T>(place, concat_pd.dst_desc().get_size()));

    return concat(concat_pd);
  }

  void SetSrcDataHandleByIndex(const std::vector<memory>& srcs, const size_t& i,
                               void* handler) {
    srcs[i].set_data_handle(handler);
  }

  void SetDstDataHandle(const memory& dst_mem, void* handler) {
    dst_mem.set_data_handle(handler);
  }

  std::vector<memory> GetSrcs() { return srcs; }

  memory GetDst() { return dst_mem.get(); }

 private:
  memory::desc CreateDstMemDescriptor(Tensor* output,
                                      const memory::data_type& dt) {
    auto dst_dims = paddle::framework::vectorize<int64_t>(output->dims());
    return memory::desc(dst_dims, dt, MKLDNNMemoryFormat::any);
  }

  void CreateSourcesDescriptors(const std::vector<const Tensor*> multi_input,
                                const mkldnn::engine& mkldnn_engine,
                                const memory::data_type& dt) {
    for (size_t i = 0; i < multi_input.size(); i++) {
      auto mem_desc = CreateMemDesc(*multi_input[i], dt);
      srcs_d.push_back(mem_desc);
      srcs.push_back(memory(mem_desc, mkldnn_engine,
                            to_void_cast(multi_input[i]->data<T>())));
    }
  }

 private:
  std::vector<memory::desc> srcs_d;
  std::vector<mkldnn::memory> srcs;
  boost::optional<mkldnn::memory> dst_mem;
};

template <typename T>
class ConcatMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    // If any of the multiple inputs of concat has an input size of 0, the
    // actual size of the multi_input will change
    auto multi_input = ReduceMultiInput(ctx.MultiInput<Tensor>("X"));
    EnforceLayouts(multi_input);
    Tensor* output = ctx.Output<Tensor>("Out");
    int concat_axis = ctx.Attr<int>("axis");
    const int rank = multi_input[0]->dims().size();
    PADDLE_ENFORCE_EQ(
        concat_axis >= -rank && concat_axis < rank, true,
        platform::errors::InvalidArgument(
            "The axis is expected to be in range of [%d, %d), but got %d",
            -rank, rank, concat_axis));
    platform::MKLDNNDeviceContext::tls().log_lib_version();

    if (ctx.HasInput("AxisTensor")) {
      auto* axis_tensor = ctx.Input<Tensor>("AxisTensor");
      concat_axis = GetDataFromTensor(axis_tensor)[0];
      auto out_dims = multi_input[0]->dims();
      for (size_t i = 1; i < multi_input.size(); ++i) {
        out_dims[concat_axis] += multi_input[i]->dims()[concat_axis];
      }
      output->Resize(out_dims);
    }

    if (concat_axis < 0) {
      concat_axis = concat_axis + rank;
    }
    auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    auto place = GetCpuPlace(ctx);

    memory::data_type dt =
        paddle::framework::ToMKLDNNDataType(multi_input[0]->type());

    ConcatPrimitiveFactory<T> prim_creator;
    std::string key =
        platform::CreateKey(dev_ctx, GetDimsForKey(multi_input),
                            multi_input.size(), ctx.OutputName("Out"), dt);
    key = platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, key);

    const std::string key_prim = key + "@concat_p";
    const std::string key_concat_pd = key + "@concat_pd";
    const std::string key_srcs = key + "@concat_srcs";
    const std::string key_dst = key + "@concat_dst";

    std::shared_ptr<concat::primitive_desc> concat_pd;
    std::shared_ptr<std::vector<memory>> srcs;
    std::shared_ptr<memory> dst_mem;
    auto concat_p = std::static_pointer_cast<concat>(dev_ctx.GetBlob(key_prim));

    const auto& mkldnn_engine = dev_ctx.GetEngine();
    if (concat_p == nullptr) {
      concat_pd = std::make_shared<concat::primitive_desc>(
          prim_creator.CreateConcatPrimDescriptor(
              multi_input, output, concat_axis, mkldnn_engine, dt));
      concat_p = std::make_shared<concat>(prim_creator.CreateConcatPrimitive(
          *concat_pd, output, place, mkldnn_engine));
      srcs = std::make_shared<std::vector<memory>>(prim_creator.GetSrcs());
      dst_mem = std::make_shared<memory>(prim_creator.GetDst());
      dev_ctx.SetBlob(key_prim, concat_p);
      dev_ctx.SetBlob(key_concat_pd, concat_pd);
      dev_ctx.SetBlob(key_srcs, srcs);
      dev_ctx.SetBlob(key_dst, dst_mem);
    } else {
      srcs = std::static_pointer_cast<std::vector<memory>>(
          dev_ctx.GetBlob(key_srcs));
      dst_mem = std::static_pointer_cast<memory>(dev_ctx.GetBlob(key_dst));
      concat_pd = std::static_pointer_cast<concat::primitive_desc>(
          dev_ctx.GetBlob(key_concat_pd));
      for (size_t i = 0; i < multi_input.size(); i++) {
        prim_creator.SetSrcDataHandleByIndex(
            *srcs, i, to_void_cast<T>(multi_input[i]->data<T>()));
      }
      prim_creator.SetDstDataHandle(
          *dst_mem,
          output->mutable_data<T>(place, concat_pd->dst_desc().get_size()));
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    std::unordered_map<int, memory> args;
    for (size_t i = 0; i < multi_input.size(); ++i) {
      args.insert({MKLDNN_ARG_MULTIPLE_SRC + i, (*srcs).at(i)});
    }
    args.insert({MKLDNN_ARG_DST, *dst_mem});

    concat_p->execute(astream, args);
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_mem));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(concat, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::ConcatMKLDNNOpKernel<float>,
                   ops::ConcatMKLDNNOpKernel<paddle::platform::bfloat16>,
                   ops::ConcatMKLDNNOpKernel<int8_t>,
                   ops::ConcatMKLDNNOpKernel<uint8_t>);
