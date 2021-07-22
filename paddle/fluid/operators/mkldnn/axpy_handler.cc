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

#include <cinttypes>
#include <memory>
#include <string>
#include <vector>

#include "mkldnn.hpp"
#include "paddle/fluid/operators/mkldnn/axpy_handler.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

namespace plat = paddle::platform;

namespace {

template <typename T>
class AXPYMKLDNNHandler : public plat::MKLDNNHandlerT<T, dnnl::reorder> {
 public:
  AXPYMKLDNNHandler(const plat::MKLDNNDeviceContext &dev_ctx,
                    const dnnl::engine mkldnn_engine, plat::Place cpu_place,
                    int n, float alpha)
      : plat::MKLDNNHandlerT<T, dnnl::reorder>(
            dev_ctx, mkldnn_engine, cpu_place,
            plat::CreateKey(dev_ctx, static_cast<int64_t>(n),
                            plat::MKLDNNGetDataType<T>(), alpha, "-axpy")),
        alpha_(alpha),
        n_(n) {}

  std::shared_ptr<dnnl::memory> AcquireMemory(void *ptr,
                                              const std::string &suffix) {
    /*Generate key*/
    auto local_key = this->key_ + suffix;
    auto mem_p = std::static_pointer_cast<dnnl::memory>(
        this->dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto md = dnnl::memory::desc({n_}, plat::MKLDNNGetDataType<T>(),
                                   dnnl::memory::format_tag::x);
      mem_p = std::make_shared<dnnl::memory>(md, this->engine_, ptr);
      this->dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<dnnl::memory> AcquireSrcMemory(const T *x) {
    return this->AcquireMemory(plat::to_void_cast(x), "@user_src_mem_p");
  }

  std::shared_ptr<dnnl::memory> AcquireDstMemory(T *y) {
    return this->AcquireMemory(y, "@user_dst_mem_p");
  }

  std::shared_ptr<dnnl::reorder> AcquireReorder(
      std::shared_ptr<dnnl::memory> dst_memory_p,
      std::shared_ptr<dnnl::memory> src_memory_p) {
    auto prim_key = this->key_ + "@reorder_p";
    auto reorder_p = std::static_pointer_cast<dnnl::reorder>(
        this->dev_ctx_.GetBlob(prim_key));
    if (reorder_p == nullptr) {
      // Here we pass Postops to mimick y -> a*X + y
      dnnl::primitive_attr reorder_attr;
      dnnl::post_ops post_operations;
      if (this->alpha_ != 1.f) {
        std::vector<float> scales(1, this->alpha_);
        reorder_attr.set_output_scales(0, scales);
      }
      post_operations.append_sum(1.0f);

      reorder_attr.set_post_ops(post_operations);
      reorder_p = std::make_shared<dnnl::reorder>(
          *(src_memory_p), *(dst_memory_p), reorder_attr);
      this->dev_ctx_.SetBlob(prim_key, reorder_p);
    }
    return reorder_p;
  }

 private:
  float alpha_;
  int n_;
};

template class AXPYMKLDNNHandler<float>;
template class AXPYMKLDNNHandler<plat::bfloat16>;

}  // anonnymouse namespace

template <typename T>
static void naive_axpy(int n, float alpha, const T *x, T *y) {
  while (n-- > 0) {
    *y += alpha * *x;
    ++y;
    ++x;
  }
}

template <typename T>
void onednn_handler_axpy(int n, float alpha, const T *x, T *y) {
  // fallback to naive version
  if (n < 100) {
    naive_axpy(n, alpha, x, y);
    return;
  }

  auto &pool = plat::DeviceContextPool::Instance();
  auto cpu_place = plat::CPUPlace();
  auto *dev_ctx =
      dynamic_cast<plat::MKLDNNDeviceContext *>(pool.Get(cpu_place));
  auto &cpu_engine = dev_ctx->GetEngine();

  AXPYMKLDNNHandler<T> handler(*dev_ctx, cpu_engine, cpu_place, n, alpha);

  auto reorder_src_memory_p = handler.AcquireSrcMemory(x);
  auto reorder_dst_memory_p = handler.AcquireDstMemory(y);
  auto reorder_p =
      handler.AcquireReorder(reorder_dst_memory_p, reorder_src_memory_p);

  auto &astream = plat::MKLDNNDeviceContext::tls().get_stream();
  plat::RecordEvent record_reorder("axpy_int_reorder",
                                   plat::EventRole::kUniqueOp);
  reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
  astream.wait();
}

template void onednn_handler_axpy<float>(int, float, const float *, float *);
template void onednn_handler_axpy<plat::bfloat16>(int, float,
                                                  const plat::bfloat16 *,
                                                  plat::bfloat16 *);

}  // namespace operators
}  // namespace paddle
