/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include "boost/optional.hpp"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace platform {

using framework::DataLayout;
using framework::Tensor;
using user_function = std::function<std::shared_ptr<float>(const float*)>;
using memory = mkldnn::memory;

template <typename T, typename TForward,
          typename TBackward = mkldnn_dummy_primitive,
          typename TBackward_params = mkldnn_dummy_primitive>
class MKLDNNHandlerT {
 public:
  MKLDNNHandlerT(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                 platform::Place cpu_place, const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        place_(cpu_place),
        key_common_(base_key),
        key_(platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, base_key)),
        fwd_pd_(nullptr),
        bwd_pd_(nullptr) {
    platform::MKLDNNDeviceContext::tls().log_lib_version();
  }

  std::shared_ptr<TForward> AcquireForwardPrimitive() {
    const std::string key_p = key_ + "@fwd_p";
    auto forward_p =
        std::static_pointer_cast<TForward>(dev_ctx_.GetBlob(key_p));
    if (forward_p == nullptr) {
      forward_p = std::make_shared<TForward>(*fwd_pd_);
      dev_ctx_.SetBlob(key_p, forward_p);
    }
    return forward_p;
  }

  std::shared_ptr<TBackward> AcquireBackwardPrimitive() {
    const std::string key_p = key_ + "@bwd_p";
    auto backward_p =
        std::static_pointer_cast<TBackward>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      backward_p = std::make_shared<TBackward>(*bwd_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<TBackward_params> AcquireBackwardWeightsPrimitive() {
    const std::string key_p = key_ + "@bwd_w_p";
    auto backward_p =
        std::static_pointer_cast<TBackward_params>(dev_ctx_.GetBlob(key_p));
    if (backward_p == nullptr) {
      PADDLE_ENFORCE_NOT_NULL(bwd_w_pd_, platform::errors::Unavailable(
                                             "Error: BWD_PD should be set when "
                                             "getting BWD prim witk key: %s .",
                                             key_p));
      backward_p = std::make_shared<TBackward_params>(*bwd_w_pd_);
      dev_ctx_.SetBlob(key_p, backward_p);
    }
    return backward_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        fwd_pd_->src_desc(), to_void_cast<T>(input_data), "@src_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr =
        output->mutable_data<T_out>(place_, fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), ptr,
                                            "@dst_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<mkldnn::memory> AcquireDstMemory(void) {
    return this->AcquireMemoryFromPrimitive(fwd_pd_->dst_desc(), "@dstt_mem_p");
  }

  template <typename T_out = T>
  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const framework::Tensor* output) {
    const T_out* output_data = output->data<T_out>();
    return this->AcquireMemoryFromPrimitive(bwd_pd_->dst_desc(),
                                            to_void_cast<T_out>(output_data),
                                            "@bwd-dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const framework::Tensor* diffdst) {
    const T* ptr = diffdst->data<T>();
    return this->AcquireMemoryFromPrimitive(
        bwd_pd_->diff_dst_desc(), to_void_cast<T>(ptr), "@diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      framework::Tensor* diffsrc) {
    T* ptr =
        diffsrc->mutable_data<T>(place_, bwd_pd_->diff_src_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_pd_->diff_src_desc(), ptr,
                                            "@diff_src_mem_p");
  }

  // Buffer of given Tensor is used for oneDNN computation
  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemory(
      framework::Tensor* diff_weights) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        platform::errors::Unavailable(
            "Error: BWD_W_PD should be set when getting BWD grad of weights."));
    T* ptr = diff_weights->mutable_data<T>(
        place_, bwd_w_pd_->diff_weights_desc().get_size());
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(), ptr,
                                            "@diff_wei_mem_p");
  }

  // Buffer is allocated by oneDNN to store computation results
  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemory(void) {
    PADDLE_ENFORCE_NOT_NULL(
        bwd_w_pd_,
        platform::errors::Unavailable(
            "Error: BWD_W_PD should be set when getting BWD grad of weights."));
    return this->AcquireMemoryFromPrimitive(bwd_w_pd_->diff_weights_desc(),
                                            "@diff_wei_mem_p");
  }

 protected:
  bool isCached() {
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    return (fwd_pd_ != nullptr);
  }

  bool isBwdCached() {
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));

    if (bwd_pd_ == nullptr) {
      return false;
    } else {
      // When BWD is cached then still we need to Get FWD PD
      const std::string key_fpd = key_ + "@fwd_pd";
      fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
          dev_ctx_.GetBlob(key_fpd));
      PADDLE_ENFORCE_NOT_NULL(
          fwd_pd_, platform::errors::Unavailable(
                       "Error: FWD PD should be set when BWD PD is cached."));
      return true;
    }
  }

  // If your primitive descriptor requires attributes, pass them as a
  // first argument and paramters to descriptor constructor in the following
  // arguments. Otherwise, all arguments will be forwarded to descriptor
  // constructor, including the first one.
  template <typename Arg, typename... Args>
  void AcquireForwardPrimitiveDescriptor(Arg&& first_arg, Args&&... args) {
    // This is used when we can recreate FWD PD in BWD so
    // we do not need to pass FWD to BWD
    const std::string key_pd = key_ + "@fwd_pd";
    fwd_pd_ = std::static_pointer_cast<typename TForward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (fwd_pd_ == nullptr) {
      CreateForwardPrimitiveDescriptor(first_arg, std::forward<Args>(args)...);
      dev_ctx_.SetBlob(key_pd, fwd_pd_);
    }
  }

  // Using sfinae to specialise variadic function. Workaround for not having
  // if constexpr in C++ 11.
  template <class First, class... Args>
  typename std::enable_if<std::is_same<typename std::decay<First>::type,
                                       dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<Args>(args)...);
    fwd_pd_ = std::make_shared<typename TForward::primitive_desc>(
        fwd_desc, first, engine_);
  }

  template <class First, class... Args>
  typename std::enable_if<!std::is_same<typename std::decay<First>::type,
                                        dnnl::primitive_attr>::value>::type
  CreateForwardPrimitiveDescriptor(First&& first, Args&&... args) {
    auto fwd_desc = typename TForward::desc(std::forward<First>(first),
                                            std::forward<Args>(args)...);
    fwd_pd_ =
        std::make_shared<typename TForward::primitive_desc>(fwd_desc, engine_);
  }

  template <typename... Args>
  void AcquireBackwardPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        platform::errors::Unavailable("Get MKLDNN Forward primitive %s failed.",
                                      key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_pd";
    bwd_pd_ = std::static_pointer_cast<typename TBackward::primitive_desc>(
        dev_ctx_.GetBlob(key_pd));
    if (bwd_pd_ == nullptr) {
      auto bwd_desc = typename TBackward::desc(std::forward<Args>(args)...);
      bwd_pd_ = std::make_shared<typename TBackward::primitive_desc>(
          bwd_desc, engine_, *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_pd_);
    }
  }

  template <typename... Args>
  void AcquireBackwardWeightsPrimitiveDescriptor(Args&&... args) {
    // fwd_pd_ is set during grad by calling
    // AcquireForwardPrimitiveDescriptor
    PADDLE_ENFORCE_NOT_NULL(
        fwd_pd_,
        platform::errors::Unavailable("Get MKLDNN Forward primitive %s failed.",
                                      key_ + "@fwd_pd"));
    const std::string key_pd = key_ + "@bwd_w_pd";
    bwd_w_pd_ =
        std::static_pointer_cast<typename TBackward_params::primitive_desc>(
            dev_ctx_.GetBlob(key_pd));
    if (bwd_w_pd_ == nullptr) {
      auto bwd_desc =
          typename TBackward_params::desc(std::forward<Args>(args)...);
      bwd_w_pd_ = std::make_shared<typename TBackward_params::primitive_desc>(
          bwd_desc, engine_, *fwd_pd_);
      dev_ctx_.SetBlob(key_pd, bwd_w_pd_);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      const std::string& suffix) {
    return std::static_pointer_cast<mkldnn::memory>(
        dev_ctx_.GetBlob(key_ + suffix));
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::desc md, void* ptr, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::desc md, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(md, engine_);
      dev_ctx_.SetBlob(local_key, mem_p);
    }
    return mem_p;
  }

  void AcquireReorder(const std::shared_ptr<mkldnn::memory>& user_memory_p,
                      const std::shared_ptr<mkldnn::memory>& target_memory_p,
                      const std::string& suffix) {
    const auto key_reorder_p = key_ + suffix + "reorder_p";

    auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
        dev_ctx_.GetBlob(key_reorder_p));

    if (reorder_p == nullptr) {
      reorder_p =
          std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
      dev_ctx_.SetBlob(key_reorder_p, reorder_p);
    }

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    platform::RecordEvent record_reorder("int_reorder",
                                         platform::EventRole::kUniqueOp);
    reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                 {MKLDNN_ARG_TO, *target_memory_p}});
    astream.wait();
  }

  template <typename F = T>
  std::shared_ptr<mkldnn::memory> AcquireMemoryWithReorder(
      const mkldnn::memory::desc& user_md,
      const mkldnn::memory::desc& target_md, void* ptr,
      const std::string& suffix, bool is_persistent = false,
      std::function<std::shared_ptr<F>(const F*)> custom_reorder_func = {}) {
    const auto target_key = key_ + suffix + "_target";
    const auto key_reorder_p = key_ + suffix + "reorder_p";
    const auto user_key = key_ + suffix + "_user";

    auto target_memory_p =
        std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(target_key));

    if (target_memory_p == nullptr) {
      if (custom_reorder_func) {
        auto reordered_data =
            custom_reorder_func(reinterpret_cast<const F*>(ptr));
        dev_ctx_.SetBlob(key_reorder_p + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }
      auto user_memory_p =
          std::make_shared<dnnl::memory>(user_md, engine_, ptr);
      if (user_md != target_md) {
        target_memory_p = std::make_shared<mkldnn::memory>(target_md, engine_);
        auto reorder_p =
            std::make_shared<dnnl::reorder>(*user_memory_p, *target_memory_p);
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);

        auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      } else {
        target_memory_p = user_memory_p;
      }
      dev_ctx_.SetBlob(user_key, user_memory_p);
      dev_ctx_.SetBlob(target_key, target_memory_p);
    } else if (!is_persistent) {
      auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

      auto user_memory_p =
          std::static_pointer_cast<dnnl::memory>(dev_ctx_.GetBlob(user_key));
      user_memory_p->set_data_handle(ptr);

      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(const std::string& suffix) {
    const auto local_key = key_ + suffix;
    return std::static_pointer_cast<mkldnn::memory>(
        dev_ctx_.GetBlob(local_key));
  }

  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  platform::Place place_;
  std::string key_common_;
  std::string key_;
  std::shared_ptr<typename TForward::primitive_desc> fwd_pd_;
  std::shared_ptr<typename TBackward::primitive_desc> bwd_pd_;
  std::shared_ptr<typename TBackward_params::primitive_desc> bwd_w_pd_;
};

// TODO(grygielski) this class will be deleted later.
class MKLDNNHandler {
 public:
  MKLDNNHandler(const MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
                const std::string& base_key)
      : dev_ctx_(dev_ctx),
        engine_(engine),
        key_common_(base_key),
        key_(platform::ExtendKeyWithThreadInfoIfNeeded(dev_ctx, base_key)) {
    platform::MKLDNNDeviceContext::tls().log_lib_version();
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_diff_dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::desc md, void* ptr, const std::string& suffix) {
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemoryFromPrimitive(
      mkldnn::memory::desc md, const std::string& suffix) {
    const auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      mem_p = std::make_shared<mkldnn::memory>(md, engine_);
      dev_ctx_.SetBlob(local_key, mem_p);
    }
    return mem_p;
  }

  // This incarnation of AcquireMemory can call user function eg. custom reorder
  // or preprocessing routine if needed
  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const mkldnn::memory::desc& md, void* ptr, const std::string& suffix,
      user_function custom_func = {}) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      // Call custom reorder/preprocessing func if available
      if (custom_func) {
        auto reordered_data = custom_func(reinterpret_cast<const float*>(ptr));
        dev_ctx_.SetBlob(local_key + "-custom_reorder", reordered_data);
        ptr = reinterpret_cast<void*>(reordered_data.get());
      }

      mem_p = std::make_shared<mkldnn::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::vector<int64_t>& dims, const mkldnn::memory::data_type dtype,
      const MKLDNNMemoryFormat& fmt, void* ptr, const std::string& suffix) {
    /*Generate key*/
    auto local_key = key_ + suffix;
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto md = mkldnn::memory::desc(dims, dtype, fmt);

      mem_p = std::make_shared<mkldnn::memory>(md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      const std::shared_ptr<mkldnn::memory>& user_memory_p,
      const std::shared_ptr<mkldnn::memory>& target_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto stored_reorder_p = std::static_pointer_cast<mkldnn::reorder>(
        dev_ctx_.GetBlob(key_reorder_p));

    if (stored_reorder_p) {
      pipeline.push_back(*stored_reorder_p);
    } else {
      auto reorder_p =
          std::make_shared<mkldnn::reorder>(*user_memory_p, *target_memory_p);
      dev_ctx_.SetBlob(key_reorder_p, reorder_p);
      auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
      platform::RecordEvent record_reorder("int_reorder",
                                           platform::EventRole::kUniqueOp);
      reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                   {MKLDNN_ARG_TO, *target_memory_p}});
      astream.wait();
    }

    return target_memory_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireMemory(
      mkldnn::memory::desc& md,       // NOLINT
      mkldnn::memory::desc& user_md,  // NOLINT
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      const std::string& suffix,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f}, int mask = 0) {
    // create reorder primitive if the input format is not the preferred one
    auto local_key = key_ + suffix;
    auto key_reorder_p = key_ + suffix + "reorder_p";

    auto target_memory_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();

    if (target_memory_p == nullptr) {
      target_memory_p = user_memory_p;
      if (md != user_md) {
        target_memory_p = std::make_shared<mkldnn::memory>(md, engine_);
        std::shared_ptr<mkldnn::reorder::primitive_desc> reorder_pd;
        if (is_INT8) {
          mkldnn::primitive_attr
              attri;  // attribute for int8 weights and bias data reorder.
          attri.set_output_scales(mask, scale_data);

          reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
              new mkldnn::reorder::primitive_desc(*user_memory_p,
                                                  *target_memory_p, attri));
        } else {
          reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
              new mkldnn::reorder::primitive_desc(*user_memory_p,
                                                  *target_memory_p));
        }
        auto reorder_p =
            std::shared_ptr<mkldnn::reorder>(new mkldnn::reorder(*reorder_pd));
        dev_ctx_.SetBlob(key_reorder_p, reorder_p);

        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      }
      dev_ctx_.SetBlob(local_key, target_memory_p);
    } else if (!is_persistent) {
      // Make reorder if needed
      auto reorder_p = std::static_pointer_cast<mkldnn::reorder>(
          dev_ctx_.GetBlob(key_reorder_p));
      if (reorder_p != nullptr) {
        platform::RecordEvent record_reorder("int_reorder",
                                             platform::EventRole::kUniqueOp);
        reorder_p->execute(astream, {{MKLDNN_ARG_FROM, *user_memory_p},
                                     {MKLDNN_ARG_TO, *target_memory_p}});
        astream.wait();
      }
    }
    return target_memory_p;
  }

 protected:
  const MKLDNNDeviceContext& dev_ctx_;
  mkldnn::engine engine_;
  std::string key_common_;
  std::string key_;
};

template <typename T>
class BinaryMKLDNNHandler : public platform::MKLDNNHandlerT<T, dnnl::binary> {
 public:
  BinaryMKLDNNHandler(const dnnl::algorithm algo, const int axis,
                      const MKLDNNDeviceContext& dev_ctx,
                      const mkldnn::engine engine, platform::Place cpu_place,
                      const Tensor* x, const Tensor* y, Tensor* z,
                      float scale_x, float scale_y, float scale_z,
                      const std::string& uniq_name)
      : platform::MKLDNNHandlerT<T, dnnl::binary>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(x->dims()),
                                uniq_name)) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(
          x->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument("Wrong layout set for X tensor."));
      PADDLE_ENFORCE_NE(
          x->format(), MKLDNNMemoryFormat::undef,
          platform::errors::InvalidArgument("Wrong format set for X tensor."));

      PADDLE_ENFORCE_EQ(
          y->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument("Wrong layout set for Y tensor."));
      PADDLE_ENFORCE_NE(
          y->format(), MKLDNNMemoryFormat::undef,
          platform::errors::InvalidArgument("Wrong format set for Y tensor."));

      const auto src_x_tz = framework::vectorize(x->dims());
      const auto src_y_tz = framework::vectorize(y->dims());
      // if output tensor(z) is nullptr then we are computing into oneDNN
      // managed buffer
      auto rankdiff = x->dims().size() - y->dims().size();
      const auto dst_tz = (z == nullptr) ? (rankdiff > 0 ? src_x_tz : src_y_tz)
                                         : framework::vectorize(z->dims());

      auto src0_md = dnnl::memory::desc(
          src_x_tz, platform::MKLDNNGetDataType<T>(), x->format());
      auto src1_md = dnnl::memory::desc(
          src_y_tz, platform::MKLDNNGetDataType<T>(), y->format());
      if (rankdiff > 0) {  // Second input is of smaller rank than first
        std::vector<int64_t> dims1_ex(rankdiff, 1);
        dims1_ex.insert(next(dims1_ex.begin(), (axis == -1 ? rankdiff : axis)),
                        src_y_tz.begin(), src_y_tz.end());
        src1_md = src1_md.reshape(dims1_ex);
      } else if (rankdiff < 0) {  // First input is of smaller than second
        std::vector<int64_t> dims0_ex(-rankdiff, 1);
        dims0_ex.insert(next(dims0_ex.begin(), (axis == -1 ? -rankdiff : axis)),
                        src_x_tz.begin(), src_x_tz.end());
        src0_md = src0_md.reshape(dims0_ex);
      }
      const auto dst_md = memory::desc(dst_tz, platform::MKLDNNGetDataType<T>(),
                                       MKLDNNMemoryFormat::any);

      auto attributes = CreateAttributes(algo, scale_x, scale_y, scale_z);
      this->AcquireForwardPrimitiveDescriptor(attributes, algo, src0_md,
                                              src1_md, dst_md);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireSecondSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->src1_desc(), to_void_cast<T>(input_data), "@src1_mem_p");
  }

 private:
  static inline dnnl::primitive_attr CreateAttributes(dnnl::algorithm op,
                                                      float scale_x,
                                                      float scale_y,
                                                      float scale_z) {
    // Scales set in attributes for inputs contibute to the output equation
    // in the following way (assuming no broadcasting takes place):
    // output_i = scale_0 * x_i <+ or *> scale_1 * y_i;
    // Hence we have to create scales that will:
    // 1. Dequantize both values, by multiplying with (1.0 / scale_x_or_y)
    // 2. Quantize their result to output scale range, by multiplying with
    // (scale_z)
    // If we combine these two, we end up with following equation
    // output = scale_out * (1/scale_x * x <* or +> 1/scale_y * y)
    // Hence, to mimic such behaviour using provided interface,
    // For add operation the equation is equal to:
    // output = (scale_out / scale_x) * x + (scale_out / scale_y) * y
    //                <scale_0>                  <scale_1>
    // For mul operation on the other hand
    // output = (scale_out / scale_x) * x * (1.0 / scale_y) * y
    //                <scale_0>                 <scale_1>
    float scale_0 = scale_z / scale_x;
    float scale_1 =
        op == dnnl::algorithm::binary_add ? scale_z / scale_y : 1.0 / scale_y;
    dnnl::primitive_attr attributes;
    attributes.set_scales(/* input_x_id = */ DNNL_ARG_SRC_0, /* mask = */ 0,
                          {scale_0});
    attributes.set_scales(/* input_y_id = */ DNNL_ARG_SRC_1, /* mask = */ 0,
                          {scale_1});
    return attributes;
  }
};

template <typename T>
class BroadcastDataMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, dnnl::binary> {
 public:
  BroadcastDataMKLDNNHandler(const dnnl::algorithm algo,
                             const MKLDNNDeviceContext& dev_ctx,
                             const mkldnn::engine engine,
                             platform::Place cpu_place, const Tensor* out,
                             const Tensor* x, float scale_x, float scale_y,
                             const std::string& uniq_name,
                             const std::vector<int64_t>& input_dims)
      : platform::MKLDNNHandlerT<T, dnnl::binary>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(x->dims()),
                                uniq_name)) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(
          x->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument("Wrong layout set for X tensor."));
      PADDLE_ENFORCE_NE(
          x->format(), MKLDNNMemoryFormat::undef,
          platform::errors::InvalidArgument("Wrong format set for X tensor."));

      const auto src0_tz = framework::vectorize(out->dims());

      const auto src0_md = dnnl::memory::desc(
          src0_tz, platform::MKLDNNGetDataType<T>(), out->format());
      const auto src1_md = dnnl::memory::desc(
          input_dims, platform::MKLDNNGetDataType<T>(), out->format());

      dnnl::primitive_attr attributes;
      attributes.set_scales(DNNL_ARG_SRC_0, 0, {scale_x});
      attributes.set_scales(DNNL_ARG_SRC_1, 0, {scale_y});

      this->AcquireForwardPrimitiveDescriptor(attributes, algo, src0_md,
                                              src1_md, src0_md);
    }
  }

  template <typename T_out = T>
  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output) {
    T_out* ptr = output->mutable_data<T_out>(
        this->place_, this->fwd_pd_->dst_desc().get_size());
    ;
    memset(ptr, 0, this->fwd_pd_->dst_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->dst_desc(), ptr,
                                            "@dst_mem_p");
  }
};

template <typename T>
class ReductionMKLDNNHandler
    : public platform::MKLDNNHandlerT<T, dnnl::reduction> {
 public:
  ReductionMKLDNNHandler(const dnnl::algorithm algo, const float p,
                         const float eps, const MKLDNNDeviceContext& dev_ctx,
                         const mkldnn::engine engine, platform::Place cpu_place,
                         const Tensor* x, const Tensor* y,
                         const std::string& uniq_name,
                         std::vector<int64_t> y_tz)
      : platform::MKLDNNHandlerT<T, dnnl::reduction>(
            dev_ctx, engine, cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(x->dims()),
                                uniq_name,
                                (std::to_string(static_cast<int>(algo))))) {
    if (!this->isCached()) {
      PADDLE_ENFORCE_EQ(
          x->layout(), DataLayout::kMKLDNN,
          platform::errors::InvalidArgument("Wrong layout set for X tensor."));
      PADDLE_ENFORCE_NE(
          x->format(), MKLDNNMemoryFormat::undef,
          platform::errors::InvalidArgument("Wrong format set for X tensor."));

      const auto x_tz = framework::vectorize(x->dims());

      const auto x_md = dnnl::memory::desc(
          x_tz, platform::MKLDNNGetDataType<T>(), x->format());
      const auto y_md =
          memory::desc(y_tz, platform::MKLDNNGetDataType<T>(), x->format());

      this->AcquireForwardPrimitiveDescriptor(algo, x_md, y_md, p, eps);
    }
  }
};

template <typename T>
class ActivationMKLDNNHandler
    : public MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                            mkldnn::eltwise_backward> {
 public:
  ActivationMKLDNNHandler(mkldnn::algorithm algorithm,
                          const framework::ExecutionContext& ctx,
                          const MKLDNNDeviceContext& dev_ctx, Place cpu_place,
                          const framework::Tensor* in_x,
                          const std::string& unique_name, bool is_inplaced)
      : platform::MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                                 mkldnn::eltwise_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            is_inplaced ? platform::CreateKey(
                              dev_ctx, framework::vectorize(in_x->dims()), "a",
                              algorithm, unique_name)
                        : platform::CreateKey(
                              dev_ctx, framework::vectorize(in_x->dims()), "a",
                              unique_name)) {
    if (!this->isCached()) {
      float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 0;
      float beta = ctx.HasAttr("beta") ? ctx.Attr<float>("beta") : 0;
      // eltwise_linear means we are in scale op
      if (algorithm == mkldnn::algorithm::eltwise_linear) {
        bool bias_after_scale = ctx.Attr<bool>("bias_after_scale");
        auto* scale_tensor = ctx.Input<Tensor>("ScaleTensor");
        alpha = (scale_tensor == nullptr) ? ctx.Attr<float>("scale")
                                          : (float)*(scale_tensor->data<T>());
        beta = ctx.Attr<float>("bias");
        // if bias_after_scale == true
        //   out = scale*X + bias
        // else
        //   out = scale*(X + bias) = scale*X + scale*bias
        if (!bias_after_scale) beta *= alpha;
      } else {
        // paddle uses beta but mkldnn uses alpha for swish
        if (algorithm == mkldnn::algorithm::eltwise_swish) {
          std::swap(alpha, beta);
        } else if (algorithm == dnnl::algorithm::eltwise_bounded_relu) {
          alpha = ctx.Attr<float>("threshold");
        }
      }

      PADDLE_ENFORCE(in_x->dims().size() >= 1 || in_x->dims().size() <= 6,
                     platform::errors::Unimplemented(
                         "Input dimension size can be 1, 2, 3, 4, "
                         "5, or 6, but now the dimension size is",
                         in_x->dims().size()));

      auto src_tz = framework::vectorize<int64_t>(in_x->dims());
      auto src_fmt =
          src_tz.size() == 2 ? MKLDNNMemoryFormat::nc : in_x->format();
      auto md = mkldnn::memory::desc(src_tz, platform::MKLDNNGetDataType<T>(),
                                     src_fmt);

      this->AcquireForwardPrimitiveDescriptor(
          mkldnn::prop_kind::forward_training, algorithm, md, alpha, beta);
    }
  }

  ActivationMKLDNNHandler(mkldnn::algorithm algorithm,
                          const framework::ExecutionContext& ctx,
                          const MKLDNNDeviceContext& dev_ctx, Place cpu_place,
                          const framework::Tensor* in_x, const Tensor* out_grad,
                          const std::string& unique_name)
      : platform::MKLDNNHandlerT<T, mkldnn::eltwise_forward,
                                 mkldnn::eltwise_backward>(
            dev_ctx, dev_ctx.GetEngine(), cpu_place,
            platform::CreateKey(dev_ctx, framework::vectorize(in_x->dims()),
                                "a", unique_name)) {
    if (!this->isBwdCached()) {
      float alpha = ctx.HasAttr("alpha") ? ctx.Attr<float>("alpha") : 0;
      float beta = ctx.HasAttr("beta") ? ctx.Attr<float>("beta") : 0;

      // paddle uses beta but mkldnn uses alpha for swish
      if (algorithm == mkldnn::algorithm::eltwise_swish) {
        std::swap(alpha, beta);
      } else if (algorithm == dnnl::algorithm::eltwise_bounded_relu) {
        alpha = ctx.Attr<float>("threshold");
      }

      auto diff_dst_tz = framework::vectorize<int64_t>(out_grad->dims());

      auto src_fmt =
          diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : in_x->format();
      auto diff_fmt =
          diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : out_grad->format();

      auto dims = framework::vectorize(in_x->dims());
      auto diff_dst_md = platform::MKLDNNMemDesc(
          dims, platform::MKLDNNGetDataType<T>(), diff_fmt);
      auto src_md = platform::MKLDNNMemDesc(
          dims, platform::MKLDNNGetDataType<T>(), src_fmt);

      this->AcquireForwardPrimitiveDescriptor(
          mkldnn::prop_kind::forward_training, algorithm, src_md, alpha, beta);
      this->AcquireBackwardPrimitiveDescriptor(algorithm, diff_dst_md, src_md,
                                               alpha, beta);
    }
  }

  std::shared_ptr<mkldnn::memory> AcquireBackwardSrcMemory(
      const framework::Tensor* input) {
    const T* input_data = input->data<T>();
    return this->AcquireMemoryFromPrimitive(this->bwd_pd_->src_desc(),
                                            to_void_cast<T>(input_data),
                                            "@bwd-src_mem_p");
  }
};

template <typename T>
class TransposeMKLDNNHandler : public MKLDNNHandler {
 public:
  TransposeMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                         std::vector<int>& axis,      // NOLINT
                         const platform::MKLDNNDeviceContext& dev_ctx,
                         mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        axis_(axis),
        logical_axis_(dims.size(), 0) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const MKLDNNMemoryFormat& fmt, void* ptr) {
    auto local_key = key_ + "@user_src_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      // Make memory descriptor using input format, unless it
      // cannot be trusted (nchw) then make up memory fmt manually
      for (size_t i = 0; i < logical_axis_.size(); ++i) {
        logical_axis_[i] = i;
      }

      auto src_md = fmt != MKLDNNMemoryFormat::nchw
                        ? platform::MKLDNNMemDesc(
                              dims_, platform::MKLDNNGetDataType<T>(), fmt)
                        : Axis2MemoryDesc(dims_, logical_axis_);
      mem_p = std::make_shared<mkldnn::memory>(src_md, engine_, ptr);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      mem_p->set_data_handle(ptr);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(framework::Tensor* output,
                                                   platform::Place place) {
    auto local_key = key_ + "@user_dst_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto dst_md = Axis2MemoryDesc(dims_, axis_);

      auto dst_data = output->mutable_data<T>(place, dst_md.get_size());

      mem_p = std::make_shared<mkldnn::memory>(dst_md, engine_, dst_data);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      auto dst_data = output->mutable_data<T>(place);
      mem_p->set_data_handle(dst_data);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::reorder> AcquireTranspose(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    auto prim_key = key_ + "@transpose_p";
    auto transpose_p =
        std::static_pointer_cast<mkldnn::reorder>(dev_ctx_.GetBlob(prim_key));
    if (transpose_p == nullptr) {
      transpose_p =
          std::make_shared<mkldnn::reorder>(*(src_memory_p), *(dst_memory_p));
      dev_ctx_.SetBlob(prim_key, transpose_p);
    }
    return transpose_p;
  }

 protected:
  mkldnn::memory::desc Axis2MemoryDesc(std::vector<int64_t>& nchw_tz,  // NOLINT
                                       std::vector<int>& axis          // NOLINT
                                       ) {
    size_t ndims = axis.size();

    std::vector<int64_t> strides(ndims);
    unsigned int total_stride = 1;
    for (int i = ndims - 1; i >= 0; --i) {
      strides[axis[i]] = total_stride;
      total_stride *= nchw_tz[axis[i]];
    }
    mkldnn::memory::desc mem_d(nchw_tz, platform::MKLDNNGetDataType<T>(),
                               strides);

    return mem_d;
  }

 private:
  std::vector<int64_t> dims_;
  std::vector<int> axis_;
  std::vector<int> logical_axis_;
};

class ReorderMKLDNNHandler : public MKLDNNHandler {
 public:
  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       mkldnn::memory::data_type dtype,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype),
        dtype_(dtype),
        dtype_dst_(dtype) {}

  ReorderMKLDNNHandler(std::vector<int64_t>& dims,  // NOLINT
                       framework::proto::VarType::Type vtype,
                       mkldnn::memory::data_type dtype,
                       framework::proto::VarType::Type vtype_dst,
                       mkldnn::memory::data_type dtype_dst,
                       const platform::MKLDNNDeviceContext& dev_ctx,
                       mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        dims_(dims),
        vtype_(vtype),
        vtype_dst_(vtype_dst),
        dtype_(dtype),
        dtype_dst_(dtype_dst) {}

  std::shared_ptr<mkldnn::memory> AcquireSrcMemory(
      const MKLDNNMemoryFormat& fmt, void* ptr) {
    return this->AcquireMemory(dims_, dtype_, fmt, ptr, "@user_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcSubmemory(
      const std::vector<int64_t>& dims, const std::vector<int64_t>& offset,
      const std::shared_ptr<mkldnn::memory>& mem_p, int submemory_number) {
    std::string local_key = key_;
    local_key.append("@submem")
        .append(std::to_string(submemory_number))
        .append("_p");

    auto sub_mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (sub_mem_p == nullptr) {
      auto sub_md = mem_p->get_desc().submemory_desc(dims, {offset});
      sub_mem_p = std::make_shared<mkldnn::memory>(sub_md, engine_,
                                                   mem_p->get_data_handle());
      dev_ctx_.SetBlob(local_key, sub_mem_p);
    } else {
      sub_mem_p->set_data_handle(mem_p->get_data_handle());
    }
    return sub_mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      framework::Tensor* output, const MKLDNNMemoryFormat& fmt,
      platform::Place place) {
    auto local_key = key_ + "@user_dst_mem_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto dst_md = platform::MKLDNNMemDesc(dims_, dtype_dst_, fmt);
      auto dst_data =
          output->mutable_data(place, vtype_dst_, dst_md.get_size());

      mem_p = std::make_shared<mkldnn::memory>(dst_md, engine_, dst_data);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      // Even if memory object exists , we may be using it for diffrent tensor
      auto dst_data =
          output->mutable_data(place, vtype_dst_, mem_p->get_desc().get_size());
      mem_p->set_data_handle(dst_data);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemory(
      framework::Tensor* output, const std::vector<int64_t>& dims,
      const int memory_number, const MKLDNNMemoryFormat& fmt,
      platform::Place place) {
    auto local_key =
        key_ + "@user_dst_mem" + std::to_string(memory_number) + "_p";
    auto mem_p =
        std::static_pointer_cast<mkldnn::memory>(dev_ctx_.GetBlob(local_key));
    if (mem_p == nullptr) {
      auto dst_md = platform::MKLDNNMemDesc(dims, dtype_dst_, fmt);
      auto dst_data =
          output->mutable_data(place, vtype_dst_, dst_md.get_size());

      mem_p = std::make_shared<mkldnn::memory>(dst_md, engine_, dst_data);
      dev_ctx_.SetBlob(local_key, mem_p);
    } else {
      // Even if memory object exists , we may be using it for diffrent tensor
      auto dst_data =
          output->mutable_data(place, vtype_dst_, mem_p->get_desc().get_size());
      mem_p->set_data_handle(dst_data);
    }
    return mem_p;
  }

  std::shared_ptr<mkldnn::reorder> AcquireReorder(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p, int reorder_number) {
    auto prim_key = key_ + "@reorder" + std::to_string(reorder_number) + "_p";
    auto reorder_p =
        std::static_pointer_cast<mkldnn::reorder>(dev_ctx_.GetBlob(prim_key));
    if (reorder_p == nullptr) {
      reorder_p =
          std::make_shared<mkldnn::reorder>(*(src_memory_p), *(dst_memory_p));
      dev_ctx_.SetBlob(prim_key, reorder_p);
    }
    return reorder_p;
  }

  std::shared_ptr<mkldnn::reorder> AcquireReorder(
      std::shared_ptr<mkldnn::memory> dst_memory_p,
      std::shared_ptr<mkldnn::memory> src_memory_p) {
    auto prim_key = key_ + "@reorder_p";
    auto reorder_p =
        std::static_pointer_cast<mkldnn::reorder>(dev_ctx_.GetBlob(prim_key));
    if (reorder_p == nullptr) {
      reorder_p =
          std::make_shared<mkldnn::reorder>(*(src_memory_p), *(dst_memory_p));
      dev_ctx_.SetBlob(prim_key, reorder_p);
    }
    return reorder_p;
  }

 private:
  std::vector<int64_t> dims_;
  framework::proto::VarType::Type vtype_, vtype_dst_;
  mkldnn::memory::data_type dtype_, dtype_dst_;
};

template <typename T>
struct convolutional_algorithm;

template <>
struct convolutional_algorithm<mkldnn::convolution_forward> {
  static constexpr mkldnn::algorithm T = mkldnn::algorithm::convolution_direct;
};

template <>
struct convolutional_algorithm<mkldnn::deconvolution_forward> {
  static constexpr mkldnn::algorithm T =
      mkldnn::algorithm::deconvolution_direct;
};

template <class forward_t, class backward_data_t, class backward_weights_t>
class ConvMKLDNNTemplateHandler : public MKLDNNHandler {
 public:
  ConvMKLDNNTemplateHandler(const platform::MKLDNNDeviceContext& dev_ctx,
                            mkldnn::engine engine, const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {}

  // TODO(jczaja): remove after conv int8 is adapted
  ConvMKLDNNTemplateHandler(
      std::shared_ptr<typename forward_t::primitive_desc> conv_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key) {
    conv_pd_ = conv_pd;
  }

  ConvMKLDNNTemplateHandler(
      std::shared_ptr<typename forward_t::primitive_desc> conv_pd,
      std::shared_ptr<typename backward_data_t::primitive_desc>
          conv_bwd_data_pd,
      std::shared_ptr<typename backward_weights_t::primitive_desc>
          conv_bwd_weights_pd,
      const platform::MKLDNNDeviceContext& dev_ctx, mkldnn::engine engine,
      const std::string& base_key)
      : platform::MKLDNNHandler(dev_ctx, engine, base_key),
        conv_pd_(conv_pd),
        conv_bwd_weights_pd_(conv_bwd_weights_pd),
        conv_bwd_data_pd_(conv_bwd_data_pd) {
    // If we are in Grad operatgor then update a key with BWD suffix to
    // distinguish from FWD memory primitives
    key_ += "-BWD";
  }

  size_t GetDstMemorySize() const { return conv_pd_->dst_desc().get_size(); }

  MKLDNNMemoryFormat GetDstFormat() const {
    return paddle::platform::GetMKLDNNFormat(conv_pd_->dst_desc());
  }

  size_t GetDiffWeightsMemorySize() const {
    return conv_bwd_weights_pd_->diff_weights_desc().get_size();
  }

  size_t GetDiffSourceMemorySize() const {
    return conv_bwd_data_pd_->diff_src_desc().get_size();
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_bwd_weights_pd_->src_desc();
    auto user_pd = user_memory_p->get_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p,
                               "@weights-src_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromWeightsPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_weights_pd_->diff_dst_desc();
    auto user_pd = user_memory_p->get_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@weights-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemoryFromWeightsPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_weights_pd_->diff_weights_desc(), ptr, "@diff_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffWeightsMemoryFromWeightsPrimitive(
      void) {
    return this->AcquireMemoryFromPrimitive(
        conv_bwd_weights_pd_->diff_weights_desc(), "@diff_weights_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffDstMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto diff_dst_pd = conv_bwd_data_pd_->diff_dst_desc();
    auto user_pd = user_memory_p->get_desc();
    return this->AcquireMemory(diff_dst_pd, user_pd, user_memory_p,
                               "@data-diff_dst_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromDataPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto weights_pd = conv_bwd_data_pd_->weights_desc();
    auto user_pd = user_weights_memory_p->get_desc();
    return this->AcquireMemory(weights_pd, user_pd, user_weights_memory_p,
                               "@data-weights_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireResidualDataMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_residual_data_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromResidualDataMemory(
      const std::shared_ptr<mkldnn::memory>& user_residual_memory_p,
      void* dst_ptr,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    return this->AcquireMemory(user_residual_memory_p,
                               this->AcquireDstMemoryFromPrimitive(dst_ptr),
                               "@residual_data_mem_p", pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireDiffSrcMemoryFromDataPrimitive(
      void* ptr) {
    return this->AcquireMemoryFromPrimitive(conv_bwd_data_pd_->diff_src_desc(),
                                            ptr, "@diff_src_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireDstMemoryFromPrimitive(void* ptr) {
    return this->AcquireMemoryFromPrimitive(conv_pd_->dst_desc(), ptr,
                                            "@dst_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireSrcMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_memory_p,
      std::vector<mkldnn::primitive>& pipeline) {  // NOLINT
    auto src_pd = conv_pd_->src_desc();
    auto user_pd = user_memory_p->get_desc();
    return this->AcquireMemory(src_pd, user_pd, user_memory_p, "@src_mem_p",
                               pipeline);
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemory(
      const mkldnn::memory::desc& md, void* ptr,
      user_function custom_func = {}) {
    return this->AcquireMemory(md, ptr, "@user_weights_mem_p", custom_func);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemory(
      const mkldnn::memory::desc& md, void* ptr) {
    return this->AcquireMemory(md, ptr, "@user_bias_mem_p");
  }

  std::shared_ptr<mkldnn::memory> AcquireWeightsMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_weights_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f}, int mask = 0) {
    auto user_weights_pd = user_weights_memory_p->get_desc();
    auto weights_pd = conv_pd_->weights_desc();
    return this->AcquireMemory(
        weights_pd, user_weights_pd, user_weights_memory_p, "@weights_mem_p",
        pipeline, is_persistent, is_INT8, scale_data, mask);
  }

  std::shared_ptr<mkldnn::memory> AcquireBiasMemoryFromPrimitive(
      const std::shared_ptr<mkldnn::memory> user_bias_memory_p,
      std::vector<mkldnn::primitive>& pipeline,  // NOLINT
      bool is_persistent = false, bool is_INT8 = false,
      std::vector<float> scale_data = {1.0f},
      int mask = 0) {  // NOLINT
    auto user_bias_pd = user_bias_memory_p->get_desc();
    auto bias_pd = conv_pd_->bias_desc();
    return this->AcquireMemory(bias_pd, user_bias_pd, user_bias_memory_p,
                               "@bias_mem_p", pipeline, is_persistent, is_INT8,
                               scale_data, mask);
  }

  mkldnn::primitive_attr CreatePostOps(
      std::string fuse_activation, float fuse_alpha, float fuse_beta,
      bool fuse_residual_conn, const std::vector<float> output_shift_scale = {},
      float sum_scale = 1.0f) const {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;
    if (output_shift_scale.size() > 0) {
      int mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
      conv_attr.set_output_scales(mask, output_shift_scale);
    }
    // Fusion with Elementwise layer relies on adding a sum post-operation with
    // the scale parameter. It is assumed that when fuse_residual_connection is
    // true, the output tensor contains the data coming from residual
    // connection. The result of this post_op is:
    // Output = scale * Output + Conv_Out.
    if (fuse_residual_conn) {
      post_operations.append_sum(sum_scale);
    }
    // Fusion with ReLU layer is executed through the PostOps feature. Create a
    // PostOps object and configure it to execute an eltwise relu operation.
    if (fuse_activation == "relu" || fuse_activation == "leaky_relu") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "relu6") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale,
                                     mkldnn::algorithm::eltwise_bounded_relu,
                                     fuse_alpha, fuse_beta);
    } else if (fuse_activation == "swish") {
      constexpr float scale = 1.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_swish,
                                     fuse_alpha, fuse_beta);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

  std::shared_ptr<typename forward_t::primitive_desc>
  AcquireConvolutionPrimitiveDescriptor(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& weights,
      boost::optional<const mkldnn::memory::desc&> bias,
      const mkldnn::memory::desc& dst, const std::vector<int64_t>& strides,
      const std::vector<int64_t>& dilations,
      const std::vector<int64_t>& paddings, const mkldnn::engine& engine,
      const std::string& fuse_activation, float fuse_alpha, float fuse_beta,
      const bool fuse_residual_conn, mkldnn::prop_kind fwd_prop_kind,
      const std::vector<float> output_shift_scale = {},
      const float sum_scale = 1.0f) {
    // Conv PD has to be passed to Grad op that
    // may be exxecuted by diffrent thread, hence
    // for that one we use key that does not contain TID
    const std::string key_conv_pd = key_common_ + "@conv_pd";

    conv_pd_ = std::static_pointer_cast<typename forward_t::primitive_desc>(
        dev_ctx_.GetBlob(key_conv_pd));

    if (conv_pd_ == nullptr) {
      static std::mutex acquire_barrier;
      std::lock_guard<std::mutex> block_threads_until_finish_this_job(
          acquire_barrier);

      conv_pd_ = std::static_pointer_cast<typename forward_t::primitive_desc>(
          dev_ctx_.GetBlob(key_conv_pd));
      if (conv_pd_ == nullptr) {
        mkldnn::memory::dims stride_dims = strides;
        mkldnn::memory::dims dilations_dims = dilations;
        auto mkldnn_paddings = ToMkldnnPadding(paddings);

        auto conv_desc =
            bias ? typename forward_t::desc(
                       fwd_prop_kind, convolutional_algorithm<forward_t>::T,
                       src, weights, *bias, dst, stride_dims, dilations_dims,
                       mkldnn_paddings[0], mkldnn_paddings[1])
                 : typename forward_t::desc(
                       fwd_prop_kind, convolutional_algorithm<forward_t>::T,
                       src, weights, dst, stride_dims, dilations_dims,
                       mkldnn_paddings[0], mkldnn_paddings[1]);

        mkldnn::primitive_attr conv_attr =
            CreatePostOps(fuse_activation, fuse_alpha, fuse_beta,
                          fuse_residual_conn, output_shift_scale, sum_scale);

        conv_pd_.reset(new typename forward_t::primitive_desc(
            conv_desc, conv_attr, engine));
        // Save conv_pd/src_memory/weights_memory for backward pass
        dev_ctx_.SetBlob(key_conv_pd, conv_pd_);
      }
    }

    return conv_pd_;
  }

  std::shared_ptr<forward_t> AcquireConvolution() {
    auto prim_key = key_ + "@conv_p";
    auto conv_p =
        std::static_pointer_cast<forward_t>(dev_ctx_.GetBlob(prim_key));
    if (conv_p == nullptr) {
      conv_p = std::make_shared<forward_t>(*conv_pd_);

      dev_ctx_.SetBlob(prim_key, conv_p);
    }
    return conv_p;
  }

  std::shared_ptr<backward_weights_t> AcquireConvolutionBackwardWeights() {
    auto prim_key = key_ + "@conv_bwd_weights_p";
    auto conv_bwd_weights_p = std::static_pointer_cast<backward_weights_t>(
        dev_ctx_.GetBlob(prim_key));
    if (conv_bwd_weights_p == nullptr) {
      // create backward conv primitive for weights
      conv_bwd_weights_p =
          std::make_shared<backward_weights_t>(*conv_bwd_weights_pd_);
      dev_ctx_.SetBlob(prim_key, conv_bwd_weights_p);
    }
    return conv_bwd_weights_p;
  }

  std::shared_ptr<backward_data_t> AcquireConvolutionBackwardData() {
    auto prim_key = key_ + "@conv_bwd_data_p";
    auto conv_bwd_data_p =
        std::static_pointer_cast<backward_data_t>(dev_ctx_.GetBlob(prim_key));
    if (conv_bwd_data_p == nullptr) {
      conv_bwd_data_p = std::make_shared<backward_data_t>(*conv_bwd_data_pd_);
      dev_ctx_.SetBlob(prim_key, conv_bwd_data_p);
    }
    return conv_bwd_data_p;
  }

 private:
  std::shared_ptr<typename forward_t::primitive_desc> conv_pd_;
  std::shared_ptr<typename backward_weights_t::primitive_desc>
      conv_bwd_weights_pd_;
  std::shared_ptr<typename backward_data_t::primitive_desc> conv_bwd_data_pd_;
};

using ConvMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::convolution_forward,
                              mkldnn::convolution_backward_data,
                              mkldnn::convolution_backward_weights>;

using ConvTransposeMKLDNNHandler =
    ConvMKLDNNTemplateHandler<mkldnn::deconvolution_forward,
                              mkldnn::deconvolution_backward_data,
                              mkldnn::deconvolution_backward_weights>;

template <typename T>
static std::shared_ptr<mkldnn::memory> SetDstMemory(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    const std::shared_ptr<ConvMKLDNNHandler>& handler) {
  T* output_data =
      output->mutable_data<T>(ctx.GetPlace(), handler->GetDstMemorySize());
  std::shared_ptr<mkldnn::memory> dst_memory_p =
      handler->AcquireDstMemoryFromPrimitive(to_void_cast<T>(output_data));
  return dst_memory_p;
}

template <typename T>
static std::shared_ptr<mkldnn::memory> SetDstMemory(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    const framework::Tensor* residual_param,
    const mkldnn::memory::desc& user_residual_md,
    const std::shared_ptr<ConvMKLDNNHandler>& handler,
    std::vector<mkldnn::primitive>* pipeline) {
  const T* residual_param_data = residual_param->data<T>();
  PADDLE_ENFORCE_NOT_NULL(
      residual_param_data,
      platform::errors::PreconditionNotMet("Residual parameter is required for "
                                           "the DNNL conv+elementwise_add "
                                           "fusion, but now it is missing."));
  std::shared_ptr<mkldnn::memory> user_residual_memory_p =
      handler->AcquireResidualDataMemory(user_residual_md,
                                         to_void_cast<T>(residual_param_data));
  T* output_data = output->mutable_data<T>(ctx.GetPlace());
  std::shared_ptr<mkldnn::memory> dst_memory_p =
      handler->AcquireDstMemoryFromResidualDataMemory(
          user_residual_memory_p, to_void_cast<T>(output_data), *pipeline);
  return dst_memory_p;
}

template <typename T>
static void SetDstMemoryHandler(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    const std::shared_ptr<ConvMKLDNNHandler>& handler,
    std::shared_ptr<mkldnn::memory> dst_memory_p) {
  T* output_data =
      output->mutable_data<T>(ctx.GetPlace(), handler->GetDstMemorySize());
  dst_memory_p->set_data_handle(to_void_cast<T>(output_data));
}

template <typename T>
static void SetDstMemoryQuantized(
    const framework::ExecutionContext& ctx, framework::Tensor* output,
    std::vector<int64_t> dst_tz, const mkldnn::engine& engine,
    std::shared_ptr<mkldnn::memory::desc>& dst_md,  // NOLINT
    std::shared_ptr<mkldnn::memory>& dst_memory,    // NOLINT
    MKLDNNMemoryFormat output_format) {
  T* output_data = output->mutable_data<T>(ctx.GetPlace());
  const size_t dst_dims = dst_tz.size();
  MKLDNNMemoryFormat dst_fmt;
  PADDLE_ENFORCE_LE(dst_dims, 5, platform::errors::InvalidArgument(
                                     "Dst memory for quantization can not have "
                                     "dims > 5. But received dst_dims is %d.",
                                     dst_dims));
  dst_fmt = platform::MKLDNNFormatForSize(dst_dims, output_format);

  auto tmp_dst_md = platform::MKLDNNMemDesc(
      {dst_tz}, paddle::framework::ToMKLDNNDataType(
                    framework::DataTypeTrait<T>::DataType()),
      dst_fmt);
  dst_md.reset(new mkldnn::memory::desc(tmp_dst_md));
  dst_memory.reset(
      new mkldnn::memory(*dst_md, engine, to_void_cast<T>(output_data)));
}
}  // namespace platform
}  // namespace paddle
