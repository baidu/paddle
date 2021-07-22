//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if defined(PADDLE_WITH_XPU_BKCL)
#include "paddle/fluid/imperative/bkcl_context.h"

#include <string>
#include <utility>
#include <vector>
#include <sstream>

#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/platform/bkcl_helper.h"
#include "paddle/fluid/platform/collective_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/gen_comm_id_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/string/split.h"
#include "paddle/fluid/string/string_helper.h"

namespace paddle {
namespace imperative {

static std::string array_to_string(float* data, size_t len) {
    std::stringstream ss;
    ss << "[" << len << "]";
    for (size_t j = 0; j < len; ++j) {
          if (j % 10 == 0) {
              ss << "\n " << data[j];
          } else {
              ss << " " << data[j];
          }
    }
    return ss.str();
}

static std::string tensor_to_string(framework::Tensor* t) {
    std::stringstream ss;
    ss << "hello, tensor_to_string: numel is: [" << t->numel() << "]";
    auto& place = t->place();
    void* data = t->mutable_data(place);
    if (platform::is_xpu_place(place)) {
        ss << "[xpu]";
        // copy
        float* temp = new float[t->numel()];
        memory::Copy(platform::CPUPlace(), reinterpret_cast<void*>(temp),
                BOOST_GET_CONST(platform::XPUPlace, place),
                reinterpret_cast<void*>(data),
                t->numel() * sizeof(float));
        ss << array_to_string(temp, t->numel());
        delete[] temp;
    } else if (platform::is_cpu_place(place)) {
        ss << "[cpu]";
        ss << array_to_string(reinterpret_cast<float*>(data), t->numel());
        // no copy
    } else {
        ss << "[???]";
    }
    return ss.str();
}

static void AllReduce(const framework::Tensor &src, framework::Tensor *dst,
                      const XPUStream stream, const platform::BKCLComm *comm, int xpu_id) {
  const auto &place = src.place();
  PADDLE_ENFORCE_EQ(
      platform::is_xpu_place(place), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));

  const void *src_ptr = src.data<void>();
  dst->Resize(src.dims());
  auto *dst_ptr = dst->mutable_data(src.place(), src.type());
  auto bkcl_dtype = platform::ToBKCLDataType(src.type());

  PADDLE_ENFORCE_EQ(bkcl_all_reduce(comm->comm(), src_ptr, dst_ptr, src.numel(),
                                    bkcl_dtype, BKCL_ADD, stream),
                    BKCL_SUCCESS, platform::errors::PreconditionNotMet(
                                      "BKCL all reduce failed"));
  // houjue debug, print dest values
  static std::ofstream stored_buffer;
  std::stringstream ss;
  ss << xpu_id;
  std::string file_name = "./log/all_reduce_" + ss.str() + ".log";
  stored_buffer.open(file_name, std::ios::app);
  stored_buffer << tensor_to_string(dst);
  stored_buffer.close();
}
/*
Baidu Kunlun Communication Library(BKCL) is designed for multi Baidu Kunlun
cards communication
as NVIDIA Collective Communications Library(NCCL) in multi Nvidia GPU cards.
Please refer to bkcl.h in xpu.tar.gz linked in cmake/external/xpu.cmake.
*/
void BKCLParallelContext::BcastBKCLId(
    std::vector<BKCLUniqueId> &bkcl_ids,  // NOLINT
    int root) {
  if (strategy_.local_rank_ == root) {
    std::vector<std::string> other_trainers;
    for (auto &ep : strategy_.trainer_endpoints_) {
      if (ep != strategy_.current_endpoint_) {
        other_trainers.push_back(ep);
      }
    }
    platform::SendBroadCastCommID(other_trainers, &bkcl_ids);
  } else {
    platform::RecvBroadCastCommID(strategy_.current_endpoint_, &bkcl_ids);
  }
}

void BKCLParallelContext::Init() {
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(strategy_.nrings_);

  if (strategy_.local_rank_ == 0) {
    // generate the unique bkclid on the root worker
    for (size_t i = 0; i < bkcl_ids.size(); ++i) {
      auto ret = bkcl_get_unique_id(&bkcl_ids[i]);
      PADDLE_ENFORCE_EQ(BKCL_SUCCESS, ret,
                        platform::errors::PreconditionNotMet(
                            "BKCL get unique id failed [%d]", ret));
    }
  }
  BcastBKCLId(bkcl_ids, 0);

  int xpu_id = BOOST_GET_CONST(platform::XPUPlace, place_).device;
  for (int ring_id = 0; ring_id < strategy_.nrings_; ring_id++) {
    VLOG(0) << "init BKCL context nranks: " << strategy_.nranks_
            << " local rank: " << strategy_.local_rank_ << " xpu id: " << xpu_id
            << " ring id: " << ring_id;
    // it will assign bkcl_comm in XPUDeviceContext within ring_id
    platform::BKCLCommContext::Instance().CreateBKCLComm(
        &bkcl_ids[ring_id], strategy_.nranks_, strategy_.local_rank_, xpu_id,
        ring_id);
  }
}

void BKCLParallelContext::InitWithRingID(int ring_id) {
  std::vector<BKCLUniqueId> bkcl_ids;
  bkcl_ids.resize(1);

  if (strategy_.local_rank_ == 0) {
    // generate the unique bkclid on the root worker
    auto ret = bkcl_get_unique_id(&bkcl_ids[0]);
    PADDLE_ENFORCE_EQ(BKCL_SUCCESS, ret,
                      platform::errors::PreconditionNotMet(
                          "BKCL get unique id failed [%d]", ret));
  }
  BcastBKCLId(bkcl_ids, 0);

  int xpu_id = BOOST_GET_CONST(platform::XPUPlace, place_).device;
  VLOG(0) << "init BKCL context nranks: " << strategy_.nranks_
          << " local rank: " << strategy_.local_rank_ << " xpu id: " << xpu_id
          << " ring id: " << ring_id;
  // it will assign bkcl_comm in XPUDeviceContext within ring_id
  platform::BKCLCommContext::Instance().CreateBKCLComm(
      &bkcl_ids[0], strategy_.nranks_, strategy_.local_rank_, xpu_id, ring_id);
}

void BKCLParallelContext::AllReduceByStream(const framework::Variable &src,
                                            framework::Variable *dst,
                                            int ring_id, bool use_calc_stream) {
  PADDLE_ENFORCE_EQ(
      platform::is_xpu_place(place_), true,
      platform::errors::Unimplemented(
          "Dynamic graph mode does not support multi-CPU training yet."));
  auto place = place_;

  auto *dev_ctx = static_cast<platform::XPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place));
  platform::BKCLComm *comm =
      platform::BKCLCommContext::Instance().Get(ring_id, place);
  XPUStream stream =
      use_calc_stream ? dev_ctx->x_context()->xpu_stream : comm->stream();

  if (src.IsType<framework::LoDTensor>()) {
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    int xpu_id = BOOST_GET_CONST(platform::XPUPlace, place_).device;
    AllReduce(src.Get<framework::LoDTensor>(),
              dst->GetMutable<framework::LoDTensor>(), stream, comm, xpu_id);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "XPU unsupported variable type %s for imperative allreduce, only "
        "LoDTensor are supported.",
        platform::demangle(framework::ToTypeName(src.Type()))));
  }
}

paddle::platform::DeviceContext *BKCLParallelContext::GetDeviceContext(
    int ring_id) {
  return static_cast<platform::DeviceContext *>(
      platform::BKCLCommContext::Instance()
          .Get(ring_id, place_)
          ->dev_context());
}

void BKCLParallelContext::WaitCompute(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0,
                    platform::errors::OutOfRange(
                        "Ring id expected >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(
      ring_id, strategy_.nrings_,
      platform::errors::OutOfRange("Ring id expected < nrings,"
                                   "but got ring id = %d, nrings = %d",
                                   ring_id, strategy_.nrings_));
  auto compute_dev_ctx = static_cast<platform::XPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

void BKCLParallelContext::WaitComm(int ring_id) {
  PADDLE_ENFORCE_GE(ring_id, 0,
                    platform::errors::OutOfRange(
                        "Ring id expected >= 0, but got %d", ring_id));
  PADDLE_ENFORCE_LT(
      ring_id, strategy_.nrings_,
      platform::errors::OutOfRange("Ring id expected < nrings,"
                                   "but got ring id = %d, nrings = %d",
                                   ring_id, strategy_.nrings_));
  auto comm_dev_ctx =
      platform::BKCLCommContext::Instance().Get(ring_id, place_)->dev_context();
  comm_dev_ctx->Wait();
}

void BKCLParallelContext::SynchronizeCompute() {
  auto compute_dev_ctx = static_cast<platform::XPUDeviceContext *>(
      platform::DeviceContextPool::Instance().Get(place_));
  compute_dev_ctx->Wait();
}

}  //  namespace imperative
}  //  namespace paddle
#endif
