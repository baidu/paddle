//   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/collective_helper.h"
#include <utility>

#include "paddle/fluid/platform/cuda_resource_pool.h"

namespace paddle {
namespace platform {
#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
class NCCLCommImpl : public NCCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override {
    return BOOST_GET_CONST(CUDAPlace, dev_ctx_->GetPlace()).device;
  }

  void set_comm(ncclComm_t comm) { comm_ = comm; }
  ncclComm_t comm() const override { return comm_; }

  gpuStream_t stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<CUDADeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  CUDADeviceContext* dev_context() const override { return dev_ctx_.get(); }

  gpuEvent_t compute_event() const override { return compute_event_.get(); }

  gpuEvent_t comm_event() const override { return comm_event_.get(); }

  void set_compute_event(
      std::shared_ptr<platform::CudaEventObject>&& compute_event) {
    compute_event_ = std::move(compute_event);
  }

  void set_comm_event(std::shared_ptr<platform::CudaEventObject>&& comm_event) {
    comm_event_ = std::move(comm_event);
  }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  ncclComm_t comm_;
  std::unique_ptr<CUDADeviceContext> dev_ctx_;

  // used for comm wait compute, compute_stream-->event-->comm_stream
  std::shared_ptr<platform::CudaEventObject> compute_event_;

  // used for compute wait comm, comm_stream-->event-->compute_stream
  std::shared_ptr<platform::CudaEventObject> comm_event_;
};

NCCLComm* NCCLCommContext::CreateNCCLComm(ncclUniqueId* nccl_id, int nranks,
                                          int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(nccl_id,
                          platform::errors::InvalidArgument(
                              "The nccl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks, 1,
      platform::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank, nranks,
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank, nranks));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  ncclComm_t comm = nullptr;
  SetDeviceId(dev_id);
  PADDLE_ENFORCE_CUDA_SUCCESS(
      platform::dynload::ncclCommInitRank(&comm, nranks, *nccl_id, rank));

  auto* comm_wrapper = AssignNCCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "nccl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });

  return comm_wrapper;
}

void NCCLCommContext::CreateAllNCCLComms(const std::vector<int>& dev_ids,
                                         int ring_id) {
  PADDLE_ENFORCE_GT(
      dev_ids.size(), 0,
      platform::errors::InvalidArgument("Expected the size of dev_ids > 0. But "
                                        "received the size of dev_ids is %d.",
                                        dev_ids.size()));

  const int kDevices = dev_ids.size();
  ncclComm_t comms[kDevices];
  PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::ncclCommInitAll(
      comms, dev_ids.size(), dev_ids.data()));

  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id), 0,
                    platform::errors::InvalidArgument(
                        "Expected comm_map_.count(ring_id) = 0. But received "
                        "comm_map_.count(ring_id) is %d.",
                        comm_map_.count(ring_id)));
  for (size_t i = 0; i < dev_ids.size(); ++i) {
    AssignNCCLComm(comms[i], dev_ids.size(), i, dev_ids[i], ring_id);
    VLOG(1) << "nccl communicator of rank " << i << " in ring " << ring_id
            << " has been created on device " << dev_ids[i];
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

void NCCLCommContext::CreateNCCLCommMultiTrainer(
    const std::vector<int>& dev_ids, ncclUniqueId* nccl_id, int ntrainers,
    int train_id, int ring_id) {
  PADDLE_ENFORCE_GT(dev_ids.size(), 0);
  const int kDevices = dev_ids.size();
  VLOG(0) << "Begin CreateNCCLCommMultiTrainer. device number: " << kDevices
          << ", ntrainers: " << ntrainers << ", train_id: " << train_id
          << ", rind_id: " << ring_id;
  ncclComm_t comms[kDevices];
  {
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupStart());
    for (int i = 0; i < kDevices; i++) {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaSetDevice(i));
      platform::dynload::ncclCommInitRank(comms + i, kDevices * ntrainers,
                                          *nccl_id, train_id * kDevices + i);
      VLOG(0) << "ncclCommInitRankkkkk82: " << i;
    }
    PADDLE_ENFORCE_CUDA_SUCCESS(dynload::ncclGroupEnd());
    VLOG(0) << "nccl group end seccessss";
  }
  PADDLE_ENFORCE_EQ(comm_map_.count(ring_id), 0);
  for (int i = 0; i < kDevices; ++i) {
    AssignNCCLComm(comms[i], kDevices * ntrainers, train_id * kDevices + i,
                   dev_ids[i], ring_id);
    VLOG(0) << "nccl communicator of train_id " << train_id * kDevices + i
            << " in ring " << ring_id << " has been created on device "
            << dev_ids[i];
  }

  std::call_once(once_flag_, []() {
    std::atexit([]() { NCCLCommContext::Instance().ReleaseNCCLComms(); });
  });
}

NCCLComm* NCCLCommContext::AssignNCCLComm(ncclComm_t comm, int nranks, int rank,
                                          int dev_id, int ring_id) {
  std::unique_ptr<CUDADeviceContext> dev_ctx(
      new CUDADeviceContext(CUDAPlace(dev_id)));

  std::shared_ptr<platform::CudaEventObject> compute_event(
      platform::CudaEventResourcePool::Instance().New(dev_id));
  std::shared_ptr<platform::CudaEventObject> comm_event(
      platform::CudaEventResourcePool::Instance().New(dev_id));

  NCCLCommImpl* c = new NCCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));
  c->set_compute_event(std::move(compute_event));
  c->set_comm_event(std::move(comm_event));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<NCCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::CUDAPlace(dev_id)));
    dev_ctx->set_nccl_comm(comm);
  }

  return comm_map_[ring_id][dev_id].get();
}

void NCCLCommContext::ReleaseNCCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

#endif

#if defined(PADDLE_WITH_XPU_BKCL)

class BKCLCommImpl : public BKCLComm {
 public:
  void set_ring_id(int ring_id) { ring_id_ = ring_id; }
  int ring_id() const override { return ring_id_; }

  void set_nranks(int nranks) { nranks_ = nranks; }
  int nranks() const override { return nranks_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  int device_id() const override {
    return BOOST_GET_CONST(XPUPlace, dev_ctx_->GetPlace()).device;
  }

  void set_comm(BKCLContext_t comm) { comm_ = comm; }
  BKCLContext_t comm() const override { return comm_; }

  XPUStream stream() const override {
    return dev_ctx_->x_context()->xpu_stream;
  }

  void set_dev_ctx(std::unique_ptr<XPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  XPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  int ring_id_;
  int nranks_;
  int rank_;
  BKCLContext_t comm_;
  std::unique_ptr<XPUDeviceContext> dev_ctx_;
};

BKCLComm* BKCLCommContext::CreateBKCLComm(BKCLUniqueId* bkcl_id, int nranks,
                                          int rank, int dev_id, int ring_id) {
  PADDLE_ENFORCE_NOT_NULL(bkcl_id,
                          platform::errors::InvalidArgument(
                              "The bkcl unique id should not be null."));
  PADDLE_ENFORCE_GT(
      nranks, 1,
      platform::errors::InvalidArgument(
          "Expected nranks > 1. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_LT(
      rank, nranks,
      platform::errors::InvalidArgument(
          "Expected rank < nranks. But received rank is %d, nranks is %d.",
          rank, nranks));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  BKCLContext_t comm = nullptr;
  auto ret = xpu_set_device(dev_id);
  PADDLE_ENFORCE_EQ(
      ret, XPU_SUCCESS,
      platform::errors::PreconditionNotMet(
          "XPU API return wrong value[%d %s], please check whether "
          "Baidu Kunlun Card is properly installed.",
          ret, XPUAPIErrorMsg[ret]));
  ret = bkcl_init_rank(&comm, rank, nranks, bkcl_id);
  PADDLE_ENFORCE_EQ(ret, BKCL_SUCCESS,
                    platform::errors::PreconditionNotMet(
                        "bkcl_init_rank failed, got wrong value [%d].", ret));

  auto* comm_wrapper = AssignBKCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "bkcl communicator of rank " << rank << " in ring " << ring_id
          << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { BKCLCommContext::Instance().ReleaseBKCLComms(); });
  });

  return comm_wrapper;
}

BKCLComm* BKCLCommContext::AssignBKCLComm(BKCLContext_t comm, int nranks,
                                          int rank, int dev_id, int ring_id) {
  std::unique_ptr<XPUDeviceContext> dev_ctx(
      new XPUDeviceContext(XPUPlace(dev_id)));

  BKCLCommImpl* c = new BKCLCommImpl;
  c->set_ring_id(ring_id);
  c->set_nranks(nranks);
  c->set_rank(rank);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));

  comm_map_mutex_.lock();
  if (comm_map_.count(ring_id) == 0) {
    comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<BKCLComm>>());
  }
  auto& dev2comm = comm_map_[ring_id];

  dev2comm.emplace(dev_id, std::unique_ptr<BKCLComm>(c));
  comm_map_mutex_.unlock();

  if (ring_id == 0) {
    auto* dev_ctx = static_cast<platform::XPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(
            platform::XPUPlace(dev_id)));
    dev_ctx->set_bkcl_context(comm);
  }

  return comm_map_[ring_id][dev_id].get();
}

void BKCLCommContext::ReleaseBKCLComms() {
  for (auto& p : comm_map_) {
    for (auto& q : p.second) {
      q.second.reset();
    }
  }
}

#endif

}  // namespace platform
}  // namespace paddle
