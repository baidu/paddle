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

#if defined(PADDLE_WITH_ASCEND_CL)
#include "paddle/fluid/platform/collective_helper.h"
#include <utility>

namespace paddle {
namespace platform {

class HCCLCommImpl : public HCCLComm {
 public:
  void set_rank_table_file(std::string rank_table_file) { rank_table_file_ = rank_table_file; }
  std::string rank_table_file() const override { return rank_table_file_; }

  void set_rank(int rank) { rank_ = rank; }
  int rank() const override { return rank_; }

  void set_device_id(int device_id) { device_id_ = device_id; }
  int device_id() const override { return device_id_; }

  void set_comm(HcclComm comm) { comm_ = comm; }
  HcclComm comm() const override { return comm_; }

  aclrtStream stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<NPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  NPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  std::string rank_table_file_;
  int rank_;
  int device_id_;
  HcclComm comm_;
  std::unique_ptr<DeviceContext> dev_ctx_;
};

NCCLComm* HCCLCommContext::CreateHCCLComm(const std::string& rank_table_file,
                                          int rank, int dev_id) {
  PADDLE_ENFORCE_NOT_NULL(rank_table_file,
                          platform::errors::InvalidArgument(
                              "The rank table file should not be null."));
  PADDLE_ENFORCE_GE(rank, 0,
                    platform::errors::InvalidArgument(
                        "Expected rank >= 0. But received rank is %d.", rank));
  PADDLE_ENFORCE_GE(
      dev_id, 0,
      platform::errors::InvalidArgument(
          "Expected dev_id >= 0. But received dev_id is %d.", dev_id));

  HcclComm comm = nullptr;
  PADDLE_ENFORCE_NPU_SUCCESS(aclrtSetDevice(dev_id));
  PADDLE_ENFORCE_NPU_SUCCESS(
      platform::dynload::HcclCommInitClusterInfo(rank_table_file, rank, &comm));

  auto* comm_wrapper = AssignHCCLComm(comm, nranks, rank, dev_id, ring_id);

  VLOG(1) << "hccl communicator of rank " << rank << " has been created on device " << dev_id;

  std::call_once(once_flag_, []() {
    std::atexit([]() { HCCLCommContext::Instance().ReleaseHCCLComms(); });
  });

  return comm_wrapper;
}

NCCLComm* HCCLCommContext::AssignHCCLComm(ncclComm_t comm, string rank_table_file, int rank, int dev_id) {
  std::unique_ptr<NPUDeviceContext> dev_ctx(
      new NPUDeviceContext(NPUPlace(dev_id)));

  HCCLCommImpl* c = new HCCLCommImpl;
  c->set_rank_table_file(rank_table_file);
  c->set_rank(rank);
  c->set_device_id(dev_id);
  c->set_comm(comm);
  c->set_dev_ctx(std::move(dev_ctx));

  // comm_map_mutex_.lock();
  // if (comm_map_.count(ring_id) == 0) {
  //   comm_map_.emplace(ring_id, std::map<int, std::unique_ptr<NCCLComm>>());
  // }
  // auto& dev2comm = comm_map_[ring_id];

  // dev2comm.emplace(dev_id, std::unique_ptr<NCCLComm>(c));
  // comm_map_mutex_.unlock();

  auto* dev_ctx = static_cast<platform::NPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(platform::NPUPlace(dev_id)));
  dev_ctx->set_hccl_comm(comm);

  // return comm_map_[ring_id][dev_id].get();
  return c;
}

void HCCLCommContext::CreateHCCLGroup(const std::string& group_name, int nranks,
  const vector<int>& rank_ids) {
  PADDLE_ENFORCE_NOT_NULL(group_name,
                          platform::errors::InvalidArgument(
                              "The group name should not be null."));
  PADDLE_ENFORCE_GT(nranks, 0,
                    platform::errors::InvalidArgument(
                        "Expected nranks > 0. But received nranks is %d.", nranks));
  PADDLE_ENFORCE_NOT_NULL(rank_inds,
                          platform::errors::InvalidArgument(
                              "The rank ids should not be null."));

  PADDLE_ENFORCE_NPU_SUCCESS(
      platform::dynload::HcclCommInitClusterInfo(rank_table_file, rank, &comm));

  VLOG(1) << "hccl group with name " << group_name << " has been created;
}

void HCCLCommContext::ReleaseHCCLComms() {
  // for (auto& p : comm_map_) {
  //   for (auto& q : p.second) {
  //     q.second.reset();
  //   }
  // }
}

}  // namespace platform
}  // namespace paddle

#endif
