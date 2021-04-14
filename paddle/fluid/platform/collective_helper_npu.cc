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
  void set_rank_table_file(const std::string& rank_table_file) {
    rank_table_file_ = rank_table_file;
  }
  std::string rank_table_file() const override { return rank_table_file_; }

  void set_rank(uint32_t rank) { rank_ = rank; }
  uint32_t rank() const override { return rank_; }

  void set_device_id(uint32_t device_id) { device_id_ = device_id; }
  uint32_t device_id() const override { return device_id_; }

  aclrtStream stream() const override { return dev_ctx_->stream(); }

  void set_dev_ctx(std::unique_ptr<NPUDeviceContext>&& dev_ctx) {
    dev_ctx_ = std::move(dev_ctx);
  }
  NPUDeviceContext* dev_context() const override { return dev_ctx_.get(); }

 private:
  std::string rank_table_file_;
  uint32_t rank_;
  uint32_t device_id_;
  std::unique_ptr<NPUDeviceContext> dev_ctx_;
};

HCCLComm* HCCLCommContext::CreateHCCLComm(const std::string& rank_table_file,
                                          uint32_t rank, uint32_t device_id) {
  /*
    PADDLE_ENFORCE_NOT_NULL(rank_table_file,
                            platform::errors::InvalidArgument(
                                "The rank table file should not be null."));

    PADDLE_ENFORCE_GE(rank, 0,
        platform::errors::InvalidArgument(
            "Expected rank >= 0. But received rank is %d.", rank));

    PADDLE_ENFORCE_GE(device_id, 0,
        platform::errors::InvalidArgument(
            "Expected dev_id >= 0. But received dev_id is %d.", device_id));
  */
  auto* comm_wrapper = AssignHCCLComm(rank_table_file, rank, device_id);

  platform::dynload::hcom_init(rank_table_file.c_str(),
                               std::to_string(rank).c_str());
  platform::dynload::hcom_bind_model(comm_wrapper->stream(),
                                     comm_wrapper->stream());

  VLOG(1) << "hccl communicator of rank " << rank << " has been created";
  return comm_wrapper;
}

HCCLComm* HCCLCommContext::AssignHCCLComm(const std::string& rank_table_file,
                                          uint32_t rank, uint32_t device_id) {
  std::unique_ptr<NPUDeviceContext> dev_ctx(
      new NPUDeviceContext(NPUPlace(device_id)));

  VLOG(3) << "device_id" << device_id;
  VLOG(3) << "dev_ctx->stream()" << dev_ctx->stream();

  HCCLCommImpl* c = new HCCLCommImpl;
  c->set_rank_table_file(rank_table_file);
  c->set_rank(rank);
  c->set_device_id(device_id);
  c->set_dev_ctx(std::move(dev_ctx));
  // comm_ = c
  comm_.reset(c);
  return c;
}

void HCCLCommContext::CreateHCCLGroup(const std::string& group_name,
                                      uint32_t nranks,
                                      const std::vector<uint32_t>& rank_ids) {
  /*
    PADDLE_ENFORCE_NOT_NULL(group_name,
                            platform::errors::InvalidArgument(
                                "The group name should not be null."));
    PADDLE_ENFORCE_GT(nranks, 0,
                      platform::errors::InvalidArgument(
                          "Expected nranks > 0. But received nranks is %d.",
    nranks));
    PADDLE_ENFORCE_NOT_NULL(rank_ids,
                            platform::errors::InvalidArgument(
                                "The rank ids should not be null."));
  */
  platform::dynload::hcom_create_group(group_name.c_str(), nranks,
                                       (unsigned int*)rank_ids.data());

  VLOG(1) << "hccl group with name " << group_name << " has been created";
}

}  // namespace platform
}  // namespace paddle

#endif
