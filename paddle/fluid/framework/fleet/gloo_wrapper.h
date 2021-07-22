/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#if defined _WIN32 || defined __APPLE__
#else
#define _LINUX
#endif

#ifdef _LINUX
#include <sys/types.h>
#include <unistd.h>
#endif
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef PADDLE_WITH_GLOO
#include <gloo/allgather.h>
#include <gloo/allreduce.h>
#include <gloo/barrier.h>
#include <gloo/rendezvous/context.h>
#include <gloo/rendezvous/file_store.h>
#include <gloo/rendezvous/http_store.h>
#include <gloo/rendezvous/prefix_store.h>
#include <gloo/rendezvous/store.h>
#include <gloo/transport/tcp/device.h>
#endif
#include "paddle/fluid/framework/variable_helper.h"

namespace gloo {
class Context;
namespace transport {
class Device;
}  // namespace transport
}  // namespace gloo

namespace gloo {
namespace rendezvous {

#ifdef PADDLE_WITH_GLOO
class HdfsStore : public gloo::rendezvous::Store {
#else
class HdfsStore {
#endif
 public:  // NOLINT
  explicit HdfsStore(const std::string& path);

  virtual ~HdfsStore() {}

  virtual void set(const std::string& key, const std::vector<char>& data);

  virtual std::vector<char> get(const std::string& key);

  virtual void wait(const std::vector<std::string>& keys);

  virtual void wait(const std::vector<std::string>& keys,
                    const std::chrono::milliseconds& timeout);

  virtual void SetTimeoutSeconds(int timeout_seconds);

  std::string EncodeName(const std::string& name);

  std::string TmpPath(const std::string& name);

  std::string ObjectPath(const std::string& name);

  bool Check(const std::vector<std::string>& keys,
             std::vector<bool>* keys_check_status);

  void SetRank(int rank) { self_rank_ = rank; }

  std::string path_;
  int wait_sleep_ms_;
  std::chrono::seconds wait_timeout_;
  int retry_times_;
  int self_rank_;
};

#ifdef PADDLE_WITH_GLOO
class ParallelConnectContext : public gloo::rendezvous::Context {
 public:
  ParallelConnectContext(int rank, int size, int base = 2)
      : gloo::rendezvous::Context(rank, size, base) {}
  virtual ~ParallelConnectContext() {}
  // in gloo::rendezvous::Context wait&get one by one,
  // slowly in case big size, especialy in HdfsStore
  void connectFullMesh(Store& store,                              // NOLINT
                       std::shared_ptr<transport::Device>& dev);  // NOLINT

 protected:
  int thread_num_ = 6;
};
#endif
}  // namespace rendezvous
}  // namespace gloo

namespace paddle {
namespace framework {

enum GlooStoreType { HDFS, HTTP };

class GlooWrapper {
 public:
  static std::shared_ptr<GlooWrapper> GetInstance() {
    static auto s_instance = std::make_shared<GlooWrapper>();
    return s_instance;
  }

  GlooWrapper() {}

  virtual ~GlooWrapper() {}

  void Init();

  void SetTimeoutSeconds(int init_seconds, int run_seconds) {
    init_timeout_ = std::chrono::seconds(init_seconds);
    run_timeout_ = std::chrono::seconds(run_seconds);
  }

  int Rank() { return rank_; }

  int Size() { return size_; }

  void SetRank(int rank) { rank_ = rank; }

  void SetSize(int size) { size_ = size; }

  void SetIface(const std::string& iface) { iface_ = iface; }

  void SetPrefix(const std::string& prefix) { prefix_ = prefix; }

  void SetHdfsStore(const std::string& path, const std::string& fs_name,
                    const std::string& fs_ugi) {
    store_type_ = GlooStoreType::HDFS;
    hdfs_path_ = path;
    hdfs_name_ = fs_name;
    hdfs_ugi_ = fs_ugi;
  }

  void SetHttpStore(const std::string& ip, int port, const std::string& scope) {
    store_type_ = GlooStoreType::HTTP;
    http_ip_ = ip;
    http_port_ = port;
    http_scope_ = scope;
  }

  void Barrier() {
    CHECK_EQ(is_initialized_, true);
#ifdef PADDLE_WITH_GLOO
    gloo::BarrierOptions opts(context_);
    gloo::barrier(opts);
#else
    LOG(WARNING) << "Barrier does nothing when WITH_GLOO=OFF";
#endif
  }

  bool IsInitialized() { return is_initialized_; }
#ifdef PADDLE_WITH_GLOO
  std::shared_ptr<gloo::Context> GetContext() { return context_; }
#endif

  template <typename T>
  std::vector<T> AllReduce(std::vector<T>& sendbuf,            // NOLINT
                           const std::string& mode = "sum") {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    std::vector<T> recvbuf(sendbuf.size(), T());
    CHECK_EQ(sendbuf.size() == recvbuf.size(), true);
#ifdef PADDLE_WITH_GLOO
    gloo::AllreduceOptions opts(context_);
    opts.setInput(sendbuf.data(), sendbuf.size());
    opts.setOutput(recvbuf.data(), recvbuf.size());
    if (mode == "sum") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::sum<T>));
    } else if (mode == "max") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::max<T>));
    } else if (mode == "min") {
      opts.setReduceFunction(
          static_cast<void (*)(void*, const void*, const void*, size_t)>(
              &gloo::min<T>));
    } else {
      PADDLE_ENFORCE_EQ(0, 1, paddle::platform::errors::InvalidArgument(
                                  "AllReduce mode not known: " + mode));
    }
    gloo::allreduce(opts);
#else
    LOG(WARNING) << "AllReduce does nothing when WITH_GLOO=OFF";
#endif
    return recvbuf;
  }

  template <typename T>
  std::vector<T> AllGather(T& input) {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    std::vector<T> ret(size_, T());
#ifdef PADDLE_WITH_GLOO
    gloo::AllgatherOptions opts(context_);
    opts.setInput(&input, 1);
    opts.setOutput(ret.data(), size_);
    gloo::allgather(opts);
#else
    LOG(WARNING) << "AllGather does nothing when WITH_GLOO=OFF";
#endif
    return std::move(ret);
  }

  template <typename T>
  std::vector<T> AllGatherList(std::vector<T>& input) {  // NOLINT
    CHECK_EQ(is_initialized_, true);
    std::vector<T> ret(size_ * input.size(), T());
#ifdef PADDLE_WITH_GLOO
    gloo::AllgatherOptions opts(context_);
    opts.setInput(input.data(), input.size());
    opts.setOutput(ret.data(), ret.size());
    gloo::allgather(opts);
#else
    LOG(WARNING) << "AllGather does nothing when WITH_GLOO=OFF";
#endif
    return std::move(ret);
  }

 protected:
  bool is_initialized_ = false;
#ifdef PADDLE_WITH_GLOO
  std::shared_ptr<gloo::Context> context_ = nullptr;
#endif
  int rank_ = 0;
  int size_ = 0;
  std::chrono::seconds init_timeout_ = std::chrono::seconds(9999999);
  std::chrono::seconds run_timeout_ = std::chrono::seconds(9999999);
  std::string iface_ = "lo";
  std::string prefix_;
  GlooStoreType store_type_ = GlooStoreType::HDFS;
  // configs for hdfs store
  std::string hdfs_path_;
  std::string hdfs_name_;
  std::string hdfs_ugi_;
  std::string http_ip_;
  // configs for http store
  int http_port_;
  std::string http_scope_;
};

}  // namespace framework
}  // namespace paddle
