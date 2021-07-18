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

#include "paddle/fluid/framework/fleet/gloo_wrapper.h"
#include "paddle/fluid/framework/io/fs.h"
#include "paddle/fluid/string/string_helper.h"

namespace gloo {
namespace transport {
class Device;
}  // namespace transport
}  // namespace gloo

namespace gloo {
namespace rendezvous {

class HTTPStore;
class Store;

constexpr int kNodeSize = 136;

HdfsStore::HdfsStore(const std::string& path) {
  path_ = path;
  wait_sleep_ms_ = 10000;
  wait_timeout_ = std::chrono::seconds(999999999);
  retry_times_ = 100;
}

void HdfsStore::set(const std::string& key, const std::vector<char>& data) {
#ifdef PADDLE_WITH_GLOO
  auto tmp = TmpPath(key);
  auto path = ObjectPath(key);
  bool is_exists = paddle::framework::fs_exists(path);
  if (is_exists) {
    LOG(WARNING) << "path exists, will be removed: " << path;
    paddle::framework::fs_remove(path);
  }
  int err_no = 0;
  for (int i = 1; i <= retry_times_; ++i) {
    err_no = 0;
    std::shared_ptr<FILE> fp =
        paddle::framework::fs_open_write(tmp, &err_no, "");
    size_t write_count = fwrite_unlocked(data.data(), 1, data.size(), fp.get());
    if (write_count != data.size()) {
      VLOG(0) << "fwrite_unlocked failed, retry times " << i << " write_count "
              << write_count << " data.size() " << data.size();
      err_no = -1;
    }
    fp.reset();
    if (err_no != 0) {
      VLOG(0) << "fs_open_write failed, retry times " << i << " err no "
              << err_no;
      sleep(wait_sleep_ms_ / 1000);
      paddle::framework::fs_remove(tmp);
      if (i == retry_times_) {
        VLOG(0) << "fs_open_write failed, retry times reaches limit";
        PADDLE_THROW(paddle::platform::errors::PreconditionNotMet(
            "fs_open_write failed, retry times reaches %d limit.",
            retry_times_));
      }
    } else {
      break;
    }
  }
  paddle::framework::fs_mv(tmp, path);
#endif
}

#ifdef PADDLE_WITH_GLOO
int retry_do_func(std::function<int(void)> func, uint32_t max_try_time,
                  uint32_t retry_interval_ms) {
  for (uint32_t i = 0; i < max_try_time; ++i) {
    if (func() == 0) {
      return 0;
    }
#ifdef _LINUX
    usleep(retry_interval_ms * 1000);
#endif
  }
  return -1;
}
#endif

std::vector<char> HdfsStore::get(const std::string& key) {
  auto path = ObjectPath(key);
  std::vector<char> result;
#ifdef PADDLE_WITH_GLOO
  // block until key is set
  wait({key});
  int ret = retry_do_func(
      [&path]() { return paddle::framework::fs_exists(path) ? 0 : -1; }, 5,
      wait_sleep_ms_);
  bool is_exists = (ret == 0);
  PADDLE_ENFORCE_EQ(is_exists, true,
                    paddle::platform::errors::NotFound(
                        "HdfsStore::get, path not exists: " + path));

  int read_status = retry_do_func(
      [&path, &result]() {
        result.clear();
        int err_no = 0;
        {
          std::shared_ptr<FILE> fp =
              paddle::framework::fs_open_read(path, &err_no, "");
          char buffer = '\0';
          size_t read_count = 0;
          while (fread(&buffer, 1, 1, fp.get()) == 1) {
            ++read_count;
            result.push_back(buffer);
          }
          VLOG(3) << "HdfsStore::get read_count " << read_count;
        }
        return err_no;
      },
      5, wait_sleep_ms_);
  PADDLE_ENFORCE_EQ(read_status, 0,
                    paddle::platform::errors::Fatal(
                        "HdfsStore::get, path read faied: " + path));
#endif
  return result;
}

void HdfsStore::wait(const std::vector<std::string>& keys) {
#ifdef PADDLE_WITH_GLOO
  wait(keys, wait_timeout_);  // NOLINT
#endif
}

void HdfsStore::wait(const std::vector<std::string>& keys,
                     const std::chrono::milliseconds&) {  // NOLINT
#ifdef PADDLE_WITH_GLOO
  auto start = std::chrono::steady_clock::now();
  std::vector<bool> check_key_status(keys.size(), false);
  while (!Check(keys, &check_key_status)) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (wait_timeout_ != gloo::kNoTimeout && elapsed > wait_timeout_) {
      int32_t last_check_rank = -1;
      for (size_t i = 0; i < check_key_status.size(); ++i) {
        if (!check_key_status[i]) {
          last_check_rank = i;
          break;
        }
      }
      PADDLE_THROW(paddle::platform::errors::ExecutionTimeout(
          "TIMEOUT self_rank = %d pair_rank = %d", self_rank_,
          last_check_rank));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(wait_sleep_ms_));
  }
#endif
}

void HdfsStore::SetTimeoutSeconds(int timeout_seconds) {
  wait_timeout_ = std::chrono::seconds(timeout_seconds);
}

std::string HdfsStore::EncodeName(const std::string& name) {
  return ::paddle::string::erase_spaces(name);
}

std::string HdfsStore::TmpPath(const std::string& name) {
  return path_ + "/" + EncodeName(name) + "_tmp";
}

std::string HdfsStore::ObjectPath(const std::string& name) {
  return path_ + "/" + EncodeName(name);
}

bool HdfsStore::Check(const std::vector<std::string>& keys,
                      std::vector<bool>* keys_check_status) {
#ifdef PADDLE_WITH_GLOO
  bool ret = true;
  std::vector<std::string> paths;
  for (const auto& key : keys) {
    paths.push_back(ObjectPath(key));
  }
  for (size_t i = 0; i < paths.size(); ++i) {
    if ((*keys_check_status)[i]) {
      continue;
    }
    const auto& path = paths[i];
    bool is_exists = paddle::framework::fs_exists(path);
    VLOG(3) << "HdfsStore::Check " << is_exists << " path " << path;
    if (!is_exists) {
      ret = false;
    }
    (*keys_check_status)[i] = is_exists;
  }
  return ret;
#else
  VLOG(0) << "HdfsStore::Check does nothing when no gloo";
#endif
  return true;
}

#ifdef PADDLE_WITH_GLOO
void ParallelConnectContext::connectFullMesh(
    Store& store, std::shared_ptr<transport::Device>& dev) {
  std::vector<char> allBytes;
  // Create pairs
  auto transportContext = dev->createContext(rank, size);
  transportContext->setTimeout(getTimeout());
  for (int i = 0; i < size; i++) {
    if (i == rank) {
      continue;
    }
    auto& pair = transportContext->createPair(i);
    auto addrBytes = pair->address().bytes();
    allBytes.insert(allBytes.end(), addrBytes.begin(), addrBytes.end());
  }
  std::ostringstream storeKey;
  storeKey << rank;
  store.set(storeKey.str(), allBytes);

  auto total_add_size = kNodeSize * (size - 1);

  std::vector<std::shared_ptr<std::thread>> connect_threads(thread_num_);
  // Connect every pair
  for (uint32_t i = 0; i < connect_threads.size(); ++i) {
    connect_threads[i].reset(new std::thread(
        [&store, &transportContext, total_add_size, this](
            size_t thread_idx, size_t thread_num) -> void {
          for (int i = thread_idx; i < size; i += thread_num) {
            if (i == rank) {
              continue;
            }
            // Wait for address of other side of this pair to become available
            std::string key = std::to_string(i);
            store.wait({key}, getTimeout());

            std::vector<char> allAddrs;
            auto max_retry_times = 10;
            // Connect to other side of this pair

            while (max_retry_times > 0) {
              allAddrs = store.get(key);
              VLOG(3) << "store get all address size: " << allAddrs.size()
                      << " except: " << total_add_size;
              if (allAddrs.size() == static_cast<size_t>(total_add_size)) {
                break;
              }

              sleep(5);
              --max_retry_times;
            }

            auto addr = extractAddress(allAddrs, i);
            transportContext->getPair(i)->connect(addr);
          }
        },
        i, connect_threads.size()));
  }
  for (uint32_t i = 0; i < connect_threads.size(); ++i) {
    connect_threads[i]->join();
  }
  device_ = dev;
  transportContext_ = std::move(transportContext);
}
#endif
}  // namespace rendezvous
}  // namespace gloo

namespace paddle {
namespace framework {

void GlooWrapper::Init() {
  if (is_initialized_) {
    return;
  }
#ifdef PADDLE_WITH_GLOO
  gloo::transport::tcp::attr attr;
  attr.iface = iface_;
  std::shared_ptr<gloo::rendezvous::HdfsStore> file_store = nullptr;
  std::shared_ptr<gloo::rendezvous::HTTPStore> http_store = nullptr;
  auto dev = gloo::transport::tcp::CreateDevice(attr);

  switch (store_type_) {
    case GlooStoreType::HDFS: {
      auto context = std::make_shared<gloo::rendezvous::ParallelConnectContext>(
          rank_, size_);
      context->setTimeout(run_timeout_);
      std::string cmd = std::string("${HADOOP_HOME}/bin/hadoop fs");
      cmd += " -D fs.default.name=" + hdfs_name_;
      cmd += " -D hadoop.job.ugi=" + hdfs_ugi_;
      paddle::framework::hdfs_set_command(cmd);
      file_store = std::make_shared<gloo::rendezvous::HdfsStore>(hdfs_path_);
      file_store->SetTimeoutSeconds(init_timeout_.count());
      auto prefix_store =
          std::make_shared<gloo::rendezvous::PrefixStore>(prefix_, *file_store);
      context->connectFullMesh(*prefix_store, dev);
      context_ = std::move(context);
      break;
    }
    case GlooStoreType::HTTP: {
      auto context = std::make_shared<gloo::rendezvous::Context>(rank_, size_);
      context->setTimeout(run_timeout_);
      http_store = std::make_shared<gloo::rendezvous::HTTPStore>(
          http_ip_, http_port_, prefix_ + "_" + http_scope_, rank_);
      http_store->SetTimeoutSeconds(init_timeout_.count());
      context->connectFullMesh(*http_store, dev);
      http_store->Finalize();
      VLOG(3) << "after calling http_store->Finalize.";
      context_ = std::move(context);
      break;
    }
    default:
      LOG(ERROR) << "unknown store type " << store_type_;
      exit(-1);
  }
#endif
  is_initialized_ = true;
  VLOG(3) << "gloo initialized done.";
}

template std::vector<int64_t> GlooWrapper::AllReduce<int64_t>(
    std::vector<int64_t>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<float> GlooWrapper::AllReduce<float>(
    std::vector<float>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<double> GlooWrapper::AllReduce<double>(
    std::vector<double>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<uint64_t> GlooWrapper::AllReduce<uint64_t>(
    std::vector<uint64_t>& sendbuf,  // NOLINT
    const std::string& mode);
template std::vector<int64_t> GlooWrapper::AllGather<int64_t>(
    int64_t& input);  // NOLINT
template std::vector<uint64_t> GlooWrapper::AllGather<uint64_t>(
    uint64_t& input);  // NOLINT
template std::vector<float> GlooWrapper::AllGather<float>(
    float& input);  // NOLINT
template std::vector<double> GlooWrapper::AllGather<double>(
    double& input);  // NOLINT

}  // namespace framework
}  // namespace paddle
