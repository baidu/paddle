/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <fcntl.h>
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/dataset_factory.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/pybind/data_set_py.h"

namespace py = pybind11;
namespace pd = paddle::framework;

namespace paddle {
namespace pybind {

class IterableDatasetWrapper {
 public:
  IterableDatasetWrapper(framework::Dataset *dataset,
                         const std::vector<std::string> &slots,
                         const std::vector<platform::Place> &places,
                         size_t batch_size, bool drop_last)
      : dataset_(dataset),
        slots_(slots),
        places_(places),
        batch_size_(batch_size),
        drop_last_(drop_last) {
#if defined _WIN32
    PADDLE_THROW("Dataset is not supported on Windows");
#elif defined __APPLE__
    PADDLE_THROW("Dataset is not supported on MAC");
#else
    size_t device_num = places_.size();
    PADDLE_ENFORCE_GT(device_num, 0, "thread_num must be larger than 0");
    PADDLE_ENFORCE_GT(slots_.size(), 0, "slot_num cannot be 0");
    scopes_.reserve(device_num);
    tensors_.reserve(device_num);
    for (size_t i = 0; i < device_num; ++i) {
      scopes_.emplace_back(new framework::Scope());
      tensors_.emplace_back();
      for (auto &var_name : slots_) {
        auto *var = scopes_.back()->Var(var_name);
        auto *t = var->GetMutable<framework::LoDTensor>();
        tensors_.back().emplace_back(t);
      }
    }

    is_exhaustive_.resize(device_num);
    exhaustive_num_ = 0;
#endif
  }

  void Start() {
    PADDLE_ENFORCE_EQ(is_started_, false, "Reader has been started");
    data_feeds_ = dataset_->GetReaders();
    PADDLE_ENFORCE_EQ(data_feeds_.size(), places_.size(),
                      "Device number does not match reader number");
    for (size_t i = 0; i < places_.size(); ++i) {
      data_feeds_[i]->AssignFeedVar(*scopes_[i]);
      data_feeds_[i]->SetPlace(platform::CPUPlace());
      PADDLE_ENFORCE_EQ(data_feeds_[i]->Start(), true, "Reader start failed");
    }
    is_started_ = true;

    is_exhaustive_.assign(places_.size(), false);
    exhaustive_num_ = 0;
  }

  std::vector<std::unordered_map<std::string, framework::LoDTensor>> Next() {
    PADDLE_ENFORCE_EQ(is_started_, true, "Reader must be started");
    size_t device_num = places_.size();

    std::vector<std::unordered_map<std::string, framework::LoDTensor>> result(
        device_num);

    size_t read_num = 0;
    while (read_num < device_num && exhaustive_num_ < device_num) {
      for (size_t i = 0; i < data_feeds_.size(); ++i) {
        if (is_exhaustive_[i]) {
          continue;
        }

        bool is_success = (data_feeds_[i]->Next() > 0);
        if (!is_success) {
          is_exhaustive_[i] = true;
          ++exhaustive_num_;
          continue;
        }

        for (size_t j = 0; j < slots_.size(); ++j) {
          if (!IsValidLoDTensor(*tensors_[i][j])) {
            is_success = false;
            break;
          }

          if (tensors_[i][j]->place() == places_[read_num]) {
            result[read_num].emplace(slots_[j], std::move(*tensors_[i][j]));
          } else {
            framework::TensorCopy(std::move(*tensors_[i][j]), places_[read_num],
                                  &result[read_num][slots_[j]]);
          }
        }

        if (!is_success) {
          is_exhaustive_[i] = true;
          ++exhaustive_num_;
          continue;
        }

        ++read_num;
        if (read_num == device_num) {
          break;
        }
      }
    }

    if (UNLIKELY(read_num != device_num)) {
      is_started_ = false;
      throw py::stop_iteration();
    }

    return result;
  }

 private:
  bool IsValidLoDTensor(const framework::LoDTensor &tensor) const {
    auto &lod = tensor.lod();
    PADDLE_ENFORCE_LE(lod.size(), 1, "lod level must be not larger than 1");
    if (!drop_last_) return true;

    if (lod.empty()) {
      return static_cast<size_t>(tensor.dims()[0]) == batch_size_;
    } else {
      return lod[0].size() == batch_size_ + 1;
    }
  }

 private:
  framework::Dataset *dataset_;
  std::vector<std::string> slots_;
  std::vector<platform::Place> places_;
  size_t batch_size_;
  bool drop_last_;

  std::vector<framework::DataFeed *> data_feeds_;
  std::vector<bool> is_exhaustive_;
  size_t exhaustive_num_;

  std::vector<std::unique_ptr<framework::Scope>> scopes_;
  std::vector<std::vector<framework::LoDTensor *>> tensors_;
  bool is_started_{false};
};

void BindDataset(py::module *m) {
  py::class_<framework::Dataset, std::unique_ptr<framework::Dataset>>(*m,
                                                                      "Dataset")
      .def(py::init([](const std::string &name = "MultiSlotDataset") {
        return framework::DatasetFactory::CreateDataset(name);
      }))
      .def("init_tdm_tree", &framework::Dataset::InitTDMTree,
           py::call_guard<py::gil_scoped_release>())
      .def("tdm_sample", &framework::Dataset::TDMSample,
           py::call_guard<py::gil_scoped_release>())
      .def("tdm_dump", &framework::Dataset::TDMDump,
           py::call_guard<py::gil_scoped_release>())
      .def("set_filelist", &framework::Dataset::SetFileList,
           py::call_guard<py::gil_scoped_release>())
      .def("set_thread_num", &framework::Dataset::SetThreadNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_trainer_num", &framework::Dataset::SetTrainerNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_fleet_send_batch_size",
           &framework::Dataset::SetFleetSendBatchSize,
           py::call_guard<py::gil_scoped_release>())
      .def("set_hdfs_config", &framework::Dataset::SetHdfsConfig,
           py::call_guard<py::gil_scoped_release>())
      .def("set_download_cmd", &framework::Dataset::SetDownloadCmd,
           py::call_guard<py::gil_scoped_release>())
      .def("set_data_feed_desc", &framework::Dataset::SetDataFeedDesc,
           py::call_guard<py::gil_scoped_release>())
      .def("get_filelist", &framework::Dataset::GetFileList,
           py::call_guard<py::gil_scoped_release>())
      .def("get_thread_num", &framework::Dataset::GetThreadNum,
           py::call_guard<py::gil_scoped_release>())
      .def("get_trainer_num", &framework::Dataset::GetTrainerNum,
           py::call_guard<py::gil_scoped_release>())
      .def("get_fleet_send_batch_size",
           &framework::Dataset::GetFleetSendBatchSize,
           py::call_guard<py::gil_scoped_release>())
      .def("get_hdfs_config", &framework::Dataset::GetHdfsConfig,
           py::call_guard<py::gil_scoped_release>())
      .def("get_download_cmd", &framework::Dataset::GetDownloadCmd,
           py::call_guard<py::gil_scoped_release>())
      .def("get_data_feed_desc", &framework::Dataset::GetDataFeedDesc,
           py::call_guard<py::gil_scoped_release>())
      .def("register_client2client_msg_handler",
           &framework::Dataset::RegisterClientToClientMsgHandler,
           py::call_guard<py::gil_scoped_release>())
      .def("create_channel", &framework::Dataset::CreateChannel,
           py::call_guard<py::gil_scoped_release>())
      .def("create_readers", &framework::Dataset::CreateReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("destroy_readers", &framework::Dataset::DestroyReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("load_into_memory", &framework::Dataset::LoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("preload_into_memory", &framework::Dataset::PreLoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_preload_done", &framework::Dataset::WaitPreLoadDone,
           py::call_guard<py::gil_scoped_release>())
      .def("release_memory", &framework::Dataset::ReleaseMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("local_shuffle", &framework::Dataset::LocalShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("global_shuffle", &framework::Dataset::GlobalShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("get_memory_data_size", &framework::Dataset::GetMemoryDataSize,
           py::call_guard<py::gil_scoped_release>())
      .def("get_pv_data_size", &framework::Dataset::GetPvDataSize,
           py::call_guard<py::gil_scoped_release>())
      .def("get_shuffle_data_size", &framework::Dataset::GetShuffleDataSize,
           py::call_guard<py::gil_scoped_release>())
      .def("set_queue_num", &framework::Dataset::SetChannelNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_parse_ins_id", &framework::Dataset::SetParseInsId,
           py::call_guard<py::gil_scoped_release>())
      .def("set_parse_content", &framework::Dataset::SetParseContent,
           py::call_guard<py::gil_scoped_release>())
      .def("set_parse_logkey", &framework::Dataset::SetParseLogKey,
           py::call_guard<py::gil_scoped_release>())
      .def("set_merge_by_sid", &framework::Dataset::SetMergeBySid,
           py::call_guard<py::gil_scoped_release>())
      .def("preprocess_instance", &framework::Dataset::PreprocessInstance,
           py::call_guard<py::gil_scoped_release>())
      .def("postprocess_instance", &framework::Dataset::PostprocessInstance,
           py::call_guard<py::gil_scoped_release>())
      .def("set_current_phase", &framework::Dataset::SetCurrentPhase,
           py::call_guard<py::gil_scoped_release>())
      .def("set_enable_pv_merge", &framework::Dataset::SetEnablePvMerge,
           py::call_guard<py::gil_scoped_release>())

      .def("set_merge_by_lineid", &framework::Dataset::SetMergeByInsId,
           py::call_guard<py::gil_scoped_release>())
      .def("merge_by_lineid", &framework::Dataset::MergeByInsId,
           py::call_guard<py::gil_scoped_release>())
      .def("set_generate_unique_feasigns",
           &framework::Dataset::SetGenerateUniqueFeasign,
           py::call_guard<py::gil_scoped_release>())
      .def("generate_local_tables_unlock",
           &framework::Dataset::GenerateLocalTablesUnlock,
           py::call_guard<py::gil_scoped_release>())
      .def("slots_shuffle", &framework::Dataset::SlotsShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("set_fea_eval", &framework::Dataset::SetFeaEval,
           py::call_guard<py::gil_scoped_release>())
      .def("set_preload_thread_num", &framework::Dataset::SetPreLoadThreadNum,
           py::call_guard<py::gil_scoped_release>())
      .def("create_preload_readers", &framework::Dataset::CreatePreLoadReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("destroy_preload_readers",
           &framework::Dataset::DestroyPreLoadReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("dynamic_adjust_channel_num",
           &framework::Dataset::DynamicAdjustChannelNum,
           py::call_guard<py::gil_scoped_release>())
      .def("dynamic_adjust_readers_num",
           &framework::Dataset::DynamicAdjustReadersNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_fleet_send_sleep_seconds",
           &framework::Dataset::SetFleetSendSleepSeconds,
           py::call_guard<py::gil_scoped_release>());

  py::class_<IterableDatasetWrapper>(*m, "IterableDatasetWrapper")
      .def(py::init<framework::Dataset *, const std::vector<std::string> &,
                    const std::vector<platform::Place> &, size_t, bool>())
      .def("_start", &IterableDatasetWrapper::Start)
      .def("_next", &IterableDatasetWrapper::Next);
}

}  // namespace pybind
}  // namespace paddle
