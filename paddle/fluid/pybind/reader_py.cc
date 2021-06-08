// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/reader_py.h"

#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Python.h"
#include "boost/optional.hpp"
#include "gflags/gflags.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/stl.h"

DEFINE_bool(reader_queue_speed_test_mode, false,
            "If set true, the queue.pop will only get data from queue but not "
            "remove the data from queue for speed testing");

namespace paddle {
namespace pybind {

namespace py = pybind11;
namespace reader = operators::reader;

// Check whether the tensor shape matches the VarDesc shape
// Return the different shape if exists
static boost::optional<std::vector<int64_t>> DiffTensorShapeWithVarDesc(
    const framework::LoDTensor &tensor, const framework::VarDesc &var_desc,
    size_t num_places) {
  auto tensor_shape = tensor.dims();
  auto desc_shape = var_desc.GetShape();

  int64_t rank = tensor_shape.size();

  if (UNLIKELY(rank == 0)) {
    if (desc_shape.size() != 0) {  // Tensor rank = 0 but desc does not match
      return framework::vectorize<int64_t>(tensor_shape);
    } else {
      return boost::none;
    }
  }

  PADDLE_ENFORCE_GE(tensor_shape[0], 0,
                    platform::errors::InvalidArgument(
                        "Tensor shape at dim 0 must not be less than 0"));

  if (!tensor.lod().empty()) {
    tensor_shape[0] = -1;  // unknown shape
  } else {
    int64_t split_size = (tensor_shape[0] + num_places - 1) / num_places;
    int64_t remainder = (split_size == 0 ? 0 : tensor_shape[0] % split_size);
    tensor_shape[0] = split_size;
    if (desc_shape[0] >= 0) {  // need check dim 0
      if (tensor_shape[0] != desc_shape[0]) {
        return framework::vectorize<int64_t>(tensor_shape);
      }

      if (remainder > 0) {
        tensor_shape[0] = remainder;
        return framework::vectorize<int64_t>(tensor_shape);
      }
    }
  }

  for (int64_t idx = 1; idx < rank; ++idx) {
    PADDLE_ENFORCE_GE(
        tensor_shape[idx], 0,
        platform::errors::InvalidArgument(
            "Tensor shape at dim %d must not be less than 0", idx));
    if (desc_shape[idx] >= 0 && tensor_shape[idx] != desc_shape[idx]) {
      return framework::vectorize<int64_t>(tensor_shape);
    }
  }

  return boost::none;
}

static const std::shared_ptr<reader::LoDTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::LoDTensorBlockingQueue> &queue, size_t idx) {
  return queue;
}

static const std::shared_ptr<reader::LoDTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>
        &queue,
    size_t idx) {
  return queue->GetQueue(idx);
}

template <typename QueueType>
class MultiDeviceFeedReader {
 public:
  using ResultDictList =
      std::vector<std::unordered_map<std::string, framework::LoDTensor>>;
  using ResultList = std::vector<std::vector<framework::LoDTensor>>;

  static constexpr bool kKeepOrder =
      std::is_same<QueueType,
                   reader::OrderedMultiDeviceLoDTensorBlockingQueue>::value;

  MultiDeviceFeedReader(
      const std::shared_ptr<QueueType> &queue, std::vector<std::string> names,
      const std::vector<std::vector<int>> &shapes,
      const std::vector<framework::proto::VarType::Type> &dtypes,
      const std::vector<bool> &need_check_feed,
      const std::vector<platform::Place> &dst_places, bool use_double_buffer,
      bool drop_last, bool pin_memory = false)
      : queue_(queue),
        names_(std::move(names)),
        pool_(new ::ThreadPool(dst_places.size())),
        drop_last_(drop_last),
        pin_memory_(pin_memory) {
    std::vector<framework::DDim> dims;
    for (auto &shape : shapes) {
      dims.push_back(framework::make_ddim(shape));
    }

    auto first_reader = std::make_shared<reader::PyReader>(
        GetQueue(queue, 0), dims, dtypes, need_check_feed);

    auto create_or_get_reader = [&](size_t idx) {
      if (idx == 0 ||
          std::is_same<QueueType, reader::LoDTensorBlockingQueue>::value) {
        return first_reader;
      } else {
        return std::make_shared<reader::PyReader>(GetQueue(queue, idx), dims,
                                                  dtypes, need_check_feed);
      }
    };

    readers_.reserve(dst_places.size());
    for (size_t i = 0; i < dst_places.size(); ++i) {
      auto &p = dst_places[i];
      auto *holder = new framework::ReaderHolder();
      auto reader = create_or_get_reader(i);
      if (use_double_buffer) {
        VLOG(10) << "Creating " << i << "-th BufferedReader";
        holder->Reset(
            framework::MakeDecoratedReader<operators::reader::BufferedReader>(
                reader, p, 2, pin_memory_));
      } else {
        if (platform::is_gpu_place(p)) {
          PADDLE_THROW(platform::errors::PermissionDenied(
              "Place cannot be CUDAPlace when use_double_buffer is False"));
        }
        holder->Reset(reader);
      }
      readers_.emplace_back(holder);
    }

    futures_.resize(dst_places.size());
    ret_.resize(dst_places.size());
    exceptions_.assign(dst_places.size(), nullptr);
    ReadAsync();
  }

  bool DropLast() const { return drop_last_; }

  ResultDictList ReadNext() {
    CheckNextStatus();
    ResultDictList result;
    result.reserve(ret_.size());
    for (auto &i : ret_) {
      if (i.empty()) {
        if (!kKeepOrder) result.emplace_back();
        continue;
      }

      result.emplace_back();
      auto &ret = result.back();
      PADDLE_ENFORCE_EQ(names_.size(), i.size(),
                        platform::errors::InvalidArgument(
                            "The sample number of reader's input data and the "
                            "input number of feed list are not equal.\n"
                            "Possible reasons are:\n"
                            "  The generator is decorated by `paddle.batch` "
                            "and configured by `set_batch_generator`, but here "
                            "need to used `set_sample_list_generator`."));
      for (size_t j = 0; j < names_.size(); ++j) {
        ret.emplace(names_[j], std::move(i[j]));
      }
    }
    ReadAsync();
    return result;
  }

  ResultList ReadNextList() {
    CheckNextStatus();
    ResultList result;
    result.reserve(ret_.size());
    for (auto &i : ret_) {
      if (kKeepOrder && i.empty()) continue;
      result.emplace_back(std::move(i));
    }
    ReadAsync();
    return result;
  }

  void Reset() {
    Shutdown();
    Start();
    ReadAsync();
  }

  void Shutdown() {
    for (auto &r : readers_) r->Shutdown();
  }

  ~MultiDeviceFeedReader() {
    queue_->Close();
    pool_.reset();
  }

 private:
  enum Status {
    kSuccess = 0,   // Read next data successfully
    kEOF = 1,       // Reach EOF
    kException = 2  // Exception raises when reading
  };

  Status WaitFutures(std::exception_ptr *excep) {
    *excep = nullptr;
    size_t success_num = 0;
    for (size_t i = 0; i < futures_.size(); ++i) {
      auto each_status = futures_[i].get();
      if (UNLIKELY(each_status != Status::kSuccess)) {
        if (UNLIKELY(each_status == Status::kException)) {
          PADDLE_ENFORCE_NOT_NULL(
              exceptions_[i],
              platform::errors::NotFound("exceptions_[%d] is NULL, but the "
                                         "result status is Status::kException",
                                         i));
          *excep = exceptions_[i];
          exceptions_[i] = nullptr;
        }
      } else {
        ++success_num;
      }
    }

    if (UNLIKELY(*excep)) {
      return Status::kException;
    }

    if (drop_last_) {
      return success_num == futures_.size() ? Status::kSuccess : Status::kEOF;
    } else {
      return success_num > 0 ? Status::kSuccess : Status::kEOF;
    }
  }

  void Start() {
    for (auto &r : readers_) r->Start();
  }

  void ReadAsync() {
    for (size_t i = 0; i < readers_.size(); ++i) {
      futures_[i] = pool_->enqueue([this, i] {
        try {
          readers_[i]->ReadNext(&ret_[i]);
          return ret_[i].empty() ? Status::kEOF : Status::kSuccess;
        } catch (...) {
          exceptions_[i] = std::current_exception();
          return Status::kException;
        }
      });
    }
  }

  void CheckNextStatus() {
    std::exception_ptr excep;
    Status status = WaitFutures(&excep);

    if (UNLIKELY(excep)) {
      PADDLE_ENFORCE_EQ(status, Status::kException,
                        platform::errors::NotFound(
                            "The exception raised is not NULL, but "
                            "the result status is not Status::kException"));
      std::rethrow_exception(excep);
    }

    if (UNLIKELY(status == Status::kEOF)) {
      VLOG(2) << "Raise StopIteration Exception in Python";
      py::gil_scoped_acquire guard;
      throw py::stop_iteration();
    }

    PADDLE_ENFORCE_EQ(status, Status::kSuccess,
                      platform::errors::NotFound(
                          "The function executed sucessfully, but "
                          "the result status is not Status::kSuccess"));
  }

  std::shared_ptr<QueueType> queue_;
  std::vector<std::string> names_;
  std::unique_ptr<::ThreadPool> pool_;

  std::vector<std::unique_ptr<framework::ReaderHolder>> readers_;

  std::vector<std::future<Status>> futures_;
  std::vector<std::exception_ptr> exceptions_;

  std::vector<std::vector<framework::LoDTensor>> ret_;
  bool drop_last_;
  bool pin_memory_;
};

template <typename QueueType>
void BindMultiDeviceReader(py::module *module, const char *reader_name) {
  auto &m = *module;

  using ReaderType = MultiDeviceFeedReader<QueueType>;
  py::class_<ReaderType>(m, reader_name, "")
      .def("read_next", &ReaderType::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_list", &ReaderType::ReadNextList,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "read_next_var_list",
          [](ReaderType &self) {
            auto result_list = self.ReadNextList();
            auto &tensor_list = result_list[0];
            std::vector<std::shared_ptr<imperative::VarBase>> var_list;
            var_list.reserve(tensor_list.size());
            auto func = [](framework::LoDTensor &lod_tensor) {
              std::string act_name =
                  imperative::GetCurrentTracer()->GenerateUniqueName(
                      "generated_var");
              auto new_var = std::make_shared<imperative::VarBase>(act_name);
              new_var->SetPersistable(false);
              new_var->SetType(framework::proto::VarType::LOD_TENSOR);
              new_var->SetDataType(lod_tensor.type());
              auto *tensor =
                  new_var->MutableVar()->GetMutable<framework::LoDTensor>();
              *tensor = std::move(lod_tensor);
              return new_var;
            };
            for (auto &tensor : tensor_list) {
              var_list.emplace_back(func(tensor));
            }
            return var_list;
          },
          py::call_guard<py::gil_scoped_release>())
      .def("reset", &ReaderType::Reset,
           py::call_guard<py::gil_scoped_release>())
      .def("shutdown", &ReaderType::Shutdown,
           py::call_guard<py::gil_scoped_release>());
}

void BindReader(py::module *module) {
  auto &m = *module;

  m.def(
      "diff_tensor_shape",
      [](const framework::LoDTensor &tensor, const framework::VarDesc &var_desc,
         size_t num_places) -> py::object {
        auto diff = DiffTensorShapeWithVarDesc(tensor, var_desc, num_places);
        if (diff) {
          return py::cast(std::move(diff.get()));
        } else {
          return py::cast(nullptr);
        }
      });

  m.def(
      "init_lod_tensor_blocking_queue",
      [](framework::Variable &var, size_t capacity,
         bool is_ordered) -> py::object {
        VLOG(1) << "init_lod_tensor_blocking_queue";
        if (is_ordered) {
          auto *holder = var.GetMutable<
              reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder>();
          holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
          return py::cast(holder->GetQueue());
        } else {
          auto *holder = var.GetMutable<reader::LoDTensorBlockingQueueHolder>();
          holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
          return py::cast(holder->GetQueue());
        }
      },
      py::return_value_policy::copy);

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<reader::LoDTensorBlockingQueue,
             std::shared_ptr<reader::LoDTensorBlockingQueue>>(
      m, "LoDTensorBlockingQueue", "")
      .def(
          "push",
          [](reader::LoDTensorBlockingQueue &self,
             const std::vector<framework::LoDTensor> &lod_tensor_vec) {
            return self.Push(lod_tensor_vec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::LoDTensorBlockingQueue::Size)
      .def("capacity", &reader::LoDTensorBlockingQueue::Cap)
      .def("close", &reader::LoDTensorBlockingQueue::Close)
      .def("kill", &reader::LoDTensorBlockingQueue::Kill)
      .def("wait_for_inited", &reader::LoDTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>());

  py::class_<reader::OrderedMultiDeviceLoDTensorBlockingQueue,
             std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>>(
      m, "OrderedMultiDeviceLoDTensorBlockingQueue", "")
      .def(
          "push",
          [](reader::OrderedMultiDeviceLoDTensorBlockingQueue &self,
             const std::vector<framework::LoDTensor> &lod_tensor_vec) {
            return self.Push(lod_tensor_vec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Size)
      .def("capacity", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Cap)
      .def("close", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Close)
      .def("kill", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Kill)
      .def("wait_for_inited",
           &reader::OrderedMultiDeviceLoDTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Reset);

  BindMultiDeviceReader<reader::LoDTensorBlockingQueue>(
      module, "MultiDeviceFeedReader");
  BindMultiDeviceReader<reader::OrderedMultiDeviceLoDTensorBlockingQueue>(
      module, "OrderedMultiDeviceFeedReader");

  m.def(
      "create_py_reader",
      [](const std::shared_ptr<reader::LoDTensorBlockingQueue> &queue,
         const std::vector<std::string> &names,
         const std::vector<std::vector<int>> &shapes,
         const std::vector<framework::proto::VarType::Type> &dtypes,
         const std::vector<bool> &need_check_feed,
         const std::vector<platform::Place> &dst_places, bool use_double_buffer,
         bool drop_last, bool pin_memory) {
        return new MultiDeviceFeedReader<reader::LoDTensorBlockingQueue>(
            queue, names, shapes, dtypes, need_check_feed, dst_places,
            use_double_buffer, drop_last, pin_memory);
      },
      py::return_value_policy::take_ownership);

  m.def(
      "create_py_reader",
      [](const std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>
             &queue,
         const std::vector<std::string> &names,
         const std::vector<std::vector<int>> &shapes,
         const std::vector<framework::proto::VarType::Type> &dtypes,
         const std::vector<bool> &need_check_feed,
         const std::vector<platform::Place> &dst_places, bool use_double_buffer,
         bool drop_last, bool pin_memory) {
        queue->SetDeviceCount(dst_places.size());
        return new MultiDeviceFeedReader<
            reader::OrderedMultiDeviceLoDTensorBlockingQueue>(
            queue, names, shapes, dtypes, need_check_feed, dst_places,
            use_double_buffer, drop_last, pin_memory);
      },
      py::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
