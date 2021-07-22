// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once
#include <sys/time.h>

#include <iostream>
#include <ostream>
#include <string>
#include <thread>  // NOLINT
#include <vector>

#include <ThreadPool.h>
#include "boost/lexical_cast.hpp"
#include "glog/logging.h"
#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/dim.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/string/split.h"

constexpr int FG = 256 * 1024 * 1024;
constexpr int Q_SIZE = 10000;
constexpr int BUCKET = 10;
constexpr char XEOF[] = "EOF";

using boost::lexical_cast;

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

namespace paddle {
namespace distributed {

class ShardingMerge {
 public:
  ShardingMerge() {}
  ~ShardingMerge() {}

  void Merge(const std::vector<std::string> &inputs, const std::string &output,
             const int embedding_dim) {
    pool_.reset(new ::ThreadPool(inputs.size()));

    std::vector<std::future<int>> tasks(inputs.size());
    std::vector<std::vector<int64_t>> rows;
    rows.resize(inputs.size());

    auto begin = GetCurrentUS();
    for (int x = 0; x < inputs.size(); ++x) {
      tasks[x] = pool_->enqueue([this, x, &rows, &inputs]() -> int {
        DeserializeRowsFromFile(inputs[x], &rows[x]);
        return 0;
      });
    }

    for (size_t x = 0; x < tasks.size(); ++x) {
      tasks[x].wait();
    }

    int64_t total_rows = 0;
    for (auto x = 0; x < rows.size(); x++) {
      total_rows += rows[x].size();
    }

    auto end = GetCurrentUS();

    VLOG(0) << "got " << total_rows
            << " feasigin ids from sparse embedding using " << end - begin;

    std::vector<int64_t> total_dims = {total_rows,
                                       static_cast<int64_t>(embedding_dim)};

    std::vector<std::vector<int>> batch_buckets;
    batch_buckets.resize(inputs.size());

    for (int x = 0; x < rows.size(); ++x) {
      batch_buckets[x] = bucket(rows[x].size(), BUCKET);
    }

    std::ofstream out(output, std::ios::binary);

    begin = GetCurrentUS();
    SerializeRowsToStream(out, rows, batch_buckets, total_rows);
    end = GetCurrentUS();
    VLOG(0) << "write rows to oostrream using " << end - begin;

    begin = GetCurrentUS();
    SerializePreTensorToStream(out, total_dims);
    end = GetCurrentUS();
    VLOG(0) << "write pretensor to oostrream using " << end - begin;

    begin = GetCurrentUS();
    SerializeValueToStream(out, inputs, batch_buckets, embedding_dim);
    end = GetCurrentUS();
    VLOG(0) << "write values to oostrream using " << end - begin;
  }

 private:
  void SerializeRowsToStream(std::ostream &os,
                             const std::vector<std::vector<int64_t>> &rows,
                             const std::vector<std::vector<int>> &batch_buckets,
                             int64_t total_rows) {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }

    {
      // the 2st field, rows information
      os.write(reinterpret_cast<const char *>(&total_rows), sizeof(total_rows));

      for (int b = 0; b < BUCKET; ++b) {
        for (int x = 0; x < batch_buckets.size(); ++x) {
          auto begin = batch_buckets[x][b];
          auto end = batch_buckets[x][b + 1];

          if (end - begin == 0) continue;

          os.write(reinterpret_cast<const char *>(rows[x].data() + begin),
                   sizeof(int64_t) * (end - begin));
        }
      }

      // the 3st field, the height of SelectedRows
      int64_t height = total_rows;
      os.write(reinterpret_cast<const char *>(&height), sizeof(height));
    }
  }

  void SerializePreTensorToStream(std::ostream &os,
                                  const std::vector<int64_t> &dims) {
    {  // the 1st field, uint32_t version
      constexpr uint32_t version = 0;
      os.write(reinterpret_cast<const char *>(&version), sizeof(version));
    }
    {  // the 2nd field, tensor description
      // int32_t  size
      framework::proto::VarType::TensorDesc desc;
      desc.set_data_type(framework::proto::VarType::FP32);
      auto *pb_dims = desc.mutable_dims();
      pb_dims->Resize(static_cast<int>(dims.size()), 0);
      std::copy(dims.begin(), dims.end(), pb_dims->begin());
      int32_t size = desc.ByteSize();
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      auto out = desc.SerializeAsString();
      os.write(out.data(), size);
    }
  }

  void SerializeValueToVec(std::ifstream &in, const int batch,
                           const int embedding_dim, std::vector<float> *out) {
    auto queue =
        std::make_shared<framework::BlockingQueue<std::vector<std::string>>>();

    auto read = [batch, &in, &queue]() {
      std::string line;
      std::vector<std::string> columns;
      std::vector<std::string> values_str;

      int count = 0;

      while (std::getline(in, line)) {
        ++count;
        columns = string::Split(line, '\t');

        if (columns.size() != 5) {
          VLOG(0) << "unexpected line: " << line << ", skip it";
          continue;
        }

        values_str = string::Split(columns[4], ',');
        queue->Push(values_str);

        if (count >= batch) {
          break;
        }
      }
      queue->Push({});
    };

    auto write = [embedding_dim, &out, &queue]() {
      std::vector<std::string> values_str;
      std::string line;

      while (true) {
        queue->Pop(&values_str);

        if (values_str.size() == 0) {
          break;
        }

        for (int x = 0; x < embedding_dim; ++x) {
          float v = 0.0;
          try {
            v = lexical_cast<float>(values_str[x]);
          } catch (boost::bad_lexical_cast &e) {
            VLOG(0) << " get unexpected line: " << line;
          }
          out->push_back(v);
        }
      }
    };

    std::thread p_read(read);
    std::thread p_write(write);
    p_read.join();
    p_write.join();
  }

  void SerializeVecToStream(std::ostream &out,
                            const std::vector<float> &value) {
    out.write(reinterpret_cast<const char *>(value.data()),
              static_cast<std::streamsize>(sizeof(float) * value.size()));
  }

  void SerializeValueToStream(
      std::ostream &out, const std::vector<std::string> &ins,
      const std::vector<std::vector<int>> &batch_buckets,
      const int embedding_dim) {
    std::vector<std::shared_ptr<std::ifstream>> in_streams;

    for (int x = 0; x < ins.size(); ++x) {
      in_streams.emplace_back(std::make_shared<std::ifstream>(ins[x]));
    }

    std::vector<std::future<int>> tasks(ins.size());

    for (int b = 0; b < BUCKET; ++b) {
      std::vector<std::vector<float>> values;
      values.resize(tasks.size());

      auto begin = GetCurrentUS();

      for (int x = 0; x < tasks.size(); ++x) {
        auto batch = batch_buckets[x][b + 1] - batch_buckets[x][b];
        values[x].clear();
        values[x].reserve(batch * embedding_dim);
      }

      for (int x = 0; x < tasks.size(); ++x) {
        tasks[x] =
            pool_->enqueue([this, b, x, &out, &in_streams, &batch_buckets,
                            &values, embedding_dim]() -> int {
              auto batch = batch_buckets[x][b + 1] - batch_buckets[x][b];
              if (batch == 0) return 0;
              SerializeValueToVec(*(in_streams[x].get()), batch, embedding_dim,
                                  &values[x]);
              return 0;
            });
      }

      for (size_t x = 0; x < tasks.size(); ++x) {
        tasks[x].wait();
      }

      auto end = GetCurrentUS();

      auto begin1 = GetCurrentUS();
      for (size_t x = 0; x < tasks.size(); ++x) {
        SerializeVecToStream(out, values[x]);
      }
      auto end1 = GetCurrentUS();

      VLOG(0) << "serialize buckets " << b << " read using " << end - begin
              << ", to oostream using " << end1 - begin1;
    }
  }

  void DeserializeRowsFromFile(const std::string &input_file,
                               std::vector<int64_t> *rows) {
    std::string line;
    std::vector<std::string> columns;
    std::ifstream file(input_file);

    while (std::getline(file, line)) {
      columns = string::Split(line, '\t');
      if (columns.size() != 5) {
        VLOG(0) << "unexpected line: " << line << ", skip it";
        continue;
      }
      rows->push_back(std::stoull(columns[0]));
    }

    VLOG(0) << "parse " << rows->size() << " embedding rows from "
            << input_file;
  }

 private:
  std::unique_ptr<::ThreadPool> pool_;
};
}  // namespace distributed
}  // namespace paddle
