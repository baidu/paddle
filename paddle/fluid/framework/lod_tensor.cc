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

#include "paddle/fluid/framework/lod_tensor.h"

#include <stdint.h>

#include "paddle/fluid/framework/version.h"

namespace paddle {
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

std::ostream &operator<<(std::ostream &os, const LoD &lod) {
  os << "{";
  for (auto &v : lod) {
    os << "{";
    bool is_first = true;
    for (auto &i : v) {
      if (is_first) {
        os << i;
        is_first = false;
      } else {
        os << ", " << i;
      }
    }
    os << "}";
  }
  os << "}";

  return os;
}

std::ostream &operator<<(std::ostream &os, const LoDTensor &t) {
  if (t.lod().size() > 0) {
    os << "  - lod: " << t.lod() << "\n";
  }
  os << static_cast<Tensor>(t);
  return os;
}

std::string LoDToString(const LoD &lod) {
  std::ostringstream stream;
  stream << lod;
  return stream.str();
}

LoD SliceInLevel(const LoD &in, size_t level, size_t elem_begin,
                 size_t elem_end) {
  PADDLE_ENFORCE_LT(level, in.size(),
                    platform::errors::InvalidArgument(
                        "The input LoDTensor's lod level should be less than "
                        "the LoD size, but received level is %d, LoD is %s.",
                        level, in));
  PADDLE_ENFORCE_LT(
      elem_begin, elem_end,
      platform::errors::InvalidArgument(
          "The index to start slicing should be less than the index to end "
          "slicing, but received start index is %d, end index is %d.",
          elem_begin, elem_end));
  PADDLE_ENFORCE_LT(
      elem_end, in[level].size(),
      platform::errors::InvalidArgument(
          "The index to end slicing should be less than the input LoD size, "
          "but received end index is %d, LoD size is %d.",
          elem_end, in[level].size()));

  LoD res;
  res.resize(in.size() - level);
  // copy the first level
  res[0].assign(in[level].begin() + elem_begin,
                in[level].begin() + elem_end + 1);
  for (size_t lvl = 1; lvl < res.size(); lvl++) {
    const auto &in_level = in[level + lvl];
    const auto &above_level = res[lvl - 1];
    auto &out_level = res[lvl];
    out_level.assign(in_level.begin() + above_level.front(),
                     in_level.begin() + above_level.back() + 1);
  }
  for (size_t lvl = 0; lvl < res.size(); lvl++) {
    // to make the first offset equals 0, all the elements minus the first
    // element
    size_t front = res[lvl].front();
    for (auto &ele : res[lvl]) {
      ele -= front;
    }
  }
  return res;
}

LoD ToAbsOffset(const LoD &in) {
  // the lowest level stores relative offsets
  if (in.empty() || in.size() == 1) return in;
  LoD result = in;
  for (auto level = static_cast<int>(in.size() - 2); level >= 0; level--) {
    for (size_t i = 0; i < in[level].size(); ++i) {
      size_t index = in[level][i];
      result[level][i] = result[level + 1][index];
    }
  }
  return result;
}

bool operator==(const LoD &a, const LoD &b) {
  if (a.size() != b.size()) {
    return false;
  }

  for (size_t i = 0; i < a.size(); i++) {
    const auto &a_level = a[i];
    const auto &b_level = b[i];
    if (a_level.size() != b_level.size()) {
      return false;
    }
    for (size_t j = 0; j < a_level.size(); j++) {
      if (a_level[j] != b_level[j]) {
        return false;
      }
    }
  }
  return true;
}

bool CheckLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;
    // check: the first offset(the begin offset) of each level should be 0.
    if (level.front() != 0) return false;
    // check: all the offsets in a level should be non-descending
    if (!std::is_sorted(level.begin(), level.end())) {
      return false;
    }
  }
  // check: the lowest level's last offset should equals `tensor_height` if
  //        tensor_height>0.
  if (tensor_height > 0 && (size_t)tensor_height != in.back().back())
    return false;

  // check: the higher level's last offset should equals the lower level's
  // size-1.
  // NOTE LoD store the levels from top to bottom, so the higher level goes
  // first.
  for (size_t level = 0; level < in.size() - 1; level++) {
    if (in[level].back() != in[level + 1].size() - 1) return false;
  }
  return true;
}

bool CheckAbsLoD(const LoD &in, int tensor_height) {
  if (in.empty()) return true;
  for (const auto &level : in) {
    // check: all the offsets in a level should be ascending(no same items
    // allowed).
    if (!std::is_sorted(level.begin(), level.begin(), [](size_t a, size_t b) {
          if (a < b) return true;
          return false;
        })) {
      return false;
    }

    // check: there should be more than 2 offsets existing in each level.
    if (level.size() < 2) return false;

    // check: the first offset of each level should be 0, and the last should be
    // the same(the height of underlying tensor).
    if (level.front() != 0) return false;
    if (tensor_height < 0) {
      tensor_height = level.back();
    } else if ((size_t)tensor_height != level.back()) {
      return false;
    }
  }
  return true;
}

using LoDAndOffset = std::pair<LoD, std::pair<size_t, size_t>>;
LoDAndOffset GetSubLoDAndAbsoluteOffset(const LoD &lod, size_t start_idx,
                                        size_t end_idx, size_t start_level) {
  LoD sub_lod;

  for (size_t level_idx = start_level; level_idx < lod.size(); ++level_idx) {
    PADDLE_ENFORCE_LE(start_idx, end_idx,
                      platform::errors::InvalidArgument(
                          "The start index should be less than the end index, "
                          "but received start index is %d, end index is %d.",
                          start_idx, end_idx));
    PADDLE_ENFORCE_LT(
        end_idx, lod[level_idx].size(),
        platform::errors::InvalidArgument(
            "The end index should be less than the LoD level size, but "
            "received end index is %d, LoD level size is %d.",
            end_idx, lod[level_idx].size()));
    std::vector<size_t> level_lens;
    for (size_t i = start_idx; i < end_idx; ++i) {
      level_lens.push_back(lod[level_idx][i + 1] - lod[level_idx][i]);
    }
    sub_lod.emplace_back(level_lens);
    start_idx = lod[level_idx][start_idx];
    end_idx = lod[level_idx][end_idx];
  }

  return LoDAndOffset{sub_lod, {start_idx, end_idx}};
}

void AppendLoD(LoD *lod, const LoD &lod_length) {
  PADDLE_ENFORCE(
      lod->empty() || lod->size() == lod_length.size(),
      platform::errors::InvalidArgument(
          "The input LoD length should be equal to the appended LoD size, but "
          "received input LoD length is %d, actual LoD size is %d.",
          lod_length, lod->size()));
  if (lod->empty()) {
    for (size_t i = 0; i < lod_length.size(); ++i) {
      lod->emplace_back(1, 0);  // size = 1, value = 0;
    }
    *lod = LoD(lod_length.size(), std::vector<size_t>({0}));
  }
  for (size_t i = 0; i < lod->size(); ++i) {
    auto &level = (*lod)[i];
    for (size_t len : lod_length[i]) {
      level.push_back(level.back() + len);
    }
  }
}

void SerializeToStream(std::ostream &os, const LoDTensor &tensor,
                       const platform::DeviceContext &dev_ctx) {
  {  // the 1st field, uint32_t version for LoDTensor
    os.write(reinterpret_cast<const char *>(&kCurTensorVersion),
             sizeof(kCurTensorVersion));
  }
  {
    // the 2st field, LoD information
    // uint64_t lod_level
    // uint64_t lod_level_1 size in byte.
    // int*     lod_level_1 data
    // ...
    auto lod = tensor.lod();
    uint64_t size = lod.size();
    os.write(reinterpret_cast<const char *>(&size), sizeof(size));

    for (auto &each : lod) {
      size = each.size() * sizeof(framework::LoD::value_type::value_type);
      os.write(reinterpret_cast<const char *>(&size), sizeof(size));
      os.write(reinterpret_cast<const char *>(each.data()),
               static_cast<std::streamsize>(size));
    }
  }
  // the 3st field, Tensor
  TensorToStream(os, static_cast<Tensor>(tensor), dev_ctx);
}

void SerializeToStream(std::ostream &os, const LoDTensor &tensor) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext *dev_ctx;
  auto place = tensor.place();
  dev_ctx = pool.Get(place);
  SerializeToStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &os, LoDTensor *tensor) {
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext *dev_ctx;
  dev_ctx = pool.Get(platform::CPUPlace());
  DeserializeFromStream(os, tensor, *dev_ctx);
}

void DeserializeFromStream(std::istream &is, LoDTensor *tensor,
                           const platform::DeviceContext &dev_ctx,
                           const size_t &seek,
                           const std::vector<int64_t> &shape) {
  {
    // the 1st field, unit32_t version for LoDTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(framework::IsTensorVersionSupported(version), true,
                      platform::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version, 0U,
        platform::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
  }
  // the 3st filed, Tensor
  TensorFromStream(is, static_cast<Tensor *>(tensor), dev_ctx, seek, shape);
}

void DeserializeFromStream(std::istream &is, LoDTensor *tensor,
                           const platform::DeviceContext &dev_ctx) {
  {
    // the 1st field, unit32_t version for LoDTensor
    uint32_t version;
    is.read(reinterpret_cast<char *>(&version), sizeof(version));
    PADDLE_ENFORCE_EQ(framework::IsTensorVersionSupported(version), true,
                      platform::errors::InvalidArgument(
                          "Tensor version %u is not supported.", version));
    PADDLE_ENFORCE_EQ(
        version, 0U,
        platform::errors::InvalidArgument(
            "Deserialize to tensor failed, maybe the loaded file is "
            "not a paddle model(expected file format: 0, but %u found).",
            version));
  }
  {
    // the 2st field, LoD information
    uint64_t lod_level;
    is.read(reinterpret_cast<char *>(&lod_level), sizeof(lod_level));
    auto &lod = *tensor->mutable_lod();
    lod.resize(lod_level);
    for (uint64_t i = 0; i < lod_level; ++i) {
      uint64_t size;
      is.read(reinterpret_cast<char *>(&size), sizeof(size));
      std::vector<size_t> tmp(size / sizeof(size_t));
      is.read(reinterpret_cast<char *>(tmp.data()),
              static_cast<std::streamsize>(size));
      lod[i] = tmp;
    }
  }
  // the 3st filed, Tensor
  TensorFromStream(is, static_cast<Tensor *>(tensor), dev_ctx);
}

std::vector<LoDTensor> LoDTensor::SplitLoDTensor(
    const std::vector<platform::Place> places) const {
  PADDLE_ENFORCE_GT(places.size(), 0,
                    platform::errors::InvalidArgument(
                        "Place number cannot be empty when splitting."));
  check_memory_size();
  size_t batch_size =
      lod().empty() ? static_cast<size_t>(dims()[0]) : lod()[0].size() - 1;

  // if batch_size is 0, just return #places.size() copys of empty
  // tensors.
  if (batch_size == 0) {
    std::vector<LoDTensor> empty_results;
    empty_results.reserve(places.size());
    for (size_t i = 0; i < places.size(); ++i) {
      LoDTensor dst;
      dst.Resize(dims());
      dst.mutable_data(places[i], type());
      if (!lod().empty()) {
        dst.set_lod(lod());
      }
      empty_results.emplace_back(std::move(dst));
    }
    return empty_results;
  }

  auto step_width = (batch_size + places.size() - 1) / places.size();
  auto result_size = (batch_size + step_width - 1) / step_width;
  std::vector<LoDTensor> results;
  results.reserve(result_size);

  for (size_t i = 0; i < result_size; ++i) {
    auto begin = i * step_width;
    auto end = std::min<size_t>((i + 1) * step_width, batch_size);
    PADDLE_ENFORCE_LT(begin, end,
                      platform::errors::InvalidArgument(
                          "The begin index must be less than the end index, "
                          "but received begin index is %d, end index is %d.",
                          begin, end));

    LoDTensor dst;
    if (lod().empty()) {
      auto src = Slice(begin, end);
      auto &dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);
    } else {
      auto lod_and_offset = GetSubLoDAndAbsoluteOffset(lod(), begin, end, 0);

      auto &offset = lod_and_offset.second;
      auto src = Slice(offset.first, offset.second);
      auto &dst_place = places[i];
      framework::TensorCopy(src, dst_place, &dst);

      LoD my_lod;
      for (auto &l : lod_and_offset.first) {
        std::vector<size_t> v{0};
        for (auto &ll : l) {
          v.push_back(ll + v.back());
        }
        my_lod.emplace_back(v);
      }
      dst.set_lod(my_lod);
    }
    results.emplace_back(std::move(dst));
  }

  return results;
}

void LoDTensor::MergeLoDTensor(
    const std::vector<const LoDTensor *> &lod_tensors,
    platform::Place dst_place) {
  PADDLE_ENFORCE_EQ(lod_tensors.empty(), false,
                    platform::errors::InvalidArgument(
                        "The LoDTensors to be merged are empty."));

  framework::DDim new_dim = lod_tensors[0]->dims();
  proto::VarType::Type new_type = proto::VarType::FP32;
  framework::DataLayout new_layout = lod_tensors[0]->layout();
  for (auto *t : lod_tensors) {
    if (t->numel() && t->IsInitialized()) {
      new_dim = t->dims();
      new_type = t->type();
      new_layout = t->layout();
      break;
    }
  }

  LoD new_lod = lod_tensors[0]->lod();

  for (size_t i = 1; i < lod_tensors.size(); ++i) {
    auto *t = lod_tensors[i];
    if (t->numel() && t->IsInitialized()) {
      PADDLE_ENFORCE_EQ(
          new_type, t->type(),
          platform::errors::InvalidArgument(
              "LoDTensor data type does not match, expected type is %s, actual "
              "type is %s.",
              DataTypeToString(new_type), DataTypeToString(t->type())));
      PADDLE_ENFORCE_EQ(
          new_layout, t->layout(),
          platform::errors::InvalidArgument(
              "LoDTensor layout does not match, expected layout is %s, "
              "actual layout is %s.",
              DataLayoutToString(new_layout), DataLayoutToString(t->layout())));
      PADDLE_ENFORCE_EQ(
          framework::product(new_dim) / new_dim[0],
          framework::product(t->dims()) / t->dims()[0],
          platform::errors::InvalidArgument(
              "LoDTensor dimension does not match, all dimensions except the "
              "first dimension need to be equal,"
              "but expected dimension is %s, actual dimension is %s.",
              new_dim, t->dims()));
      new_dim[0] += t->dims()[0];
    }

    auto &lod = t->lod();
    PADDLE_ENFORCE_EQ(new_lod.size(), lod.size(),
                      platform::errors::InvalidArgument(
                          "The LoD information of LoDTensor does not match, "
                          "expected LoD is %s, actual LoD is %s.",
                          new_lod, lod));
    for (size_t j = 0; j < lod.size(); ++j) {
      auto &sub_lod = new_lod[j];
      size_t offset = sub_lod.back();
      for (size_t k = 1; k < lod[j].size(); ++k) {
        sub_lod.push_back(lod[j][k] + offset);
      }
    }
  }
  Resize(new_dim);
  set_layout(new_layout);
  set_lod(new_lod);
  mutable_data(dst_place, new_type);

  int begin = 0;
  for (auto *src : lod_tensors) {
    int end = begin + src->dims()[0];
    if (end == begin) {
      continue;
    }
    auto dst = Slice(begin, end);
    framework::TensorCopy(*src, dst_place, &dst);
    begin = end;
  }
}

LoD ConvertToLengthBasedLoD(const LoD &offset_lod) {
  LoD length_lod;
  length_lod.reserve(offset_lod.size());
  for (size_t lvl = 0; lvl < offset_lod.size(); ++lvl) {
    std::vector<size_t> level;
    if (offset_lod[lvl].size() > 0) {
      level.reserve(offset_lod[lvl].size() - 1);
    }
    for (size_t idx = 0; idx < offset_lod[lvl].size() - 1; ++idx) {
      level.push_back(offset_lod[lvl][idx + 1] - offset_lod[lvl][idx]);
    }
    length_lod.push_back(level);
  }
  return length_lod;
}

LoD ConvertToOffsetBasedLoD(const LoD &length_lod) {
  LoD offset_lod;
  offset_lod.reserve(length_lod.size());
  for (size_t lvl = 0; lvl < length_lod.size(); ++lvl) {
    std::vector<size_t> level;
    level.reserve(length_lod[lvl].size() + 1);
    size_t tmp = 0;
    level.push_back(tmp);
    for (size_t idx = 0; idx < length_lod[lvl].size(); ++idx) {
      tmp += length_lod[lvl][idx];
      level.push_back(tmp);
    }
    offset_lod.push_back(level);
  }
  return offset_lod;
}

}  // namespace framework
}  // namespace paddle
