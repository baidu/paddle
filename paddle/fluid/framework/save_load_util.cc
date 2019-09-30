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

#include "paddle/fluid/framework/save_load_util.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>

#include "paddle/fluid/imperative/layer.h"

namespace paddle {
namespace framework {

const int model_file_reserve_size = 256;
const std::string tensor_number_mark = "TNUM";  // NOLINT
const std::string tensor_name_mark = "NAME";    // NOLINT

void CheckInStreamState(std::istream& istre, size_t length) {
  if (!istre) {
    VLOG(5) << "Can't read [" << length << "] from file"
            << "file seems breakem";

    PADDLE_THROW("Model load error, file seems breaken");
  }
}

struct DeserializedDataFunctor {
  DeserializedDataFunctor(void** buf, Tensor* tensor,
                          const platform::Place& place)
      : buf_(buf), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *buf_ = tensor_->mutable_data<T>(place_);
  }

  void** buf_;
  Tensor* tensor_;
  platform::Place place_;
};

size_t ReadTensorNumber(std::istream& istre) {
  char* tensor_number_mark_buffer = new char[tensor_number_mark.size()];
  istre.read(tensor_number_mark_buffer,
             sizeof(char) * tensor_number_mark.size());
  std::string str_read_tensor_number_mark(tensor_number_mark_buffer,
                                          tensor_number_mark.size());
  PADDLE_ENFORCE_EQ(
      tensor_number_mark, str_read_tensor_number_mark,
      "Tensor number mark not match, expect [%s], but read from file is [%]",
      tensor_number_mark, str_read_tensor_number_mark);

  size_t tensor_number = 0;
  istre.read(reinterpret_cast<char*>(&tensor_number), sizeof(tensor_number));

  CheckInStreamState(istre, sizeof(tensor_number));

  delete[] tensor_number_mark_buffer;
  return tensor_number;
}

std::string ReadTensorName(std::istream& istre) {
  char* name_mark_buffer = new char[tensor_name_mark.size()];
  istre.read(name_mark_buffer, sizeof(char) * tensor_name_mark.size());
  CheckInStreamState(istre, sizeof(char) * tensor_name_mark.size());

  std::string str_read_tensor_name_mark(name_mark_buffer,
                                        tensor_name_mark.size());
  PADDLE_ENFORCE_EQ(
      tensor_name_mark, str_read_tensor_name_mark,
      "Tensor name mark not match, expect [%s], but read from file is [%]",
      tensor_name_mark, str_read_tensor_name_mark);

  size_t tensor_name_length = 0;
  istre.read(reinterpret_cast<char*>(&tensor_name_length),
             sizeof(tensor_name_length));

  CheckInStreamState(istre, sizeof(tensor_name_length));

  char* tensor_name_buffer = new char[tensor_name_length];
  istre.read(tensor_name_buffer, sizeof(char) * tensor_name_length);
  CheckInStreamState(istre, sizeof(char) * tensor_name_length);

  std::string str_tensor_name(tensor_name_buffer, tensor_name_length);

  delete[] name_mark_buffer;
  delete[] tensor_name_buffer;

  return str_tensor_name;
}

void ReadReserveBuffer(std::istream& istre) {
  char* reserve_buffer = new char[model_file_reserve_size];
  istre.read(reserve_buffer, sizeof(char) * model_file_reserve_size);
  CheckInStreamState(istre, model_file_reserve_size);

  delete[] reserve_buffer;
}

bool SaveStaticNameListToDisk(
    const std::string& file_name,
    const std::vector<std::string>& vec_tensor_name_list, const Scope& scope) {
  std::map<std::string, Tensor*> map_tensor;

  for (size_t i = 0; i < vec_tensor_name_list.size(); ++i) {
    auto var_ptr = scope.FindVar(vec_tensor_name_list[i]);
    PADDLE_ENFORCE_NE(
        var_ptr, nullptr,
        "Variable find error, when save model, can't not find vairable [%s]",
        vec_tensor_name_list[i]);
    Tensor* tensor = var_ptr->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_EQ(tensor->IsInitialized(), true,
                      "Paramter [%s] not initialzed,"
                      "Please make sure you have run StartUpProgram",
                      vec_tensor_name_list[i]);

    map_tensor[vec_tensor_name_list[i]] = tensor;
  }

  return SaveTensorToDisk(file_name, map_tensor);
}

bool SaveDygraphVarBaseListToDisk(
    const std::string& file_name,
    const std::vector<std::shared_ptr<imperative::VarBase>>&
        vec_var_base_list) {
  std::map<std::string, Tensor*> map_tensor;
  for (size_t i = 0; i < vec_var_base_list.size(); ++i) {
    auto var_ptr = vec_var_base_list[i]->MutableVar();

    Tensor* tensor = var_ptr->GetMutable<LoDTensor>();

    PADDLE_ENFORCE_EQ(tensor->IsInitialized(), true,
                      "Paramter [%s] not initialzed,"
                      "Please make sure you have run StartUpProgram",
                      vec_var_base_list[i]->Name());

    map_tensor[vec_var_base_list[i]->Name()] = tensor;
  }

  return SaveTensorToDisk(file_name, map_tensor);
}

const std::vector<std::shared_ptr<imperative::VarBase>>
LoadDygraphVarBaseListFromDisk(const std::string& file_name) {
  std::map<std::string, std::shared_ptr<Tensor>> map_load_tensor;
  LoadTensorFromDisk(file_name, map_load_tensor);

  std::vector<std::shared_ptr<imperative::VarBase>> vec_res;
  vec_res.reserve(map_load_tensor.size());
  for (auto& load_tensor : map_load_tensor) {
    std::shared_ptr<imperative::VarBase> var(
        new imperative::VarBase(load_tensor.first));

    auto* tensor = var->MutableVar()->GetMutable<framework::LoDTensor>();

    TensorCopySync(*(load_tensor.second.get()), load_tensor.second->place(),
                   tensor);

    vec_res.emplace_back(var);
  }

  return vec_res;
}

bool LoadStaticNameListFromDisk(
    const std::string& file_name,
    const std::vector<std::string>& vec_tensor_name_list, const Scope& scope) {
  std::map<std::string, std::shared_ptr<Tensor>> map_load_tensor;
  LoadTensorFromDisk(file_name, map_load_tensor);

  for (size_t i = 0; i < vec_tensor_name_list.size(); ++i) {
    auto it = map_load_tensor.find(vec_tensor_name_list[i]);
    PADDLE_ENFORCE(it != map_load_tensor.end(),
                   "Paramete not found in Model file, "
                   "Can not find [%s] in model file [%s]",
                   vec_tensor_name_list[i], file_name);
    auto var_ptr = scope.FindVar(vec_tensor_name_list[i]);

    PADDLE_ENFORCE_NE(
        var_ptr, nullptr,
        "Parameter not created, when load model, can't not find parameter [%s]",
        vec_tensor_name_list[i]);

    Tensor* tensor = var_ptr->GetMutable<LoDTensor>();
    PADDLE_ENFORCE_NE(tensor, nullptr,
                      "Paramter [%s] not initialzed "
                      "please make sure you have run startUpProgram",
                      vec_tensor_name_list[i]);

    PADDLE_ENFORCE_EQ(tensor->IsInitialized(), true,
                      "Paramter [%s] not initialzed "
                      "please make sure you have run StartUpProgram",
                      vec_tensor_name_list[i]);
    PADDLE_ENFORCE_EQ(
        tensor->dims(), it->second->dims(),
        "Shape not matching: the Program requires a parameter with a shape of "
        "(%s), "
        "while the loaded parameter (namely [ %s ]) has a shape of  (%s).",
        tensor->dims(), vec_tensor_name_list[i], it->second->dims());

    TensorCopySync(*(it->second.get()), tensor->place(), tensor);

    map_load_tensor.erase(it);
  }

  if (map_load_tensor.size() != 0) {
    std::string used_tensor_message = "There is [" +
                                      std::to_string(map_load_tensor.size()) +
                                      "] tensor in model file not used: ";

    for (auto& tensor_temp : map_load_tensor) {
      used_tensor_message += " " + tensor_temp.first;
    }

    LOG(WARNING) << used_tensor_message;
  }

  return true;
}

bool SaveTensorToDisk(const std::string& file_name,
                      const std::map<std::string, Tensor*>& map_tensor) {
  MkDirRecursively(DirName(file_name).c_str());

  std::ofstream fout(file_name, std::ios::binary);
  if (!fout) {
    PADDLE_THROW("File open error. Can not open file [%s]", file_name);
  }

  // first 256 byte for reserve for fulture upgrade
  char kReserveBuffer = new char[model_file_reserve_size];
  fout.write(kReserveBuffer, sizeof(char) * model_file_reserve_size);
  delete[] kReserveBuffer;

  fout.write(tensor_number_mark.c_str(),
             sizeof(char) * tensor_number_mark.size());
  size_t tensor_number = map_tensor.size();
  fout.write(reinterpret_cast<const char*>(&tensor_number),
             sizeof(tensor_number));

  for (auto& itera : map_tensor) {
    // first save tensor name
    fout.write(tensor_name_mark.c_str(),
               sizeof(char) * tensor_name_mark.size());
    size_t name_length = itera.first.size();
    fout.write(reinterpret_cast<const char*>(&name_length),
               sizeof(name_length));
    fout.write(itera.first.c_str(), sizeof(char) * name_length);
    // write tensor version
    constexpr uint32_t version = 0;
    fout.write(reinterpret_cast<const char*>(&version), sizeof(version));

    // the 2nd field, tensor description
    // int32_t  size
    // void*    protobuf message
    auto tensor = itera.second;

    proto::VarType::TensorDesc desc;
    desc.set_data_type(tensor->type());
    auto dims = framework::vectorize(tensor->dims());
    auto* pb_dims = desc.mutable_dims();
    pb_dims->Resize(static_cast<int>(dims.size()), 0);
    std::copy(dims.begin(), dims.end(), pb_dims->begin());
    int32_t size = desc.ByteSize();
    fout.write(reinterpret_cast<const char*>(&size), sizeof(size));
    auto out = desc.SerializeAsString();
    fout.write(out.data(), size);

    // save tensor
    uint64_t data_size =
        tensor->numel() * framework::SizeOfType(tensor->type());
    auto* data_ptr = tensor->data<void>();
    if (platform::is_gpu_place(tensor->place())) {
#ifdef PADDLE_WITH_CUDA
      framework::Tensor temp;
      TensorCopySync(*tensor, platform::CPUPlace(), &temp);
      data_ptr = temp.data<void>();
    }
#else
      PADDLE_THROW(
          "Tensor is in CUDA device, but paddle not compile with CUDA, this "
          "shoul'd not happen");
#endif
    fout.write(static_cast<const char*>(data_ptr),
               static_cast<std::streamsize>(data_size));
  }

  if (!fout) {
    PADDLE_THROW("Model save failed, data write to model file [%s] error",
                 file_name);
  }

  fout.close();

  return true;
}

bool LoadTensorFromDisk(
    const std::string& file_name,
    std::map<std::string, std::shared_ptr<Tensor>>& map_tensor) {  // NOLINT
  std::ifstream fin(file_name, std::ios::binary);

  if (!fin) {
    PADDLE_THROW("File open error. Can not open model file [%s]", file_name);
  }

  ReadReserveBuffer(fin);

  size_t tensor_number = ReadTensorNumber(fin);
  PADDLE_ENFORCE_GT(tensor_number, 0U,
                    "Tensor number must great than 0, file [%s] seems breaken",
                    file_name);

  for (size_t i = 0; i < tensor_number; ++i) {
    std::string str_tensor_name = ReadTensorName(fin);

    std::shared_ptr<Tensor> tensor_temp(new Tensor());
    uint32_t version;
    fin.read(reinterpret_cast<char*>(&version), sizeof(version));
    CheckInStreamState(fin, sizeof(version));
    PADDLE_ENFORCE_EQ(version, 0U, "Only version 0 is supported");
    proto::VarType::TensorDesc desc;
    {
      // int32_t size
      // proto buffer
      int32_t size;
      fin.read(reinterpret_cast<char*>(&size), sizeof(size));
      CheckInStreamState(fin, sizeof(size));
      std::unique_ptr<char[]> buf(new char[size]);
      fin.read(reinterpret_cast<char*>(buf.get()), size);
      CheckInStreamState(fin, sizeof(size));
      PADDLE_ENFORCE(desc.ParseFromArray(buf.get(), size),
                     "Cannot parse tensor desc");
    }

    {  // read tensor
      std::vector<int64_t> dims;
      dims.reserve(static_cast<size_t>(desc.dims().size()));
      std::copy(desc.dims().begin(), desc.dims().end(),
                std::back_inserter(dims));
      auto new_dim = framework::make_ddim(dims);
      tensor_temp->Resize(new_dim);
      float* buf = tensor_temp->mutable_data<float>(platform::CPUPlace());
      size_t size =
          tensor_temp->numel() * framework::SizeOfType(desc.data_type());

      fin.read(reinterpret_cast<char*>(buf), size);
      CheckInStreamState(fin, size);
    }

    map_tensor[str_tensor_name] = tensor_temp;
  }

  return true;
}

}  // namespace framework
}  // namespace paddle
