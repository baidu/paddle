/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/io.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace py = pybind11;
namespace paddle {
namespace pybind {

void BindIO(pybind11::module *m) {
  m->def("save_lod_tensor", [](const paddle::framework::LoDTensor &tensor,
                               const std::string &str_file_name) {
    std::ofstream fout(str_file_name, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                      platform::errors::Unavailable(
                          "Cannot open %s to save variables.", str_file_name));
    paddle::framework::SerializeToStream(fout, tensor);

    int64_t tellp = fout.tellp();
    fout.close();
    return tellp;
  });

  m->def("load_lod_tensor", [](paddle::framework::LoDTensor &tensor,
                               const std::string &str_file_name) {
    std::ifstream fin(str_file_name, std::ios::binary);
    PADDLE_ENFORCE_EQ(static_cast<bool>(fin), true,
                      platform::errors::Unavailable(
                          "Cannot open %s to load variables.", str_file_name));

    paddle::framework::DeserializeFromStream(fin, &tensor);
    int64_t tellg = fin.tellg();
    fin.close();
    return tellg;
  });

  m->def("save_selected_rows",
         [](const paddle::framework::SelectedRows &selected_rows,
            const std::string &str_file_name) {
           std::ofstream fout(str_file_name, std::ios::binary);
           PADDLE_ENFORCE_EQ(
               static_cast<bool>(fout), true,
               platform::errors::Unavailable(
                   "Cannot open %s to save SelectedRows.", str_file_name));

           paddle::framework::SerializeToStream(fout, selected_rows);
           int64_t tellp = fout.tellp();
           fout.close();
           return tellp;
         });

  m->def("load_selected_rows",
         [](paddle::framework::SelectedRows &selected_rows,
            const std::string &str_file_name) {
           std::ifstream fin(str_file_name, std::ios::binary);
           PADDLE_ENFORCE_EQ(
               static_cast<bool>(fin), true,
               platform::errors::Unavailable(
                   "Cannot open %s to load SelectedRows.", str_file_name));

           paddle::framework::DeserializeFromStream(fin, &selected_rows);
           int64_t tellg = fin.tellg();
           fin.close();
           return tellg;
         });

  m->def("save_lod_tensor_to_memory",
         [](const paddle::framework::LoDTensor &tensor) -> py::bytes {
           std::ostringstream ss;
           paddle::framework::SerializeToStream(ss, tensor);
           return ss.str();
         });

  m->def("load_lod_tensor_from_memory", [](paddle::framework::LoDTensor &tensor,
                                           const std::string &tensor_bytes) {
    std::istringstream fin(tensor_bytes, std::ios::in | std::ios::binary);
    paddle::framework::DeserializeFromStream(fin, &tensor);
  });

  m->def("save_selected_rows_to_memory",
         [](const paddle::framework::SelectedRows &selected_rows) -> py::bytes {
           std::ostringstream ss;
           paddle::framework::SerializeToStream(ss, selected_rows);
           return ss.str();
         });

  m->def("load_selected_rows_from_memory",
         [](paddle::framework::SelectedRows &selected_rows,
            const std::string &selected_rows_bytes) {
           std::istringstream fin(selected_rows_bytes,
                                  std::ios::in | std::ios::binary);
           paddle::framework::DeserializeFromStream(fin, &selected_rows);
         });
}
}  // namespace pybind
}  // namespace paddle
