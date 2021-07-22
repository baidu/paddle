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
#pragma once

namespace paddle {
namespace operators {

///
/// @brief      Helper function to execute AXPY using oneDNN.
///
/// @param[in]  n      The number of elements in tensor (assumed 1D)
/// @param[in]  alpha  The alpha coefficient.
/// @param[in]  x      The pointer to input X tensor.
/// @param      y      The pointer to output Y tensor.
///
/// @tparam     T      Data type.
///
template <typename T>
void onednn_handler_axpy(int n, float alpha, const T *x, T *y);

}  // namespace operators
}  // namespace paddle
