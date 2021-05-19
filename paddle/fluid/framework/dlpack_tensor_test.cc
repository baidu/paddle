// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/dlpack_tensor.h"
#include <glog/logging.h>
#include <gtest/gtest.h>

namespace paddle {
namespace platform {
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace framework {

namespace {  // NOLINT
template <typename T>
constexpr uint8_t GetDLDataTypeCode() {
  if (std::is_same<T, platform::complex<float>>::value ||
      std::is_same<T, platform::complex<double>>::value ||
      std::is_same<T, platform::complex64>::value ||
      std::is_same<T, platform::complex128>::value) {
    return static_cast<uint8_t>(5);
  }

  return std::is_same<platform::float16, T>::value ||
                 std::is_floating_point<T>::value
             ? static_cast<uint8_t>(kDLFloat)
             : (std::is_unsigned<T>::value
                    ? static_cast<uint8_t>(kDLUInt)
                    : (std::is_integral<T>::value ? static_cast<uint8_t>(kDLInt)
                                                  : static_cast<uint8_t>(-1)));
}
}  // NOLINT

template <typename T>
void TestMain(const platform::Place &place, uint16_t lanes) {
  DDim dims{4, 5, 6, 7};
  Tensor tensor;
  tensor.Resize(dims);
  void *p = tensor.mutable_data<T>(place);

  DLPackTensor dlpack_tensor(tensor, lanes);
  ::DLTensor &dl_tensor = dlpack_tensor;

  CHECK_EQ(p, dl_tensor.data);
  if (platform::is_cpu_place(place)) {
    CHECK_EQ(kDLCPU, dl_tensor.ctx.device_type);
    CHECK_EQ(0, dl_tensor.ctx.device_id);
  } else if (platform::is_gpu_place(place)) {
    CHECK_EQ(kDLGPU, dl_tensor.ctx.device_type);
    CHECK_EQ(BOOST_GET_CONST(platform::CUDAPlace, place).device,
             dl_tensor.ctx.device_id);
  } else if (platform::is_cuda_pinned_place(place)) {
    CHECK_EQ(kDLCPUPinned, dl_tensor.ctx.device_type);
    CHECK_EQ(0, dl_tensor.ctx.device_id);
  } else {
    CHECK_EQ(false, true);
  }

  CHECK_EQ(dims.size(), dl_tensor.ndim);
  for (auto i = 0; i < dims.size(); ++i) {
    CHECK_EQ(dims[i], dl_tensor.shape[i]);
  }

  CHECK_EQ(dl_tensor.strides == nullptr, true);
  CHECK_EQ(static_cast<uint64_t>(0), dl_tensor.byte_offset);

  CHECK_EQ(lanes, dl_tensor.dtype.lanes);
  CHECK_EQ(sizeof(T) * 8, dl_tensor.dtype.bits);

  CHECK_EQ(GetDLDataTypeCode<T>(), dl_tensor.dtype.code);
}

template <typename T>
void TestToCudfCompatibleDLManagedTensor(const platform::Place &place,
                                         uint16_t lanes) {
  DDim dims{6, 7};
  Tensor tensor;
  tensor.Resize(dims);
  tensor.mutable_data<T>(place);

  DLPackTensor dlpack_tensor(tensor, lanes);

  ::DLManagedTensor *dl_managed_tensor =
      dlpack_tensor.ToCudfCompatibleDLManagedTensor();

  CHECK_EQ(dl_managed_tensor->manager_ctx == nullptr, true);

  for (auto i = 0; i < dims.size(); ++i) {
    CHECK_EQ(dims[i], dl_managed_tensor->dl_tensor.shape[i]);
  }

  CHECK_EQ(dl_managed_tensor->dl_tensor.strides[0] == 1, true);

  dl_managed_tensor->deleter(dl_managed_tensor);
}

template <typename T>
void TestMainLoop() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  std::vector<platform::Place> places{platform::CPUPlace(),
                                      platform::CUDAPlace(0),
                                      platform::CUDAPinnedPlace()};
  if (platform::GetCUDADeviceCount() > 1) {
    places.emplace_back(platform::CUDAPlace(1));
  }
#else
  std::vector<platform::Place> places{platform::CPUPlace()};
#endif
  std::vector<uint16_t> lanes{1, 2};
  for (auto &p : places) {
    for (auto &l : lanes) {
      TestMain<T>(p, l);
      TestToCudfCompatibleDLManagedTensor<T>(p, l);
    }
  }
}
TEST(dlpack, test_all) {
#define TestCallback(cpp_type, proto_type) TestMainLoop<cpp_type>()

  _ForEachDataType_(TestCallback);
}

}  // namespace framework
}  // namespace paddle
