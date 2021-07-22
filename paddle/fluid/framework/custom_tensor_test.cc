// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/extension/include/ext_all.h"
#include "paddle/fluid/framework/custom_tensor_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"

template <typename T>
paddle::Tensor InitCPUTensorForTest() {
  std::vector<int64_t> tensor_shape{5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  auto* p_data_ptr = t1.mutable_data<T>(paddle::PlaceType::kCPU);
  for (int64_t i = 0; i < t1.size(); i++) {
    p_data_ptr[i] = T(5);
  }
  return t1;
}

template <typename T>
void TestCopyTensor() {
  auto t1 = InitCPUTensorForTest<T>();
  auto t1_cpu_cp = t1.template copy_to<T>(paddle::PlaceType::kCPU);
  CHECK((paddle::PlaceType::kCPU == t1_cpu_cp.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_cpu_cp.template data<T>()[i], T(5));
  }
#ifdef PADDLE_WITH_CUDA
  VLOG(2) << "Do GPU copy test";
  auto t1_gpu_cp = t1_cpu_cp.template copy_to<T>(paddle::PlaceType::kGPU);
  CHECK((paddle::PlaceType::kGPU == t1_gpu_cp.place()));
  auto t1_gpu_cp_cp = t1_gpu_cp.template copy_to<T>(paddle::PlaceType::kGPU);
  CHECK((paddle::PlaceType::kGPU == t1_gpu_cp_cp.place()));
  auto t1_gpu_cp_cp_cpu =
      t1_gpu_cp_cp.template copy_to<T>(paddle::PlaceType::kCPU);
  CHECK((paddle::PlaceType::kCPU == t1_gpu_cp_cp_cpu.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_gpu_cp_cp_cpu.template data<T>()[i], T(5));
  }
#elif defined(PADDLE_WITH_HIP)
  VLOG(2) << "Do HIP copy test";
  auto t1_gpu_cp = t1_cpu_cp.template copy_to<T>(paddle::PlaceType::kHIP);
  CHECK((paddle::PlaceType::kHIP == t1_gpu_cp.place()));
  auto t1_gpu_cp_cp = t1_gpu_cp.template copy_to<T>(paddle::PlaceType::kHIP);
  CHECK((paddle::PlaceType::kHIP == t1_gpu_cp_cp.place()));
  auto t1_gpu_cp_cp_cpu =
      t1_gpu_cp_cp.template copy_to<T>(paddle::PlaceType::kCPU);
  CHECK((paddle::PlaceType::kCPU == t1_gpu_cp_cp_cpu.place()));
  for (int64_t i = 0; i < t1.size(); i++) {
    CHECK_EQ(t1_gpu_cp_cp_cpu.template data<T>()[i], T(5));
  }
#endif
}

void TestAPIPlace() {
  std::vector<int64_t> tensor_shape = {5, 5};
#ifdef PADDLE_WITH_CUDA
  auto t1 = paddle::Tensor(paddle::PlaceType::kGPU);
  t1.reshape(tensor_shape);
  t1.mutable_data<float>();
  CHECK((paddle::PlaceType::kGPU == t1.place()));
#elif defined(PADDLE_WITH_HIP)
  auto t1 = paddle::Tensor(paddle::PlaceType::kHIP);
  t1.reshape(tensor_shape);
  t1.mutable_data<float>();
  CHECK((paddle::PlaceType::kHIP == t1.place()));
#endif
  auto t2 = paddle::Tensor(paddle::PlaceType::kCPU);
  t2.reshape(tensor_shape);
  t2.mutable_data<float>();
  CHECK((paddle::PlaceType::kCPU == t2.place()));
}

void TestAPISizeAndShape() {
  std::vector<int64_t> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  CHECK_EQ(t1.size(), 25);
  CHECK(t1.shape() == tensor_shape);
}

void TestAPISlice() {
  std::vector<int64_t> tensor_shape_origin1 = {5, 5};
  std::vector<int64_t> tensor_shape_sub1 = {3, 5};
  std::vector<int64_t> tensor_shape_origin2 = {5, 5, 5};
  std::vector<int64_t> tensor_shape_sub2 = {1, 5, 5};
#ifdef PADDLE_WITH_CUDA
  auto t1 = paddle::Tensor(paddle::PlaceType::kGPU, tensor_shape_origin1);
  t1.mutable_data<float>();
  CHECK(t1.slice(0, 5).shape() == tensor_shape_origin1);
  CHECK(t1.slice(0, 3).shape() == tensor_shape_sub1);
  auto t2 = paddle::Tensor(paddle::PlaceType::kGPU, tensor_shape_origin2);
  t2.mutable_data<float>();
  CHECK(t2.slice(4, 5).shape() == tensor_shape_sub2);
#endif
  auto t3 = paddle::Tensor(paddle::PlaceType::kCPU, tensor_shape_origin1);
  t3.mutable_data<float>();
  CHECK(t3.slice(0, 5).shape() == tensor_shape_origin1);
  CHECK(t3.slice(0, 3).shape() == tensor_shape_sub1);
  auto t4 = paddle::Tensor(paddle::PlaceType::kCPU, tensor_shape_origin2);
  t4.mutable_data<float>();
  CHECK(t4.slice(4, 5).shape() == tensor_shape_sub2);

  // Test writing function for sliced tensor
  auto t = InitCPUTensorForTest<float>();
  auto t_sliced = t.slice(0, 1);
  auto* t_sliced_data_ptr = t_sliced.mutable_data<float>();
  for (int64_t i = 0; i < t_sliced.size(); i++) {
    t_sliced_data_ptr[i] += static_cast<float>(5);
  }
  auto* t_data_ptr = t.mutable_data<float>();
  for (int64_t i = 0; i < t_sliced.size(); i++) {
    CHECK_EQ(t_data_ptr[i], static_cast<float>(10));
  }
}

template <typename T>
paddle::DataType TestDtype() {
  std::vector<int64_t> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  t1.template mutable_data<T>();
  return t1.type();
}

template <typename T>
void TestCast(paddle::DataType data_type) {
  std::vector<int64_t> tensor_shape = {5, 5};
  auto t1 = paddle::Tensor(paddle::PlaceType::kCPU);
  t1.reshape(tensor_shape);
  t1.template mutable_data<T>();
  auto t2 = t1.cast(data_type);
  CHECK(t2.type() == data_type);
}

void GroupTestCopy() {
  VLOG(2) << "Float cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<float>();
  VLOG(2) << "Double cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<double>();
  VLOG(2) << "int cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int>();
  VLOG(2) << "int64 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int64_t>();
  VLOG(2) << "int16 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int16_t>();
  VLOG(2) << "int8 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<int8_t>();
  VLOG(2) << "uint8 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<uint8_t>();
  VLOG(2) << "complex<float> cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::complex64>();
  VLOG(2) << "complex<double> cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::complex128>();
  VLOG(2) << "Fp16 cpu-cpu-gpu-gpu-cpu";
  TestCopyTensor<paddle::float16>();
}

void GroupTestCast() {
  VLOG(2) << "int cast";
  TestCast<int>(paddle::DataType::FLOAT32);
  VLOG(2) << "int32 cast";
  TestCast<int32_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "int64 cast";
  TestCast<int64_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "double cast";
  TestCast<double>(paddle::DataType::FLOAT32);
  VLOG(2) << "bool cast";
  TestCast<bool>(paddle::DataType::FLOAT32);
  VLOG(2) << "uint8 cast";
  TestCast<uint8_t>(paddle::DataType::FLOAT32);
  VLOG(2) << "float cast";
  TestCast<float>(paddle::DataType::FLOAT32);
  VLOG(2) << "complex<float> cast";
  TestCast<paddle::complex64>(paddle::DataType::FLOAT32);
  VLOG(2) << "complex<double> cast";
  TestCast<paddle::complex128>(paddle::DataType::FLOAT32);
  VLOG(2) << "float16 cast";
  TestCast<paddle::float16>(paddle::DataType::FLOAT16);
}

void GroupTestDtype() {
  CHECK(TestDtype<float>() == paddle::DataType::FLOAT32);
  CHECK(TestDtype<double>() == paddle::DataType::FLOAT64);
  CHECK(TestDtype<int>() == paddle::DataType::INT32);
  CHECK(TestDtype<int64_t>() == paddle::DataType::INT64);
  CHECK(TestDtype<int16_t>() == paddle::DataType::INT16);
  CHECK(TestDtype<int8_t>() == paddle::DataType::INT8);
  CHECK(TestDtype<uint8_t>() == paddle::DataType::UINT8);
  CHECK(TestDtype<paddle::complex64>() == paddle::DataType::COMPLEX64);
  CHECK(TestDtype<paddle::complex128>() == paddle::DataType::COMPLEX128);
  CHECK(TestDtype<paddle::float16>() == paddle::DataType::FLOAT16);
}

void GroupTestDtypeConvert() {
  // enum -> proto
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::FLOAT64) ==
        paddle::framework::proto::VarType::FP64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::FLOAT32) ==
        paddle::framework::proto::VarType::FP32);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::UINT8) ==
        paddle::framework::proto::VarType::UINT8);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::INT8) == paddle::framework::proto::VarType::INT8);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::INT32) ==
        paddle::framework::proto::VarType::INT32);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::INT64) ==
        paddle::framework::proto::VarType::INT64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::INT16) ==
        paddle::framework::proto::VarType::INT16);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::BOOL) == paddle::framework::proto::VarType::BOOL);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::COMPLEX64) ==
        paddle::framework::proto::VarType::COMPLEX64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::COMPLEX128) ==
        paddle::framework::proto::VarType::COMPLEX128);
  CHECK(paddle::framework::CustomTensorUtils::ConvertEnumDTypeToInnerDType(
            paddle::DataType::FLOAT16) ==
        paddle::framework::proto::VarType::FP16);
  // proto -> enum
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::FP64) ==
        paddle::DataType::FLOAT64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::FP32) ==
        paddle::DataType::FLOAT32);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::INT64) ==
        paddle::DataType::INT64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::INT32) ==
        paddle::DataType::INT32);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::INT8) == paddle::DataType::INT8);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::UINT8) ==
        paddle::DataType::UINT8);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::INT16) ==
        paddle::DataType::INT16);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::BOOL) == paddle::DataType::BOOL);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::COMPLEX64) ==
        paddle::DataType::COMPLEX64);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::COMPLEX128) ==
        paddle::DataType::COMPLEX128);
  CHECK(paddle::framework::CustomTensorUtils::ConvertInnerDTypeToEnumDType(
            paddle::framework::proto::VarType::FP16) ==
        paddle::DataType::FLOAT16);
}

void TestInitilized() {
  paddle::Tensor test_tensor(paddle::PlaceType::kCPU);
  CHECK(test_tensor.is_initialized() == false);
  test_tensor.reshape({1, 1});
  test_tensor.mutable_data<float>();
  CHECK(test_tensor.is_initialized() == true);
  float* tensor_data = test_tensor.data<float>();
  for (int i = 0; i < test_tensor.size(); i++) {
    tensor_data[i] = 0.5;
  }
  for (int i = 0; i < test_tensor.size(); i++) {
    CHECK(tensor_data[i] == 0.5);
  }
}

TEST(CustomTensor, copyTest) {
  VLOG(2) << "TestCopy";
  GroupTestCopy();
  VLOG(2) << "TestDtype";
  GroupTestDtype();
  VLOG(2) << "TestShape";
  TestAPISizeAndShape();
  VLOG(2) << "TestPlace";
  TestAPIPlace();
  VLOG(2) << "TestSlice";
  TestAPISlice();
  VLOG(2) << "TestCast";
  GroupTestCast();
  VLOG(2) << "TestDtypeConvert";
  GroupTestDtypeConvert();
  VLOG(2) << "TestInitilized";
  TestInitilized();
}
