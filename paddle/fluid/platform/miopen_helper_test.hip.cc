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

#define GLOG_NO_ABBREVIATED_SEVERITIES
#define GOOGLE_GLOG_DLL_DECL

#include "paddle/fluid/platform/miopen_helper.h"
#include <gtest/gtest.h>

TEST(CudnnHelper, ScopedTensorDescriptor) {
  using paddle::platform::ScopedTensorDescriptor;
  using paddle::platform::DataLayout;

  ScopedTensorDescriptor tensor_desc;
  std::vector<int> shape = {2, 4, 6, 6};
  auto desc = tensor_desc.descriptor<float>(DataLayout::kNCHW, shape);

  miopenDataType_t type;
  //  int nd;
  std::vector<int> dims(4);
  std::vector<int> strides(4);
  paddle::platform::dynload::miopenGetTensorDescriptor(desc, &type, dims.data(),
                                                       strides.data());

  //  EXPECT_EQ(nd, 4);
  for (size_t i = 0; i < dims.size(); ++i) {
    EXPECT_EQ(dims[i], shape[i]);
  }
  EXPECT_EQ(strides[3], 1);
  EXPECT_EQ(strides[2], 6);
  EXPECT_EQ(strides[1], 36);
  EXPECT_EQ(strides[0], 144);

  // test tensor5d: ScopedTensorDescriptor
  ScopedTensorDescriptor tensor5d_desc;
  std::vector<int> shape_5d = {2, 4, 6, 6, 6};
  auto desc_5d = tensor5d_desc.descriptor<float>(DataLayout::kNCDHW, shape_5d);

  std::vector<int> dims_5d(5);
  std::vector<int> strides_5d(5);
  paddle::platform::dynload::miopenGetTensorDescriptor(
      desc_5d, &type, dims_5d.data(), strides_5d.data());

  //  EXPECT_EQ(nd, 5);
  for (size_t i = 0; i < dims_5d.size(); ++i) {
    EXPECT_EQ(dims_5d[i], shape_5d[i]);
  }
  EXPECT_EQ(strides_5d[4], 1);
  EXPECT_EQ(strides_5d[3], 6);
  EXPECT_EQ(strides_5d[2], 36);
  EXPECT_EQ(strides_5d[1], 216);
  EXPECT_EQ(strides_5d[0], 864);
}

// TEST(CudnnHelper, ScopedFilterDescriptor) {
//  using paddle::platform::ScopedFilterDescriptor;
//  using paddle::platform::DataLayout;
//
//  ScopedFilterDescriptor filter_desc;
//  std::vector<int> shape = {2, 3, 3};
//  auto desc = filter_desc.descriptor<float>(DataLayout::kNCHW, shape);
//
//  miopenDataType_t type;
//  int nd;
//  miopenTensorFormat_t format;
//  std::vector<int> kernel(3);
//  paddle::platform::dynload::miopenGetTensorDescriptor(desc, &type, &format,
//                                                        kernel.data());
//
//  EXPECT_EQ(GetCudnnTensorFormat(DataLayout::kNCHW), format);
//  EXPECT_EQ(nd, 3);
//  for (size_t i = 0; i < shape.size(); ++i) {
//    EXPECT_EQ(kernel[i], shape[i]);
//  }
//
//  ScopedFilterDescriptor filter_desc_4d;
//  std::vector<int> shape_4d = {2, 3, 3, 3};
//  auto desc_4d = filter_desc.descriptor<float>(DataLayout::kNCDHW, shape_4d);
//
//  std::vector<int> kernel_4d(4);
//  paddle::platform::dynload::miopenGetTensorDescriptor(
//      desc_4d, 4, &type, &format, &nd, kernel_4d.data());
//
//  EXPECT_EQ(GetCudnnTensorFormat(DataLayout::kNCHW), format);
//  EXPECT_EQ(nd, 4);
//  for (size_t i = 0; i < shape_4d.size(); ++i) {
//    EXPECT_EQ(kernel_4d[i], shape_4d[i]);
//  }
//}

TEST(CudnnHelper, ScopedConvolutionDescriptor) {
  using paddle::platform::ScopedConvolutionDescriptor;

  ScopedConvolutionDescriptor conv_desc;
  std::vector<int> src_pads = {2, 2, 2};
  std::vector<int> src_strides = {1, 1, 1};
  std::vector<int> src_dilations = {1, 1, 1};
  auto desc = conv_desc.descriptor<float>(src_pads, src_strides, src_dilations);

  miopenDataType_t type;
  miopenConvolutionMode_t mode;
  int nd;
  std::vector<int> pads(3);
  std::vector<int> strides(3);
  std::vector<int> dilations(3);
  paddle::platform::dynload::miopenGetConvolutionNdDescriptor(
      desc, 3, &nd, pads.data(), strides.data(), dilations.data(), &mode);

  EXPECT_EQ(nd, 3);
  for (size_t i = 0; i < src_pads.size(); ++i) {
    EXPECT_EQ(pads[i], src_pads[i]);
    EXPECT_EQ(strides[i], src_strides[i]);
    EXPECT_EQ(dilations[i], src_dilations[i]);
  }
  EXPECT_EQ(mode, miopenConvolution);
}

TEST(CudnnHelper, ScopedPoolingDescriptor) {
  using paddle::platform::ScopedPoolingDescriptor;
  using paddle::platform::PoolingMode;

  ScopedPoolingDescriptor pool_desc;
  std::vector<int> src_kernel = {3, 3};
  std::vector<int> src_pads = {1, 1};
  std::vector<int> src_strides = {2, 2};
  auto desc = pool_desc.descriptor(PoolingMode::kMaximum, src_kernel, src_pads,
                                   src_strides);

  miopenPoolingMode_t mode;
  // miopenNanPropagation_t nan_t = HIPDNN_PROPAGATE_NAN;
  //  int nd;
  std::vector<int> kernel(2);
  std::vector<int> pads(2);
  std::vector<int> strides(2);
  paddle::platform::dynload::miopenGet2dPoolingDescriptor(
      desc, &mode, kernel.data(), kernel.data() + 1, pads.data(),
      pads.data() + 1, strides.data(), strides.data() + 1);

  //  EXPECT_EQ(nd, 2);
  for (size_t i = 0; i < src_pads.size(); ++i) {
    EXPECT_EQ(kernel[i], src_kernel[i]);
    EXPECT_EQ(pads[i], src_pads[i]);
    EXPECT_EQ(strides[i], src_strides[i]);
  }
  EXPECT_EQ(mode, miopenPoolingMax);
}
