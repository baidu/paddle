//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_layout_transform.h"

#include "gtest/gtest.h"
#include "paddle/fluid/platform/device_context.h"

TEST(DataTransform, DataLayoutFunction) {
  auto place = paddle::platform::CPUPlace();
  paddle::framework::Tensor in = paddle::framework::Tensor();
  paddle::framework::Tensor out = paddle::framework::Tensor();
  in.mutable_data<double>(paddle::framework::make_ddim({2, 3, 1, 2}), place);
  in.set_layout(paddle::framework::DataLayout::kNHWC);

  auto kernel_nhwc = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::FP32, place,
      paddle::framework::DataLayout::kNHWC,
      paddle::framework::LibraryType::kPlain);
  auto kernel_ncwh = paddle::framework::OpKernelType(
      paddle::framework::proto::VarType::FP32, place,
      paddle::framework::DataLayout::kNCHW,
      paddle::framework::LibraryType::kPlain);

  paddle::framework::TransDataLayout(kernel_nhwc, kernel_ncwh, in, &out);

  EXPECT_TRUE(out.layout() == paddle::framework::DataLayout::kNCHW)
  EXPECT_TRUE(out.dims() == paddle::framework::make_ddim({2, 2, 3, 1}))

  TransDataLayout(kernel_ncwh, kernel_nhwc, in, &out);

  EXPECT_TRUE(in.layout() == paddle::framework::DataLayout::kNHWC);
  EXPECT_TRUE(in.dims() == paddle::framework::make_ddim({2, 3, 1, 2}));
}
