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

#include "paddle/fluid/platform/miopen_desc.h"
#include <gtest/gtest.h>

namespace paddle {
namespace platform {

TEST(TensorDescriptor, Empty) {
  ActivationDescriptor a;
  TensorDescriptor t;
  TensorDescriptor t1;
  TensorDescriptor *t11 = new TensorDescriptor();
  delete t11;
  std::unique_ptr<TensorDescriptor> tt(new TensorDescriptor());
}

TEST(TensorDescriptor, Normal) {
  framework::Tensor tt;
  tt.Resize({2, 3, 4});
  tt.mutable_data<float>(platform::CPUPlace());

  TensorDescriptor desc;
  desc.set((const framework::Tensor)tt);
  EXPECT_TRUE(desc.desc() != nullptr);
}

}  // namespace platform
}  // namespace paddle
