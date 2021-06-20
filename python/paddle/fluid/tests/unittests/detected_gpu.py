# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import sys
import paddle.fluid as fluid

print("compile with cuda:", fluid.core.is_compiled_with_cuda())
if paddle.is_compiled_with_cuda():
    print("get_cuda_device_count:", fluid.core.get_cuda_device_count())

if fluid.core.is_compiled_with_cuda() and fluid.core.get_cuda_device_count(
) > 0:
    sys.exit(0)
else:
    sys.exit(1)
