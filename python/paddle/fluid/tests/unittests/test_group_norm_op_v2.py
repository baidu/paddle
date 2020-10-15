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

import os
import unittest
import numpy as np
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from op_test import OpTest, _set_use_system_allocator
from paddle.fluid.framework import grad_var_name
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


class TestDygraphGroupNormv2(unittest.TestCase):
    def test_dygraph(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            shape = [2, 2, 2, 2]

            def compute_v1(x):
                with fluid.dygraph.guard(p):
                    gn = fluid.dygraph.GroupNorm(channels=2, groups=2)
                    y = gn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def compute_v2(x):
                with fluid.dygraph.guard(p):
                    gn = paddle.nn.GroupNorm(num_channels=2, num_groups=2)
                    y = gn(fluid.dygraph.to_variable(x))
                return y.numpy()

            def test_weight_bias_false():
                with fluid.dygraph.guard(p):
                    gn = paddle.nn.GroupNorm(
                        num_channels=2,
                        num_groups=2,
                        weight_attr=False,
                        bias_attr=False)

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            result = np.allclose(y1, y2, atol=1e-5)
            if not result:
                print("y1:", y1, "\ty2:", y2)
            self.assertTrue(result)
            test_weight_bias_false()

    def test_static(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda() and core.op_support_gpu("group_norm"):
            places.append(fluid.CUDAPlace(0))
        for p in places:
            exe = fluid.Executor(p)
            shape = [2, 6, 2, 2]

            def compute_v1(x_np):
                with program_guard(Program(), Program()):
                    gn = fluid.dygraph.GroupNorm(channels=6, groups=2)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = gn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            def compute_v2(x_np):
                with program_guard(Program(), Program()):
                    gn = paddle.nn.GroupNorm(num_channels=6, num_groups=2)
                    x = fluid.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
                    y = gn(x)
                    exe.run(fluid.default_startup_program())
                    r = exe.run(feed={'x': x_np}, fetch_list=[y])[0]
                return r

            x = np.random.randn(*shape).astype("float32")
            y1 = compute_v1(x)
            y2 = compute_v2(x)
            self.assertTrue(np.allclose(y1, y2))


if __name__ == '__main__':
    unittest.main()
