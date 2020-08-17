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

from test_pool2d_op import adaptive_start_index, adaptive_end_index, pool2D_forward_naive
import unittest
from op_test import OpTest
import numpy as np
import paddle.fluid.core as core
from paddle.nn.functional import *
import paddle.fluid as fluid
import paddle


class TestPool2d_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_avg_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(
                name="input", shape=[2, 3, 32, 32], dtype="float32")
            result = avg_pool2d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg')

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_avg_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = avg_pool2d(input, kernel_size=2, stride=2, padding=0)

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='avg')
            self.assertTrue(np.allclose(result.numpy(), result_np))

            avg_pool2d_dg = paddle.nn.layer.AvgPool2d(
                kernel_size=2, stride=2, padding=0)
            result = avg_pool2d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_max_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(
                name="input", shape=[2, 3, 32, 32], dtype="float32")
            result = max_pool2d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max')

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_max_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = max_pool2d(
                input, kernel_size=2, stride=2, padding=0, return_indices=False)

            result_np = pool2D_forward_naive(
                input_np,
                ksize=[2, 2],
                strides=[2, 2],
                paddings=[0, 0],
                pool_type='max')
            self.assertTrue(np.allclose(result.numpy(), result_np))

            max_pool2d_dg = paddle.nn.layer.MaxPool2d(
                kernel_size=2, stride=2, padding=0)
            result = max_pool2d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def test_pool2d(self):
        for place in self.places:

            self.check_max_dygraph_results(place)
            self.check_avg_dygraph_results(place)
            self.check_max_static_results(place)
            self.check_avg_static_results(place)


if __name__ == '__main__':
    unittest.main()
