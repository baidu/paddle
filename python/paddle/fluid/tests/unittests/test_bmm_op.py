#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid import Program, program_guard


class TestBmmOp(OpTest):
    def setUp(self):
        self.op_type = "bmm"
        X = np.random.random((10, 3, 4)).astype("float64")
        Y = np.random.random((10, 4, 5)).astype("float64")
        self.inputs = {'X': X, 'Y': Y}
        Out = np.matmul(X, Y)
        self.outputs = {'Out': Out}

    def test_check_output(self):
        self.check_output()

    def test_checkout_grad(self):
        self.check_grad(['X', 'Y'], 'Out')


class API_TestBmm(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data(
                'data1', shape=[-1, 3, 4], dtype='float64')
            data2 = fluid.layers.data(
                'data2', shape=[-1, 4, 5], dtype='float64')
            result_bmm = paddle.bmm(data1, data2)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([10, 3, 4]).astype('float64')
            input2 = np.random.random([10, 4, 5]).astype('float64')
            result, = exe.run(feed={"data1": input1,
                                    "data2": input2},
                              fetch_list=[result_bmm])
            expected_result = np.matmul(input1, input2)
        self.assertTrue(np.allclose(expected_result, result))


class API_TestDygraphBmm(unittest.TestCase):
    def test_out(self):
        input1 = np.array([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                           [[3.0, 3.0, 3.0], [4.0, 4.0, 4.0]]])
        input2 = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
                           [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]])
        with fluid.dygraph.guard():
            x = fluid.dygraph.to_variable(input1)
            y = fluid.dygraph.to_variable(input2)
            out = paddle.bmm(x, y)
            out_np = out.numpy()
        expected_result = np.matmul(input1, input2)
        self.assertTrue(np.allclose(expected_result, out_np))


class TestBmmAPIError(unittest.TestCase):
    def test_api_error(self):
        x_data = np.arange(24, dtype='float32').reshape((2, 3, 4))
        y_data = np.arange(16, dtype='float32').reshape((2, 4, 2))
        y_data_wrong1 = np.arange(16, dtype='float32').reshape((2, 2, 4))
        y_data_wrong2 = np.arange(16, dtype='float32').reshape((2, 2, 2, 2))
        y_data_wrong3 = np.arange(24, dtype='float32').reshape((3, 4, 2))
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong1)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong2)
        self.assertRaises(ValueError, paddle.bmm, x_data, y_data_wrong3)


if __name__ == "__main__":
    unittest.main()
