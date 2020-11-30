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

import unittest
import paddle
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dg


class TestComplexMatMulLayer(unittest.TestCase):
    def setUp(self):
        self._places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(fluid.CUDAPlace(0))

    def compare(self, x, y):
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x)
                y_var = dg.to_variable(y)
                result = paddle.complex.matmul(x_var, y_var)
        np_result = np.matmul(x, y)
        self.assertTrue(np.allclose(result.numpy(), np_result))

    def compare_op(self, x, y):
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x)
                y_var = dg.to_variable(y)
                result = x_var.matmul(y_var)
        np_result = np.matmul(x, y)
        self.assertTrue(np.allclose(result.numpy(), np_result))

    def test_complex_xy(self):
        x = np.random.random(
            (2, 3, 4, 5)).astype("float32") + 1J * np.random.random(
                (2, 3, 4, 5)).astype("float32")
        y = np.random.random(
            (2, 3, 5, 4)).astype("float32") + 1J * np.random.random(
                (2, 3, 5, 4)).astype("float32")
        self.compare(x, y)
        self.compare_op(x, y)

    # Todo: not support now
    '''
    def test_complex_x(self):
        x = np.random.random(
            (2, 3, 4, 5)).astype("float32") + 1J * np.random.random(
                (2, 3, 4, 5)).astype("float32")
        y = np.random.random((2, 3, 5, 4)).astype("float32")
        self.compare(x, y)
        self.compare_op(x, y)

    def test_complex_y(self):
        x = np.random.random((2, 3, 4, 5)).astype("float32")
        y = np.random.random(
            (2, 3, 5, 4)).astype("float32") + 1J * np.random.random(
                (2, 3, 5, 4)).astype("float32")
        self.compare(x, y)
    '''


if __name__ == '__main__':
    unittest.main()
