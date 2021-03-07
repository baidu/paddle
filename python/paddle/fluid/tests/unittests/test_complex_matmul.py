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
        self._dtypes = ["float32", "float64"]
        self._places = [fluid.CPUPlace()]
        if fluid.core.is_compiled_with_cuda():
            self._places.append(fluid.CUDAPlace(0))

    def compare_by_basic_api(self, x, y, np_result):
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x)
                y_var = dg.to_variable(y)
                result = paddle.matmul(x_var, y_var)
                pd_result = result.numpy()
                self.assertEqual(1, 2)
                self.assertTrue(
                    np.allclose(pd_result, np_result),
                    "\nplace: {}\npaddle diff result:\n {}\nnumpy diff result:\n {}\n".
                    format(place, pd_result[~np.isclose(pd_result, np_result)],
                           np_result[~np.isclose(pd_result, np_result)]))

    def compare_op_by_basic_api(self, x, y, np_result):
        for place in self._places:
            with dg.guard(place):
                x_var = dg.to_variable(x)
                y_var = dg.to_variable(y)
                result = x_var.matmul(y_var)
                pd_result = result.numpy()
                self.assertTrue(
                    np.allclose(pd_result, np_result),
                    "\nplace: {}\npaddle diff result:\n {}\nnumpy diff result:\n {}\n".
                    format(place, pd_result[~np.isclose(pd_result, np_result)],
                           np_result[~np.isclose(pd_result, np_result)]))

    def test_complex_xy(self):
        for dtype in self._dtypes:
            x = np.random.random(
                (2, 3, 4, 5)).astype(dtype) + 1J * np.random.random(
                    (2, 3, 4, 5)).astype(dtype)
            y = np.random.random(
                (2, 3, 5, 4)).astype(dtype) + 1J * np.random.random(
                    (2, 3, 5, 4)).astype(dtype)

            np_result = np.matmul(x, y)

            self.compare_by_basic_api(x, y, np_result)
            self.compare_op_by_basic_api(x, y, np_result)

    def test_complex_x_real_y(self):
        for dtype in self._dtypes:
            x = np.random.random(
                (2, 3, 4, 5)).astype(dtype) + 1J * np.random.random(
                    (2, 3, 4, 5)).astype(dtype)
            y = np.random.random((2, 3, 5, 4)).astype(dtype)

            np_result = np.matmul(x, y)

            # float -> complex type promotion
            self.compare_by_basic_api(x, y, np_result)
            self.compare_op_by_basic_api(x, y, np_result)

    def test_real_x_complex_y(self):
        for dtype in self._dtypes:
            x = np.random.random((2, 3, 4, 5)).astype(dtype)
            y = np.random.random(
                (2, 3, 5, 4)).astype(dtype) + 1J * np.random.random(
                    (2, 3, 5, 4)).astype(dtype)

            np_result = np.matmul(x, y)

            # float -> complex type promotion
            self.compare_by_basic_api(x, y, np_result)
            self.compare_op_by_basic_api(x, y, np_result)

    # for coverage
    def test_complex_xy_gemv(self):
        for dtype in self._dtypes:
            x = np.random.random(
                (2, 1, 100)).astype(dtype) + 1J * np.random.random(
                    (2, 1, 100)).astype(dtype)
            y = np.random.random((100)).astype(dtype) + 1J * np.random.random(
                (100)).astype(dtype)

            np_result = np.matmul(x, y)

            self.compare_by_basic_api(x, y, np_result)
            self.compare_op_by_basic_api(x, y, np_result)

    # for coverage
    def test_complex_xy_gemm(self):
        for dtype in self._dtypes:
            x = np.random.random(
                (1, 2, 50)).astype(dtype) + 1J * np.random.random(
                    (1, 2, 50)).astype(dtype)
            y = np.random.random(
                (1, 50, 2)).astype(dtype) + 1J * np.random.random(
                    (1, 50, 2)).astype(dtype)

            np_result = np.matmul(x, y)

            self.compare_by_basic_api(x, y, np_result)
            self.compare_op_by_basic_api(x, y, np_result)


if __name__ == '__main__':
    unittest.main()
