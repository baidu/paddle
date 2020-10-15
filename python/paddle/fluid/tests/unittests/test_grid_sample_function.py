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

import numpy as np
import paddle
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import unittest


class GridSampleTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 x_shape=[2, 2, 3, 3],
                 grid_shape=[2, 3, 3, 2],
                 mode="bilinear",
                 padding_mode="zeros",
                 align_corners=False):
        super(GridSampleTestCase, self).__init__(methodName)
        self.padding_mode = padding_mode
        self.x_shape = x_shape
        self.grid_shape = grid_shape
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.dtype = "float64"

    def setUp(self):
        self.x = np.random.randn(*(self.x_shape)).astype(self.dtype)
        self.grid = np.random.uniform(-1, 1, self.grid_shape).astype(self.dtype)

    def static_functional(self, place):
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                x = fluid.data("x", self.x_shape, dtype=self.dtype)
                grid = fluid.data("grid", self.grid_shape, dtype=self.dtype)
                y_var = F.grid_sample(
                    x,
                    grid,
                    mode=self.mode,
                    padding_mode=self.padding_mode,
                    align_corners=self.align_corners)
        feed_dict = {"x": self.x, "grid": self.grid}
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def dynamic_functional(self):
        x_t = paddle.to_tensor(self.x)
        grid_t = paddle.to_tensor(self.grid)
        y_t = F.grid_sample(
            x_t,
            grid_t,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=self.align_corners)
        y_np = y_t.numpy()
        return y_np

    def _test_equivalence(self, place):
        result1 = self.static_functional(place)
        with dg.guard(place):
            result2 = self.dynamic_functional()
        np.testing.assert_array_almost_equal(result1, result2)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)

        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self._test_equivalence(place)


class GridSampleErrorTestCase(GridSampleTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with self.assertRaises(ValueError):
            self.static_functional(place)


def add_cases(suite):
    suite.addTest(GridSampleTestCase(methodName='runTest'))
    suite.addTest(
        GridSampleTestCase(
            methodName='runTest',
            mode='bilinear',
            padding_mode='reflection',
            align_corners=True))
    suite.addTest(
        GridSampleTestCase(
            methodName='runTest',
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True))


def add_error_cases(suite):
    suite.addTest(
        GridSampleErrorTestCase(
            methodName='runTest', padding_mode="VALID"))
    suite.addTest(
        GridSampleErrorTestCase(
            methodName='runTest', align_corners="VALID"))
    suite.addTest(GridSampleErrorTestCase(methodName='runTest', mode="VALID"))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


class TestGridSampleAPI(unittest.TestCase):
    def test_errors(self):
        with self.assertRaises(ValueError):
            x = paddle.randn([1, 1, 3, 3])
            F.grid_sample(x, 1.0)
        with self.assertRaises(ValueError):
            x = paddle.randn([1, 1, 3, 3])
            F.grid_sample(1.0, x)


if __name__ == '__main__':
    unittest.main()
