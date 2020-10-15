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
import numpy as np
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.fluid.initializer as I
import unittest


class AffineGridTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 theta_shape=(20, 2, 3),
                 output_shape=[20, 2, 5, 7],
                 align_corners=True,
                 dtype="float32",
                 invalid_theta=False,
                 variable_output_shape=False):
        super(AffineGridTestCase, self).__init__(methodName)

        self.theta_shape = theta_shape
        self.output_shape = output_shape
        self.align_corners = align_corners
        self.dtype = dtype
        self.invalid_theta = invalid_theta
        self.variable_output_shape = variable_output_shape

    def setUp(self):
        self.theta = np.random.randn(*(self.theta_shape)).astype(self.dtype)

    def fluid_layer(self, place):
        paddle.enable_static()
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                theta_var = fluid.data(
                    "input", self.theta_shape, dtype=self.dtype)
                y_var = fluid.layers.affine_grid(theta_var, self.output_shape)
        feed_dict = {"input": self.theta}
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def functional(self, place):
        paddle.enable_static()
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                theta_var = fluid.data(
                    "input", self.theta_shape, dtype=self.dtype)
                y_var = F.affine_grid(
                    theta_var,
                    self.output_shape,
                    align_corners=self.align_corners)
        feed_dict = {"input": self.theta}
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_dygraph_layer(self):
        paddle.disable_static()
        theta_var = dg.to_variable(
            self.theta) if not self.invalid_theta else "invalid"
        output_shape = dg.to_variable(
            self.
            output_shape) if self.variable_output_shape else self.output_shape
        y_var = F.affine_grid(
            theta_var, output_shape, align_corners=self.align_corners)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        place = fluid.CPUPlace()
        result1 = self.fluid_layer(place)
        result2 = self.functional(place)
        result3 = self.paddle_dygraph_layer()
        if self.align_corners:
            np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)

        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self._test_equivalence(place)


class AffineGridErrorTestCase(AffineGridTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_dygraph_layer()


def add_cases(suite):
    suite.addTest(AffineGridTestCase(methodName='runTest'))
    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=True))

    suite.addTest(AffineGridTestCase(methodName='runTest', align_corners=False))
    suite.addTest(
        AffineGridTestCase(
            methodName='runTest', variable_output_shape=True))

    suite.addTest(
        AffineGridTestCase(
            methodName='runTest',
            theta_shape=(20, 2, 3),
            output_shape=[20, 1, 7, 7],
            align_corners=True))


def add_error_cases(suite):
    suite.addTest(
        AffineGridErrorTestCase(
            methodName='runTest', output_shape="not_valid"))
    suite.addTest(
        AffineGridErrorTestCase(
            methodName='runTest',
            invalid_theta=True))  # to test theta not variable error checking


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
