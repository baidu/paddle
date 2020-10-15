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
import paddle.fluid.initializer as I
import unittest


class LabelSmoothTestCase(unittest.TestCase):
    def __init__(self,
                 methodName='runTest',
                 label_shape=(20, 1),
                 prior_dist=None,
                 epsilon=0.1,
                 dtype="float32"):
        super(LabelSmoothTestCase, self).__init__(methodName)

        self.label_shape = label_shape
        self.prior_dist = prior_dist
        self.dtype = dtype
        self.epsilon = epsilon

    def setUp(self):
        self.label = np.random.randn(*(self.label_shape)).astype(self.dtype)

    def fluid_layer(self, place):
        paddle.enable_static()
        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                label_var = fluid.data(
                    "input", self.label_shape, dtype=self.dtype)
                y_var = fluid.layers.label_smooth(
                    label_var,
                    prior_dist=self.prior_dist,
                    epsilon=self.epsilon,
                    dtype=self.dtype)
        feed_dict = {"input": self.label}
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
                label_var = fluid.data(
                    "input", self.label_shape, dtype=self.dtype)
                y_var = F.label_smooth(
                    label_var, prior_dist=self.prior_dist, epsilon=self.epsilon)
        feed_dict = {"input": self.label}
        exe = fluid.Executor(place)
        exe.run(start)
        y_np, = exe.run(main, feed=feed_dict, fetch_list=[y_var])
        return y_np

    def paddle_dygraph_layer(self):
        paddle.disable_static()
        label_var = dg.to_variable(self.label)
        y_var = F.label_smooth(
            label_var, prior_dist=self.prior_dist, epsilon=self.epsilon)
        y_np = y_var.numpy()
        return y_np

    def _test_equivalence(self, place):
        place = fluid.CPUPlace()
        result1 = self.fluid_layer(place)
        result2 = self.functional(place)
        result3 = self.paddle_dygraph_layer()
        np.testing.assert_array_almost_equal(result1, result2)
        np.testing.assert_array_almost_equal(result2, result3)

    def runTest(self):
        place = fluid.CPUPlace()
        self._test_equivalence(place)
        if fluid.core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self._test_equivalence(place)


class LabelSmoothErrorTestCase(LabelSmoothTestCase):
    def runTest(self):
        place = fluid.CPUPlace()
        with dg.guard(place):
            with self.assertRaises(ValueError):
                self.paddle_dygraph_layer()


def add_cases(suite):
    suite.addTest(LabelSmoothTestCase(methodName='runTest'))
    suite.addTest(
        LabelSmoothTestCase(
            methodName='runTest', label_shape=[2, 3, 1]))


def add_error_cases(suite):
    suite.addTest(LabelSmoothErrorTestCase(methodName='runTest', epsilon=2))


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    add_cases(suite)
    add_error_cases(suite)
    return suite


if __name__ == '__main__':
    unittest.main()
