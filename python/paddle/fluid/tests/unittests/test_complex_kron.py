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

from paddle import fluid, tensor
import paddle
import paddle.fluid.dygraph as dg
import numpy as np
import unittest


class ComplexKronTestCase(unittest.TestCase):
    def __init__(self, methodName='runTest', x=None, y=None):
        super(ComplexKronTestCase, self).__init__(methodName)
        self.x = x
        self.y = y

    def setUp(self):
        self.ref_result = np.kron(self.x, self.y)

    def runTest(self):
        place = fluid.CPUPlace()
        self.test_identity(place)

        if fluid.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.test_identity(place)

    def test_identity(self, place):
        with dg.guard(place):
            x_var = dg.to_variable(self.x)
            y_var = dg.to_variable(self.y)
            out_var = paddle.complex.kron(x_var, y_var)
            np.testing.assert_allclose(out_var.numpy(), self.ref_result)


def load_tests(loader, standard_tests, pattern):
    suite = unittest.TestSuite()
    suite.addTest(
        ComplexKronTestCase(
            x=np.random.randn(2, 2) + 1j * np.random.randn(2, 2),
            y=np.random.randn(3, 3) + 1j * np.random.randn(3, 3)))
    # Todo: not support now
    '''
    suite.addTest(
        ComplexKronTestCase(
            x=np.random.randn(2, 2),
            y=np.random.randn(3, 3) + 1j * np.random.randn(3, 3)))
    suite.addTest(
        ComplexKronTestCase(
            x=np.random.randn(2, 2) + 1j * np.random.randn(2, 2),
            y=np.random.randn(3, 3)))

    suite.addTest(
        ComplexKronTestCase(
            x=np.random.randn(2, 2) + 1j * np.random.randn(2, 2),
            y=np.random.randn(2, 2, 3)))
            '''
    return suite


if __name__ == '__main__':
    unittest.main()
