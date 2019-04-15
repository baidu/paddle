#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import gradient_checker

from decorators import prog_scope


class TestGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        pass

    def test_grad(self):
        for p in [fluid.CPUPlace(), fluid.CUDAPlace(0)]:
            self.func(p)


class TestMulGradCheck(TestGradCheck):
    @prog_scope()
    def func(self, place):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            x = layers.create_parameter(dtype="float64", shape=[2, 8], name='x')
            y = layers.create_parameter(dtype="float64", shape=[8, 4], name='y')
            z = layers.mul(x=x, y=y)
            gradient_checker.grad_check([x, y], z, place=place)


class TestReluDoubleGradCheck(TestGradCheck):
    @prog_scope()
    def func(self, place):
        # the shape of input variable shoule be clearly specified, not inlcude -1.
        shape = [2, 8]
        eps = 0.005
        dtype = np.float64

        #x = layers.data('x', shape, False, dtype)
        x = layers.create_global_var(shape, 0, dtype, True, name='x')
        x.persistable = True
        y = layers.relu(x)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        x_arr[np.abs(x_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x], y, x_init=x_arr, place=place, eps=eps)


if __name__ == "__main__":
    unittest.main()
