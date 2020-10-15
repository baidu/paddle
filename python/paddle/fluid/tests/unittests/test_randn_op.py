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
import paddle
import paddle.fluid.core as core
from paddle.static import program_guard, Program


class TestRandnOp(unittest.TestCase):
    def test_api(self):
        shape = [1000, 784]
        train_program = Program()
        startup_program = Program()
        with program_guard(train_program, startup_program):
            x1 = paddle.randn(shape, 'float32')
            x2 = paddle.randn(shape, 'float64')

            dim_1 = paddle.fluid.layers.fill_constant([1], "int64", 20)
            dim_2 = paddle.fluid.layers.fill_constant([1], "int32", 50)
            x3 = paddle.randn([dim_1, dim_2, 784])

            var_shape = paddle.static.data('X', [2], 'int32')
            x4 = paddle.randn(var_shape)

        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        res = exe.run(train_program,
                      feed={'X': np.array(
                          shape, dtype='int32')},
                      fetch_list=[x1, x2, x3, x4])

        for out in res:
            self.assertAlmostEqual(np.mean(out), .0, delta=0.1)
            self.assertAlmostEqual(np.std(out), 1., delta=0.1)


class TestRandnOpForDygraph(unittest.TestCase):
    def test_api(self):
        shape = [1000, 784]
        place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()
        paddle.disable_static(place)
        x1 = paddle.randn(shape, 'float32')
        x2 = paddle.randn(shape, 'float64')

        dim_1 = paddle.fluid.layers.fill_constant([1], "int64", 20)
        dim_2 = paddle.fluid.layers.fill_constant([1], "int32", 50)
        x3 = paddle.randn(shape=[dim_1, dim_2, 784])

        var_shape = paddle.to_tensor(np.array(shape))
        x4 = paddle.randn(var_shape)

        for out in [x1, x2, x3, x4]:
            self.assertAlmostEqual(np.mean(out.numpy()), .0, delta=0.1)
            self.assertAlmostEqual(np.std(out.numpy()), 1., delta=0.1)
        paddle.enable_static()


class TestRandnOpError(unittest.TestCase):
    def test_error(self):
        with program_guard(Program(), Program()):
            # The argument shape's size of randn_op should not be 0.
            self.assertRaises(AssertionError, paddle.randn, [])

            # The argument shape's type of randn_op should be list or tuple.
            self.assertRaises(TypeError, paddle.randn, 1)

            # The argument dtype of randn_op should be float32 or float64.
            self.assertRaises(TypeError, paddle.randn, [1, 2], 'int32')


if __name__ == "__main__":
    unittest.main()
