#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
from paddle import zeros_like
from paddle.fluid import core, Program, program_guard


class TestZerosLikeAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x = paddle.fluid.data('x', [3, 4])
            self.assertRaises(TypeError, zeros_like, x, 'int8')


class TestZerosLikeAPI(unittest.TestCase):
    def test_api(self):
        shape = [3, 4]
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            x = paddle.fluid.data('X', shape)

            # 'bool', 'float32', 'float64', 'int32', 'int64'
            out1 = zeros_like(x)
            out2 = zeros_like(x, np.bool)
            out3 = zeros_like(x, 'float64')
            out4 = zeros_like(x, 'int32')
            out5 = zeros_like(x, 'int64')

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        outs = exe.run(train_program,
                       feed={'X': np.ones(shape).astype('float32')},
                       fetch_list=[out1, out2, out3, out4, out5])

        for i, dtype in enumerate(
            [np.float32, np.bool, np.float64, np.int32, np.int64]):
            self.assertEqual(outs[i].dtype, dtype)
            self.assertEqual((outs[i] == np.zeros(shape, dtype)).all(), True)


class TestZerosLikeImpeartive(unittest.TestCase):
    def test_out(self):
        shape = [3, 4]
        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        paddle.disable_static(place)
        x = paddle.to_tensor(np.ones(shape))
        for dtype in [np.bool, np.float32, np.float64, np.int32, np.int64]:
            out = zeros_like(x, dtype)
            self.assertEqual((out.numpy() == np.zeros(shape, dtype)).all(),
                             True)

        out = paddle.tensor.zeros_like(x)
        self.assertEqual((out.numpy() == np.zeros(shape, dtype)).all(), True)

        out = paddle.tensor.creation.zeros_like(x)
        self.assertEqual((out.numpy() == np.zeros(shape, dtype)).all(), True)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
