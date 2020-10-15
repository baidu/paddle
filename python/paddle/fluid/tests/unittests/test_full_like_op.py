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

import paddle
import paddle.fluid.core as core
from paddle.static import program_guard, Program
import paddle.compat as cpt
import unittest
import numpy as np
from op_test import OpTest


class TestFullOp(unittest.TestCase):
    """ Test fill_any_like op(whose API is full_like) for attr out. """

    def test_attr_tensor_API(self):
        startup_program = Program()
        train_program = Program()
        with program_guard(train_program, startup_program):
            fill_value = 2.0
            input = paddle.fluid.data(name='input', dtype='float32', shape=[2, 3])
            output = paddle.full_like(input, fill_value)
            output_dtype = paddle.full_like(input, fill_value, dtype='float32')

            place = paddle.CPUPlace()
            if core.is_compiled_with_cuda():
                place = paddle.CUDAPlace(0)
            exe = paddle.static.Executor(place)
            exe.run(startup_program)

            img = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)

            res = exe.run(train_program,
                          feed={'input': img},
                          fetch_list=[output])

            out_np = np.array(res[0])
            self.assertTrue(
                not (out_np - np.full_like(img, fill_value)).any(),
                msg="full_like output is wrong, out = " + str(out_np))

    def test_full_like_imperative(self):
        paddle.disable_static()
        input = paddle.arange(6, 10, dtype='float32')
        out = paddle.full_like(input, fill_value=888.88, dtype='float32')
        out_numpy = np.random.random((4)).astype("float32")
        out_numpy.fill(888.88)
        self.assertTrue((out.numpy() == out_numpy).all(), True)
        paddle.enable_static()


class TestFullOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            #for ci coverage

            input_data = paddle.fluid.data(
                name='input', dtype='float32', shape=[2, 3])
            output = paddle.full_like(input_data, 2.0)

            def test_input_dtype():
                paddle.full_like

            self.assertRaises(
                TypeError,
                paddle.full_like,
                x=input_data,
                fill_value=2,
                dtype='uint4')
            self.assertRaises(
                TypeError,
                paddle.full_like,
                x=input_data,
                fill_value=2,
                dtype='int16')


if __name__ == "__main__":
    unittest.main()
