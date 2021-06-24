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

from __future__ import print_function, division

import unittest
import numpy as np

import paddle
from paddle.static import program_guard
from paddle.static import Program

# Support types are ref from `paddle.tensor.math`
# - Related paddle dtypes:
#  - int type: int64, (no test here: uint8, int8, int16, int32)
#  - float type: float32, (no test here: float64)
# - Python scalar dtypes: 
#  - int(64)
#  - float(64)


class TestTensorScalarTypePromotionStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def check_operation(self, a, b, c, op):
        exe = paddle.static.Executor()

        if op == '+':
            c_rlt = a + b
        elif op == '-':
            c_rlt = a - b
        elif op == '*':
            c_rlt = a * b
        elif op == '/':
            c_rlt = a / b
        elif op == '**':
            c_rlt = a**b
        elif op == '//':
            c_rlt = a // b
        elif op == '%':
            c_rlt = a % b
        else:
            raise ValueError("Unsupported operation.")

        rlt = exe.run(fetch_list=[c_rlt.name, c.name])

        self.assertEqual(rlt[0].dtype, rlt[1].dtype)
        self.assertTrue(np.array_equal(rlt[0], rlt[1]))

    def test_tensor_add_scalar(self):
        # tensor(int64) + scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype="int64")
            self.check_operation(a, b, c, '+')

        # tensor(float32) + scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(int64) + scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(int64) + scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(float32) + scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype="float32")
            self.check_operation(a, b, c, '+')

    def test_tensor_sub_scalar(self):
        # tensor(int64) - scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '-')

        # tensor(float32) - scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(int64) - scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(int64) - scalar(float, .5)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(float32) - scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

    def test_scalar_sub_tensor(self):
        # scalar(int) - tensor(int64)
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '-')

        # scalar(int) - tensor(float32)
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='float32')
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float, .0) - tensor(int64)
        with program_guard(Program()):
            a = 1.0
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float, .5) - tensor(int64)
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], -0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float) - tensor(float32)
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], -0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

    def test_tensor_mul_tensor(self):
        # tensor(int64) * scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.ones([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '*')

        # tensor(float32) * scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.ones([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(int64) * scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.ones([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(int64) * scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(float32) * scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype="float32")
            self.check_operation(a, b, c, '*')

    def test_tensor_div_scalar(self):
        # tensor(int64) / scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(float32) / scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

            # this behavior should be equal to elementwise_div Op
            a = paddle.full([2, 2, 2], 99, dtype="float32")
            b = 100
            c = a / paddle.full([2, 2, 2], 100, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(int64) / scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(int64) / scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(float32) / scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

    def test_scalar_div_tensor(self):
        # scalar(int) / tensor(int64)
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(int) / tensor(float32)
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(float) / tensor(int64)
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(float) / tensor(float32)
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

    def test_tensor_pow_scalar(self):
        # tensor(int64) ** scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype="int64")
            self.check_operation(a, b, c, '**')

        # tensor(int64) ** scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

    def test_scalar_pow_tensor(self):
        # scalar(int) ** tensor(int64)
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype="int64")
            self.check_operation(a, b, c, '**')

        # scalar(float) ** tensor(int64)
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

        # scalar(int) ** tensor(float32)
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(float)
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

    # ## TODO: floordiv op kernel doesn't support float
    def test_tensor_floordiv_scalar(self):
        # tensor(int64) // scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="int64")
            self.check_operation(a, b, c, '//')

    def test_tensor_mod_scalar(self):
        # tensor(int64) % scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="int64")
            self.check_operation(a, b, c, '%')

        # tensor(int64) % scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')

        # tensor(float32) % scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')

        # tensor(float32) % scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')


if __name__ == '__main__':
    unittest.main()
