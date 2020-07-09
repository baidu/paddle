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
import paddle.fluid.core as core
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


class TestMulOp(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'Y': np.random.random((5, 21)).astype(self.dtype)}
        self.outputs = {'Out': np.dot(self.inputs['X'], self.inputs['Y'])}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


class TestMulOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of mul_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.mul, x1, x2)
            # The input dtype of mul_op must be float32 or float64.
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="int32")
            x4 = fluid.layers.data(name='x4', shape=[4], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.mul, x3, x4)


class TestMulOp2(OpTest):
    def setUp(self):
        self.op_type = "mul"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {
            'X': np.random.random((3, 4, 2, 9)).astype(self.dtype),
            'Y': np.random.random((3, 6, 1, 2, 3)).astype(self.dtype)
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
        }
        result = np.dot(self.inputs['X'].reshape(3 * 4, 2 * 9),
                        self.inputs['Y'].reshape(3 * 6, 1 * 2 * 3))
        result = result.reshape(3, 4, 1, 2, 3)
        self.outputs = {'Out': result}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set('X'))

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16MulOp1(TestMulOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=1e-1)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['X', 'Y'], 'Out', max_relative_error=0.5)

    def test_check_grad_ingore_x(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['Y'],
                'Out',
                max_relative_error=0.5,
                no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=0.5,
                no_grad_set=set('Y'))


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16MulOp2(TestMulOp2):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=2e-1)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['X', 'Y'], 'Out', max_relative_error=0.9)

    def test_check_grad_ingore_x(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['Y'],
                'Out',
                max_relative_error=0.5,
                no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ['X'],
                'Out',
                max_relative_error=0.9,
                no_grad_set=set('Y'))


class TestMulOpAttr(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[3, 2], dtype='float32')

            res = fluid.data(name="output", shape=[2, 2], dtype="float32")
            y_1 = paddle.mul(x, y, out=res)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            data1 = np.array([[1, 2, 3], [4, 5, 6]], dtype='float32')
            data2 = np.array([[1, 2], [1, 2], [1, 2]], dtype='float32')
            np_res, np_y_1 = exe.run(feed={'x': data1,
                                           'y': data2},
                                     fetch_list=[res, y_1])

            self.assertEqual((np_res == np_y_1).all(), True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[3, 2], dtype='float32')

            res = fluid.data(name="output", shape=[2, 2], dtype="float32")
            y_1 = paddle.mul(x, y, name='mul_res')
            y_2 = paddle.mul(x, y, out=res, name='mul_res')
            self.assertEqual(('mul_res' in y_1.name), True)


if __name__ == "__main__":
    unittest.main()
