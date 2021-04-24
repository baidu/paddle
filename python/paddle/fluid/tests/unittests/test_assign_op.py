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

import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward


class TestAssignOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype('float64')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out')


class TestAssignFP16Op(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype('float16')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        self.check_output()

    def test_backward(self):
        self.check_grad(['X'], 'Out')


class TestAssignOpWithLoDTensorArray(unittest.TestCase):
    def test_assign_LoDTensorArray(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program):
            x = fluid.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.fill_constant(
                shape=[100, 10], dtype='float32', value=1)
            z = fluid.layers.elementwise_add(x=x, y=y)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = fluid.layers.array_write(x=z, i=i)
            array = fluid.layers.assign(init_array)
            sums = fluid.layers.array_read(array=init_array, i=i)
            mean = fluid.layers.mean(sums)
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(main_program,
                      feed={'x': feed_x},
                      fetch_list=[sums.name, x.grad_name])
        self.assertTrue(np.allclose(res[0], feed_add))
        self.assertTrue(np.allclose(res[1], ones / 1000.0))


class TestAssignOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.assign, x1)
            # When the type of input is Variable, the dtype of input must be float16, float32, float64, int32, int64, bool.
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, fluid.layers.assign, x3)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, float64, int32, int64, bool.
            x4 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, fluid.layers.assign, x4)


class TestAssignOApi(unittest.TestCase):
    def test_assign_LoDTensorArray(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program):
            x = fluid.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.fill_constant(
                shape=[100, 10], dtype='float32', value=1)
            z = fluid.layers.elementwise_add(x=x, y=y)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = fluid.layers.array_write(x=z, i=i)
            array = paddle.assign(init_array)
            sums = fluid.layers.array_read(array=init_array, i=i)
            mean = fluid.layers.mean(sums)
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(main_program,
                      feed={'x': feed_x},
                      fetch_list=[sums.name, x.grad_name])
        self.assertTrue(np.allclose(res[0], feed_add))
        self.assertTrue(np.allclose(res[1], ones / 1000.0))

    def test_assign_NumpyArray(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.bool)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        self.assertTrue(np.allclose(result1.numpy(), array))

    def test_assign_NumpyArray1(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.float32)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        self.assertTrue(np.allclose(result1.numpy(), array))

    def test_assign_NumpyArray2(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.int32)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        self.assertTrue(np.allclose(result1.numpy(), array))

    def test_assign_NumpyArray3(self):
        with fluid.dygraph.guard():
            array = np.random.random(size=(100, 10)).astype(np.int64)
            result1 = paddle.zeros(shape=[3, 3], dtype='float32')
            paddle.assign(array, result1)
        self.assertTrue(np.allclose(result1.numpy(), array))


class TestAssignOpErrorApi(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, paddle.assign, x1)
            # When the type of input is Variable, the dtype of input must be float16, float32, float64, int32, int64, bool.
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="uint8")
            self.assertRaises(TypeError, paddle.assign, x3)
            # When the type of input is numpy.ndarray, the dtype of input must be float32, int32.
            x4 = np.array([[2.5, 2.5]], dtype='float64')
            self.assertRaises(TypeError, paddle.assign, x4)
            x5 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, paddle.assign, x5)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
