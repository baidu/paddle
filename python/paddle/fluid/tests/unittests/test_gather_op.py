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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.framework import core


def gather_numpy(x, index, axis):
    x_transpose = np.swapaxes(x, 0, axis)
    tmp_gather = x_transpose[index, ...]
    gather = np.swapaxes(tmp_gather, 0, axis)
    return gather


class TestGatherOp(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        self.inputs = {
            'X': xnp,
            'Index': np.array(self.index).astype(self.index_type)
        }
        self.outputs = {'Out': self.inputs["X"][self.inputs["Index"]]}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase1(TestGatherOp):
    def config(self):
        """
        For one dimension input
        """
        self.x_shape = (100)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int32"


class TestCase2(TestGatherOp):
    def config(self):
        """
        For int64_t index type
        """
        self.x_shape = (100)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase3(TestGatherOp):
    def config(self):
        """
        For other input type
        """
        self.x_shape = (10, 20)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"


class TestCase4(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.x_type = "double"
        self.index = [1, 1]
        self.index_type = "int32"


class TestCase5(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': False}
        self.x_type = "float64"
        self.index = [1, 1, 3]
        self.index_type = "int32"


class TestCase6(TestGatherOp):
    def config(self):
        self.x_shape = (10, 20)
        self.attrs = {'overwrite': True}
        self.x_type = "float64"
        self.index = [1, 3]
        self.index_type = "int32"


class TestGatherOp1(OpTest):
    def setUp(self):
        self.op_type = "gather"
        self.config()
        xnp = np.random.random(self.x_shape).astype(self.x_type)
        axis_np = np.array(self.axis).astype(self.index_type)
        index_np = np.array(self.index).astype(self.index_type)
        out = gather_numpy(xnp, index_np, axis_np[0])
        self.inputs = {'X': xnp, 'Index': index_np, 'Axis': axis_np}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 88, 3)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int32"
        self.axis = [1]
        self.axis_type = "int32"


class TestGatherOp2(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 88, 10)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"
        self.axis = [0]
        self.axis_type = "int32"


class TestGatherOp3(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (10, 88, 10)
        self.x_type = "float64"
        self.index = [1, 3, 5]
        self.index_type = "int64"
        self.axis = [2]
        self.axis_type = "int32"


class TestGatherOp4(TestGatherOp1):
    def config(self):
        """
        For multi-dimension input
        """
        self.x_shape = (3, 100, 10)
        self.x_type = "float64"
        self.index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.index_type = "int64"
        self.axis = [0]
        self.axis_type = "int32"
        self.attrs = {'overwrite': False}


class API_TestGather(unittest.TestCase):
    def test_out1(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[-1, 2], dtype='float64')
            index = fluid.layers.data('index', shape=[-1, 1], dtype='int32')
            out = paddle.fluid.layers.gather(data1, index)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input = np.array([[1, 2], [3, 4], [5, 6]])
            index_1 = np.array([1, 2])
            result, = exe.run(feed={"data1": input,
                                    "index": index_1},
                              fetch_list=[out])
            expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(result, expected_output))

    def test_out2(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            x = paddle.fluid.data('x', shape=[-1, 2], dtype='float64')
            index = paddle.fluid.data('index', shape=[-1, 1], dtype='int32')
            axis = paddle.fluid.data('axis', shape=[1], dtype='int32')
            out = paddle.gather(x, index, axis)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.array([[1, 2], [3, 4], [5, 6]]).astype('float64')
            index_np = np.array([1, 1]).astype('int32')
            axis_np = np.array([1]).astype('int32')
            result, = exe.run(
                feed={"x": x_np,
                      "index": index_np,
                      'axis': axis_np},
                fetch_list=[out])
            expected_output = gather_numpy(x_np, index_np, axis_np[0])
        self.assertTrue(np.allclose(result, expected_output))


class API_TestDygraphGather(unittest.TestCase):
    def test_out1(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        input = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.fluid.layers.gather(input, index)
        output_np = output.numpy()
        expected_output = np.array([[3, 4], [5, 6]])
        self.assertTrue(np.allclose(output_np, expected_output))
        paddle.enable_static()

    def test_out12(self):
        paddle.disable_static()
        input_1 = np.array([[1, 2], [3, 4], [5, 6]])
        index_1 = np.array([1, 2])
        x = paddle.to_tensor(input_1)
        index = paddle.to_tensor(index_1)
        output = paddle.gather(x, index, axis=0)
        output_np = output.numpy()
        expected_output = gather_numpy(input_1, index_1, axis=0)
        self.assertTrue(np.allclose(output_np, expected_output))
        paddle.enable_static()


class TestGathertError(unittest.TestCase):
    def test_error1(self):
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):

            shape = [8, 9, 6]
            x = paddle.fluid.data(shape=shape, dtype='int8', name='x')
            axis = paddle.fluid.data(shape=[1], dtype='float32', name='axis')
            index = paddle.fluid.data(shape=shape, dtype='int32', name='index')
            index_float = paddle.fluid.data(
                shape=shape, dtype='float32', name='index_float')

            def test_x_type():
                paddle.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)

            def test_axis_dtype():
                paddle.gather(x, index, axis=1.11)

            self.assertRaises(TypeError, test_axis_dtype)

            def test_axis_dtype():
                paddle.gather(x, index, axis=axis)

            self.assertRaises(TypeError, test_axis_dtype)

    def test_error2(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):

            shape = [8, 9, 6]
            x = fluid.data(shape=shape, dtype='int8', name='x')
            index = fluid.data(shape=shape, dtype='int32', name='mask')
            index_float = fluid.data(
                shape=shape, dtype='float32', name='index_float')

            def test_x_type():
                paddle.fluid.layers.gather(x, index)

            self.assertRaises(TypeError, test_x_type)

            def test_index_type():
                paddle.fluid.layers.gather(x, index_float)

            self.assertRaises(TypeError, test_index_type)


class TestCheckOutType(unittest.TestCase):
    def test_out_type(self):
        data = paddle.static.data(shape=[16, 10], dtype='int64', name='x')
        index = paddle.static.data(shape=[4], dtype='int64', name='index')
        out = paddle.gather(data, index)
        self.assertTrue(out.dtype == core.VarDesc.VarType.INT64)


if __name__ == "__main__":
    unittest.main()
