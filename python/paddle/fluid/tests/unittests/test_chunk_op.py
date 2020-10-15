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
from op_test import OpTest
import numpy as np
from paddle.fluid import Program, program_guard
from paddle import fluid
import paddle


class TestChunkOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of axis in chunk_op should be int or Variable.
            def test_axis_type():
                x1 = paddle.fluid.data(shape=[4], dtype='float16', name='x3')
                paddle.chunk(x=x1, chunks=2, axis=3.2)

            self.assertRaises(TypeError, test_axis_type)

            # The type of axis in chunk op should be int or Variable.
            def test_axis_variable_type():
                x2 = paddle.fluid.data(shape=[4], dtype='float16', name='x9')
                x3 = paddle.fluid.data(shape=[1], dtype='float16', name='x10')
                paddle.chunk(input=x2, chunks=2, axis=x3)

            self.assertRaises(TypeError, test_axis_variable_type)

            # The type of num_or_sections in chunk_op should be int, tuple or list.
            def test_chunks_type():
                x4 = paddle.fluid.data(shape=[4], dtype='float16', name='x4')
                paddle.chunk(input=x4, chunks=2.1, axis=3)

            self.assertRaises(TypeError, test_chunks_type)

            def test_axis_type_tensor():
                x5 = paddle.fluid.data(shape=[4], dtype='float16', name='x6')
                paddle.chunk(input=x5, chunks=2, axis=3.2)

            self.assertRaises(TypeError, test_axis_type_tensor)


class API_TestChunk(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = paddle.fluid.data('data1', shape=[4, 6, 6], dtype='float64')
            data2 = paddle.fluid.data('data2', shape=[1], dtype='int32')
            x0, x1, x2 = paddle.chunk(data1, chunks=3, axis=data2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            input2 = np.array([2]).astype('int32')
            r0, r1, r2, = exe.run(feed={"data1": input1,
                                        "data2": input2},
                                  fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.array_split(input1, 3, axis=2)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))
            self.assertTrue(np.allclose(ex_x2, r2))


class API_TestChunk1(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = paddle.fluid.data('data1', shape=[4, 6, 6], dtype='float64')
            x0, x1, x2 = paddle.chunk(data1, chunks=3, axis=2)
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            r0, r1, r2, = exe.run(feed={"data1": input1},
                                  fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.array_split(input1, 3, axis=2)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))
            self.assertTrue(np.allclose(ex_x2, r2))


class API_TestDygraphChunk(unittest.TestCase):
    def test_out1(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = fluid.dygraph.to_variable(input_1)
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))

    def test_out2(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("bool")
            # input is a variable which shape is [4, 6, 6]
            input = fluid.dygraph.to_variable(input_1)
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))

    def test_axis_tensor_input(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = fluid.dygraph.to_variable(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            x0, x1, x2 = paddle.chunk(input, chunks=3, axis=num1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.array_split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))


if __name__ == '__main__':
    unittest.main()
