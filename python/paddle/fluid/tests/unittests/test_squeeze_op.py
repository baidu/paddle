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
from paddle.fluid import compiler, Program, program_guard
import paddle
from op_test import OpTest
paddle.enable_static()


# Correct: General.
class TestSqueezeOp(OpTest):
    def setUp(self):
        self.op_type = "squeeze"
        self.init_test_case()
        self.inputs = {"X": np.random.random(self.ori_shape).astype("float64")}
        self.init_attrs()
        self.outputs = {"Out": self.inputs["X"].reshape(self.new_shape), }

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(["X"], "Out")

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def init_attrs(self):
        self.attrs = {"axes": self.axes}


# Correct: There is mins axis.
class TestSqueezeOp1(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, -2)
        self.new_shape = (3, 40)


# Correct: No axes input.
class TestSqueezeOp2(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


# Correct: Just part of axes be squeezed. 
class TestSqueezeOp3(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (6, 5, 1, 4)


# Correct: The demension of axis is not of size 1 remains unchanged.
class TestSqueezeOp4(TestSqueezeOp):
    def init_test_case(self):
        self.ori_shape = (6, 1, 5, 1, 4, 1)
        self.axes = (1, 2)
        self.new_shape = (6, 5, 1, 4, 1)


class TestSqueezeOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of softmax_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.squeeze, x1)
            # The input axes of squeeze must be list.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.squeeze, x2, axes=0)
            # The input dtype of squeeze not support float16.
            x3 = fluid.layers.data(name='x3', shape=[4], dtype="float16")
            self.assertRaises(TypeError, fluid.layers.squeeze, x3, axes=0)


class API_TestSqueeze(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data(
                'data1', shape=[-1, 1, 10], dtype='float64')
            result_squeeze = paddle.squeeze(data1, axis=[1])
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([5, 1, 10]).astype('float64')
            result, = exe.run(feed={"data1": input1},
                              fetch_list=[result_squeeze])
            expected_result = np.squeeze(input1, axis=1)
            self.assertTrue(np.allclose(expected_result, result))


class API_TestDygraphSqueeze(unittest.TestCase):
    def test_out(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = fluid.dygraph.to_variable(input_1)
            output = paddle.squeeze(input, axis=[1])
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            self.assertTrue(np.allclose(expected_out, out_np))

    def test_axis_not_list(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = fluid.dygraph.to_variable(input_1)
            output = paddle.squeeze(input, axis=1)
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            self.assertTrue(np.allclose(expected_out, out_np))

    def test_dimension_not_1(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([5, 1, 10]).astype("int32")
            input = fluid.dygraph.to_variable(input_1)
            output = paddle.squeeze(input, axis=(1, 2))
            out_np = output.numpy()
            expected_out = np.squeeze(input_1, axis=1)
            self.assertTrue(np.allclose(expected_out, out_np))


if __name__ == "__main__":
    unittest.main()
