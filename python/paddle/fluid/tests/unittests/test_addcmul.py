#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
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
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard
from op_test import OpTest, skip_check_grad_ci


class TestAddcmulLayer(unittest.TestCase):
    def setUp(self):
        self._dtype = "float64"
        self.input = np.random.uniform(0.1, 1, [3, 100]).astype(self._dtype)
        self.tensor1 = np.random.uniform(0.1, 1, [100]).astype(self._dtype)
        self.tensor2 = np.random.uniform(0.1, 1, [3, 100]).astype(self._dtype)

    def static(self, value=1.0):
        prog = fluid.Program()
        with fluid.program_guard(prog):
            input = fluid.data(name="input", dtype=self._dtype, shape=[3, 100])
            tensor1 = fluid.data(name="tensor1", dtype=self._dtype, shape=[100])
            tensor2 = fluid.data(
                name="tensor2", dtype=self._dtype, shape=[3, 100])
            out = paddle.tensor.math.addcmul(input, tensor1, tensor2, value)

        exe = fluid.Executor(self._place)
        return exe.run(feed={
            "input": self.input,
            "tensor1": self.tensor1,
            "tensor2": self.tensor2
        },
                       program=prog,
                       fetch_list=[out])[0]

    def dynamic(self, value=1.0):
        with fluid.dygraph.guard(self._place):
            input = fluid.dygraph.to_variable(self.input)
            tensor1 = fluid.dygraph.to_variable(self.tensor1)
            tensor2 = fluid.dygraph.to_variable(self.tensor2)
            out = paddle.tensor.math.addcmul(input, tensor1, tensor2, value)
            return out.numpy()

    def numpy(self, value=1.0):
        self.out = np.add(self.input,
                          np.multiply(self.tensor1, self.tensor2) * value)
        return self.out

    def test_equal(self):
        places = []
        if fluid.core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            self._place = place
            self.assertTrue(np.allclose(self.numpy(), self.static()))
            self.assertTrue(
                np.allclose(
                    self.numpy(value=0.9), self.dynamic(value=0.9)))
            self.assertTrue(
                np.allclose(
                    self.numpy(value=0), self.dynamic(value=0)))


class TestAddcmul(unittest.TestCase):
    def test_addcmul(self):
        program = Program()
        with program_guard(program):
            data_shape = [3, 64, 64]
            input = fluid.data(name='in', shape=data_shape, dtype='float32')
            tensor1 = fluid.data(name='t1', shape=data_shape, dtype='float32')
            tensor2 = fluid.data(name='t2', shape=data_shape, dtype='float32')

            out = paddle.tensor.math.addcmul(input, tensor1, tensor2)
            self.assertEqual(out.shape, input.shape)

    def test_addcmul_with_broadcast0(self):
        program = Program()
        with program_guard(program):
            input = fluid.data(name='in', shape=[3, 100], dtype='float32')
            tensor1 = fluid.data(name='t1', shape=[3, 100], dtype='float32')
            tensor2 = fluid.data(name='t2', shape=[100], dtype='float32')

            out = paddle.tensor.math.addcmul(input, tensor1, tensor2)
            self.assertEqual(out.shape, input.shape)

    def test_addcmul_with_broadcast1(self):
        program = Program()
        with program_guard(program):
            input = fluid.data(name='in', shape=[4, 100], dtype='float32')
            tensor1 = fluid.data(name='t1', shape=[100], dtype='float32')
            tensor2 = fluid.data(name='t2', shape=[4, 100], dtype='float32')

            out = paddle.tensor.math.addcmul(input, tensor1, tensor2)
            self.assertEqual(out.shape, input.shape)

    def test_addcmul_with_broadcast2(self):
        program = Program()
        with program_guard(program):
            input = fluid.data(name='in', shape=[4, 100], dtype='float32')
            tensor1 = fluid.data(name='t1', shape=[100], dtype='float32')
            tensor2 = fluid.data(name='t2', shape=[100], dtype='float32')

            out = paddle.tensor.math.addcmul(input, tensor1, tensor2)
            self.assertEqual(out.shape, input.shape)


class InvalidInputTest(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            program = Program()
            with program_guard(program):
                input = [20, 20]
                tensor1 = fluid.data(
                    name='tensor1', shape=[20, 20], dtype='float32')
                tensor2 = fluid.data(
                    name='tensor2', shape=[20, 20], dtype='float32')
                out = paddle.tensor.math.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_tensor1():
            program = Program()
            with program_guard(program):
                input = fluid.data(
                    name='input', shape=[20, 20], dtype='float32')
                tensor1 = [20, 20]
                tensor2 = fluid.data(
                    name='tensor2', shape=[20, 20], dtype='float32')
                out = paddle.tensor.math.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_tensor1)

        def test_invalid_tensor2():
            program = Program()
            with program_guard(program):
                input = fluid.data(
                    name='input', shape=[20, 20], dtype='float32')
                tensor1 = fluid.data(
                    name='tensor1', shape=[20, 20], dtype='float32')
                tensor2 = [20, 20]
                out = paddle.tensor.math.addcmul(input, tensor1, tensor2)

        self.assertRaises(TypeError, test_invalid_tensor2)

        def test_invalid_value_int():
            program = Program()
            with program_guard(program):
                input = fluid.data(
                    name='input', shape=[20, 20], dtype='float32')
                tensor1 = fluid.data(
                    name='tensor1', shape=[20, 20], dtype='float32')
                tensor2 = fluid.data(
                    name='tensor2', shape=[20, 20], dtype='float32')
                out = paddle.tensor.math.addcmul(input, tensor1, tensor2, value=1)

        self.assertRaises(TypeError, test_invalid_value_int)

        def test_invalid_value_float():
            program = Program()
            with program_guard(program):
                input = fluid.data(name='input', shape=[20, 20], dtype='int32')
                tensor1 = fluid.data(
                    name='tensor1', shape=[20, 20], dtype='int32')
                tensor2 = fluid.data(
                    name='tensor2', shape=[20, 20], dtype='int32')
                out = paddle.tensor.math.addcmul(input, tensor1, tensor2, value=1.0)

        self.assertRaises(TypeError, test_invalid_value_float)


if __name__ == '__main__':
    unittest.main()
