# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")
from op_test import OpTest
from paddle.fluid import Program, program_guard
import paddle.fluid.dygraph as dg
import paddle.static as static
from numpy.random import random as rand

paddle.enable_static()


class TestConjOp(OpTest):
    def setUp(self):
        self.op_type = "conj"
        self.init_dtype_type()
        self.init_input_output()
        self.init_grad_input_output()

    def init_dtype_type(self):
        self.dtype = np.complex64

    def init_input_output(self):
        x = (np.random.random((12, 14)) + 1j * np.random.random(
            (12, 14))).astype(self.dtype)
        out = np.conj(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_grad_input_output(self):
        self.grad_out = (np.ones((12, 14)) + 1j * np.ones(
            (12, 14))).astype(self.dtype)
        self.grad_in = np.conj(self.grad_out)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X'],
            'Out',
            user_defined_grads=[self.grad_in],
            user_defined_grad_outputs=[self.grad_out])


class TestComplexConjOp(unittest.TestCase):
    def setUp(self):
        self._dtypes = ["float32", "float64"]
        self._places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self._places.append(paddle.CUDAPlace(0))

    def test_conj_api(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = paddle.conj(var_x).numpy()
                    target = np.conj(input)
                    self.assertTrue(np.array_equal(result, target))
                    self.assertEqual(1, 2)

    def test_conj_operator(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = var_x.conj().numpy()
                    target = np.conj(input)
                    self.assertTrue(np.array_equal(result, target))

    def test_conj_static_mode(self):
        def init_input_output(dtype):
            input = rand([2, 20, 2, 3]).astype(dtype) + 1j * rand(
                [2, 20, 2, 3]).astype(dtype)
            return {'x': input}, np.conj(input)

        for dtype in self._dtypes:
            input_dict, np_res = init_input_output(dtype)
            for place in self._places:
                with static.program_guard(static.Program()):
                    x_dtype = np.complex64 if dtype == "float32" else np.complex128
                    x = static.data(
                        name="x", shape=[2, 20, 2, 3], dtype=x_dtype)
                    out = paddle.conj(x)

                    exe = static.Executor(place)
                    out_value = exe.run(feed=input_dict, fetch_list=[out.name])
                    self.assertTrue(np.array_equal(np_res, out_value[0]))

    def test_conj_api_real_number(self):
        for dtype in self._dtypes:
            input = rand([2, 20, 2, 3]).astype(dtype)
            for place in self._places:
                with dg.guard(place):
                    var_x = paddle.to_tensor(input)
                    result = paddle.conj(var_x).numpy()
                    target = np.conj(input)
                    self.assertTrue(np.array_equal(result, target))


if __name__ == "__main__":
    unittest.main()
