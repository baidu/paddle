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
from op_test import OpTest
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.tensor as tensor

paddle.enable_static()


class TestDiagonalOp(OpTest):
    def setUp(self):
        self.op_type = "diagonal"
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out')

    def init_config(self):
        self.case = np.random.randn(10, 5, 2).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])


class TestDiagonalOpCase1(TestDiagonalOp):
    def init_config(self):
        self.case = np.random.randn(4, 2, 4, 4).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 3, 'axis2': 0}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])


class TestDiagonalOpCase2(TestDiagonalOp):
    def init_config(self):
        self.case = np.random.randn(100, 100).astype('int64')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': 0, 'axis1': 0, 'axis2': 1}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])
        self.grad_x = np.eye(100).astype('int64')
        self.grad_out = np.ones(100).astype('int64')

    def test_check_grad(self):
        self.check_grad(
            ['Input'],
            'Out',
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out])


class TestDiagonalOpCase3(TestDiagonalOp):
    def init_config(self):
        self.case = np.random.randint(0, 2, (4, 2, 4, 4)).astype('bool')
        self.inputs = {'Input': self.case}
        self.attrs = {'offset': -2, 'axis1': 3, 'axis2': 0}
        self.target = np.diagonal(
            self.inputs['Input'],
            offset=self.attrs['offset'],
            axis1=self.attrs['axis1'],
            axis2=self.attrs['axis2'])

    def test_check_grad(self):
        pass


class TestDiagonalAPI(unittest.TestCase):
    def setUp(self):
        self.shape = [10, 3, 4]
        self.x = np.random.random((10, 3, 4)).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            out = paddle.diagonal(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.diagonal(self.x)
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-08), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.diagonal(x_tensor)
        out_ref = np.diagonal(self.x)
        self.assertEqual(np.allclose(out.numpy(), out_ref, rtol=1e-08), True)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
