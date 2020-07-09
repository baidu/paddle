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
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.tensor as tensor


class TestTraceOp(OpTest):
    def setUp(self):
        self.op_type = "trace"
        self.init_config()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out')

    def init_config(self):
        self.case = np.random.randn(20, 6).astype('float64')
        self.inputs = {'Input': self.case}
        self.attrs = {'diagonal': 0, 'start_axis': 0, 'stop_axis': 1}
        self.target = np.trace(self.inputs['Input'])


class TestTraceOpCase1(TestTraceOp):
    def init_config(self):
        self.case = np.random.randn(2, 20, 2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'diagonal': 1, 'start_axis': 0, 'stop_axis': 2}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['diagonal'],
            axis1=self.attrs['start_axis'],
            axis2=self.attrs['stop_axis'])


class TestTraceOpCase2(TestTraceOp):
    def init_config(self):
        self.case = np.random.randn(2, 20, 2, 3).astype('float32')
        self.inputs = {'Input': self.case}
        self.attrs = {'diagonal': -5, 'start_axis': 1, 'stop_axis': -1}
        self.target = np.trace(
            self.inputs['Input'],
            offset=self.attrs['diagonal'],
            axis1=self.attrs['start_axis'],
            axis2=self.attrs['stop_axis'])


class TestTraceAPICase(unittest.TestCase):
    def test_case1(self):
        case = np.random.randn(2, 20, 2, 3).astype('float32')
        data1 = fluid.data(name='data1', shape=[2, 20, 2, 3], dtype='float32')
        out1 = tensor.trace(data1)
        out2 = tensor.trace(data1, diagonal=-5, start_axis=1, stop_axis=-1)

        place = core.CPUPlace()
        exe = fluid.Executor(place)
        results = exe.run(fluid.default_main_program(),
                          feed={"data1": case},
                          fetch_list=[out1, out2],
                          return_numpy=True)
        target1 = np.trace(case)
        target2 = np.trace(case, offset=-5, axis1=1, axis2=-1)
        self.assertTrue(np.allclose(results[0], target1))
        self.assertTrue(np.allclose(results[1], target2))


if __name__ == "__main__":
    unittest.main()
