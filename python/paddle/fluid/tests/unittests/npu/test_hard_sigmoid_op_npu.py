#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.fluid import Program, program_guard

paddle.enable_static()
SEED = 2021


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.), 0.).astype(x.dtype)


class TestNPUHardSigmoid(OpTest):
    def setUp(self):
        paddle.enable_static()

        self.op_type = "hard_sigmoid"
        self.set_npu()
        self.init_dtype()
        self.set_attrs()

        x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
        lower_threshold = -self.offset / self.slope
        upper_threshold = (1. - self.offset) / self.slope

        # Same reason as TestAbs
        delta = 0.005
        x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
        x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

        out = ref_hardsigmoid(x, self.slope, self.offset)

        self.attrs = {'slope': self.slope, 'offset': self.offset}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-5)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return

        self.check_grad_with_place(self.place, ['X'], 'Out')

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def set_attrs(self):
        self.slope = 0.166666666666667
        self.offset = 0.5


class TestNPUHardSigmoid2(TestNPUHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.5


class TestNPUHardSigmoid3(TestNPUHardSigmoid):
    def set_attrs(self):
        self.slope = 0.2
        self.offset = 0.4


class TestNPUHardSigmoidFp16(TestNPUHardSigmoid):
    def test_check_output(self):
        self.check_output_with_place(self.place, atol=1e-3)

    def init_dtype(self):
        self.dtype = np.float16


class TestHardsigmoidAPI(unittest.TestCase):
    # test paddle.nn.Hardsigmoid, paddle.nn.functional.hardsigmoid
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [10, 12]).astype(np.float32)
        self.place = paddle.NPUPlace(0)

    def test_static_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.hardsigmoid(x)
            m = paddle.nn.Hardsigmoid()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_hardsigmoid(self.x_np)
        for r in res:
            self.assertTrue(np.allclose(out_ref, r))

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.hardsigmoid(x)
        m = paddle.nn.Hardsigmoid()
        out2 = m(x)
        out_ref = ref_hardsigmoid(self.x_np)
        for r in [out1, out2]:
            self.assertTrue(np.allclose(out_ref, r.numpy()))
        paddle.enable_static()

    def test_fluid_api(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.hard_sigmoid(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_hardsigmoid(self.x_np, 0.2, 0.5)
        self.assertTrue(np.allclose(out_ref, res[0]))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out = paddle.fluid.layers.hard_sigmoid(x)
        self.assertTrue(np.allclose(out_ref, out.numpy()))
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.hardsigmoid, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.hardsigmoid, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.hardsigmoid(x_fp16)


if __name__ == '__main__':
    unittest.main()
