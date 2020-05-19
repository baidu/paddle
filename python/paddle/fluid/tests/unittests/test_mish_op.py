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
import paddle.fluid as fluid
import six
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from op_test import OpTest, skip_check_grad_ci


class TestMishOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.mish, 0.1, 20)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.mish, x_int32, 20)
            # support the input dtype is float32
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float32')
            fluid.layers.mish(x_fp16, threshold=20)


class MishTest(OpTest):
    def setUp(self):
        self.init_input_shape()
        self.op_type = "mish"

        # mish op contain calculation like: tanh, exp, log, while tanh
        # may have diff on CPUPlace(see test_activation_op.py::TestTanh)
        # set dtype as float32 here.
        x_np = np.random.uniform(-11, 11, self.x_shape).astype('float32')
        self.inputs = {'X': x_np}

        self.threshold = 10.
        softplus = x_np * (x_np > self.threshold) + np.exp(x_np) * \
                    (x_np < -self.threshold) + np.log(np.exp(x_np) + 1.) * \
                    (x_np >= -self.threshold) * (x_np <= self.threshold)
        out_np = x_np * np.tanh(softplus)

        self.outputs = {'Out': out_np}
        self.attrs = {'threshold': self.threshold}

    def init_input_shape(self):
        self.x_shape = (10, 12)

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


if __name__ == "__main__":
    unittest.main()
