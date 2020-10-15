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
import sys
import os
sys.path.append("..")
from op_test import OpTest
import paddle
from paddle.fluid import core
from paddle.fluid.op import Operator


class TestMomentumOp1(OpTest):
    def setUp(self):
        self.op_type = "momentum"
        self.dtype = np.float32
        self.init_dtype()

        param = np.random.random((123, 321)).astype(self.dtype)
        grad = np.random.random((123, 321)).astype(self.dtype)
        velocity = np.zeros((123, 321)).astype(self.dtype)
        learning_rate = np.array([0.001]).astype(self.dtype)
        mu = 0.0001
        use_nesterov = False

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Velocity': velocity,
            'LearningRate': learning_rate
        }

        self.attrs = {'mu': mu}

        velocity_out = mu * velocity + grad
        if use_nesterov:
            param_out = param - grad * learning_rate - \
                        velocity_out * mu * learning_rate
        else:
            param_out = param - learning_rate * velocity_out

        self.outputs = {'ParamOut': param_out, 'VelocityOut': velocity_out}

    def init_dtype(self):
        pass

    def test_check_output_with_place(self):
        self.check_output_with_place(paddle.XPUPlace(0))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
