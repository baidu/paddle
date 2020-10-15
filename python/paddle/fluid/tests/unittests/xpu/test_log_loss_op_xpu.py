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

import sys
sys.path.append("..")
import paddle.fluid.core as core
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


class TestXPULogLossOp(OpTest):
    def setUp(self):
        self.op_type = 'log_loss'
        samples_num = 100

        x = np.random.random((samples_num, 1)).astype("float32")
        predicted = sigmoid_array(x)
        labels = np.random.randint(0, 2, (samples_num, 1)).astype("float32")
        epsilon = 1e-7
        self.inputs = {
            'Predicted': predicted,
            'Labels': labels,
        }

        self.attrs = {'epsilon': epsilon}
        loss = -labels * np.log(predicted + epsilon) - (
            1 - labels) * np.log(1 - predicted + epsilon)
        self.outputs = {'Loss': loss}

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad(['Predicted'], 'Loss')


if __name__ == '__main__':
    unittest.main()
