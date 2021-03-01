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
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
from paddle.fluid import core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
import paddle


class TestLambOp1(XPUOpTest):
    def set_attrs(self):
        self.attrs = {
            'epsilon': 1e-6,
            'beta1': 0.9,
            'beta2': 0.999,
            'weight_decay': 0.01
        }

    def setUp(self):
        '''Test Lamb Op with supplied attributes
        '''
        self.op_type = "lamb"
        param = np.random.uniform(-1, 1, 10000).astype("float32")
        grad = np.random.uniform(-1, 1, 10000).astype("float32")
        moment1 = np.random.uniform(-1, 1, 10000).astype("float32")
        moment2 = np.random.random(10000).astype("float32")

        self.set_attrs()
        learning_rate = 0.001
        beta1_pow = self.attrs['beta1']
        beta2_pow = self.attrs['beta2']

        self.inputs = {
            'Param': param,
            'Grad': grad,
            'Moment1': moment1,
            'Moment2': moment2,
            'LearningRate': np.array([learning_rate]).astype("float32"),
            'Beta1Pow': np.array([beta1_pow]).astype("float32"),
            'Beta2Pow': np.array([beta2_pow]).astype("float32")
        }

        param_out, moment1_out, moment2_out, \
            beta1_pow_out, beta2_pow_out = lamb_step(self.inputs, self.attrs)

        self.outputs = {
            'Moment1Out': moment1_out,
            'Moment2Out': moment2_out,
            'ParamOut': param_out,
            'Beta1PowOut': beta1_pow_out,
            'Beta2PowOut': beta2_pow_out
        }

    def test_check_output(self):
        self.check_output_with_place(paddle.XPUPlace(0))


def lamb_step(inputs, attributes):
    '''
    Simulate one step of the lamb optimizer
    :param inputs: dict of inputs
    :param attributes: dict of attributes
    :return tuple: tuple of output param, moment1, moment2,
    beta1 power accumulator and beta2 power accumulator
    '''
    param = inputs['Param']
    grad = inputs['Grad']
    moment1 = inputs['Moment1']
    moment2 = inputs['Moment2']
    lr = inputs['LearningRate']
    beta1_pow = inputs['Beta1Pow']
    beta2_pow = inputs['Beta2Pow']

    beta1 = attributes['beta1']
    beta2 = attributes['beta2']
    epsilon = attributes['epsilon']
    weight_decay = attributes['weight_decay']

    moment1_out = beta1 * moment1 + (1 - beta1) * grad
    moment2_out = beta2 * moment2 + (1 - beta2) * np.square(grad)

    moment1_unbiased = moment1_out / (1 - beta1_pow)
    moment2_unbiased = moment2_out / (1 - beta2_pow)

    r_1 = np.linalg.norm(param)
    r_2 = np.linalg.norm(moment1_unbiased / (np.sqrt(moment2_unbiased) + epsilon
                                             ) + weight_decay * param)
    if r_1 > 0.0 and r_2 > 0.0:
        lr_t = lr * r_1 / r_2
    else:
        lr_t = 1.0

    param_out = param - lr_t * (moment1_unbiased / (
        np.sqrt(moment2_unbiased) + epsilon) + weight_decay * param)

    beta1_pow_out = beta1_pow * beta1
    beta2_pow_out = beta2_pow * beta2

    return param_out, moment1_out, moment2_out, beta1_pow_out, beta2_pow_out


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
