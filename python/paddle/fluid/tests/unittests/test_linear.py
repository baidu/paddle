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
import paddle.fluid.core as core
from op_test import OpTest
import paddle
from paddle import fluid, nn
import paddle.fluid.dygraph as dg
import paddle.nn.functional as F
import paddle.fluid.initializer as I


class LinearTestCase(unittest.TestCase):
    def setUp(self):
        self.dtype = 'float32'
        self.input = np.ones((3, 1, 2)).astype(self.dtype)
        self.weight = np.ones((2, 2)).astype(self.dtype)
        self.bias = np.ones((2)).astype(self.dtype)
        self.place = paddle.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else paddle.CPUPlace()

    def functional(self, place):
        paddle.disable_static(place)
        input = paddle.to_tensor(self.input)
        weight = paddle.to_tensor(self.weight)
        bias = paddle.to_tensor(self.bias)
        out = F.linear(input, weight, bias)
        return out.numpy()

    def paddle_nn_layer(self, place):
        paddle.disable_static(place)
        input = paddle.to_tensor(self.input)
        weight_attr = fluid.ParamAttr(
            name="linear_weight",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.fluid.initializer.ConstantInitializer(value=1.0))
        bias_attr = fluid.ParamAttr(
            name="linear_bias",
            learning_rate=1.0,
            trainable=False,
            regularizer=None,
            initializer=paddle.fluid.initializer.ConstantInitializer(value=1.0))
        linear = paddle.nn.Linear(
            2, 2, weight_attr=weight_attr, bias_attr=bias_attr)
        y = linear(input)
        return y.numpy()

    def numpy_cal(self):
        res = np.matmul(self.input, self.weight) + self.bias
        return res

    def test_error(self, place=paddle.CPUPlace()):
        res_f = self.functional(place)
        res_nn = self.paddle_nn_layer(place)
        res_np = self.numpy_cal()
        self.assertEqual(1, 2)
        np.testing.assert_array_almost_equal(res_f, res_nn)
        np.testing.assert_array_almost_equal(res_nn, res_np)


if __name__ == "__main__":
    unittest.main()
