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


class TestFunctionalLayers(unittest.TestCase):
    """
    """

    def setUp(self):
        paddle.disable_static()
        np.random.seed(1)

        shape = [3, 100, 120]
        self.x = paddle.to_tensor(np.random.random(shape))
        self.y = paddle.to_tensor(np.random.random(shape))

    def check(self, x, y):
        self.assertTrue(np.allclose(x.numpy(), y.numpy()))

    def test_quant_add(self):
        out_1 = paddle.add(self.x, self.y)
        out_2 = paddle.nn.quant.add()(self.x, self.y)
        self.check(out_1, out_2)

    def test_quant_subtract(self):
        out_1 = paddle.subtract(self.x, self.y)
        out_2 = paddle.nn.quant.subtract()(self.x, self.y)
        self.check(out_1, out_2)

    def test_quant_multiply(self):
        out_1 = paddle.multiply(self.x, self.y)
        out_2 = paddle.nn.quant.multiply()(self.x, self.y)
        self.check(out_1, out_2)

    def test_quant_divide(self):
        out_1 = paddle.divide(self.x, self.y)
        out_2 = paddle.nn.quant.divide()(self.x, self.y)
        self.check(out_1, out_2)

    def test_quant_reshape(self):
        reshape = [120, 300]
        out_1 = paddle.reshape(self.x, reshape)
        out_2 = paddle.nn.quant.reshape()(self.x.clone(), reshape)
        self.check(out_1, out_2)
        self.assertTrue(out_1.shape == out_2.shape)

    def test_quant_transpose(self):
        perm = [1, 2, 0]
        out_1 = paddle.transpose(self.x, perm)
        out_2 = paddle.nn.quant.transpose()(self.x.clone(), perm)
        self.check(out_1, out_2)
        self.assertTrue(out_1.shape == out_2.shape)

    def test_quant_concat(self):
        out_1 = paddle.concat([self.x, self.y], axis=0)
        out_2 = paddle.nn.quant.concat()([self.x, self.y], 0)
        self.check(out_1, out_2)
        self.assertTrue(out_1.shape == out_2.shape)

    def test_quant_flatten(self):
        start_axis = 1
        end_axis = 2
        out_1 = paddle.flatten(self.x, start_axis, end_axis)
        out_2 = paddle.nn.quant.flatten()(self.x.clone(), start_axis, end_axis)
        self.check(out_1, out_2)
        self.assertTrue(out_1.shape == out_2.shape)


if __name__ == '__main__':
    unittest.main()
