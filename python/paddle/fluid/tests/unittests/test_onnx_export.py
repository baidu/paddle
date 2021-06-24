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

import os
import pickle
import unittest
import numpy as np
import paddle
from paddle.static import InputSpec


class LinearNet(paddle.nn.Layer):
    def __init__(self):
        super(LinearNet, self).__init__()
        self._linear = paddle.nn.Linear(128, 10)

    def forward(self, x):
        return self._linear(x)


class Logic(paddle.nn.Layer):
    def __init__(self):
        super(Logic, self).__init__()

    def forward(self, x, y, z):
        if z:
            return x
        else:
            return y


class TestExportWithTensor(unittest.TestCase):
    def setUp(self):
        self.x_spec = paddle.static.InputSpec(
            shape=[None, 128], dtype='float32')

    def test_with_tensor(self):
        model = LinearNet()
        paddle.onnx.export(model, 'linear_net', input_spec=[self.x_spec])


class TestExportWithTensor(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor(np.random.random((1, 128)))

    def test_with_tensor(self):
        model = LinearNet()
        paddle.onnx.export(model, 'linear_net', input_spec=[self.x])


class TestExportPrunedGraph(unittest.TestCase):
    def setUp(self):
        self.x = paddle.to_tensor(np.array([1]))
        self.y = paddle.to_tensor(np.array([-1]))

    def test_prune_graph(self):
        model = Logic()
        paddle.jit.to_static(model)
        out = model(self.x, self.y, z=True)
        paddle.onnx.export(
            model, 'pruned', input_spec=[self.x], output_spec=[out])


if __name__ == '__main__':
    unittest.main()
