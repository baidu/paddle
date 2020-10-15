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

import paddle
import paddle.nn as nn
import paddle.fluid as fluid

import numpy as np


class LeNetDygraph(fluid.dygraph.Layer):
    def __init__(self, num_classes=10, classifier_activation='softmax'):
        super(LeNetDygraph, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
            nn.Conv2d(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2))

        if num_classes > 0:
            self.fc = nn.Sequential(
                nn.Linear(400, 120),
                nn.Linear(120, 84), nn.Linear(84, 10),
                nn.Softmax())  #Todo: accept any activation

    def forward(self, inputs):
        x = self.features(inputs)

        if self.num_classes > 0:
            x = fluid.layers.flatten(x, 1)
            x = self.fc(x)
        return x


def init_weights(layer):
    if type(layer) == nn.Linear:
        new_weight = paddle.fluid.layers.fill_constant(
            layer.weight.shape, layer.weight.dtype, value=0.9)
        layer.weight.set_value(new_weight)
        new_bias = paddle.fluid.layers.fill_constant(
            layer.bias.shape, layer.bias.dtype, value=-0.1)
        layer.bias.set_value(new_bias)
    elif type(layer) == nn.Conv2d:
        new_weight = paddle.fluid.layers.fill_constant(
            layer.weight.shape, layer.weight.dtype, value=0.7)
        layer.weight.set_value(new_weight)
        new_bias = paddle.fluid.layers.fill_constant(
            layer.bias.shape, layer.bias.dtype, value=-0.2)
        layer.bias.set_value(new_bias)


class TestLayerApply(unittest.TestCase):
    def test_apply_init_weight(self):
        with fluid.dygraph.guard():
            net = LeNetDygraph()

            net.apply(init_weights)

            for layer in net.sublayers():
                if type(layer) == nn.Linear:
                    np.testing.assert_allclose(layer.weight.numpy(), 0.9)
                    np.testing.assert_allclose(layer.bias.numpy(), -0.1)
                elif type(layer) == nn.Conv2d:
                    np.testing.assert_allclose(layer.weight.numpy(), 0.7)
                    np.testing.assert_allclose(layer.bias.numpy(), -0.2)


if __name__ == '__main__':
    unittest.main()
