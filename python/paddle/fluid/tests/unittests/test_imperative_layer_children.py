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
    def __init__(self):
        super(LeNetDygraph, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 6, 3, stride=1, padding=1),
            nn.ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2),
            nn.Conv2d(
                6, 16, 5, stride=1, padding=0),
            nn.ReLU(),
            paddle.fluid.dygraph.Pool2D(2, 'max', 2))

    def forward(self, inputs):
        x = self.features(inputs)

        return x


class TestLayerChildren(unittest.TestCase):
    def test_apply_init_weight(self):
        with fluid.dygraph.guard():
            net = LeNetDygraph()
            net.eval()

            net_layers = nn.Sequential(*list(net.children()))
            net_layers.eval()

            x = paddle.rand([2, 1, 28, 28])

            y1 = net(x)
            y2 = net_layers(x)

            np.testing.assert_allclose(y1.numpy(), y2.numpy())


if __name__ == '__main__':
    unittest.main()
