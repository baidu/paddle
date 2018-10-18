#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest


def PolygonBoxRestore(input):
    shape = input.shape
    batch_size = shape[0]
    geo_channels = shape[1]
    h = shape[2]
    w = shape[3]
    h_indexes = np.array(list(range(h)) * w).reshape(
        [w, h]).transpose()[np.newaxis, :]  # [1, h, w]
    w_indexes = np.array(list(range(w)) * h).reshape(
        [h, w])[np.newaxis, :]  # [1, h, w]
    indexes = np.concatenate(
        (w_indexes, h_indexes))[np.newaxis, :]  # [1, 2, h, w]
    indexes = indexes.repeat(
        [geo_channels / 2],
        axis=0)[np.newaxis, :]  # [1, geo_channels/2, 2, h, w]
    indexes = indexes.repeat(
        [batch_size], axis=0)  # [batch_size, geo_channels/2, 2, h, w]
    return indexes.reshape(
        input.shape) * 4 - input  # [batch_size, geo_channels, h, w]


class TestPolygonBoxRestoreOp(OpTest):
    def config(self):
        self.input_shape = (1, 8, 2, 2)

    def setUp(self):
        self.config()
        self.op_type = "polygon_box_transform"
        input = np.random.random(self.input_shape).astype("float32")
        self.inputs = {'Input': input}
        output = PolygonBoxRestore(input)
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output()


class TestCase1(TestPolygonBoxRestoreOp):
    def config(self):
        self.input_shape = (2, 10, 3, 2)


class TestCase2(TestPolygonBoxRestoreOp):
    def config(self):
        self.input_shape = (3, 12, 4, 5)


if __name__ == '__main__':
    unittest.main()
