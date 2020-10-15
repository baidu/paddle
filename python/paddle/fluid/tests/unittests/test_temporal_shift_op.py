#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import division

import unittest
import numpy as np
from op_test import OpTest

import paddle
from paddle.fluid import core


def temporal_shift(x, seg_num, shift_ratio):
    shape = x.shape
    reshape_x = x.reshape((-1, seg_num, shape[1], shape[2], shape[3]))
    pad_x = np.pad(reshape_x, ((0, 0), (1, 1), (0, 0), (0, 0), (0, 0)),
                   'constant')
    c1 = int(shape[1] * shift_ratio)
    c2 = int(shape[1] * 2 * shift_ratio)
    slice1 = pad_x[:, :seg_num, :c1, :, :]
    slice2 = pad_x[:, 2:seg_num + 2, c1:c2, :, :]
    slice3 = pad_x[:, 1:seg_num + 1, c2:, :, :]
    concat_x = np.concatenate([slice1, slice2, slice3], axis=2)
    return concat_x.reshape(shape)


class TestTemporalShift(OpTest):
    def setUp(self):
        self.initTestCase()
        self.op_type = 'temporal_shift'
        x = np.random.random(self.x_shape).astype('float64')

        self.attrs = {
            "seg_num": self.seg_num,
            "shift_ratio": self.shift_ratio,
        }

        self.inputs = {"X": x, }

        output = temporal_shift(x, self.seg_num, self.shift_ratio)
        self.outputs = {"Out": output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_ignore_uv(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.x_shape = (6, 4, 4, 4)
        self.seg_num = 3
        self.shift_ratio = 0.25


class TestTemporalShift2(TestTemporalShift):
    def initTestCase(self):
        self.x_shape = (4, 9, 7, 7)
        self.seg_num = 2
        self.shift_ratio = 0.2


class TestTemporalShift3(TestTemporalShift):
    def initTestCase(self):
        self.x_shape = (3, 10, 5, 5)
        self.seg_num = 1
        self.shift_ratio = 0.3


class TestTemporalShiftAPI(unittest.TestCase):
    def test_api(self):
        input = paddle.randn([6, 4, 2, 2])
        out = paddle.fluid.layers.temporal_shift(
            x=input, seg_num=2, shift_ratio=0.2)


if __name__ == "__main__":
    unittest.main()
