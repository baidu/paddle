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

import paddle
paddle.set_default_dtype("float64")
from paddle.fluid.layers import sequence_mask

import numpy as np
import unittest

from convert import convert_params_for_cell
from rnn_numpy import GRUCell, RNN, BiRNN


class TestRNNWrapper(unittest.TestCase):
    def __init__(self, time_major=True, direction="forward", place="cpu"):
        super(TestRNNWrapper, self).__init__("runTest")
        self.time_major = time_major
        self.direction = direction
        self.place = paddle.CPUPlace() if place == "cpu" \
            else paddle.CUDAPlace(0)

    def setUp(self):
        paddle.disable_static(self.place)
        cell1 = GRUCell(16, 32)
        cell2 = paddle.nn.GRUCell(16, 32)
        convert_params_for_cell(cell1, cell2)
        rnn1 = RNN(cell1,
                   is_reverse=self.direction == "backward",
                   time_major=self.time_major)
        rnn2 = paddle.nn.RNN(cell2,
                             is_reverse=self.direction == "backward",
                             time_major=self.time_major)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        prev_h = np.random.randn(4, 32)

        y1, h1 = rnn1(x, prev_h)
        y2, h2 = rnn2(paddle.to_tensor(x), paddle.to_tensor(prev_h))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, h1 = rnn1(x)
        y2, h2 = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_input_lengths(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)

        y1, h1 = rnn1(x, sequence_length=sequence_length)

        seq_len = paddle.to_tensor(sequence_length)
        mask = sequence_mask(seq_len, dtype=paddle.get_default_dtype())
        if self.time_major:
            mask = paddle.transpose(mask, [1, 0])
        y2, h2 = rnn2(paddle.to_tensor(x), sequence_length=seq_len)
        y2 = paddle.multiply(y2, mask, axis=0)

        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(h1, h2.numpy(), atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()


class TestBiRNNWrapper(unittest.TestCase):
    def __init__(self, time_major=True, place="cpu"):
        super(TestBiRNNWrapper, self).__init__("runTest")
        self.time_major = time_major
        self.place = paddle.CPUPlace() if place == "cpu" \
            else paddle.CUDAPlace(0)

    def setUp(self):
        paddle.disable_static(self.place)
        fw_cell1 = GRUCell(16, 32)
        bw_cell1 = GRUCell(16, 32)
        fw_cell2 = paddle.nn.GRUCell(16, 32)
        bw_cell2 = paddle.nn.GRUCell(16, 32)
        convert_params_for_cell(fw_cell1, fw_cell2)
        convert_params_for_cell(bw_cell1, bw_cell2)
        rnn1 = BiRNN(fw_cell1, bw_cell1, time_major=self.time_major)
        rnn2 = paddle.nn.BiRNN(fw_cell2, bw_cell2, time_major=self.time_major)

        self.rnn1 = rnn1
        self.rnn2 = rnn2

    def test_with_initial_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        fw_prev_h = np.random.randn(4, 32)
        bw_prev_h = np.random.randn(4, 32)

        y1, (fw_h1, bw_h1) = rnn1(x, (fw_prev_h, bw_prev_h))
        y2, (fw_h2, bw_h2) = rnn2(
            paddle.to_tensor(x),
            (paddle.to_tensor(fw_prev_h), paddle.to_tensor(bw_prev_h)))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(fw_h1, fw_h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(bw_h1, bw_h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_zero_state(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])

        y1, (fw_h1, bw_h1) = rnn1(x)
        y2, (fw_h2, bw_h2) = rnn2(paddle.to_tensor(x))
        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(fw_h1, fw_h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(bw_h1, bw_h2.numpy(), atol=1e-8, rtol=1e-5)

    def test_with_input_lengths(self):
        rnn1 = self.rnn1
        rnn2 = self.rnn2

        x = np.random.randn(12, 4, 16)
        if not self.time_major:
            x = np.transpose(x, [1, 0, 2])
        sequence_length = np.array([12, 10, 9, 8], dtype=np.int64)

        y1, (fw_h1, bw_h1) = rnn1(x, sequence_length=sequence_length)

        seq_len = paddle.to_tensor(sequence_length)
        mask = sequence_mask(seq_len, dtype=paddle.get_default_dtype())
        if self.time_major:
            mask = paddle.transpose(mask, [1, 0])
        y2, (fw_h2, bw_h2) = rnn2(paddle.to_tensor(x), sequence_length=seq_len)
        y2 = paddle.multiply(y2, mask, axis=0)

        np.testing.assert_allclose(y1, y2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(fw_h1, fw_h2.numpy(), atol=1e-8, rtol=1e-5)
        np.testing.assert_allclose(bw_h1, bw_h2.numpy(), atol=1e-8, rtol=1e-5)

    def runTest(self):
        self.test_with_initial_state()
        self.test_with_zero_state()
        self.test_with_input_lengths()


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    devices = ["cpu", "gpu"] if paddle.fluid.is_compiled_with_cuda() \
        else ["cpu"]
    for direction in ["forward", "backward"]:
        for device in devices:
            for time_major in [False]:
                suite.addTest(TestRNNWrapper(time_major, direction, device))
            suite.addTest(TestBiRNNWrapper(time_major, device))
    return suite
