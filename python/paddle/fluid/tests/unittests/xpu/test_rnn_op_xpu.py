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
import math
import paddle.fluid.core as core
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
import sys

sys.path.append("..")
from op_test_xpu import XPUOpTest
sys.path.append("../rnn")
from rnn_numpy import SimpleRNN, LSTM, GRU
from convert import get_params_for_net

random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()


class TestRNNOp(XPUOpTest):
    def init_size(self):
        self.seq_length = 1
        self.batch_size = 1
        self.input_size = 5
        self.hidden_size = 16

    def get_weight_names(self):
        weight_names = []
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.weight_{}".format(i, j))
        for i in range(self.num_layers):
            for j in range(0, 2 * self.direction_num):
                weight_names.append("{}.bias_{}".format(i, j))
        return weight_names

    def setUp(self):
        self.init_size()
        self.op_type = "rnn"
        self.dtype = np.float32
        self.sequence_length = np.ones(
            (self.batch_size, ), dtype=np.int32) * self.seq_length
        self.num_layers = 1
        self.is_bidirec = False
        self.mode = "LSTM"
        self.is_test = False
        self.dropout = 0.0
        self.set_attrs()

        self.direction_num = 2 if self.is_bidirec else 1
        direction = "bidirectional" if self.is_bidirec else "forward"

        input = np.random.uniform(
            low=-0.1,
            high=0.1,
            size=(self.seq_length, self.batch_size,
                  self.input_size)).astype(self.dtype)

        rnn1 = LSTM(
            self.input_size,
            self.hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction=direction,
            dropout=self.dropout,
            dtype="float32")

        flat_w = get_params_for_net(rnn1)
        output, (last_hidden, last_cell) = rnn1(
            input, sequence_length=self.sequence_length)

        init_h = np.zeros(
            (self.num_layers * self.direction_num, self.batch_size,
             self.hidden_size)).astype(self.dtype)
        init_c = np.zeros(
            (self.num_layers * self.direction_num, self.batch_size,
             self.hidden_size)).astype(self.dtype)
        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'PreState': [('init_h', init_h), ('init_c', init_c)],
            'SequenceLength': self.sequence_length
        }
        if self.sequence_length is None:
            self.inputs = {
                'Input': input,
                'WeightList': flat_w,
                'PreState': [('init_h', init_h), ('init_c', init_c)],
            }
        self.attrs = {
            'dropout_prob': self.dropout,
            'is_bidirec': self.is_bidirec,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'mode': self.mode,
            'is_test': self.is_test
        }
        self.outputs = {
            'Out': output,
            "State": [('last_hidden', last_hidden), ('last_cell', last_cell)],
            'Reserve': np.ndarray((400)).astype("uint8"),
            'DropoutState': state_out
        }

    def test_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(
                place, atol=0.01, no_check_set=['Reserve', 'DropoutState'])

    def set_attrs(self):
        pass

    def test_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            if not self.is_test:
                var_name_list = self.get_weight_names()
                grad_check_list = ['Input', 'init_h', 'init_c']
                grad_check_list.extend(var_name_list)
                self.check_grad_with_place(
                    place,
                    set(grad_check_list), ['Out', 'last_hidden', 'last_cell'],
                    max_relative_error=0.1)


class TestRNNOpCase0(TestRNNOp):
    def init_size(self):
        self.seq_length = 2
        self.batch_size = 4
        self.input_size = 10
        self.hidden_size = 32


class TestRNNOpCase1(TestRNNOp):
    def init_size(self):
        self.seq_length = 5
        self.batch_size = 16
        self.input_size = 30
        self.hidden_size = 64


class TestRNNOpCase2(TestRNNOp):
    def init_size(self):
        self.seq_length = 10
        self.batch_size = 64
        self.input_size = 50
        self.hidden_size = 64


class TestRNNOpCase3(TestRNNOp):
    def init_size(self):
        self.seq_length = 2
        self.batch_size = 4
        self.input_size = 10
        self.hidden_size = 32

    def set_attrs(self):
        self.is_bidirec = True


class TestRNNOpCase4(TestRNNOp):
    def init_size(self):
        self.seq_length = 5
        self.batch_size = 16
        self.input_size = 30
        self.hidden_size = 64

    def set_attrs(self):
        self.is_bidirec = True


class TestRNNOpCase5(TestRNNOp):
    def init_size(self):
        self.seq_length = 10
        self.batch_size = 64
        self.input_size = 50
        self.hidden_size = 64

    def set_attrs(self):
        self.is_bidirec = True


if __name__ == '__main__':
    unittest.main()
