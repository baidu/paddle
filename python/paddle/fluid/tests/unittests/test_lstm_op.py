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
from paddle import fluid
from paddle.fluid.layers import lstm as LSTM
from paddle.fluid.layers import fill_constant
from paddle.fluid.framework import program_guard, Program

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


def identity(x):
    return x


def sigmoid(x):
    y = np.copy(x)
    y[x < SIGMOID_THRESHOLD_MIN] = SIGMOID_THRESHOLD_MIN
    y[x > SIGMOID_THRESHOLD_MAX] = SIGMOID_THRESHOLD_MAX
    return 1. / (1. + np.exp(-y))


def tanh(x):
    y = -2. * x
    y[y > EXP_MAX_INPUT] = EXP_MAX_INPUT
    return (2. / (1. + np.exp(y))) - 1.


def relu(x):
    return np.maximum(x, 0)


ACTIVATION = {
    'identity': identity,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu
}


def lstm(
        input,  # T x 4D
        lod,  # 1 x N
        h0=None,  # N x D
        c0=None,  # N x D
        w_h=None,  # D x 4D
        w_b=None,  # 1 x 4D
        w_c=None,  # 1 x 3D
        is_reverse=False,
        act_gate=None,
        act_cell=None,
        act_cand=None):
    def _step(x, w_h, w_c, h_pre, c_pre, act_gate, act_cell, act_cand):
        g = np.dot(h_pre, w_h)  # 1 x 4D
        g = g + x
        g = np.reshape(g, (1, g.size))
        c, g_i, g_f, g_o = np.split(g, 4, axis=1)
        if w_c is None:
            g_i = act_gate(g_i)  # 1 x D
            g_f = act_gate(g_f)  # 1 x D
        else:
            w_ic, w_fc, w_oc = np.split(w_c, 3, axis=1)
            g_i = act_gate(g_i + w_ic * c_pre)  # 1 x D
            g_f = act_gate(g_f + w_fc * c_pre)  # 1 x D
        c = g_f * c_pre + g_i * act_cand(c)  # 1 x D

        if w_c is None:
            g_o = act_gate(g_o)  # 1 x D
        else:
            _, _, w_oc = np.split(w_c, 3, axis=1)
            g_o = act_gate(g_o + w_oc * c)  # 1 x D
        h = g_o * act_cell(c)
        return h, c

    def _reverse(x, offset):
        y = np.zeros_like(x)
        for i in range(len(offset) - 1):
            b, e = offset[i], offset[i + 1]
            y[b:e, :] = np.flip(x[b:e, :], 0)
        return y

    offset = [0]
    for l in lod[0]:
        offset.append(offset[-1] + l)
    batch_size = len(lod[0])
    hidden = []
    cell = []
    input = _reverse(input, offset) if is_reverse else input
    if w_b is not None:
        input = input + np.tile(w_b, (offset[-1], 1))
    for i in range(batch_size):
        # compute one sequence
        seq_len = lod[0][i]
        x = input[offset[i]:offset[i + 1], :]
        h_pre = h0[i]  # 1 x D
        c_pre = c0[i]  # 1 x D
        for j in range(seq_len):
            # compute one step
            h_pre, c_pre = _step(x[j], w_h, w_c, h_pre, c_pre, act_gate,
                                 act_cell, act_cand)
            hidden.append(h_pre.flatten())
            cell.append(c_pre.flatten())

    hidden = np.array(hidden).astype('float64')
    cell = np.array(cell).astype('float64')

    hidden = _reverse(hidden, offset) if is_reverse else hidden
    cell = _reverse(cell, offset) if is_reverse else cell

    assert hidden.shape == (input.shape[0], input.shape[1] / 4)
    assert cell.shape == (input.shape[0], input.shape[1] / 4)
    return hidden, cell


class LstmUnitTestError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            batch_size = 20
            seq_len = 100
            dropout_prob = 0.2
            hidden_size = 150
            num_layers = 1
            input = fluid.data(
                name='input',
                shape=[batch_size, seq_len, hidden_size],
                dtype='float32')
            pre_hidden = fill_constant([num_layers, batch_size, hidden_size],
                                       'float32', 0.0)
            pre_cell = fill_constant([num_layers, batch_size, hidden_size],
                                     'float32', 0.0)

            np_input = np.random.uniform(
                -0.1, 0.1, (batch_size, seq_len, hidden_size)).astype('float64')
            np_pre_hidden = np.random.uniform(
                -0.1, 0.1,
                (num_layers, batch_size, hidden_size)).astype('float64')
            np_pre_cell = np.random.uniform(
                -0.1, 0.1,
                (num_layers, batch_size, hidden_size)).astype('float64')

            def test_input_Variable():
                LSTM(np_input, pre_hidden, pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_input_Variable)

            def test_pre_hidden_Variable():
                LSTM(np_input, np_pre_hidden, pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_pre_hidden_Variable)

            def test_pre_cell_Variable():
                LSTM(np_input, pre_hidden, np_pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_pre_cell_Variable)

            def test_input_type():
                error_input = fluid.data(
                    name='error_input',
                    shape=[None, hidden_size * 3],
                    dtype='int32')
                LSTM(error_input, pre_hidden, pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_input_type)

            def test_pre_hidden_type():
                error_pre_hidden = fluid.data(
                    name='error_pre_hidden',
                    shape=[None, hidden_size],
                    dtype='int32')
                LSTM(input, error_pre_hidden, pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_pre_hidden_type)

            def test_pre_cell_type():
                error_pre_cell = fluid.data(
                    name='error_pre_cell',
                    shape=[None, hidden_size],
                    dtype='int32')
                LSTM(input, pre_hidden, error_pre_cell, \
                    seq_len, hidden_size, num_layers, \
                    dropout_prob=dropout_prob)

            self.assertRaises(TypeError, test_pre_cell_type)


class TestLstmOp(OpTest):
    def set_lod(self):
        self.lod = [[2, 3, 2]]

    def set_argument(self):
        self.set_lod()
        self.D = 16

        self.act_gate = 'sigmoid'
        self.act_cell = 'tanh'
        self.act_cand = 'tanh'

        self.has_initial_state = False
        self.is_reverse = False
        self.use_peepholes = True

    def setUp(self):
        self.set_argument()
        self.op_type = 'lstm'
        T = sum(self.lod[0])
        N = len(self.lod[0])

        x = np.random.normal(size=(T, 4 * self.D)).astype('float64')
        if self.has_initial_state:
            h0 = np.random.normal(size=(N, self.D)).astype('float64')
            c0 = np.random.normal(size=(N, self.D)).astype('float64')
        else:
            h0 = np.zeros((N, self.D)).astype('float64')
            c0 = np.zeros((N, self.D)).astype('float64')
        w = np.random.normal(size=(self.D, 4 * self.D)).astype('float64')
        if self.use_peepholes:
            b = np.random.normal(size=(1, 7 * self.D)).astype('float64')
        else:
            b = np.random.normal(size=(1, 4 * self.D)).astype('float64')

        w_b = b[:, 0:4 * self.D]
        w_c = b[:, 4 * self.D:] if self.use_peepholes else None
        h, c = lstm(x, self.lod, h0, c0, w, w_b, w_c, self.is_reverse,
                    ACTIVATION[self.act_gate], ACTIVATION[self.act_cell],
                    ACTIVATION[self.act_cand])

        self.inputs = {'Input': (x, self.lod), 'Weight': w}

        self.inputs['Bias'] = b

        if self.has_initial_state:
            self.inputs['H0'] = h0
            self.inputs['C0'] = c0

        self.outputs = {
            'Hidden': (h, self.lod),
            'Cell': (c, self.lod),
        }
        self.attrs = {
            'use_peepholes': self.use_peepholes,
            'is_reverse': self.is_reverse,
            'gate_activation': self.act_gate,
            'cell_activation': self.act_cell,
            'candidate_activation': self.act_cand
        }

    def test_check_output(self):
        self.check_output(atol=1e-8, check_dygraph=False)

    def test_check_grad(self):
        # TODO(qingqing) remove folowing lines after the check_grad is refined.
        N = len(self.lod[0])
        self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
        self.outputs['BatchCellPreAct'] = np.zeros(
            (N, self.D)).astype('float64')
        self.check_grad(
            ['Input', 'Weight', 'Bias'], ['Hidden'],
            max_relative_error=5e-4,
            check_dygraph=False)


class TestLstmOpCase1(TestLstmOp):
    def set_lod(self):
        self.lod = [[0, 3, 2]]


class TestLstmOpCase2(TestLstmOp):
    def set_lod(self):
        self.lod = [[0, 3, 0]]


class TestLstmOpCase3(TestLstmOp):
    def set_lod(self):
        self.lod = [[2, 0, 4]]


class TestLstmOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                input_data = np.random.random((1, 2048)).astype("float32")
                fluid.layers.dynamic_lstm(
                    input=input_data, size=2048, use_peepholes=False)

            self.assertRaises(TypeError, test_Variable)

            def test_h_0():
                in_data = fluid.data(
                    name="input", shape=[None, 2048], dtype="float32")
                h = fluid.data(name="h", shape=[None, 512], dtype="int32")
                c = fluid.data(name="c", shape=[None, 512], dtype="float32")
                fluid.layers.dynamic_lstm(
                    input=in_data, size=2048, use_peepholes=False, h_0=h, c_0=c)

            self.assertRaises(TypeError, test_h_0)

            def test_c_0():
                in_data_ = fluid.data(
                    name="input_", shape=[None, 2048], dtype="float32")
                h_ = fluid.data(name="h_", shape=[None, 512], dtype="float32")
                c_ = fluid.data(name="c_", shape=[None, 512], dtype="int32")
                fluid.layers.dynamic_lstm(
                    input=in_data_,
                    size=2048,
                    use_peepholes=False,
                    h_0=h_,
                    c_0=c_)

            self.assertRaises(TypeError, test_c_0)


# class TestLstmOpHasInitial(TestLstmOp):
#     def set_argument(self):
#         self.lod = [[2, 3, 2]]
#         self.D = 16

#         self.act_gate = 'sigmoid'
#         self.act_cell = 'tanh'
#         self.act_cand = 'tanh'

#         self.has_initial_state = True
#         self.is_reverse = True
#         self.use_peepholes = True

#     def test_check_grad(self):
#         # TODO(qingqing) remove folowing lines after the check_grad is refined.
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Input', 'Weight', 'Bias', 'H0', 'C0'], ['Hidden'],
#             max_relative_error=5e-4)

#     def test_check_grad_ingore_bias(self):
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Input', 'Weight'], ['Hidden'],
#             max_relative_error=5e-4,
#             no_grad_set=set('Bias'))

#     def test_check_grad_ingore_weight(self):
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Input', 'Bias'], ['Hidden'],
#             max_relative_error=5e-4,
#             no_grad_set=set('Weight'))

#     def test_check_grad_ingore_input(self):
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Weight', 'Bias'], ['Hidden'],
#             max_relative_error=5e-4,
#             no_grad_set=set('Input'))

#     def test_check_grad_ingore_h0(self):
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Input', 'Weight', 'Bias', 'C0'], ['Hidden'],
#             max_relative_error=5e-4,
#             no_grad_set=set('H0'))

#     def test_check_grad_ingore_c0(self):
#         N = len(self.lod[0])
#         self.outputs['BatchGate'] = np.zeros((N, 4 * self.D)).astype('float64')
#         self.outputs['BatchCellPreAct'] = np.zeros(
#             (N, self.D)).astype('float64')
#         self.check_grad(
#             ['Input', 'Weight', 'Bias', 'H0'], ['Hidden'],
#             max_relative_error=5e-4,
#             no_grad_set=set('C0'))

# class TestLstmOpRerverse(TestLstmOp):
#     def set_argument(self):
#         self.lod = [[2, 3, 2]]
#         self.D = 16

#         self.act_gate = 'sigmoid'
#         self.act_cell = 'tanh'
#         self.act_cand = 'tanh'

#         self.has_initial_state = False
#         self.is_reverse = True
#         self.use_peepholes = True

# class TestLstmOpNotUsePeepholes(TestLstmOp):
#     def set_argument(self):
#         self.lod = [[2, 3, 2]]
#         self.D = 16

#         self.act_gate = 'sigmoid'
#         self.act_cell = 'tanh'
#         self.act_cand = 'tanh'

#         self.has_initial_state = False
#         self.is_reverse = True
#         self.use_peepholes = False

if __name__ == '__main__':
    unittest.main()
