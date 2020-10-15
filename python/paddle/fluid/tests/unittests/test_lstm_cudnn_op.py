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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import random
random.seed(2)
np.set_printoptions(threshold=np.inf)
paddle.enable_static()

SIGMOID_THRESHOLD_MIN = -40.0
SIGMOID_THRESHOLD_MAX = 13.0
EXP_MAX_INPUT = 40.0


class RandomWeight:
    def __init__(self):
        pass

    def updata_weight(self, hidden_size, input_size, dtype):
        std = 1.0 / math.sqrt(hidden_size)
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.dtype = dtype

        self.weight_ih = np.random.uniform(
            low=-std, high=std, size=(4 * self.hidden_size,
                                      self.input_size)).astype(dtype)
        self.weight_hh = np.random.uniform(
            low=-std, high=std, size=(4 * self.hidden_size,
                                      self.hidden_size)).astype(dtype)
        self.bias_ih = np.random.uniform(
            low=-std, high=std, size=(4 * self.hidden_size)).astype(dtype)
        self.bias_hh = np.random.uniform(
            low=-std, high=std, size=(4 * self.hidden_size)).astype(dtype)


weight = RandomWeight()


class LayerMixin(object):
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LayerListMixin(LayerMixin):
    def __init__(self, layers=None):
        self._layers = list(layers) if layers else []

    def append(self, layer):
        self._layers.append(layer)

    def __iter__(self):
        return iter(self._layers)


class LSTMCell(LayerMixin):
    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.dtype = np.float64
        self.parameters = dict()
        self.weight_ih = weight.weight_ih
        self.weight_hh = weight.weight_hh
        self.parameters['weight_ih'] = self.weight_ih
        self.parameters['weight_hh'] = self.weight_hh
        if bias:
            self.bias_ih = weight.bias_ih
            self.bias_hh = weight.bias_hh
            self.parameters['bias_ih'] = self.bias_ih
            self.parameters['bias_hh'] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs):
        batch_size = inputs.shape[0]
        init_h = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        init_c = np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)
        return init_h, init_c

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = self.init_state(inputs)
        pre_hidden, pre_cell = hx
        gates = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            gates = gates + self.bias_ih
        gates += np.matmul(pre_hidden, self.weight_hh.T)
        if self.bias_hh is not None:
            gates = gates + self.bias_hh

        chunked_gates = np.split(gates, 4, -1)

        i = 1.0 / (1.0 + np.exp(-chunked_gates[0]))
        f = 1.0 / (1.0 + np.exp(-chunked_gates[1]))
        o = 1.0 / (1.0 + np.exp(-chunked_gates[3]))
        c = f * pre_cell + i * np.tanh(chunked_gates[2])
        h = o * np.tanh(c)

        return h, (h, c)


def sequence_mask(lengths, max_len=None):
    if max_len is None:
        max_len = np.max(lengths)
    else:
        assert max_len >= np.max(lengths)
    return np.arange(max_len) < np.expand_dims(lengths, -1)


def update_state(mask, new, old):
    if not isinstance(old, (tuple, list)):
        return np.where(mask, new, old)
    else:
        return tuple(map(lambda x, y: np.where(mask, x, y), new, old))


def rnn(cell,
        inputs,
        initial_states,
        sequence_length=None,
        time_major=False,
        is_reverse=False):
    if not time_major:
        inputs = np.transpose(inputs, [1, 0, 2])
    if is_reverse:
        inputs = np.flip(inputs, 0)

    if sequence_length is None:
        mask = None
    else:
        mask = np.transpose(sequence_mask(sequence_length), [1, 0])
        mask = np.expand_dims(mask, -1)
        if is_reverse:
            mask = np.flip(mask, 0)

    time_steps = inputs.shape[0]
    state = initial_states
    outputs = []
    for t in range(time_steps):
        x_t = inputs[t]
        if mask is not None:
            m_t = mask[t]
            y, new_state = cell(x_t, state)
            y = np.where(m_t, y, 0.)
            outputs.append(y)
            state = update_state(m_t, new_state, state)
        else:
            y, new_state = cell(x_t, state)
            outputs.append(y)
            state = new_state

    outputs = np.stack(outputs)
    final_state = state

    if is_reverse:
        outputs = np.flip(outputs, 0)
    if not time_major:
        outputs = np.transpose(outputs, [1, 0, 2])
    return outputs, final_state


def birnn(cell_fw,
          cell_bw,
          inputs,
          initial_states,
          sequence_length=None,
          time_major=False):
    states_fw, states_bw = initial_states
    outputs_fw, states_fw = rnn(cell_fw,
                                inputs,
                                states_fw,
                                sequence_length,
                                time_major=time_major)

    outputs_bw, states_bw = rnn(cell_bw,
                                inputs,
                                states_bw,
                                sequence_length,
                                time_major=time_major,
                                is_reverse=True)

    outputs = np.concatenate((outputs_fw, outputs_bw), -1)
    final_states = (states_fw, states_bw)
    return outputs, final_states


def flatten(nested):
    return list(_flatten(nested))


def _flatten(nested):
    for item in nested:
        if isinstance(item, (list, tuple)):
            for subitem in _flatten(item):
                yield subitem
        else:
            yield item


def unstack(array, axis=0):
    num = array.shape[axis]
    sub_arrays = np.split(array, num, axis)
    return [np.squeeze(sub_array, axis) for sub_array in sub_arrays]


def dropout(array, p=0.0):
    if p == 0.0:
        return array

    mask = (np.random.uniform(size=array.shape) < (1 - p)).astype(array.dtype)
    return array * (mask / (1 - p))


def split_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        states = unstack(states)
        if not bidirectional:
            return states
        else:
            return list(zip(states[::2], states[1::2]))
    else:
        assert len(states) == state_components
        states = tuple([unstack(item) for item in states])
        if not bidirectional:
            return list(zip(*states))
        else:
            states = list(zip(*states))
            return list(zip(states[::2], states[1::2]))


def concat_states(states, bidirectional=False, state_components=1):
    if state_components == 1:
        return np.stack(flatten(states))
    else:
        states = flatten(states)
        componnets = []
        for i in range(state_components):
            componnets.append(states[i::state_components])
        return [np.stack(item) for item in componnets]


class RNN(LayerMixin):
    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major

    def forward(self, inputs, initial_states=None, sequence_length=None):
        final_outputs, final_states = rnn(self.cell,
                                          inputs,
                                          initial_states=initial_states,
                                          sequence_length=sequence_length,
                                          time_major=self.time_major,
                                          is_reverse=self.is_reverse)
        return final_outputs, final_states


class BiRNN(LayerMixin):
    def __init__(self, cell_fw, cell_bw, time_major=False):
        super(BiRNN, self).__init__()
        self.cell_fw = cell_fw
        self.cell_bw = cell_bw
        self.time_major = time_major

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        if isinstance(initial_states, (list, tuple)):
            assert len(initial_states) == 2, \
                "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]

        outputs, final_states = birnn(self.cell_fw, self.cell_bw, inputs,
                                      initial_states, sequence_length,
                                      self.time_major)
        return outputs, final_states


class RNNMixin(LayerListMixin):
    def forward(self, inputs, initial_states=None, sequence_length=None):
        batch_index = 1 if self.time_major else 0
        batch_size = inputs.shape[batch_index]
        dtype = inputs.dtype
        if initial_states is None:
            state_shape = (self.num_layers * self.num_directions, batch_size,
                           self.hidden_size)
            if self.state_components == 1:
                initial_states = np.zeros(state_shape, dtype)
            else:
                initial_states = tuple([
                    np.zeros(state_shape, dtype)
                    for _ in range(self.state_components)
                ])

        states = split_states(initial_states, self.num_directions == 2,
                              self.state_components)
        final_states = []

        for i, rnn_layer in enumerate(self):
            if i > 0:
                inputs = dropout(inputs, self.dropout)
            outputs, final_state = rnn_layer(inputs, states[i], sequence_length)
            final_states.append(final_state)
            inputs = outputs

        final_states = concat_states(final_states, self.num_directions == 2,
                                     self.state_components)
        return outputs, final_states


class LSTM(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False):
        super(LSTM, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = LSTMCell(input_size, hidden_size)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = LSTMCell(hidden_size, hidden_size)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = LSTMCell(input_size, hidden_size)
            cell_bw = LSTMCell(input_size, hidden_size)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = LSTMCell(2 * hidden_size, hidden_size)
                cell_bw = LSTMCell(2 * hidden_size, hidden_size)
                self.append(BiRNN(cell_fw, cell_bw, time_major))
        else:
            raise ValueError(
                "direction should be forward, backward or bidirectional, "
                "received direction = {}".format(direction))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_directions = 2 if direction == "bidirectional" else 1
        self.time_major = time_major
        self.num_layers = num_layers
        self.state_components = 2


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNLstmOp(OpTest):
    def get_weight_names(self):
        weight_names = []
        for i in range(2 * self.num_layers):
            weight_names.append('weight{}'.format(i))
        for i in range(2 * self.num_layers):
            weight_names.append('bias{}'.format(i))
        return weight_names

    def setUp(self):
        self.op_type = "cudnn_lstm"
        self.dtype = np.float64
        self.sequence_length = np.array([12, 11, 10, 9, 8], dtype=np.int32)
        self.num_layers = 1
        self.set_attrs()

        seq_length = 12
        batch_size = 5
        input_size = 21
        hidden_size = 21

        input = np.random.uniform(
            low=-0.1, high=0.1,
            size=(seq_length, batch_size, input_size)).astype(self.dtype)
        input[11][1:][:] = 0
        input[10][2:][:] = 0
        input[9][3:][:] = 0
        input[8][4:][:] = 0

        weight.updata_weight(hidden_size, input_size, self.dtype)
        rnn1 = LSTM(
            input_size,
            hidden_size,
            num_layers=self.num_layers,
            time_major=True,
            direction="forward")

        output, (last_hidden, last_cell) = rnn1(
            input, sequence_length=self.sequence_length)

        flat_w = []
        num = 0
        for i in range(self.num_layers):
            if i == 0:
                weight_ih = weight.weight_ih
            else:
                weight_ih = weight.weight_hh
            flat_w.append(("weight" + str(num), weight_ih))
            num += 1
        for i in range(self.num_layers):
            weight_hh = weight.weight_hh
            flat_w.append(("weight" + str(num), weight_hh))
            num += 1
        num = 0
        for i in range(self.num_layers):
            bias_ih = weight.bias_ih
            flat_w.append(("bias" + str(num), bias_ih))
            num += 1
        for i in range(self.num_layers):
            bias_hh = weight.bias_hh
            flat_w.append(("bias" + str(num), bias_hh))
            num += 1
        init_h = np.zeros((self.num_layers, batch_size,
                           hidden_size)).astype(self.dtype)
        init_c = np.zeros((self.num_layers, batch_size,
                           hidden_size)).astype(self.dtype)
        state_out = np.ndarray((300)).astype("uint8")

        self.inputs = {
            'Input': input,
            'WeightList': flat_w,
            'InitH': init_h,
            'InitC': init_c,
            'SequenceLength': self.sequence_length
        }
        self.attrs = {
            'dropout_prob': 0.0,
            'is_bidirec': False,
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': self.num_layers,
        }
        self.outputs = {
            'Out': output,
            "LastH": last_hidden,
            'LastC': last_cell,
            'Reserve': np.ndarray((400)).astype("uint8"),
            'StateOut': state_out
        }

    def set_attrs(self):
        pass

    def test_output_with_place(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, no_check_set=['Reserve', 'StateOut'])

    def test_grad_with_place(self):
        place = core.CUDAPlace(0)
        var_name_list = self.get_weight_names()
        for var_name in var_name_list:
            self.check_grad_with_place(
                place,
                set(['Input', var_name, 'InitH', 'InitC']),
                ['Out', 'LastH', 'LastC'])


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNlstmAPI(unittest.TestCase):
    def test_lstm(self):
        seq_len = 20
        batch_size = 5
        hidden_size = 20
        dropout_prob = 0.0
        num_layers = 1
        input = fluid.data(
            name='input',
            shape=[seq_len, batch_size, hidden_size],
            dtype='float64')
        init_h = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        init_c = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        rnn_out, last_h, last_c = layers.lstm(input, init_h, init_c, seq_len,
                                              hidden_size, num_layers,
                                              dropout_prob, False)
        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(fluid.default_startup_program())
        input_i = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_len, batch_size,
                                      hidden_size)).astype("float64")
        out = exe.run(fluid.default_main_program(),
                      feed={'input': input_i},
                      fetch_list=[rnn_out, last_h, last_c, 'cudnn_lstm_0.w_0'])


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDNNlstmAPI(unittest.TestCase):
    def test_lstm(self):
        seq_len = 20
        batch_size = 5
        hidden_size = 20
        dropout_prob = 0.0
        num_layers = 2
        input = fluid.data(
            name='input',
            shape=[seq_len, batch_size, hidden_size],
            dtype='float64')
        init_h = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        init_c = layers.fill_constant([num_layers, batch_size, hidden_size],
                                      'float64', 0.0)
        rnn_out, last_h, last_c = layers.lstm(input, init_h, init_c, seq_len,
                                              hidden_size, num_layers,
                                              dropout_prob, False, True)
        exe = fluid.Executor(fluid.CUDAPlace(0))
        exe.run(fluid.default_startup_program())
        input_i = np.random.uniform(
            low=-0.1, high=0.1, size=(seq_len, batch_size,
                                      hidden_size)).astype("float64")
        out = exe.run(fluid.default_main_program(),
                      feed={'input': input_i},
                      fetch_list=[rnn_out, last_h, last_c, 'cudnn_lstm_0.w_0'])


if __name__ == '__main__':
    unittest.main()
