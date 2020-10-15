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

import numpy as np
import math


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


class SimpleRNNCell(LayerMixin):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity="tanh"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        if nonlinearity == 'tanh':
            self.nonlinearity = np.tanh
        else:
            self.nonlinearity = lambda x: np.maximum(x, 0.)

        self.parameters = dict()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(-std, std, (
            hidden_size, input_size)).astype('float64')
        self.weight_hh = np.random.uniform(-std, std, (
            hidden_size, hidden_size)).astype('float64')
        self.parameters['weight_ih'] = self.weight_ih
        self.parameters['weight_hh'] = self.weight_hh
        if bias:
            self.bias_ih = np.random.uniform(-std, std,
                                             (hidden_size, )).astype('float64')
            self.bias_hh = np.random.uniform(-std, std,
                                             (hidden_size, )).astype('float64')
            self.parameters['bias_ih'] = self.bias_ih
            self.parameters['bias_hh'] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs, batch_dim_index=0):
        batch_size = inputs.shape[batch_dim_index]
        return np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = self.init_state(inputs)
        pre_h = hx
        i2h = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            i2h += self.bias_ih
        h2h = np.matmul(pre_h, self.weight_hh.T)
        if self.bias_hh is not None:
            h2h += self.bias_hh
        h = self.nonlinearity(i2h + h2h)
        return h, h


class GRUCell(LayerMixin):
    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.parameters = dict()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(-std, std, (
            3 * hidden_size, input_size)).astype('float64')
        self.weight_hh = np.random.uniform(-std, std, (
            3 * hidden_size, hidden_size)).astype('float64')
        self.parameters['weight_ih'] = self.weight_ih
        self.parameters['weight_hh'] = self.weight_hh
        if bias:
            self.bias_ih = np.random.uniform(-std, std, (
                3 * hidden_size)).astype('float64')
            self.bias_hh = np.random.uniform(-std, std, (
                3 * hidden_size)).astype('float64')
            self.parameters['bias_ih'] = self.bias_ih
            self.parameters['bias_hh'] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs, batch_dim_index=0):
        batch_size = inputs.shape[batch_dim_index]
        return np.zeros((batch_size, self.hidden_size), dtype=inputs.dtype)

    def forward(self, inputs, hx=None):
        if hx is None:
            hx = self.init_state(inputs)
        pre_hidden = hx
        x_gates = np.matmul(inputs, self.weight_ih.T)
        if self.bias_ih is not None:
            x_gates = x_gates + self.bias_ih
        h_gates = np.matmul(pre_hidden, self.weight_hh.T)
        if self.bias_hh is not None:
            h_gates = h_gates + self.bias_hh
        x_r, x_z, x_c = np.split(x_gates, 3, 1)
        h_r, h_z, h_c = np.split(h_gates, 3, 1)

        r = 1.0 / (1.0 + np.exp(-(x_r + h_r)))
        z = 1.0 / (1.0 + np.exp(-(x_z + h_z)))
        c = np.tanh(x_c + r * h_c)  # apply reset gate after mm
        h = (pre_hidden - c) * z + c
        return h, h


class LSTMCell(LayerMixin):
    def __init__(self, input_size, hidden_size, bias=True):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.parameters = dict()
        std = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = np.random.uniform(-std, std, (
            4 * hidden_size, input_size)).astype('float64')
        self.weight_hh = np.random.uniform(-std, std, (
            4 * hidden_size, hidden_size)).astype('float64')
        self.parameters['weight_ih'] = self.weight_ih
        self.parameters['weight_hh'] = self.weight_hh
        if bias:
            self.bias_ih = np.random.uniform(-std, std, (
                4 * hidden_size)).astype('float64')
            self.bias_hh = np.random.uniform(-std, std, (
                4 * hidden_size)).astype('float64')
            self.parameters['bias_ih'] = self.bias_ih
            self.parameters['bias_hh'] = self.bias_hh
        else:
            self.bias_ih = None
            self.bias_hh = None

    def init_state(self, inputs, batch_dim_index=0):
        batch_size = inputs.shape[batch_dim_index]
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

    if initial_states is None:
        initial_states = cell.init_state(inputs, 1)

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


def dropout(array, p=0.5):
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


class SimpleRNN(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 nonlinearity="tanh",
                 direction="forward",
                 dropout=0.,
                 time_major=False):
        super(SimpleRNN, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = SimpleRNNCell(input_size, hidden_size, nonlinearity)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = SimpleRNNCell(hidden_size, hidden_size, nonlinearity)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = SimpleRNNCell(input_size, hidden_size, nonlinearity)
            cell_bw = SimpleRNNCell(input_size, hidden_size, nonlinearity)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = SimpleRNNCell(2 * hidden_size, hidden_size,
                                        nonlinearity)
                cell_bw = SimpleRNNCell(2 * hidden_size, hidden_size,
                                        nonlinearity)
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
        self.state_components = 1


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


class GRU(RNNMixin):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 direction="forward",
                 dropout=0.,
                 time_major=False):
        super(GRU, self).__init__()

        if direction in ["forward", "backward"]:
            is_reverse = direction == "backward"
            cell = GRUCell(input_size, hidden_size)
            self.append(RNN(cell, is_reverse, time_major))
            for i in range(1, num_layers):
                cell = GRUCell(hidden_size, hidden_size)
                self.append(RNN(cell, is_reverse, time_major))
        elif direction == "bidirectional":
            cell_fw = GRUCell(input_size, hidden_size)
            cell_bw = GRUCell(input_size, hidden_size)
            self.append(BiRNN(cell_fw, cell_bw, time_major))
            for i in range(1, num_layers):
                cell_fw = GRUCell(2 * hidden_size, hidden_size)
                cell_bw = GRUCell(2 * hidden_size, hidden_size)
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
        self.state_components = 1
