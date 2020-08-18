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

# TODO: define classes of recurrent neural network

__all__ = [
    'RNNCellBase',
    'LSTMCell',
    'GRUCell',
    'StackedRNNCell',
    'StackedLSTMCell',
    'stackedGRUCell',
    'RNN',
    'BidirectionalRNN',
    'LSTM',
    'GRU',
]

import copy
import collections
import itertools
import six
import sys
import warnings
from functools import partial, reduce

import numpy as np

from ... import fluid
from ...fluid import layers
from ...fluid.data_feeder import convert_dtype
from ...fluid.dygraph import Layer, LayerList
from ...fluid.param_attr import ParamAttr
from ...fluid.layers import utils, BeamSearchDecoder
from ...fluid.layers.utils import map_structure, flatten, pack_sequence_as


class RNNCellBase(Layer):
    """
    RNNCellBase is the base class for abstraction representing the calculations
    mapping the input and state to the output and new state. It is suitable to
    and mostly used in RNN.
    """

    def get_initial_states(self,
                           batch_ref,
                           shape=None,
                           dtype=None,
                           init_value=0,
                           batch_dim_idx=0):
        """
        Generate initialized states according to provided shape, data type and
        value.
        Parameters:
            batch_ref: A (possibly nested structure of) tensor variable[s].
                The first dimension of the tensor will be used as batch size to
                initialize states.
            shape: A (possibly nested structure of) shape[s], where a shape is
                represented as a list/tuple of integer). -1(for batch size) will
                beautomatically inserted if shape is not started with it. If None,
                property `state_shape` will be used. The default value is None.
            dtype: A (possibly nested structure of) data type[s]. The structure
                must be same as that of `shape`, except when all tensors' in states
                has the same data type, a single data type can be used. If None and
                property `cell.state_shape` is not available, float32 will be used
                as the data type. The default value is None.
            init_value: A float value used to initialize states.
            batch_dim_idx: An integer indicating which dimension of the tensor in
                inputs represents batch size.  The default value is 0.
        Returns:
            Variable: tensor variable[s] packed in the same structure provided \
                by shape, representing the initialized states.
        """
        # TODO: use inputs and batch_size
        batch_ref = flatten(batch_ref)[0]

        def _is_shape_sequence(seq):
            if sys.version_info < (3, ):
                integer_types = (
                    int,
                    long, )
            else:
                integer_types = (int, )
            """For shape, list/tuple of integer is the finest-grained objection"""
            if (isinstance(seq, list) or isinstance(seq, tuple)):
                if reduce(lambda flag, x: isinstance(x, integer_types) and flag,
                          seq, True):
                    return False
            # TODO: Add check for the illegal
            if isinstance(seq, dict):
                return True
            return (isinstance(seq, collections.Sequence) and
                    not isinstance(seq, six.string_types))

        class Shape(object):
            def __init__(self, shape):
                self.shape = shape if shape[0] == -1 else ([-1] + list(shape))

        # nested structure of shapes
        states_shapes = self.state_shape if shape is None else shape
        is_sequence_ori = utils.is_sequence
        utils.is_sequence = _is_shape_sequence
        states_shapes = map_structure(lambda shape: Shape(shape), states_shapes)
        utils.is_sequence = is_sequence_ori

        # nested structure of dtypes
        try:
            states_dtypes = self.state_dtype if dtype is None else dtype
        except NotImplementedError:  # use fp32 as default
            states_dtypes = "float32"
        if len(flatten(states_dtypes)) == 1:
            dtype = flatten(states_dtypes)[0]
            states_dtypes = map_structure(lambda shape: dtype, states_shapes)

        init_states = map_structure(
            lambda shape, dtype: layers.fill_constant_batch_size_like(
                input=batch_ref,
                shape=shape.shape,
                dtype=dtype,
                value=init_value,
                input_dim_idx=batch_dim_idx), states_shapes, states_dtypes)
        return init_states

    @property
    def state_shape(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) shape[s], where a shape is represented
        as a list/tuple of integers (-1 for batch size would be automatically
        inserted into a shape if shape is not started with it).
        Not necessary to be implemented if states are not initialized by
        `get_initial_states` or the `shape` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_shape` in the used cell.")

    @property
    def state_dtype(self):
        """
        Abstract method (property).
        Used to initialize states.
        A (possiblely nested structure of) data types[s]. The structure must be
        same as that of `shape`, except when all tensors' in states has the same
        data type, a signle data type can be used.
        Not necessary to be implemented if states are not initialized
        by `get_initial_states` or the `dtype` argument is provided when using
        `get_initial_states`.
        """
        raise NotImplementedError(
            "Please add implementaion for `state_dtype` in the used cell.")


class LSTMCell(RNNCellBase):
    """
    Long-Short Term Memory(LSTM) RNN cell.

    The formula used is as follows:

    .. math::
        i_{t} & = \sigma(W_{x_{i}}x_{t} + b_{x_{i}} + W_{h_{i}}h_{t-1} + b_{h_{i}})
        f_{t} & = \sigma(W_{x_{f}}x_{t} + b_{x_{f}} + W_{h_{f}}h_{t-1} + b_{h_{f}})
        o_{t} & = \sigma(W_{x_{o}}x_{t} + b_{x_{o}} + W_{h_{o}}h_{t-1} + b_{h_{o}})
        c_{t} & = f_{t}c_{t-1} + i_{t} \\tanh (W_{x_{c}}x_{t} + b_{x_{c}} + W_{h_{c}}h_{t-1} + b_{h_{c}})
        h_{t} & = o_{t} \\tanh (c_{t})

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size in the LSTM cell.
        hidden_size (int): The hidden size in the LSTM cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        dtype(string, optional): The data type used in this cell. Default float32.

    Examples:
        .. code-block:: python
            import paddle
            inputs = paddle.rand((2, 4, 32))
            cell = paddle.LSTMCell(input_size=32, hidden_size=64)
            rnn = paddle.RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(LSTMCell, self).__init__(dtype)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = getattr(layers, "sigmoid")
        self._activation = getattr(layers, "tanh")
        self._param_attr = ParamAttr._to_attr(param_attr)
        self._bias_attr = ParamAttr._to_attr(bias_attr)
        self._dtype = dtype

        if self._param_attr and self._param_attr.name is not None:
            weight_ih_param_attr = copy.deepcopy(self._param_attr)
            weight_hh_param_attr = copy.deepcopy(self._param_attr)
            weight_ih_param_attr.name += "_weight_ih"
            weight_hh_param_attr.name += "_weight_hh"
        else:
            weight_ih_param_attr = self._param_attr
            weight_hh_param_attr = self._param_attr

        if self._bias_attr and self._bias_attr.name is not None:
            bias_ih_param_attr = copy.deepcopy(self._bias_attr)
            bias_hh_param_attr = copy.deepcopy(self._bias_attr)
            bias_ih_param_attr.name += "_bias_ih"
            bias_hh_param_attr.name += "_bias_hh"
        else:
            bias_ih_param_attr = self._bias_attr
            bias_hh_param_attr = self._bias_attr

        self.weight_ih = self.create_parameter(
            attr=weight_ih_param_attr,
            shape=[4 * hidden_size, input_size],
            dtype=dtype)

        self.weight_hh = self.create_parameter(
            attr=weight_hh_param_attr,
            shape=[4 * hidden_size, hidden_size],
            dtype=dtype)

        self.bias_ih = self.create_parameter(
            attr=bias_ih_param_attr,
            shape=[4 * hidden_size],
            dtype=dtype,
            is_bias=True)
        self.bias_hh = self.create_parameter(
            attr=bias_hh_param_attr,
            shape=[4 * hidden_size],
            dtype=dtype,
            is_bias=True)

    def forward(self, inputs, states):
        """
        Performs single step LSTM calculations.
        Parameters:
            inputs (Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states (Variable): A list of containing two tensors, each shaped
                `[batch_size, hidden_size]`, corresponding to :math:`h_{t-1}, c_{t-1}`
                in the formula. The data type should be float32 or float64.
        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula; `new_states` is a list containing \
                two tenser variables shaped `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}, c_{t}` in the formula. The data type of these \
                tensors all is same as that of `states`.
        """
        pre_hidden, pre_cell = states
        gates = layers.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih:
            gates = gates + self.bias_ih
        gates += layers.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh:
            gates = gates + self.bias_hh

        chunked_gates = layers.split(gates, num_or_sections=4, dim=1)

        i = self._gate_activation(chunked_gates[0])
        f = self._gate_activation(chunked_gates[1])
        o = self._gate_activation(chunked_gates[3])
        c = f * pre_cell + i * self._activation(chunked_gates[2])
        h = o * self._activation(c)

        return h, [h, c]

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicLSTMCell is a list with two shapes: `[[hidden_size], [hidden_size]]`
        (-1 for batch size would be automatically inserted into shape). These two
        shapes correspond to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return [[self.hidden_size], [self.hidden_size]]


class GRUCell(RNNCellBase):
    """
    Gated Recurrent Unit (GRU) RNN cell.

    The formula for GRU used is as follows:

    .. math::

        u_t & = \sigma(W_{x_{u}}x_{t} + b_{x_{u}} + W_{h_{u}}h_{t-1} + b_{h_{u}})

        r_t & = \sigma(W_{x_{r}}x_{t} + b_{x_{r}} + W_{h_{r}}h_{t-1} + b_{h_{r}})

        \\tilde{h_t} & = \\tanh(W_{x_{c}}x_{t} + r_t \odot (W_{h_{c}}h_{t-1} + b_{h_{c}})

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        dtype(string, optional): The data type used in this cell. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            inputs = paddle.rand((2, 4, 32))
            cell = BasicGRUCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(GRUCell, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self._gate_activation = getattr(layers, "sigmoid")
        self._activation = getattr(layers, "tanh")
        self._param_attr = ParamAttr._to_attr(param_attr)
        self._bias_attr = ParamAttr._to_attr(bias_attr)
        self._dtype = dtype

        if self._param_attr and self._param_attr.name is not None:
            weight_ih_param_attr = copy.deepcopy(self._param_attr)
            weight_hh_param_attr = copy.deepcopy(self._param_attr)
            weight_ih_param_attr.name += "_weight_ih"
            weight_hh_param_attr.name += "_weight_hh"
        else:
            weight_ih_param_attr = self._param_attr
            weight_hh_param_attr = self._param_attr

        if self._bias_attr and self._bias_attr.name is not None:
            bias_ih_param_attr = copy.deepcopy(self._bias_attr)
            bias_hh_param_attr = copy.deepcopy(self._bias_attr)
            bias_ih_param_attr.name += "_bias_ih"
            bias_hh_param_attr.name += "_bias_hh"
        else:
            bias_ih_param_attr = self._bias_attr
            bias_hh_param_attr = self._bias_attr

        self.weight_ih = self.create_parameter(
            attr=weight_ih_param_attr,
            shape=[3 * hidden_size, input_size],
            dtype=dtype)

        self.weight_hh = self.create_parameter(
            attr=weight_hh_param_attr,
            shape=[3 * hidden_size, hidden_size],
            dtype=dtype)

        self.bias_ih = self.create_parameter(
            attr=bias_ih_param_attr,
            shape=[3 * hidden_size],
            dtype=dtype,
            is_bias=True)
        self.bias_hh = self.create_parameter(
            attr=bias_hh_param_attr,
            shape=[3 * hidden_size],
            dtype=dtype,
            is_bias=True)

    def forward(self, inputs, states):
        """
        Performs single step GRU calculations.

        Parameters:
            inputs (Variable): A tensor with shape `[batch_size, input_size]`,
                corresponding to :math:`x_t` in the formula. The data type
                should be float32 or float64.
            states (Variable): A tensor with shape `[batch_size, hidden_size]`.
                corresponding to :math:`h_{t-1}` in the formula. The data type
                should be float32 or float64.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` and \
                `new_states` is the same tensor shaped `[batch_size, hidden_size]`, \
                corresponding to :math:`h_t` in the formula. The data type of the \
                tensor is same as that of `states`.        
        """
        pre_hidden = states

        x_gates = layers.matmul(inputs, self.weight_ih, transpose_y=True)
        if self.bias_ih:
            x_gates = x_gates + self.bias_ih
        h_gates = layers.matmul(pre_hidden, self.weight_hh, transpose_y=True)
        if self.bias_hh:
            h_gates = h_gates + self.bias_hh

        x_u, x_r, x_c = layers.split(x_gates, num_or_sections=3, dim=1)
        h_u, h_r, h_c = layers.split(x_gates, num_or_sections=3, dim=1)

        u = self._gate_activation(x_u + h_u)
        r = self._gate_activation(x_r + h_r)
        h_c = r * h_c
        c = self._activation(x_c + h_c)
        h = u * pre_hidden + (1 - u) * c

        return h, h

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicGRUCell is a shape `[hidden_size]` (-1 for batch
        size would be automatically inserted into shape). The shape corresponds
        to :math:`h_{t-1}`.
        """
        return [self._hidden_size]


class StackedRNNCell(RNNCellBase):
    """
    Wrapper allowing a stack of RNN cells to behave as a single cell. It is used
    to implement stacked RNNs.

    Parameters:
        cells (list|tuple): List of RNN cell instances.

    Examples:
        .. code-block:: python
            from paddle import LSTMCell, StackedRNNCell
            cells = [LSTMCell(32, 32), LSTMCell(32, 32)]
            stack_rnn = StackedRNNCell(cells)
    """

    def __init__(self, cells):
        super(StackedRNNCell, self).__init__()
        self.cells = []
        for i, cell in enumerate(cells):
            self.cells.append(self.add_sublayer("cell_%d" % i, cell))

    def forward(self, inputs, states, **kwargs):
        """
        Performs :code:`cell.forward` for all including cells sequentially.
        Each cell's `inputs` is the `outputs` of the previous cell. And each
        cell's `states` is the corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. Mostly it is a
                float32 or float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ). `outputs` is the \
                `outputs` of the last cell. `new_states` is a list composed \
                of all cells' `new_states`, and its structure and data type is \
                same as that of `states` argument.
        """
        new_states = []
        for cell, state in zip(self.cells, states):
            outputs, new_state = cell(inputs, state, **kwargs)
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @staticmethod
    def stack_param_attr(param_attr, n):
        """
        If `param_attr` is a list or tuple, convert every element in it to a
        ParamAttr instance. Otherwise, repeat `param_attr` `n` times to
        construct a list, and rename every one by appending a increasing index
        suffix to avoid having same names when `param_attr` contains a name.
        Parameters:
            param_attr (list|tuple|ParamAttr): A list, tuple or something can be
                converted to a ParamAttr instance by `ParamAttr._to_attr`.
            n (int): The times to repeat to construct a list when `param_attr`
                is not a list or tuple.
        Returns:
            list: A list composed of each including cell's `param_attr`.
        """
        if isinstance(param_attr, (list, tuple)):
            assert len(param_attr) == n, (
                "length of param_attr should be %d when it is a list/tuple" % n)
            param_attrs = [ParamAttr._to_attr(attr) for attr in param_attr]
        else:
            param_attrs = []
            attr = ParamAttr._to_attr(param_attr)
            for i in range(n):
                attr_i = copy.deepcopy(attr)
                if attr.name:
                    attr_i.name = attr_i.name + "_" + str(i)
                param_attrs.append(attr_i)
        return param_attrs

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedRNNCell is a list composed of each including
        cell's `state_shape`.
        Returns:
            list: A list composed of each including cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class StackedLSTMCell(RNNCellBase):
    """
    Wrapper allowing a stack of LSTM cells to behave as a single cell. It is used
    to implement stacked LSTM.

    The formula for LSTM used here is as follows:

    .. math::
        i_{t} & = \sigma(W_{x_{i}}x_{t} + b_{x_{i}} + W_{h_{i}}h_{t-1} + b_{h_{i}})
        f_{t} & = \sigma(W_{x_{f}}x_{t} + b_{x_{f}} + W_{h_{f}}h_{t-1} + b_{h_{f}})
        o_{t} & = \sigma(W_{x_{o}}x_{t} + b_{x_{o}} + W_{h_{o}}h_{t-1} + b_{h_{o}})
        c_{t} & = f_{t}c_{t-1} + i_{t} \\tanh (W_{x_{c}}x_{t} + b_{x_{c}} + W_{h_{c}}h_{t-1} + b_{h_{c}})
        h_{t} & = o_{t} \\tanh (c_{t})

    Parameters:
        input_size (int): The input size for the first LSTM cell.
        hidden_size (int): The hidden size for every LSTM cell.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float, optional): The dropout probability applied on the outputs
            of each LSTM cell except the last one. 0 for no dropout. Default 0.0
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:
        .. code-block:: python
            import paddle
            inputs = paddle.rand((2, 4, 32))
            cell = paddle.StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = paddle.RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        param_attrs = StackedRNNCell.stack_param_attr(param_attr, num_layers)
        bias_attrs = StackedRNNCell.stack_param_attr(bias_attr, num_layers)

        self.cells = []
        for i in range(num_layers):
            self.cells.append(
                self.add_sublayer(
                    "lstm_%d" % i,
                    LSTMCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        param_attr=param_attrs[i],
                        bias_attr=bias_attrs[i],
                        dtype=dtype)))

    def forward(self, inputs, states):
        """
        Performs the stacked LSTM cells sequentially. Each cell's `inputs` is
        the `outputs` of the previous cell. And each cell's `states` is the
        corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. It is a float32 or
                float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula of the last LSTM; `new_states` \
                is a list composed of every LSTM `new_states` which is a pair \
                of tensors standing for :math:`h_{t}, c_{t}` in the formula, \
                and the data type and structure of these tensors all is same \
                as that of `states`.
        """
        new_states = []
        for i, cell in enumerate(self.cells):
            outputs, new_state = cell(inputs, states[i])
            outputs = layers.dropout(
                outputs,
                self.dropout,
                dropout_implementation='upscale_in_train'
            ) if self.dropout and i != (self.num_layers - 1) else outputs
            inputs = outputs
            new_states.append(new_state)
        # TODO(guosheng): maybe should stack list of states as one tensor
        return outputs, new_states

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedLSTMCell is a list composed of each including
        LSTM cell's `state_shape`.
        Returns:
            list: A list composed of each including LSTM cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class StackedGRUCell(RNNCellBase):
    """
    Wrapper allowing a stack of GRU cells to behave as a single cell. It is used
    to implement stacked GRU.

    The formula for GRU used here is as follows:

    .. math::

        u_t & = \sigma(W_{x_{u}}x_{t} + b_{x_{u}} + W_{h_{u}}h_{t-1} + b_{h_{u}})

        r_t & = \sigma(W_{x_{r}}x_{t} + b_{x_{r}} + W_{h_{r}}h_{t-1} + b_{h_{r}})

        \\tilde{h_t} & = \\tanh(W_{x_{c}}x_{t} + r_t \odot (W_{h_{c}}h_{t-1} + b_{h_{c}})

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    Parameters:
        input_size (int): The input size for the first GRU cell.
        hidden_size (int): The hidden size for every GRU cell.
        num_layers(int, optional): The number of GRU to be stacked. Default 1.
        dropout(float, optional): The dropout probability applied on the outputs
            of each GRU cell except the last one. 0 for no dropout. Default 0.0
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.

    Examples:

        .. code-block:: python

            import paddle

            inputs = paddle.rand((2, 4, 32))
            cell = paddle.StackedGRUCell(input_size=32, hidden_size=64)
            rnn = paddle.RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedGRUCell, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.dropout = dropout
        param_attrs = StackedRNNCell.stack_param_attr(param_attr, num_layers)
        bias_attrs = StackedRNNCell.stack_param_attr(bias_attr, num_layers)

        self.cells = []
        for i in range(num_layers):
            self.cells.append(
                self.add_sublayer(
                    "gru_%d" % i,
                    GRUCell(
                        input_size=input_size if i == 0 else hidden_size,
                        hidden_size=hidden_size,
                        param_attr=param_attrs[i],
                        bias_attr=bias_attrs[i],
                        dtype=dtype)))

    def forward(self, inputs, states):
        """
        Performs the stacked GRU cells sequentially. Each cell's `inputs` is
        the `outputs` of the previous cell. And each cell's `states` is the
        corresponding one in `states`.

        Parameters:
            inputs (Variable): The inputs for the first cell. It is a float32 or
                float64 tensor with shape `[batch_size, input_size]`.
            states (list): A list containing states for all cells orderly.

        Returns:
            tuple: A tuple( :code:`(outputs, new_states)` ), where `outputs` is \
                a tensor with shape `[batch_size, hidden_size]`, corresponding \
                to :math:`h_{t}` in the formula of the last GRU; `new_states` \
                is a list composed of every GRU `new_states` which is also \
                :math:`h_{t}` in the formula, and the data type and structure \
                of these tensors all is same as that of `states`.
        """
        new_states = []
        for i, cell in enumerate(self.cells):
            outputs, new_state = cell(inputs, states[i])
            outputs = layers.dropout(
                outputs,
                self.dropout,
                dropout_implementation='upscale_in_train'
            ) if self.dropout and i != (self.num_layers - 1) else outputs
            inputs = outputs
            new_states.append(new_state)
        return outputs, new_states

    @property
    def state_shape(self):
        """
        The `state_shape` of StackedGRUCell is a list composed of each including
        GRU cell's `state_shape`.

        Returns:
            list: A list composed of each including GRU cell's `state_shape`.
        """
        return [cell.state_shape for cell in self.cells]


class RNN(Layer):
    """
    RNN creates a recurrent neural network specified by RNNCell `cell`, which
    performs :code:`cell.forward()` repeatedly until reaches to the maximum
    length of `inputs`.

    Parameters:
        cell(RNNCell): An instance of `RNNCell`.
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.

    Examples:
        .. code-block:: python
            import paddle
            inputs = paddle.rand((2, 4, 32))
            cell = paddle.StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = paddle.RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
            # for non-dygraph mode, `rnn` api uses cell.call
            self.cell.call = self.cell.forward
        self.is_reverse = is_reverse
        self.time_major = time_major
        self.batch_index, self.time_step_index = (1, 0) if time_major else (0,
                                                                            1)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        """
        Performs :code:`cell.forward()` repeatedly until reaches to the maximum
        length of `inputs`.
        Parameters:
            inputs (Variable): A (possibly nested structure of) tensor variable[s]. 
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be unrolled
                in RNN.
            initial_states (Variable, optional): A (possibly nested structure of)
                tensor variable[s], representing the initial state for RNN. 
                If not provided, `cell.get_initial_states` would be used to produce
                the initial state. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
            **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`. 
        Returns:
            tuple: A tuple( :code:`(final_outputs, final_states)` ) including the final \
                outputs and states, both are Tensor or nested structure of Tensor. \
                `final_outputs` has the same structure and data types as \
                the returned `outputs` of :code:`cell.forward` , and each Tenser in `final_outputs` \
                stacks all time steps' counterpart in `outputs` thus has shape `[batch_size, sequence_length, ...]` \
                for `time_major == False` or `[sequence_length, batch_size, ...]` for `time_major == True`. \
                `final_states` is the counterpart at last time step of initial states, \
                thus has the same structure with it and has tensors with same shapes \
                and data types.
        """
        flat_inputs = flatten(inputs)
        batch_size, time_steps = (flat_inputs[0].shape[self.batch_index],
                                  flat_inputs[0].shape[self.time_step_index])

        if initial_states is None:
            initial_states = self.cell.get_initial_states(
                batch_ref=inputs,
                dtype=self.cell.dtype if hasattr(self.cell, "dtype") else
                self.cell.parameters()[0].dtype,
                batch_dim_idx=self.batch_index)

        if fluid.in_dygraph_mode():

            class ArrayWrapper(object):
                def __init__(self, x):
                    self.array = [x]

                def append(self, x):
                    self.array.append(x)
                    return self

            def _maybe_copy(state, new_state, step_mask):
                # TODO: use where_op
                new_state = layers.elementwise_mul(
                    new_state, step_mask, axis=0) - layers.elementwise_mul(
                        state, (step_mask - 1), axis=0)
                return new_state

            if not self.time_major:
                inputs = map_structure(
                    lambda x: layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(lambda x: layers.reverse(x, axis=[0]),
                                       inputs)
                mask = layers.reverse(
                    mask, axis=[0]) if sequence_length is not None else None

            states = initial_states
            outputs = []
            for i in range(time_steps):
                step_inputs = map_structure(lambda x: x[i], inputs)
                step_outputs, new_states = self.cell(step_inputs, states,
                                                     **kwargs)
                if sequence_length is not None:
                    new_states = map_structure(
                        partial(
                            _maybe_copy, step_mask=mask[i]),
                        states,
                        new_states)
                states = new_states
                outputs = map_structure(
                    lambda x: ArrayWrapper(x),
                    step_outputs) if i == 0 else map_structure(
                        lambda x, x_array: x_array.append(x), step_outputs,
                        outputs)

            final_outputs = map_structure(
                lambda x: layers.stack(x.array, axis=self.time_step_index),
                outputs)

            if self.is_reverse:
                final_outputs = map_structure(
                    lambda x: layers.reverse(x, axis=self.time_step_index),
                    final_outputs)

            final_states = new_states
        else:
            final_outputs, final_states = layers.rnn(
                self.cell,
                inputs,
                initial_states=initial_states,
                sequence_length=sequence_length,
                time_major=self.time_major,
                is_reverse=self.is_reverse,
                **kwargs)
        return final_outputs, final_states


class BidirectionalRNN(Layer):
    """
    Wrapper for bidirectional RNN. It assembles two RNNCell instances to perform
    forward and backward RNN separately, and merge outputs of these two RNN
    according to `merge_mode`.
    Parameters:
        cell_fw (RNNCell): A RNNCell instance used for forward RNN.
        cell_bw (RNNCell): A RNNCell instance used for backward RNN.
        merge_mode (str|None, optional): The way to merget outputs of forward and
            backward RNN. It can be `concat`, `sum`, `ave`, `mul`, `zip` and None,
            where None stands for make the two `outputs` as a tuple, `zip` stands
            for make each two corresponding tensors of the two `outputs` as a tuple.
            Default `concat`
    Examples:
        .. code-block:: python
            import paddle
            from paddle.incubate.hapi.text import StackedLSTMCell, BidirectionalRNN
            inputs = paddle.rand((2, 4, 32))
            cell_fw = StackedLSTMCell(32, 64)
            cell_bw = StackedLSTMCell(32, 64)
            bi_rnn = BidirectionalRNN(cell_fw, cell_bw)
            outputs, _ = bi_rnn(inputs)  # [2, 4, 128]
    """

    def __init__(self,
                 cell_fw,
                 cell_bw,
                 merge_mode='concat',
                 time_major=False,
                 cell_cls=None,
                 **kwargs):
        super(BidirectionalRNN, self).__init__()
        self.rnn_fw = RNN(cell_fw, is_reverse=False, time_major=time_major)
        self.rnn_bw = RNN(cell_bw, is_reverse=True, time_major=time_major)
        if merge_mode == 'concat':
            self.merge_func = lambda x, y: layers.concat([x, y], -1)
        elif merge_mode == 'sum':
            self.merge_func = lambda x, y: layers.elementwise_add(x, y)
        elif merge_mode == 'ave':
            self.merge_func = lambda x, y: layers.scale(
                layers.elementwise_add(x, y), 0.5)
        elif merge_mode == 'mul':
            self.merge_func = lambda x, y: layers.elementwise_mul(x, y)
        elif merge_mode == 'zip':
            self.merge_func = lambda x, y: (x, y)
        elif merge_mode is None:
            self.merge_func = None
        else:
            raise ValueError('Unsupported value for `merge_mode`: %s' %
                             merge_mode)

    def forward(self,
                inputs,
                initial_states=None,
                sequence_length=None,
                **kwargs):
        """
        Performs forward and backward RNN separately, and merge outputs of these
        two RNN according to `merge_mode`.
        Parameters:
            inputs (Variable): A (possibly nested structure of) tensor variable[s]. 
                The shape of tensor should be `[batch_size, sequence_length, ...]`
                for `time_major == False` or `[sequence_length, batch_size, ...]`
                for `time_major == True`. It represents the inputs to be unrolled
                in both forward and backward RNN.
            initial_states (Variable|list|tuple): If it is a list or tuple, its
                length should be 2 to include initial states of forward and backward
                RNN separately. Otherwise it would be used twice for the two RNN. 
                If None, `cell.get_initial_states` would be used to produce the initial
                states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
            **kwargs: Additional keyword arguments. Arguments passed to `cell.forward`.
        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is produced by merge outputs of forward and backward RNN according \
                to `merge_mode`; similarly, `final_states` is produced by merge \
                `final_states` of forward and backward RNN.
        """
        if isinstance(initial_states, (list, tuple)):
            assert len(
                initial_states
            ) == 2, "length of initial_states should be 2 when it is a list/tuple"
        else:
            initial_states = [initial_states, initial_states]
        outputs_fw, states_fw = self.rnn_fw(inputs, initial_states[0],
                                            sequence_length, **kwargs)
        outputs_bw, states_bw = self.rnn_bw(inputs, initial_states[1],
                                            sequence_length, **kwargs)
        outputs = map_structure(self.merge_func, outputs_fw,
                                outputs_bw) if self.merge_func else (outputs_fw,
                                                                     outputs_bw)
        final_states = map_structure(
            self.merge_func, states_fw,
            states_bw) if self.merge_func else (states_fw, states_bw)
        return outputs, final_states

    @staticmethod
    def bidirect_param_attr(param_attr):
        """
        Converts `param_attr` to a pair of `param_attr` when it is not a list
        or tuple with length 2, also rename every one by appending a suffix to
        avoid having same names when `param_attr` contains a name.

        Parameters:
            param_attr (list|tuple|ParamAttr): A list, tuple or something can be
                converted to a ParamAttr instance by `ParamAttr._to_attr`. When
                it is a list or tuple, its length must be 2.

        Returns:
            list: A pair composed of forward and backward RNN cell's `param_attr`.
        """
        if isinstance(param_attr, (list, tuple)):
            assert len(
                param_attr
            ) == 2, "length of param_attr should be 2 when it is a list/tuple"
            param_attrs = param_attr
        else:
            param_attrs = []
            attr = ParamAttr._to_attr(param_attr)
            attr_fw = copy.deepcopy(attr)
            if attr.name:
                attr_fw.name = attr_fw.name + "_fw"
            param_attrs.append(attr_fw)
            attr_bw = copy.deepcopy(attr)
            if attr.name:
                attr_bw.name = attr_bw.name + "_bw"
            param_attrs.append(attr_bw)
        return param_attrs


class LSTM(Layer):
    """
    Applies a stacked multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    The formula for LSTM used here is as follows:

    .. math::
        i_{t} & = \sigma(W_{x_{i}}x_{t} + b_{x_{i}} + W_{h_{i}}h_{t-1} + b_{h_{i}})
        f_{t} & = \sigma(W_{x_{f}}x_{t} + b_{x_{f}} + W_{h_{f}}h_{t-1} + b_{h_{f}})
        o_{t} & = \sigma(W_{x_{o}}x_{t} + b_{x_{o}} + W_{h_{o}}h_{t-1} + b_{h_{o}})
        c_{t} & = f_{t}c_{t-1} + i_{t} \\tanh (W_{x_{c}}x_{t} + b_{x_{c}} + W_{h_{c}}h_{t-1} + b_{h_{c}})
        h_{t} & = o_{t} \\tanh (c_{t})

    Parameters:
        input_size (int): The input feature size for the first LSTM.
        hidden_size (int): The hidden size for every LSTM.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float, optional): The dropout probability applied on the outputs
            of each LSTM except the last one. 0 for not dropout. Default 0.0
        direction (str, optional): Indicate the direction for LSTM calculation
            applying on the input sequences. It can be `forward`, `backward` or
            `bidirect`. If it is `backward`, calculate in the reverse order of
            input sequences. If it is `bidirect`, each layer would be a 
            bidirectional LSTM composed of a `forward` LSTM and `backward` LSTM,
            and it concatenates their outputs as outputs. Default: `forward`.
        time_major (bool, optional): Indicate the data layout of Tensor included
            in `input` and `output` tensors. If `False`, the data layout would
            be batch major with shape `[batch_size, sequence_length, ...]`.  If
            `True`, the data layout would be time major with shape
            `[sequence_length, batch_size, ...]`. Default: `False`.
        param_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(param_attr, num_layers)`.
            Default None.
        bias_attr (list|tuple|ParamAttr): A list, tuple or something can be
            converted to a ParamAttr instance by `ParamAttr._to_attr`. If it is
            a list or tuple, it's length must equal to `num_layers`. Otherwise,
            construct a list by `StackedRNNCell.stack_param_attr(bias_attr, num_layers)`.
            Default None.
        dtype(string, optional): The data type used in this cell. It can be
            float32 or float64. Default float32.
    Examples:
        .. code-block:: python
            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import LSTM
            inputs = paddle.rand((2, 4, 32))
            lstm = LSTM(input_size=32, hidden_size=64, num_layers=2)
            outputs, _ = lstm(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 dropout=0.0,
                 direction="forward",
                 time_major=False,
                 param_attr=None,
                 bias_attr=None,
                 dtype='float32'):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.direction = direction
        self.num_directions = 2 if direction == 'bidirect' else 1
        self.time_major = time_major

        if direction == 'bidirect':
            param_attrs = BidirectionalRNN.bidirect_param_attr(param_attr)
            bias_attrs = BidirectionalRNN.bidirect_param_attr(bias_attr)
            fw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[0],
                                                             num_layers)
            bw_param_attrs = StackedRNNCell.stack_param_attr(param_attrs[1],
                                                             num_layers)
            fw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[0],
                                                            num_layers)
            bw_bias_attrs = StackedRNNCell.stack_param_attr(bias_attrs[1],
                                                            num_layers)

            # maybe design cell including both forward and backward later
            merge_mode = 'concat'
            rnns = []
            for i in range(num_layers):
                cell_fw = StackedLSTMCell(input_size if i == 0 else (
                    hidden_size * 2 if merge_mode == 'concat' else
                    hidden_size), hidden_size, 1, dropout, fw_param_attrs[i],
                                          fw_bias_attrs[i], dtype)
                cell_bw = StackedLSTMCell(input_size if i == 0 else (
                    hidden_size * 2 if merge_mode == 'concat' else
                    hidden_size), hidden_size, 1, dropout, bw_param_attrs[i],
                                          bw_bias_attrs[i], dtype)
                rnns.append(
                    BidirectionalRNN(
                        cell_fw,
                        cell_bw,
                        merge_mode=merge_mode,
                        time_major=time_major))
            self.lstm = LayerList(rnns)
        else:
            lstm_cell = StackedLSTMCell(input_size, hidden_size, num_layers,
                                        dropout, param_attr, bias_attr, dtype)
            self.lstm = RNN(lstm_cell,
                            is_reverse=(direction == "backward"),
                            time_major=time_major)

    def forward(self, input, initial_states=None, sequence_length=None):
        """
        Performs the stacked multi-layer LSTM layer by layer. Each LSTM's `outputs`
        is the `inputs` of the subsequent one.
        Parameters:
            inputs (Variable): The inputs for the first LSTM. It is a float32
                or float64 tensor shaped `[batch_size, sequence_length, input_size]`.
            initial_states (list|None, optional): A list containing initial states 
                of all stacked LSTM, and the initial states of each LSTM is a pair
                of tensors shaped `[batch_size, hidden_size]`. If not provided,
                use 0 as initial states. Default None.
            sequence_length (Variable, optional): A tensor with shape `[batch_size]`.
                It stores real length of each instance, thus enables users to extract
                the last valid state when past a batch element's sequence length for
                correctness. If not provided, the paddings would be treated same as
                non-padding inputs. Default None.
        Returns:
            tuple: A tuple( :code:`(outputs, final_states)` ), where `outputs` \
                is the output of last LSTM and it is a tensor with shape \
                `[batch_size, sequence_length, hidden_size]` and has the same \
                data type as `inputs`, `final_states` is the counterpart of \
                `initial_states` at last time step, thus has the same structure \
                with it and has tensors with same shapes data types. 
        """
        if not isinstance(self.lstm, LayerList):
            return self.lstm(input, initial_states, sequence_length)
        else:
            if isinstance(initial_states, (list, tuple)):
                assert len(initial_states) == self.num_layers, (
                    "length of initial_states should be %d when it is a list|tuple"
                    % self.num_layers)
            else:
                initial_states = [initial_states] * self.num_layers
            stacked_states = []
            for i in range(self.num_layers):
                output, states = self.lstm[i](input, initial_states[i],
                                              sequence_length)
                input = output
                stacked_states.append(states)
            return output, stacked_states
