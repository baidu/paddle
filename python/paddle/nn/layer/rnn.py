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
    #       'RNNCell',
    #       'GRUCell',
    #       'LSTMCell'
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
from ...fluid.layers import utils, BeamSearchDecoder
from ...fluid.layers.utils import map_structure, flatten, pack_sequence_as


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
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self, cell, is_reverse=False, time_major=False):
        super(RNN, self).__init__()
        self.cell = cell
        if not hasattr(self.cell, "call"):
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
                new_state = fluid.layers.elementwise_mul(
                    new_state, step_mask,
                    axis=0) - fluid.layers.elementwise_mul(
                        state, (step_mask - 1), axis=0)
                return new_state

            if not self.time_major:
                inputs = map_structure(
                    lambda x: fluid.layers.transpose(x, [1, 0] + list(
                        range(2, len(x.shape)))), inputs)

            if sequence_length is not None:
                mask = fluid.layers.sequence_mask(
                    sequence_length,
                    maxlen=time_steps,
                    dtype=flatten(initial_states)[0].dtype)
                mask = fluid.layers.transpose(mask, [1, 0])

            if self.is_reverse:
                inputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=[0]), inputs)
                mask = fluid.layers.reverse(
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
                lambda x: fluid.layers.stack(x.array, axis=self.time_step_index
                                             ), outputs)

            if self.is_reverse:
                final_outputs = map_structure(
                    lambda x: fluid.layers.reverse(x, axis=self.time_step_index
                                                   ), final_outputs)

            final_states = new_states
        else:
            final_outputs, final_states = fluid.layers.rnn(
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
                to `merge_mode`, `final_states` is a pair including `final_states` \
                of forward and backward RNN.
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
        return outputs, (states_fw, states_bw)

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
            attr = fluid.ParamAttr._to_attr(param_attr)
            attr_fw = copy.deepcopy(attr)
            if attr.name:
                attr_fw.name = attr_fw.name + "_fw"
            param_attrs.append(attr_fw)
            attr_bw = copy.deepcopy(attr)
            if attr.name:
                attr_bw.name = attr_bw.name + "_bw"
            param_attrs.append(attr_bw)
        return param_attrs


class RNNCell(Layer):
    """
    RNNCell is the base class for abstraction representing the calculations
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
            lambda shape, dtype: fluid.layers.fill_constant_batch_size_like(
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


class StackedRNNCell(RNNCell):
    """
    Wrapper allowing a stack of RNN cells to behave as a single cell. It is used
    to implement stacked RNNs.

    Parameters:
        cells (list|tuple): List of RNN cell instances.

    Examples:

        .. code-block:: python

            from paddle.incubate.hapi.text import BasicLSTMCell, StackedRNNCell

            cells = [BasicLSTMCell(32, 32), BasicLSTMCell(32, 32)]
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
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

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
            param_attrs = [
                fluid.ParamAttr._to_attr(attr) for attr in param_attr
            ]
        else:
            param_attrs = []
            attr = fluid.ParamAttr._to_attr(param_attr)
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


class LSTMCell(RNNCell):
    """
    Long-Short Term Memory(LSTM) RNN cell.

    The formula used is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + b_{x_{i}} + W_{h_{i}}h_{t-1} + b_{h_{i}})

        f_{t} & = act_g(W_{x_{f}}x_{t} + b_{x_{f}} + W_{h_{f}}h_{t-1} + b_{h_{f}})

        o_{t} & = act_g(W_{x_{o}}x_{t} + b_{x_{o}} + W_{h_{o}}h_{t-1} + b_{h_{o}})

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + b_{x_{c}} + W_{h_{c}}h_{t-1} + b_{h_{c}})

        h_{t} & = o_{t} act_c (c_{t})

    Please refer to `An Empirical Exploration of Recurrent Network Architectures
    <http://proceedings.mlr.press/v37/jozefowicz15.pdf>`_ for more details.

    Parameters:
        input_size (int): The input size in the LSTM cell.
        hidden_size (int): The hidden size in the LSTM cell.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        param_attr(ParamAttr, optional): The parameter attribute for the learnable
            weight matrix. Default: None.
        bias_attr (ParamAttr, optional): The parameter attribute for the bias
            of LSTM. Default: None.
        dtype(string, optional): The data type used in this cell. Default float32.

    Examples:

        .. code-block:: python

            import paddle
            import paddle.fluid as fluid
            from paddle.incubate.hapi.text import BasicLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = BasicLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 param_attr=None,
                 bias_attr=None,
                 gate_activation="sigmoid",
                 activation="tanh",
                 use_cudnn_impl=True,
                 dtype="float32"):
        super(LSTMCell, self).__init__(dtype)

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gate_activation = getattr(fluid.layers, gate_activation)
        self.activation = getattr(fluid.layers, activation)
        self.use_cudnn_impl = use_cudnn_impl
        self._param_attr = param_attr
        self._bias_attr = bias_attr
        self._dtype = dtype

        if self._param_attr is not None and self._param_attr.name is not None:
            weight_ih_param_attr = copy.deepcopy(self._param_attr)
            weight_hh_param_attr = copy.deepcopy(self._param_attr)
            weight_ih_param_attr.name += "_weight_ih"
            weight_hh_param_attr.name += "_weight_hh"
        else:
            weight_ih_param_attr = self._param_attr
            weight_hh_param_attr = self._param_attr

        if self._bias_attr is not None and self._bias_attr.name is not None:
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
        gates = layers.matmul(
            inputs, self.weight_ih, transpose_y=True) + self.bias_ih
        gates += layers.matmul(
            pre_hidden, self.weight_hh, transpose_y=True) + self.bias_hh

        chunked_gates = layers.split(gates, num_or_sections=4, dim=1)

        i = self.gate_activation(chunked_gates[0])
        f = self.gate_activation(chunked_gates[1])
        o = self.gate_activation(chunked_gates[3])
        c = f * pre_cell + i * self.activation(chunked_gates[2])
        h = o * self.activation(c)

        return h, [h, c]

    @property
    def state_shape(self):
        """
        The `state_shape` of BasicLSTMCell is a list with two shapes: `[[hidden_size], [hidden_size]]`
        (-1 for batch size would be automatically inserted into shape). These two
        shapes correspond to :math:`h_{t-1}` and :math:`c_{t-1}` separately.
        """
        return [[self.hidden_size], [self.hidden_size]]


class StackedLSTMCell(RNNCell):
    """
    Wrapper allowing a stack of LSTM cells to behave as a single cell. It is used
    to implement stacked LSTM.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + b_{x_{i}} + W_{h_{i}}h_{t-1} + b_{h_{i}})

        f_{t} & = act_g(W_{x_{f}}x_{t} + b_{x_{f}} + W_{h_{f}}h_{t-1} + b_{h_{f}})

        o_{t} & = act_g(W_{x_{o}}x_{t} + b_{x_{o}} + W_{h_{o}}h_{t-1} + b_{h_{o}})

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + b_{x_{c}} + W_{h_{c}}h_{t-1} + b_{h_{c}})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input size for the first LSTM cell.
        hidden_size (int): The hidden size for every LSTM cell.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
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
            from paddle.incubate.hapi.text import StackedLSTMCell, RNN

            inputs = paddle.rand((2, 4, 32))
            cell = StackedLSTMCell(input_size=32, hidden_size=64)
            rnn = RNN(cell=cell)
            outputs, _ = rnn(inputs)  # [2, 4, 64]
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 gate_activation="sigmoid",
                 activation="tanh",
                 num_layers=1,
                 dropout=0.0,
                 param_attr=None,
                 bias_attr=None,
                 dtype="float32"):
        super(StackedLSTMCell, self).__init__()
        self.dropout = utils.convert_to_list(dropout, num_layers, "dropout",
                                             float)
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
                        gate_activation=gate_activation,
                        activation=activation,
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
            **kwargs: Additional keyword arguments, which passed to `cell.forward`
                for all including cells.

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
                self.dropout[i],
                dropout_implementation='upscale_in_train') if self.dropout[
                    i] > 0 else outputs
            inputs = outputs
            new_states.append(new_state)
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


class _CudnnRnnAdaptor(object):
    pass


class LSTM(Layer):
    """
    Applies a stacked multi-layer long short-term memory (LSTM) RNN to an input
    sequence.

    The formula for LSTM used here is as follows:

    .. math::

        i_{t} & = act_g(W_{x_{i}}x_{t} + W_{h_{i}}h_{t-1} + b_{i})

        f_{t} & = act_g(W_{x_{f}}x_{t} + W_{h_{f}}h_{t-1} + b_{f} + forget\\_bias)

        c_{t} & = f_{t}c_{t-1} + i_{t} act_c (W_{x_{c}}x_{t} + W_{h_{c}}h_{t-1} + b_{c})

        o_{t} & = act_g(W_{x_{o}}x_{t} + W_{h_{o}}h_{t-1} + b_{o})

        h_{t} & = o_{t} act_c (c_{t})


    Parameters:
        input_size (int): The input feature size for the first LSTM.
        hidden_size (int): The hidden size for every LSTM.
        gate_activation (function, optional): The activation function for gates
            of LSTM, that is :math:`act_g` in the formula. Default: None,
            representing for `fluid.layers.sigmoid`.
        activation (function, optional): The non-gate activation function of
            LSTM, that is :math:`act_c` in the formula. Default: None,
            representing for 'fluid.layers.tanh'.
        num_layers(int, optional): The number of LSTM to be stacked. Default 1.
        dropout(float|list|tuple, optional): The dropout probability after each
            LSTM. It also can be a list or tuple, including dropout probabilities
            for the corresponding LSTM. Default 0.0
        is_reverse (bool, optional): Indicate whether to calculate in the reverse
            order of input sequences. Default: `False`.
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
                 gate_activation="sigmoid",
                 activation="tanh",
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
                    hidden_size), hidden_size, gate_activation, activation, 1,
                                          dropout, fw_param_attrs[i],
                                          fw_bias_attrs[i], dtype)
                cell_bw = StackedLSTMCell(input_size if i == 0 else (
                    hidden_size * 2 if merge_mode == 'concat' else
                    hidden_size), hidden_size, gate_activation, activation, 1,
                                          dropout, bw_param_attrs[i],
                                          bw_bias_attrs[i], dtype)
                rnns.append(
                    BidirectionalRNN(
                        cell_fw,
                        cell_bw,
                        merge_mode=merge_mode,
                        time_major=time_major))
            self.lstm = LayerList(rnns)
        else:
            lstm_cell = StackedLSTMCell(input_size, hidden_size,
                                        gate_activation, activation, num_layers,
                                        dropout, param_attr, bias_attr, dtype)
            self.lstm = RNN(lstm_cell,
                            is_reverse=(direction == "backward"),
                            time_major=time_major)

        # TODO(guosheng): need more elaborate condition for cudnn, including
        # place and cudnn support
        self.could_use_cudnn = fluid.is_compiled_with_cuda()
        self.could_use_cudnn &= (
            activation == 'tanh' and gate_activation == 'sigmoid' and
            direction != 'backward' and
            fluid.data_feeder.convert_dtype(dtype) == 'float32')
        self.could_use_cudnn &= len(self.lstm.parameters(
        )) == num_layers * 4 * (2 if direction == 'bidirect' else 1)
        # flatten parameters to use cudnn
        # TODO(guosheng): refine this to share storage between the newly
        # created parameter and original parameters when support sharing
        if self.could_use_cudnn:
            shape = [np.prod(param.shape) for param in self.lstm.parameters()]
            self.flat_weight = self.create_parameter(
                shape=[np.sum(shape)], dtype=dtype)
            with fluid.program_guard(fluid.default_startup_program(),
                                     fluid.default_startup_program()):
                with fluid.dygraph.no_grad():
                    # layer.parameters() is depth first and ordered
                    # for i in layer: for j in direct: w_ih, w_hh, b_ih, b_hh
                    params = self.lstm.parameters()
                    flat_params = []
                    for param in params:
                        # no_grad guard does not work for static-graph, thus
                        # stop gradient manually. Maybe remove it since only
                        # add to startup program
                        param.stop_gradient = True
                        # TODO: maybe forbid assignment usage by ast
                        flat_params.append(layers.reshape(param, [-1]))
                    param_buf = layers.concat(flat_params)
                    param_buf.stop_gradient = True
                    layers.assign(param_buf, self.flat_weight)
                    #self.lstm = None

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
        if self.could_use_cudnn:
            # stop_gradient for init but not for forward
            self.flat_weight.stop_gradient = False

            out = self._helper.create_variable_for_type_inference(input.dtype)
            last_h = self._helper.create_variable_for_type_inference(
                input.dtype)
            last_c = self._helper.create_variable_for_type_inference(
                input.dtype)
            reserve = self._helper.create_variable_for_type_inference(
                dtype=fluid.core.VarDesc.VarType.UINT8, stop_gradient=True)
            state = self._helper.create_variable_for_type_inference(
                dtype=fluid.core.VarDesc.VarType.UINT8, stop_gradient=True)
            state.persistable = True
            state_out = state

            inputs = {
                'Input': input,
                'W': self.flat_weight,
                'InitH': layers.fill_constant_batch_size_like(input, [
                    self.num_layers * self.num_directions, -1, self.hidden_size
                ], input.dtype, 0, 1, 1)
                if initial_states is None else initial_states[0],
                'InitC': layers.fill_constant_batch_size_like(input, [
                    self.num_layers * self.num_directions, -1, self.hidden_size
                ], input.dtype, 0, 1, 1),
                'State': state,
            }
            attrs = {
                'max_len': 100,
                'dropout_prob': self.dropout,
                'is_bidirec': self.num_directions == 2,
                'input_size': self.input_size,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
            }

            outputs = {
                'Out': out,
                'LastH': last_h,
                'LastC': last_c,
                'Reserve': reserve,
                'StateOut': state_out,
            }

            self._helper.append_op(
                type="cudnn_lstm", inputs=inputs, outputs=outputs, attrs=attrs)
            return out, (last_h, last_c)

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

    def __getattr__(self, name):
        if self.__dict__.get("could_use_cudnn",
                             None) and name == self.__class__.__name__.lower():
            # make self.lstm private to avoid internal usages causing warning
            caller = sys._getframe(1).f_code.co_name
            if not caller in dir(self):
                warnings.warn(  # warnings.warn also ignores the same warning 
                    "A flattened parameter is created to use cudnn, and the "
                    "original parameters of {} would not be used and updated. "
                    "*** Please do not use these {} parameters directly, such "
                    "as sharing these weights. To set parameter values, use "
                    "`set_parameter_values`, which would also copy to the new "
                    "parameter. To copy latest weight values back to the original "
                    "parameters, such as for saving, use `state_dict()` to sync. ***"
                    .format(self.__class__.__name__,
                            self.lstm.state_dict().keys()))
        return super(self.__class__, self).__getattr__(name)

    def __setattr__(self, name, value):
        if self.__dict__.get("could_use_cudnn",
                             None) and name == self.__class__.__name__.lower():
            # make self.lstm private to avoid internal usages causing warning
            caller = sys._getframe(1).f_code.co_name
            if not caller in dir(self):
                warnings.warn(  # warnings.warn also ignores the same warning 
                    "A flattened parameter is created to use cudnn, and the "
                    "original parameters of {} would not be used and updated. "
                    "*** Please do not use these {} parameters directly, such "
                    "as sharing these weights. To set parameter values, use "
                    "`set_parameter_values`, which would also copy to the new "
                    "parameter. To copy latest weight values back to the original "
                    "parameters, such as for saving, use `state_dict()` to sync. ***"
                    .format(self.__class__.__name__,
                            self.lstm.state_dict().keys()))
        super(self.__class__, self).__setattr__(name, value)

    def state_dict(self):
        def to_numpy(var):
            if isinstance(var, np.ndarray):
                return var
            if isinstance(var, fluid.core.VarBase):
                return var.numpy()
            assert fluid.executor.global_scope(
            ).find_var(var.name) and fluid.executor.global_scope().find_var(
                var.name).get_tensor(), "Please do parameter initialization."
            t = fluid.executor.global_scope().find_var(var.name).get_tensor()
            return np.array(t)

        def to_tensor(var, val):
            def set_var(var, ndarray):
                assert fluid.executor.global_scope().find_var(
                    var.name) and fluid.executor.global_scope().find_var(
                        var.name).get_tensor(
                        ), "Please do parameter initialization."
                t = fluid.executor.global_scope().find_var(var.name).get_tensor(
                )
                p = t._place()
                if p.is_cpu_place():
                    place = fluid.CPUPlace()
                elif p.is_cuda_pinned_place():
                    place = fluid.CUDAPinnedPlace()
                else:
                    p = fluid.core.Place()
                    p.set_place(t._place())
                    place = fluid.CUDAPlace(p.gpu_device_id())
                t.set(ndarray, place)

            val = to_numpy(val)
            if isinstance(var, fluid.core.VarBase):
                return var.set_value(val)
            set_var(var, val)
            return var

        if self.could_use_cudnn:
            if fluid.in_dygraph_mode() or (fluid.executor.global_scope(
            ).find_var(self.flat_weight.name) and fluid.executor.global_scope(
            ).find_var(self.flat_weight.name).get_tensor()._is_initialized()):
                params = self.lstm.parameters()
                flat_weight = to_numpy(self.flat_weight)
                offset = 0
                for param in params:
                    mat_numel = np.prod(param.shape)
                    to_tensor(param, flat_weight[offset:offset + mat_numel]
                              .reshape(param.shape))
                    offset += mat_numel
        return super(LSTM, self).state_dict()

    def set_parameter_values(self, values):
        """
        Copies the values into parameters. This provides a way to reset the
        weights' values.
        """

        def to_numpy(var):
            if isinstance(var, np.ndarray):
                return var
            if isinstance(var, fluid.core.VarBase):
                return var.numpy()
            assert fluid.executor.global_scope(
            ).find_var(var.name) and fluid.executor.global_scope().find_var(
                var.name).get_tensor(), "Please do parameter initialization."
            t = fluid.executor.global_scope().find_var(var.name).get_tensor()
            return np.array(t)

        def to_tensor(var, val):
            def set_var(var, ndarray):
                assert fluid.executor.global_scope().find_var(
                    var.name) and fluid.executor.global_scope().find_var(
                        var.name).get_tensor(
                        ), "Please do parameter initialization."
                t = fluid.executor.global_scope().find_var(var.name).get_tensor(
                )
                p = t._place()
                if p.is_cpu_place():
                    place = fluid.CPUPlace()
                elif p.is_cuda_pinned_place():
                    place = fluid.CUDAPinnedPlace()
                else:
                    p = fluid.core.Place()
                    p.set_place(t._place())
                    place = fluid.CUDAPlace(p.gpu_device_id())
                t.set(ndarray, place)

            val = to_numpy(val)
            if isinstance(var, fluid.core.VarBase):
                return var.set_value(val)
            set_var(var, val)
            return var

        # not for sharing weights, but for setting weight values
        params = self.lstm.parameters()
        if isinstance(values, (list, tuple)):
            if self.could_use_cudnn:
                flat_weight = to_numpy(self.flat_weight)
                offset = 0
            for param, value in zip(params, values):
                mat_numel = np.prod(param.shape)
                np_val = to_numpy(value)
                to_tensor(param, np_val)
                if self.could_use_cudnn:
                    flat_weight[offset:offset + mat_numel] = np_val.reshape(
                        [-1])
                    offset += mat_numel
            if self.could_use_cudnn:
                to_tensor(self.flat_weight, flat_weight)
        else:  # set the value for flattened parameter
            if self.could_use_cudnn:
                flat_weight = to_numpy(values)
                to_tensor(self.flat_weight, flat_weight)
                offset = 0
                for param in params:
                    mat_numel = np.prod(param.shape)
                    to_tensor(param, flat_weight[offset:offset + mat_numel]
                              .reshape(param.shape))
                    offset += mat_numel
