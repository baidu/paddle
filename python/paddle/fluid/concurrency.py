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

from .layers.control_flow import BlockGuard, equal
from .framework import Operator
from .layer_helper import LayerHelper, unique_name
from .layers import fill_constant
from . import core

__all__ = [
    'Go', 'make_channel', 'channel_send', 'channel_recv', 'channel_close',
    'Select'
]


class Go(BlockGuard):
    def __init__(self, name=None):
        self.helper = LayerHelper("go", name=name)
        super(Go, self).__init__(self.helper.main_program)

    def __enter__(self):
        super(Go, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.construct_go_op()
        return super(Go, self).__exit__(exc_type, exc_val, exc_tb)

    def construct_go_op(self):
        main_program = self.helper.main_program
        go_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        inner_outputs = set()
        x_name_list = set()
        for op in go_block.ops:
            # Iterate over all operators, get all the inputs
            # and add as input to the Go operator.
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in inner_outputs:
                        x_name_list.add(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    inner_outputs.add(out_var_name)

        # Iterate over all operators , get all the outputs
        # add to the output list of Go operator only if
        # they exist in the parent block.
        out_vars = []
        for inner_out_name in inner_outputs:
            if inner_out_name in parent_block.vars:
                out_vars.append(parent_block.var(inner_out_name))

        parent_block.append_op(
            type='go',
            inputs={
                'X': [
                    parent_block._var_recursive(x_name)
                    for x_name in x_name_list
                ]
            },
            outputs={},
            attrs={'sub_block': go_block})


class SelectCase(object):
    DEFAULT = 0
    SEND = 1
    RECEIVE = 2

    def __init__(self,
                 select,
                 case_idx,
                 case_to_execute,
                 channel_action_fn=None,
                 channel=None,
                 value=None,
                 is_copy=False):
        self.select = select
        self.helper = LayerHelper('conditional_block')
        self.main_program = self.helper.main_program
        self.is_scalar_condition = True

        self.case_to_execute = case_to_execute
        self.idx = case_idx

        # Since we aren't going to use the `channel_send` or `channel_recv`
        # functions directly, we just need to capture the name.
        self.action = (self.SEND
                       if channel_action_fn.__name__ == ('channel_send') else
                       self.RECEIVE) if channel_action_fn else self.DEFAULT

        X = value
        if self.action == self.SEND and is_copy:
            # We create of copy of the data we want to send
            copied_X = self.select.parent_block.create_var(
                name=unique_name.generate(value.name + '_copy'),
                type=value.type,
                dtype=value.dtype,
                shape=value.shape,
                lod_level=value.lod_level,
                capacity=value.capacity
                if hasattr(value, 'capacity') else None, )

            self.select.parent_block.append_op(
                type="assign", inputs={"X": value}, outputs={"Out": copied_X})
            X = copied_X

        self.value = X
        self.channel = channel

    def __enter__(self):
        self.block = self.main_program.create_block()

    def construct_op(self):
        main_program = self.helper.main_program
        cases_block = main_program.current_block()

        inner_outputs = set()
        input_set = set()
        params = set()

        for op in self.block.ops:
            # Iterate over all operators, get all the inputs
            # and add as input to the SelectCase operator.
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in inner_outputs:
                        input_set.add(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    inner_outputs.add(out_var_name)

        param_list = [
            cases_block.var(each_name) for each_name in params
            if each_name not in input_set
        ]

        # Iterate over all operators, get all the outputs
        # add to the output list of SelectCase operator only if
        # they exist in the parent block.
        out_vars = []
        for inner_out_name in inner_outputs:
            if inner_out_name in cases_block.vars:
                out_vars.append(cases_block.var(inner_out_name))

        # First, create an op that will determine whether or not this is the
        # conditional variable to execute.
        should_execute_block = equal(
            fill_constant(
                shape=[1], dtype=core.VarDesc.VarType.INT32, value=self.idx),
            self.case_to_execute)

        step_scope = cases_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        cases_block.append_op(
            type='conditional_block',
            inputs={'X': [should_execute_block],
                    'Params': param_list},
            outputs={'Out': out_vars,
                     'Scope': [step_scope]},
            attrs={
                'sub_block': self.block,
                'is_scalar_condition': self.is_scalar_condition
            })

        return '%s,%s,%s,%s' % (self.idx, self.action, self.channel.name
                                if self.channel else '', self.value.name
                                if self.value else '')

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class Select(BlockGuard):
    def __init__(self, name=None):
        self.helper = LayerHelper('select', name=name)
        self.parent_block = self.helper.main_program.current_block()
        self.cases = []

        super(Select, self).__init__(self.helper.main_program)
        self.case_to_execute = fill_constant(
            shape=[1], dtype=core.VarDesc.VarType.INT32, value=-1)

    def __enter__(self):
        super(Select, self).__enter__()
        return self

    def case(self, channel_action_fn, channel, value, is_copy=False):
        """Create a new block for this condition.
        """
        select_case = SelectCase(self,
                                 len(self.cases), self.case_to_execute,
                                 channel_action_fn, channel, value, is_copy)

        self.cases.append(select_case)

        return select_case

    def default(self):
        """Create a default case block for this condition.
        """
        default_case = SelectCase(self, len(self.cases), self.case_to_execute)

        self.cases.append(default_case)

        return default_case

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False

        # Create a select op and another block to wrap its
        # case blocks.
        select_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(select_block.parent_idx)

        # Construct each case op, inside the newly created select block.
        serialized_cases = []
        for case in self.cases:
            serialized_cases.append(case.construct_op())

        intermediate = set()
        params = set()

        for case_block in select_block.ops:
            if case_block.attrs and 'sub_block' in case_block.attrs:
                for each_op in case_block.attrs['sub_block'].ops:
                    assert isinstance(each_op, Operator)
                    for iname in each_op.input_names:
                        for in_var_name in each_op.input(iname):
                            if in_var_name not in intermediate:
                                params.add(in_var_name)

                    for oname in each_op.output_names:
                        for out_var_name in each_op.output(oname):
                            intermediate.add(out_var_name)

        out_list = [
            parent_block.var(var_name) for var_name in parent_block.vars
            if var_name in intermediate
        ]

        X = [select_block._var_recursive(x_name) for x_name in params]

        # Needs to be used by `equal` inside the cases block.
        X.append(self.case_to_execute)

        # Construct the select op.
        parent_block.append_op(
            type='select',
            inputs={'X': X,
                    'case_to_execute': self.case_to_execute},
            attrs={'sub_block': select_block,
                   'cases': serialized_cases},
            outputs={'Out': out_list})

        return super(Select, self).__exit__(exc_type, exc_val, exc_tb)


def make_channel(dtype, capacity=0):
    """
    Helps implementation of a concurrent program by creating a "channel" of
    a defined data type. Channels allow for the passing of data in
    concurrent scenarios - such as when using threads to divide computation.
    Channels can be used to "send" and "receive" such data concurrently.

    There are two kinds of channels: unbuffered and buffered. Unbuffered
    channels have no capacity - and thus, block on send and only unblock only
    once what they have sent has been received.

    On the other hand, buffered channels are initialized with a capacity -
    and do not block on sends.

    Use this method in combination with `channel_send`, `channel_recv`,
    `channel_close`, and `Go` to design a concurrent Paddle program.

    Args:
        dtype (ParamAttr|string): Data type of the data sent in the channel.
        This data type should be the string name of a numpy data type.
        capacity (ParamAttr|int): Size of the channel. Defaults to 0 for
        to create an unbuffered channel.

    Returns:
        Variable: The channel variable that can be used to send an receive data
                  of the defined dtype.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to execute in a Go block, which receives the channel data.
          fluid.channel_send(ch, 100)
          fluid.channel_close(ch)
    """
    helper = LayerHelper('channel_create', **locals())
    main_program = helper.main_program
    make_channel_block = main_program.current_block()

    # Make a channel variable (using the channel data type) and make sure it
    # persists into the global scope.
    channel = helper.create_variable(
        name=unique_name.generate('channel'),
        type=core.VarDesc.VarType.CHANNEL,
        persistable=True)

    create_channel_op = make_channel_block.append_op(
        type="channel_create",
        outputs={"Out": channel},
        attrs={"data_type": dtype,
               "capacity": capacity})

    return channel


def channel_send(channel, value, is_copy=False):
    """
    Sends a value through a channel variable. Used by an unbuffered or buffered
    channel to pass data from within or to a concurrent Go block, where
    `channel_recv` to used to get the passed value.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.
        value (Variable): Value to send to channel
        is_copy (bool): Copy data while channel send. If False, then data
        is moved. The input cannot be used after move. (default False)
    Returns:
        Variable: The boolean status on whether or not the channel
                  successfully sent the passed value.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to execute in a Go block, which receives the channel data.
          fluid.channel_send(ch, 100)
    """
    helper = LayerHelper('channel_send', **locals())
    main_program = helper.main_program
    channel_send_block = main_program.current_block()

    X = value

    if is_copy:
        copied_X = helper.create_variable(
            name=unique_name.generate(value.name + '_copy'),
            type=value.type,
            dtype=value.dtype,
            shape=value.shape,
            lod_level=value.lod_level,
            capacity=value.capacity if hasattr(value, 'capacity') else None)

        assign_op = channel_send_block.append_op(
            type="assign", inputs={"X": value}, outputs={"Out": copied_X})
        X = copied_X

    channel_send_block.append_op(
        type="channel_send", inputs={
            "Channel": channel,
            "X": X,
        })


def channel_recv(channel, return_value):
    """
    Receives a value through a channel variable. Used by an unbuffered or
    buffered channel within a concurrent Go block to get data from originally
    sent using `channel_send`, or from outside such a block where
    `channel_send` is used to send the value.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.
        return_value (Variable): Variable to set as a result of running channel_recv_op

    Returns:
        Variable: The received value from the channel.
        Variable: The boolean status on whether or not the channel
                  successfully received the passed value.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          with fluid.Go():
            returned_value, return_status = fluid.channel_recv(ch, 'int32')

          # Code to send data through the channel.
    """
    helper = LayerHelper('channel_recv', **locals())
    main_program = helper.main_program
    channel_recv_block = main_program.current_block()

    status = helper.create_variable(
        name=unique_name.generate('status'),
        type=core.VarDesc.VarType.LOD_TENSOR,
        dtype=core.VarDesc.VarType.BOOL)

    channel_recv_op = channel_recv_block.append_op(
        type="channel_recv",
        inputs={"Channel": channel},
        outputs={"Out": return_value,
                 "Status": status})

    return return_value, status


def channel_close(channel):
    """
    Closes a channel created using `make_channel`.

    Args:
        channel (Variable|Channel): Channel variable created using
        `make_channel`.

    Examples:
        .. code-block:: python

          ch = fluid.make_channel(dtype='int32', capacity=10)
          ...
          # Code to receive and send data through a channel
          ...
          fluid.channel_close(ch)
    """
    helper = LayerHelper('channel_close', **locals())
    main_program = helper.main_program
    channel_close_block = main_program.current_block()

    channel_close_op = channel_close_block.append_op(
        type="channel_close", inputs={"Channel": channel})
