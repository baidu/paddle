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
import contextlib

from layer_function_generator import autodoc
from tensor import assign, fill_constant
from .. import core
from ..framework import Program, Variable, Operator, Block
from ..layer_helper import LayerHelper, unique_name
from ops import logical_and, logical_not, logical_or

__all__ = [
    'split_lod_tensor',
    'merge_lod_tensor',
    'BlockGuard',
    'BlockGuardWithCompletion',
    'StaticRNNMemoryLink',
    'WhileGuard',
    'While',
    'Switch',
    'Select',
    'lod_rank_table',
    'max_sequence_len',
    'topk',
    'lod_tensor_to_array',
    'array_to_lod_tensor',
    'increment',
    'array_write',
    'create_array',
    'less_than',
    'equal',
    'array_read',
    'shrink_memory',
    'array_length',
    'IfElse',
    'DynamicRNN',
    'ConditionalBlock',
    'StaticRNN',
    'reorder_lod_tensor_by_rank',
    'ParallelDo',
    'Print',
]


def split_lod_tensor(input, mask, level=0):
    """
    **split_lod_tensor**

    This function takes in an input that contains the complete lod information,
    and takes in a mask which is used to mask certain parts of the input.
    The output is the true branch and the false branch with the mask applied to
    the input at a certain level in the tensor.

    Args:
        input(tuple|list|None): The input tensor that contains complete
                                lod information needed to construct the output.
        mask(list): A bool column vector which masks the input.
        level(int): The specific lod level to rank.

    Returns:
        Variable: The true branch of tensor as per the mask applied to input.
        Variable: The false branch of tensor as per the mask applied to input.

    Examples:
        .. code-block:: python

          x = layers.data(name='x', shape=[1])
          x.persistable = True

          y = layers.data(name='y', shape=[1])
          y.persistable = True

          out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
    """
    helper = LayerHelper('split_lod_tensor', **locals())
    out_true = helper.create_tmp_variable(dtype=input.dtype)
    out_false = helper.create_tmp_variable(dtype=input.dtype)
    helper.append_op(
        type='split_lod_tensor',
        inputs={
            'X': input,
            'Mask': mask,
        },
        outputs={'OutTrue': out_true,
                 'OutFalse': out_false},
        attrs={'level': level})
    return out_true, out_false


def merge_lod_tensor(in_true, in_false, x, mask, level=0):
    """
    **merge_lod_tensor**

    This function takes in an input :math:`x`, the True branch, the False
    branch and a binary :math:`mask`. Using this information, this function
    merges the True and False branches of the tensor into a single Output
    at a certain lod level indiacted by :math:`level`.

    Args:
        in_true(tuple|list|None): The True branch to be merged.
        in_false(tuple|list|None): The False branch to be merged.
        x(tuple|list|None): The input tensor that contains complete
                            lod information needed to construct the output.
        mask(list): A bool column vector which masks the input.
        level(int): The specific lod level to rank.

    Returns:
        Variable: The merged output tensor.

    Examples:
        .. code-block:: python

          x = layers.data(
                      name='x', shape=[1], dtype='float32', stop_gradient=False)
          y = layers.data(
                name='y', shape=[1], dtype='bool', stop_gradient=False)

          level = 0

          out_true, out_false = layers.split_lod_tensor(
                input=x, mask=y, level=level)
          out = layers.merge_lod_tensor(
                in_true=out_true, in_false=out_false, mask=y, x=x, level=level)
    """
    helper = LayerHelper('merge_lod_tensor', **locals())
    out = helper.create_tmp_variable(dtype=in_true.dtype)
    helper.append_op(
        type='merge_lod_tensor',
        inputs={'X': x,
                'Mask': mask,
                'InTrue': in_true,
                'InFalse': in_false},
        outputs={'Out': out},
        attrs={'level': level})
    return out


def Print(input,
          first_n=-1,
          message=None,
          summarize=-1,
          print_tensor_name=True,
          print_tensor_type=True,
          print_tensor_shape=True,
          print_tensor_lod=True,
          print_phase='both'):
    '''
    **Print operator**

    This creates a print op that will print when a tensor is accessed.

    Wraps the tensor passed in so that whenever that a tensor is accessed,
    the message `message` is printed, along with the current value of the
    tensor `t`.

    Args:
        input (Variable): A Tensor to print.
        summarize (int): Print this number of elements in the tensor, will print
                all if left is negative.
        message (str): A string message to print as a prefix.
        first_n (int): Only log `first_n` number of times.
        print_tensor_name (bool): Print the tensor name.
        print_tensor_type (bool): Print the tensor type.
        print_tensor_shape (bool): Print the tensor shape.
        print_tensor_lod (bool): Print the tensor lod.
        print_phase (str): Which phase to displace, including 'forward',
                'backward' and 'both'. If set to 'backward' or 'both', will
                print the gradients of input tensor.

    Returns:
        Variable: Output tensor, same data with input tensor.

    Examples:
        .. code-block:: python

        value = some_layer(...)
        Print(value, summarize=10,
              message="The content of some_layer: ")
    '''
    helper = LayerHelper('print', **locals())
    out = helper.create_tmp_variable(dtype=helper.input_dtype())
    helper.append_op(
        type='print',
        inputs={'In': input},
        attrs={
            'first_n': first_n,
            'summarize': summarize,
            'message': message or "",
            'print_tensor_name': print_tensor_name,
            'print_tensor_type': print_tensor_type,
            'print_tensor_shape': print_tensor_shape,
            'print_tensor_lod': print_tensor_lod,
            'print_phase': print_phase.upper()
        },
        outputs={'Out': out})
    return out


class BlockGuard(object):
    """
    BlockGuard class.

    BlockGuard class is used to create a sub-block in a program by
    using the Python `with` keyword.
    """

    def __init__(self, main_program):
        if not isinstance(main_program, Program):
            raise TypeError("BlockGuard takes a program")
        self.main_program = main_program

    def __enter__(self):
        self.main_program.create_block()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class ParallelDo(object):
    """
    ParallelDo class.

    ParallelDo class is used to create a ParallelDo.
    """

    def __init__(self, places, use_nccl=False, name=None):
        self.helper = LayerHelper("parallel_do", name=name)
        self.inputs = []
        self.places = places
        self.outputs = []
        self.status = StaticRNN.BEFORE_RNN_BLOCK
        self.use_nccl = use_nccl

    def do(self):
        return BlockGuardWithCompletion(self)

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def __call__(self, *args, **kwargs):
        if self.status != StaticRNN.AFTER_RNN_BLOCK:
            raise ValueError("RNN output can only be retrieved after rnn block")
        if len(self.outputs) == 0:
            raise ValueError("RNN has no output")
        elif len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def read_input(self, var):
        self.inputs.append(var)
        return var

    def write_output(self, var):
        self.outputs.append(var)

    def get_parameters(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()

        local_inputs = set()
        params = list()
        for var in self.inputs:
            local_inputs.add(var.name)

        for op in current_block.ops:
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    local_inputs.add(out_var_name)

        params = list(set(params))

        return [parent_block.var(name) for name in params]

    def complete_op(self):
        main_program = self.helper.main_program
        current_block = main_program.current_block()
        parent_block = self.parent_block()

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        self.outputs = [
            parent_block.create_var(
                name=o.name,
                shape=o.shape,
                dtype=o.dtype,
                lod_level=o.lod_level,
                persistable=o.persistable,
                stop_gradient=o.stop_gradient) for o in self.outputs
        ]

        inputs = [parent_block.var(i.name) for i in self.inputs]
        outputs = [parent_block.var(o.name) for o in self.outputs]

        parent_block.append_op(
            type='parallel_do',
            inputs={
                'inputs': inputs,
                'parameters': self.get_parameters(),
                'places': self.places
            },
            outputs={'outputs': outputs,
                     'parallel_scopes': [step_scope]},
            attrs={'sub_block': current_block,
                   'use_nccl': self.use_nccl})


class BlockGuardWithCompletion(BlockGuard):
    """
    BlockGuardWithCompletion class.

    BlockGuardWithCompletion class is used to create an op with a block in a program.
    """

    def __init__(self, rnn):
        if not (isinstance(rnn, StaticRNN) or isinstance(rnn, ParallelDo)):
            raise TypeError(
                "BlockGuardWithCompletion takes a StaticRNN or ParallelDo")
        super(BlockGuardWithCompletion, self).__init__(rnn.helper.main_program)
        self.rnn = rnn

    def __enter__(self):
        self.rnn.status = StaticRNN.IN_RNN_BLOCK
        return super(BlockGuardWithCompletion, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.rnn.status = StaticRNN.AFTER_RNN_BLOCK
        self.rnn.complete_op()
        return super(BlockGuardWithCompletion, self).__exit__(exc_type, exc_val,
                                                              exc_tb)


class StaticRNNMemoryLink(object):
    """
    StaticRNNMemoryLink class.

    Args:
        init: the initial variable for Memory
        init: Variable
        pre_mem: the memory variable in previous time step
        pre_mem: Variable
        mem: the memory variable in current time step
        mem: Variable

    StaticRNNMemoryLink class is used to create a link between two
    memory cells of a StaticRNN.
    """

    def __init__(self, init, pre_mem, mem=None):
        self.init = init
        self.pre_mem = pre_mem
        self.mem = mem


class StaticRNN(object):
    """
    StaticRNN class.

    StaticRNN class is used to create a StaticRNN. The RNN will have its
    own parameters like inputs, outputs, memories, status and length.
    """
    BEFORE_RNN_BLOCK = 0
    IN_RNN_BLOCK = 1
    AFTER_RNN_BLOCK = 2

    def __init__(self, name=None):
        self.helper = LayerHelper("static_rnn", name=name)
        self.memories = {}  # memory map, from pre_mem.name --> MemoryLink
        self.inputs = []  # input variable list in current block
        self.outputs = []  # output variable list in parent block
        self.status = StaticRNN.BEFORE_RNN_BLOCK  # status flag.
        # sequence length, since it is a static RNN, sequence length are fixed.
        self.seq_len = None

    def step(self):
        return BlockGuardWithCompletion(self)

    def _assert_in_rnn_block_(self, method):
        if self.status != StaticRNN.IN_RNN_BLOCK:
            raise ValueError("You must invoke {0} in rnn block".format(method))

    def memory(self,
               init=None,
               shape=None,
               batch_ref=None,
               init_value=0.0,
               init_batch_dim_idx=0,
               ref_batch_dim_idx=1):
        """
        Args:
            init: boot memory, if not set, a shape, batch_ref must be provided
            shape: shape of the boot memory
            batch_ref: batch size reference variable
            init_value: the init value of boot memory
            init_batch_dim_idx: the index of batch size in init's dimension
            ref_batch_dim_idx: the index of batch size in batch_ref's dimension
        """
        self._assert_in_rnn_block_('memory')
        if init is None:
            if shape is None or batch_ref is None:
                raise ValueError(
                    "if init is None, memory at least need shape and batch_ref")
            parent_block = self.parent_block()
            var_name = unique_name.generate("@".join(
                [self.helper.name, "memory_boot"]))
            boot_var = parent_block.create_var(
                name=var_name,
                shape=shape,
                dtype=batch_ref.dtype,
                persistable=False)

            parent_block.append_op(
                type="fill_constant_batch_size_like",
                inputs={'Input': [batch_ref]},
                outputs={'Out': [boot_var]},
                attrs={
                    'value': init_value,
                    'shape': boot_var.shape,
                    'dtype': boot_var.dtype,
                    'input_dim_idx': ref_batch_dim_idx,
                    'output_dim_idx': init_batch_dim_idx
                })

            return self.memory(init=boot_var)
        else:
            pre_mem = self.helper.create_variable(
                name=unique_name.generate("@".join([self.helper.name, "mem"])),
                dtype=init.dtype,
                shape=init.shape)
            self.memories[pre_mem.name] = StaticRNNMemoryLink(
                init=init, pre_mem=pre_mem)
            return pre_mem

    def step_input(self, x):
        self._assert_in_rnn_block_('step_input')
        if not isinstance(x, Variable):
            raise TypeError("step input takes a Variable")
        if self.seq_len is None:
            self.seq_len = x.shape[0]
        elif self.seq_len != x.shape[0]:
            raise ValueError("Static RNN only take fix seq_len input")

        ipt = self.helper.create_variable(
            name=x.name, dtype=x.dtype, shape=list(x.shape[1:]), type=x.type)
        self.inputs.append(ipt)
        return ipt

    def step_output(self, o):
        self._assert_in_rnn_block_('step_output')
        if not isinstance(o, Variable):
            raise TypeError("step output takes a Variable")

        tmp_o = self.helper.create_tmp_variable(dtype=o.dtype)
        self.helper.append_op(
            type='rnn_memory_helper',
            inputs={'X': [o]},
            outputs={'Out': tmp_o},
            attrs={'dtype': o.dtype})

        out_var = self.parent_block().create_var(
            name=tmp_o.name,
            shape=[self.seq_len] + list(tmp_o.shape),
            dtype=tmp_o.dtype)

        self.outputs.append(out_var)

    def output(self, *outputs):
        for each in outputs:
            self.step_output(each)

    def update_memory(self, mem, var):
        if not isinstance(mem, Variable) or not isinstance(var, Variable):
            raise TypeError("update memory should take variables")
        self.memories[mem.name].mem = var

    def parent_block(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)
        return parent_block

    def __call__(self, *args, **kwargs):
        if self.status != StaticRNN.AFTER_RNN_BLOCK:
            raise ValueError("RNN output can only be retrieved after rnn block")
        if len(self.outputs) == 0:
            raise ValueError("RNN has no output")
        elif len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def complete_op(self):
        main_program = self.helper.main_program
        rnn_block = main_program.current_block()
        parent_block = self.parent_block()

        local_inputs = set()

        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    local_inputs.add(out_var_name)

        for var in self.inputs:
            local_inputs.add(var.name)
        for m in self.memories:
            local_inputs.add(m)

        params = list()
        for op in rnn_block.ops:
            assert isinstance(op, Operator)
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in local_inputs:
                        params.append(in_var_name)

        parameters = [parent_block.var(name) for name in params]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        inlinks = [parent_block.var(i.name) for i in self.inputs]
        outlinks = self.outputs

        boot_memories = []
        pre_memories = []
        memories = []
        for _, mem in self.memories.iteritems():
            boot_memories.append(mem.init)
            pre_memories.append(mem.pre_mem.name)
            mem_var = rnn_block.var(mem.mem.name)
            assert isinstance(mem_var, Variable)
            new_mem = self.helper.create_tmp_variable(dtype=mem_var.dtype)

            rnn_block.append_op(
                type='rnn_memory_helper',
                inputs={'X': [mem_var]},
                outputs={'Out': [new_mem]},
                attrs={'dtype': mem_var.dtype})

            memories.append(new_mem.name)

        parent_block.append_op(
            type='recurrent',
            inputs={
                'inputs': inlinks,
                'initial_states': boot_memories,
                'parameters': parameters
            },
            outputs={'outputs': outlinks,
                     'step_scopes': [step_scope]},
            attrs={
                'ex_states': pre_memories,
                'states': memories,
                'sub_block': rnn_block
            })


class WhileGuard(BlockGuard):
    def __init__(self, while_op):
        if not isinstance(while_op, While):
            raise TypeError("WhileGuard takes a while op")
        super(WhileGuard, self).__init__(while_op.helper.main_program)
        self.while_op = while_op

    def __enter__(self):
        self.while_op.status = While.IN_WHILE_BLOCK
        return super(WhileGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            return False
        self.while_op.status = While.AFTER_WHILE_BLOCK
        self.while_op.complete()
        return super(WhileGuard, self).__exit__(exc_type, exc_val, exc_tb)


class While(object):
    BEFORE_WHILE_BLOCK = 0
    IN_WHILE_BLOCK = 1
    AFTER_WHILE_BLOCK = 2

    def __init__(self, cond, name=None):
        self.helper = LayerHelper("while", name=name)
        self.status = While.BEFORE_WHILE_BLOCK
        if not isinstance(cond, Variable):
            raise TypeError("condition should be a variable")
        assert isinstance(cond, Variable)
        if cond.dtype != core.VarDesc.VarType.BOOL:
            raise TypeError("condition should be a bool variable")
        if reduce(lambda a, b: a * b, cond.shape, 1) != 1:
            raise TypeError("condition should be a bool scalar")
        self.cond_var = cond

    def block(self):
        return WhileGuard(self)

    def complete(self):
        main_program = self.helper.main_program
        while_block = main_program.current_block()
        parent_block = main_program.block(main_program.current_block()
                                          .parent_idx)

        inner_outputs = {self.cond_var.name}
        x_name_list = set()
        for op in while_block.ops:
            for iname in op.input_names:
                for in_var_name in op.input(iname):
                    if in_var_name not in inner_outputs:
                        x_name_list.add(in_var_name)

            for oname in op.output_names:
                for out_var_name in op.output(oname):
                    inner_outputs.add(out_var_name)

        out_vars = []
        for inner_out_name in inner_outputs:
            if inner_out_name in parent_block.vars:
                out_vars.append(parent_block.var(inner_out_name))

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        parent_block.append_op(
            type='while',
            inputs={
                'X':
                [parent_block.var_recursive(x_name) for x_name in x_name_list],
                'Condition': [self.cond_var]
            },
            outputs={'Out': out_vars,
                     'StepScopes': [step_scope]},
            attrs={'sub_block': while_block})


def lod_rank_table(x, level=0):
    """LoD Rank Table Operator. Given an input variable **x** and a level number
    of LoD, this layer creates a LodRankTable object. A LoDRankTable object
    contains a list of bi-element tuples. Each tuple consists of an index and
    a length, both of which are int type. Refering to specified level of LoD,
    the index is the sequence index number and the length representes the
    sequence length. Please note that the list is ranked in descending order by
    the length. The following is an example:

        .. code-block:: text

            x is a LoDTensor:
                x.lod = [[0,                2, 3],
                         [0,             5, 6, 7]]
                x.data = [a, b, c, d, e, f, g]

            1. set level to 0:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=0)

                Get:
                    lod_rank_table_obj.items() = [(0, 2), (1, 1)]

            2. set level to 1:
                Create lod rank table:
                    lod_rank_table_obj = lod_rank_table(x, level=1)

                Get:
                    lod_rank_table_obj.items() = [(0, 5), (1, 1), (2, 1)]

    Args:
        x (Variable): Input variable, a LoDTensor based which to create the lod
            rank table.
        level (int): Specify the LoD level, on which to create the lod rank
            table.

    Returns:
        Variable: The created LoDRankTable object.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[10],
                            dtype='float32', lod_level=1)
            out = layers.lod_rank_table(x=x, level=0)
    """
    helper = LayerHelper("lod_rank_table", **locals())
    table = helper.create_variable(
        type=core.VarDesc.VarType.LOD_RANK_TABLE,
        name=unique_name.generate("lod_rank_table"))
    helper.append_op(
        type='lod_rank_table',
        inputs={'X': x},
        outputs={'Out': table},
        attrs={'level': level})
    return table


def max_sequence_len(rank_table):
    """Max Sequence Len Operator. Given a LoDRankTable object, this layer
    returns the max length of a batch of sequences. In fact, a LoDRankTable
    object contains a list of tuples(<sequence index, sequence length>) and
    the list is already sorted by sequence length in descending order, so the
    operator just returns the sequence length of the first tuple element.

    Args:
        rank_table (Variable): Input variable which is a LoDRankTable object.

    Returns:
        Variable: The max length of sequence.

    Examples:
        .. code-block:: python

            x = fluid.layers.data(name='x', shape=[10],
                            dtype='float32', lod_level=1)
            rank_table = layers.lod_rank_table(x=x, level=0)
            max_seq_len = layers.max_sequence_len(rank_table)
    """
    helper = LayerHelper("max_seqence_len", **locals())
    res = helper.create_tmp_variable(dtype="int64")
    helper.append_op(
        type="max_sequence_len",
        inputs={"RankTable": rank_table},
        outputs={"Out": res})
    return res


def topk(input, k):
    """
    **topk**

    This function performs the operation that selects the k entries in the input
    vector and outputs their values and indices as vectors. Thus topk_out[j] is
    the j-th largest entry in input, and its index is topk_indices[j]

    Args:
        input (Variable|list): The input tensor that has all the data.
        k (int): The number of top elements that the function will pick.

    Returns:
        Variable: The variable of type array that contains the k largest entries
                  from input.
        Variable: The variable of type array that contains the indices of k
                  largest entries from input.

    Examples:
        .. code-block:: python

          x = fluid.layers.data(name='x', shape=[10])
          k = 5
          array = fluid.layers.topk(x, k)
    """
    helper = LayerHelper('topk', **locals())
    topk_out = helper.create_tmp_variable(dtype=input.dtype)
    topk_indices = helper.create_tmp_variable(dtype='int64')
    helper.append_op(
        type='top_k',
        inputs={'X': [input]},
        outputs={'Out': [topk_out],
                 'Indices': [topk_indices]},
        attrs={'k': k})
    return topk_out, topk_indices


def lod_tensor_to_array(x, table):
    """ Convert a LOD_TENSOR to an LOD_TENSOR_ARRAY.

    Args:
        x (Variable|list): The LOD tensor to be converted to a LOD tensor array.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order.

    Returns:
        Variable: The variable of type array that has been converted from a
                  tensor.

    Examples:
        .. code-block:: python

          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
    """
    helper = LayerHelper("lod_tensor_to_array", **locals())
    array = helper.create_variable(
        name=unique_name.generate("lod_tensor_to_array"),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=x.dtype)
    helper.append_op(
        type='lod_tensor_to_array',
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': array})
    return array


def array_to_lod_tensor(x, table):
    """Convert a LoD_Tensor_Aarry to an LoDTensor.

    Args:
        x (Variable|list): The lod tensor array to be converted to a tensor.
        table (ParamAttr|list): The variable that stores the level of lod
                                which is ordered by sequence length in
                                descending order.

    Returns:
        Variable: The variable of type tensor that has been converted
                  from an array.

    Examples:
        .. code-block:: python

          x = fluid.layers.data(name='x', shape=[10])
          table = fluid.layers.lod_rank_table(x, level=0)
          array = fluid.layers.lod_tensor_to_array(x, table)
          lod_tensor = fluid.layers.array_to_lod_tensor(array, table)
    """
    helper = LayerHelper("array_to_lod_tensor", **locals())
    tmp = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type="array_to_lod_tensor",
        inputs={'X': x,
                'RankTable': table},
        outputs={'Out': tmp})
    return tmp


def increment(x, value=1.0, in_place=True):
    """
    This function performs an operation that increments each value in the
    input :math:`x` by an amount: :math:`value` as mentioned in the input
    parameter. This operation is performed in-place by default.

    Args:
        x (Variable|list): The tensor that has the input values.
        value (float): The amount by which the values should be incremented.
        in_place (bool): If the increment should be performed in-place.

    Returns:
        Variable: The tensor variable storing the transformation of
                  element-wise increment of each value in the input.

    Examples:
        .. code-block:: python

          data = fluid.layers.data(name='data', shape=[32, 32], dtype='float32')
          data = fluid.layers.increment(x=data, value=3.0, in_place=True)
    """
    helper = LayerHelper("increment", **locals())
    if not in_place:
        out = helper.create_tmp_variable(dtype=x.dtype)
    else:
        out = x
    helper.append_op(
        type='increment',
        inputs={'X': [x]},
        outputs={'Out': [out]},
        attrs={'step': float(value)})
    return out


def array_write(x, i, array=None):
    """
    This function writes the given input variable to the specified position
    indicating by the arrary index to an output LOD_TENSOR_ARRAY. If the
    output LOD_TENSOR_ARRAY is not given(None), a new one will be created and
    returned.

    Args:
        x (Variable|list): The input tensor from which the data will be read.
        i (Variable|list): The index of the output LOD_TENSOR_ARRAY, pointing to
                           the position to which the input tensor will be
                           written.
        array (Variable|list): The output LOD_TENSOR_ARRAY to which the input
                               tensor will be written. If this parameter is
                               NONE, a new LOD_TENSOR_ARRAY will be created and
                               returned.

    Returns:
        Variable: The output LOD_TENSOR_ARRAY where the input tensor is written.

    Examples:
        .. code-block::python

          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          arr = layers.array_write(tmp, i=i)
    """
    helper = LayerHelper('array_write', **locals())
    if array is None:
        array = helper.create_variable(
            name="{0}.out".format(helper.name),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
    helper.append_op(
        type='write_to_array',
        inputs={'X': [x],
                'I': [i]},
        outputs={'Out': [array]})
    return array


def create_array(dtype):
    """This function creates an array of type :math:`LOD_TENSOR_ARRAY` using the
    LayerHelper.

    Args:
        dtype (int|float): The data type of the elements in the array.

    Returns:
        Variable: The tensor variable storing the elements of data type.

    Examples:
        .. code-block:: python

          data = fluid.layers.create_array(dtype='float32')

    """
    helper = LayerHelper("array", **locals())
    return helper.create_variable(
        name="{0}.out".format(helper.name),
        type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
        dtype=dtype)


def less_than(x, y, cond=None, **ignored):
    """
    **Less than**

    This layer returns the truth value of :math:`x < y` elementwise.

    Args:
        x(Variable): First operand of *less_than*
        y(Variable): Second operand of *less_than*
        cond(Variable|None): Optional output variable to store the result of *less_than*

    Returns:
        Variable: The tensor variable storing the output of *less_than*.

    Examples:
        .. code-block:: python

          less = fluid.layers.less_than(x=label, y=limit)
    """
    helper = LayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_tmp_variable(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='less_than', inputs={'X': [x],
                                  'Y': [y]}, outputs={'Out': [cond]})
    return cond


def equal(x, y, cond=None, **ignored):
    """
    **equal**

    This layer returns the truth value of :math:`x == y` elementwise.

    Args:
        x(Variable): First operand of *equal*
        y(Variable): Second operand of *equal*
        cond(Variable|None): Optional output variable to store the result of *equal*

    Returns:
        Variable: The tensor variable storing the output of *equal*.

    Examples:
        .. code-block:: python

          less = fluid.layers.equal(x=label, y=limit)
    """
    helper = LayerHelper("equal", **locals())
    if cond is None:
        cond = helper.create_tmp_variable(dtype='bool')
        cond.stop_gradient = True

    helper.append_op(
        type='equal', inputs={'X': [x],
                              'Y': [y]}, outputs={'Out': [cond]})
    return cond


def array_read(array, i):
    """This function performs the operation to read the data in as an
    LOD_TENSOR_ARRAY.
    Args:
        array (Variable|list): The input tensor that will be written to an array.
        i (Variable|list): The subscript index in tensor array, that points the
                           place where data will be written to.
    Returns:
        Variable: The tensor type variable that has the data written to it.
    Examples:
        .. code-block::python
          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          arr = layers.array_read(tmp, i=i)
    """
    helper = LayerHelper('array_read', **locals())
    if not isinstance(
            array,
            Variable) or array.type != core.VarDesc.VarType.LOD_TENSOR_ARRAY:
        raise TypeError("array should be tensor array vairable")
    out = helper.create_tmp_variable(dtype=array.dtype)
    helper.append_op(
        type='read_from_array',
        inputs={'X': [array],
                'I': [i]},
        outputs={'Out': [out]})
    return out


def shrink_memory(x, i, table):
    """
    This function creates an operator to shrink_rnn_memory using the RankTable
    as mentioned in the input parameter.
    """
    helper = LayerHelper('shrink_memory', **locals())
    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='shrink_rnn_memory',
        inputs={'X': [x],
                'I': [i],
                'RankTable': [table]},
        outputs={'Out': [out]},
        attrs={})
    return out


def array_length(array):
    """This function performs the operation to find the length of the input
    LOD_TENSOR_ARRAY.

    Args:
        array (LOD_TENSOR_ARRAY): The input array that will be used
                                  to compute the length.

    Returns:
        Variable: The length of the input LoDTensorArray.

    Examples:
        .. code-block::python

          tmp = fluid.layers.zeros(shape=[10], dtype='int32')
          i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=10)
          arr = fluid.layers.array_write(tmp, i=i)
          arr_len = fluid.layers.array_length(arr)
    """
    helper = LayerHelper('array_length', **locals())
    tmp = helper.create_tmp_variable(dtype='int64')
    tmp.stop_gradient = True
    helper.append_op(
        type='lod_array_length', inputs={'X': [array]}, outputs={'Out': [tmp]})
    return tmp


class ConditionalBlockGuard(BlockGuard):
    def __init__(self, block):
        if not isinstance(block, ConditionalBlock):
            raise TypeError("block should be conditional block")
        super(ConditionalBlockGuard, self).__init__(block.helper.main_program)
        self.block = block

    def __enter__(self):
        return super(ConditionalBlockGuard, self).__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.block.complete()
        return super(ConditionalBlockGuard, self).__exit__(exc_type, exc_val,
                                                           exc_tb)


class ConditionalBlock(object):
    def __init__(self, inputs, is_scalar_condition=False, name=None):
        for each_input in inputs:
            if not isinstance(each_input, Variable):
                raise TypeError("Each input should be variable")
        self.inputs = inputs
        self.is_scalar_condition = is_scalar_condition
        self.helper = LayerHelper('conditional_block', name=name)

    def block(self):
        return ConditionalBlockGuard(self)

    def complete(self):
        inside_block = self.helper.main_program.current_block()
        parent_block = self.helper.main_program.block(inside_block.parent_idx)

        intermediate = set()
        params = set()

        for each_op in inside_block.ops:
            assert isinstance(each_op, Operator)
            for iname in each_op.input_names:
                for in_var_name in each_op.input(iname):
                    if in_var_name not in intermediate:
                        params.add(in_var_name)

            for oname in each_op.output_names:
                for out_var_name in each_op.output(oname):
                    intermediate.add(out_var_name)
        input_set = set([ipt.name for ipt in self.inputs])

        param_list = [
            parent_block.var(each_name) for each_name in params
            if each_name not in input_set
        ]

        out_list = [
            parent_block.var(var_name) for var_name in parent_block.vars
            if var_name in intermediate
        ]

        step_scope = parent_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)
        parent_block.append_op(
            type='conditional_block',
            inputs={
                'X': self.inputs,
                'Params': param_list,
            },
            outputs={'Out': out_list,
                     'Scope': [step_scope]},
            attrs={
                'sub_block': inside_block,
                'is_scalar_condition': self.is_scalar_condition
            })


class Switch(object):
    def __init__(self, name=None):
        self.helper = LayerHelper('switch', name=name)
        self.inside_scope = False
        self.pre_not_conditions = []

    def case(self, condition):
        """create a new block for this condition
        """
        if not self.inside_scope:
            raise ValueError("case should be called inside with")

        if len(self.pre_not_conditions) == 0:
            cond_block = ConditionalBlock([condition], is_scalar_condition=True)
            not_cond = logical_not(x=condition)
            self.pre_not_conditions.append(not_cond)
        else:
            pre_cond_num = len(self.pre_not_conditions)
            pre_not_cond = self.pre_not_conditions[pre_cond_num - 1]
            new_not_cond = logical_and(
                x=pre_not_cond, y=logical_not(x=condition))
            self.pre_not_conditions.append(new_not_cond)
            cond_block = ConditionalBlock(
                [logical_and(
                    x=pre_not_cond, y=condition)],
                is_scalar_condition=True)

        return ConditionalBlockGuard(cond_block)

    def default(self):
        """create a default case for this switch
        """
        pre_cond_num = len(self.pre_not_conditions)
        if pre_cond_num == 0:
            raise ValueError("there should be at least one condition")
        cond_block = ConditionalBlock(
            [self.pre_not_conditions[pre_cond_num - 1]],
            is_scalar_condition=True)
        return ConditionalBlockGuard(cond_block)

    def __enter__(self):
        """
        set flag that now is inside switch.block {}
        :return:
        """
        self.inside_scope = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.inside_scope = False
        if exc_type is not None:
            return False  # re-raise exception

        return True


class SelectCase(object):
    def __init__(self, case_idx, case_to_execute, channel_action_fn=None,
                 channel=None, value=None):
        self.helper = LayerHelper('conditional_block')
        self.main_program = self.helper.main_program
        self.is_scalar_condition = True

        self.case_to_execute = case_to_execute
        self.idx = case_idx

        # Since we aren't going to use the `channel_send` or `channel_recv`
        # functions directly, we just need to capture the name.
        self.action = channel_action_fn.__name__ if channel_action_fn else None
        self.value = value
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
            fill_constant(shape=[1], dtype=core.VarDesc.VarType.LOD_TENSOR,
                          value=self.idx),
            self.case_to_execute
        )

        step_scope = cases_block.create_var(
            type=core.VarDesc.VarType.STEP_SCOPES)

        cases_block.append_op(
            type='conditional_block',
            inputs={
                'X': [should_execute_block],
                'Params': param_list
            },
            outputs={
                'Out': out_vars,
                'Scope': [step_scope]
            },
            attrs={
                'sub_block': self.block,
                'is_scalar_condition': self.is_scalar_condition,
                'case_idx': self.idx,
                'case_action': self.action,
                'case_channel': self.channel.name if self.channel else None,
                'case_channel_value': self.value.name if self.value else None
            })

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.main_program.rollback()
        if exc_type is not None:
            return False  # re-raise exception
        return True


class Select(BlockGuard):
    def __init__(self, name=None):
        self.helper = LayerHelper('select', name=name)
        self.cases = []

        super(Select, self).__init__(self.helper.main_program)

        self.case_to_execute = self.helper.main_program.current_block().create_var(
            type=core.VarDesc.VarType.LOD_TENSOR)

    def __enter__(self):
        super(Select, self).__enter__()
        return self

    def case(self, channel_action_fn, channel, value):
        """Create a new block for this condition.
        """
        select_case = SelectCase(
            len(self.cases), self.case_to_execute, channel_action_fn, channel, value)

        self.cases.append(select_case)

        return select_case

    def default(self):
        """Create a default case block for this condition.
        """
        default_case = SelectCase(len(self.cases), self.case_to_execute)

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
        for case in self.cases:
            case.construct_op()

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

        # TODO(varunarora): Figure out if defining output is needed.
        out_list = [
            parent_block.var(var_name) for var_name in parent_block.vars
            if var_name in intermediate
        ]

        X = [select_block.var_recursive(x_name) for x_name in params]

        # Needs to be used by `equal` inside the cases block.
        X.append(self.case_to_execute)

        # Construct the select op.
        select_block.append_op(
            type="select",
            inputs={
                'X': X
            },
            attrs={
                'case_to_execute': self.case_to_execute.name,
                'sub_block': select_block
            },
            outputs={}
        )

        return super(Select, self).__exit__(exc_type, exc_val, exc_tb)


class IfElseBlockGuard(object):
    def __init__(self, is_true, ifelse):
        if not isinstance(ifelse, IfElse):
            raise TypeError("ifelse must be an instance of IfElse class")

        if ifelse.status != IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("You cannot invoke IfElse.block() inside a block")

        self.is_true = is_true
        self.ie = ifelse
        if is_true:
            self.cond_block = ifelse.conditional_true_block
        else:
            self.cond_block = ifelse.conditional_false_block

        if not isinstance(self.cond_block, ConditionalBlock):
            raise TypeError("Unexpected situation")

        self.cond_block = self.cond_block.block()

    def __enter__(self):
        self.ie.status = IfElse.IN_IF_ELSE_TRUE_BLOCKS if self.is_true else IfElse.IN_IF_ELSE_FALSE_BLOCKS
        self.cond_block.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cond_block.__exit__(exc_type, exc_val, exc_tb):
            # re-raise inside exception
            return False
        if len(self.ie.output_table[1 if self.is_true else 0]) == 0:
            raise ValueError("Must set output inside block")
        self.ie.status = IfElse.OUT_IF_ELSE_BLOCKS


class IfElse(object):
    OUT_IF_ELSE_BLOCKS = 0
    IN_IF_ELSE_TRUE_BLOCKS = 1
    IN_IF_ELSE_FALSE_BLOCKS = 2

    def __init__(self, cond, name=None):
        if not isinstance(cond, Variable):
            raise TypeError("cond must be a Variable")
        self.helper = LayerHelper('ifelse', name=name)
        self.cond = cond
        self.input_table = {}
        self.status = IfElse.OUT_IF_ELSE_BLOCKS
        self.conditional_true_block = ConditionalBlock(inputs=[self.cond])
        self.conditional_false_block = ConditionalBlock(inputs=[self.cond])
        self.output_table = ([], [])  # (true_outs, false_outs)

    def input(self, x):
        if self.status == IfElse.OUT_IF_ELSE_BLOCKS:
            raise ValueError("input must in true/false blocks")
        if id(x) not in self.input_table:
            parent_block = self.parent_block()
            out_true = parent_block.create_var(
                name=unique_name.generate('ifelse_input' + self.helper.name),
                dtype=x.dtype)

            out_false = parent_block.create_var(
                name=unique_name.generate('ifelse_input' + self.helper.name),
                dtype=x.dtype)
            parent_block.append_op(
                type='split_lod_tensor',
                inputs={
                    'X': x,
                    'Mask': self.cond,
                },
                outputs={'OutTrue': out_true,
                         'OutFalse': out_false},
                attrs={'level': 0})
            self.input_table[id(x)] = (out_true, out_false)
        else:
            out_true, out_false = self.input_table[id(x)]

        if self.status == IfElse.IN_IF_ELSE_TRUE_BLOCKS:
            return out_true
        else:
            return out_false

    def parent_block(self):
        current_block = self.helper.main_program.current_block()
        return self.helper.main_program.block(current_block.parent_idx)

    def true_block(self):
        return IfElseBlockGuard(True, self)

    def false_block(self):
        return IfElseBlockGuard(False, self)

    def output(self, *outs):
        if self.status == self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("output can only be invoked in the sub-block")

        out_table = self.output_table[1 if self.status ==
                                      self.IN_IF_ELSE_TRUE_BLOCKS else 0]
        parent_block = self.parent_block()
        for each_out in outs:
            if not isinstance(each_out, Variable):
                raise TypeError("Each output should be a variable")
            # create outside tensor
            outside_out = parent_block.create_var(
                name=unique_name.generate("_".join(
                    [self.helper.name, 'output'])),
                dtype=each_out.dtype)
            out_table.append(outside_out)

            # assign local var to outside
            assign(input=each_out, output=outside_out)

    def __call__(self):
        if self.status != self.OUT_IF_ELSE_BLOCKS:
            raise ValueError("IfElse::__call__ must be out of sub-block")
        false_len, true_len = map(len, self.output_table)
        if false_len == 0 and true_len == 0:
            raise ValueError("Must invoke true_block/false_block before "
                             "__call__")
        elif false_len != true_len and false_len != 0 and true_len != 0:
            raise ValueError("The output side must be same")
        elif false_len == 0 or true_len == 0:
            return self.output_table[0 if false_len != 0 else 1]

        # else none of false_len/true_len is zero
        # merge together
        rlist = []
        for false_var, true_var in zip(*self.output_table):
            rlist.append(
                merge_lod_tensor(
                    in_true=true_var,
                    in_false=false_var,
                    mask=self.cond,
                    x=self.cond,
                    level=0))
        return rlist


class DynamicRNN(object):
    BEFORE_RNN = 0
    IN_RNN = 1
    AFTER_RNN = 2

    def __init__(self, name=None):
        self.helper = LayerHelper('dynamic_rnn', name=name)
        self.status = DynamicRNN.BEFORE_RNN
        self.lod_rank_table = None
        self.max_seq_len = None
        self.step_idx = None
        self.zero_idx = fill_constant(
            shape=[1], value=0, dtype='int64', force_cpu=True)
        self.mem_dict = dict()
        self.output_array = []
        self.outputs = []
        self.cond = self.helper.create_tmp_variable(dtype='bool')
        self.cond.stop_gradient = False
        self.while_op = While(self.cond)
        self.input_array = []
        self.mem_link = []

    def step_input(self, x):
        self._assert_in_rnn_block_("step_input")
        if not isinstance(x, Variable):
            raise TypeError(
                "step_input() can only take a Variable as its input.")
        parent_block = self._parent_block_()
        if self.lod_rank_table is None:
            self.lod_rank_table = parent_block.create_var(
                name=unique_name.generate('lod_rank_table'),
                type=core.VarDesc.VarType.LOD_RANK_TABLE)
            self.lod_rank_table.stop_gradient = True
            parent_block.append_op(
                type='lod_rank_table',
                inputs={"X": x},
                outputs={"Out": self.lod_rank_table})
            self.max_seq_len = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_max_seq_len'),
                dtype='int64')
            self.max_seq_len.stop_gradient = False
            parent_block.append_op(
                type='max_sequence_len',
                inputs={'RankTable': self.lod_rank_table},
                outputs={"Out": self.max_seq_len})
            self.cond.stop_gradient = True
            parent_block.append_op(
                type='less_than',
                inputs={'X': self.step_idx,
                        'Y': self.max_seq_len},
                outputs={'Out': self.cond})

        input_array = parent_block.create_var(
            name=unique_name.generate('dynamic_rnn_input_array'),
            type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
            dtype=x.dtype)
        self.input_array.append((input_array, x.dtype))
        parent_block.append_op(
            type='lod_tensor_to_array',
            inputs={'X': x,
                    'RankTable': self.lod_rank_table},
            outputs={'Out': input_array})
        return array_read(array=input_array, i=self.step_idx)

    def static_input(self, x):
        self._assert_in_rnn_block_("static_input")
        if not isinstance(x, Variable):
            raise TypeError(
                "static_input() can only take a Variable as its input")
        if self.lod_rank_table is None:
            raise RuntimeError(
                "static_input() must be called after step_input().")
        parent_block = self._parent_block_()
        x_reordered = parent_block.create_var(
            name=unique_name.generate("dynamic_rnn_static_input_reordered"),
            type=core.VarDesc.VarType.LOD_TENSOR,
            dtype=x.dtype)
        parent_block.append_op(
            type='reorder_lod_tensor_by_rank',
            inputs={'X': [x],
                    'RankTable': [self.lod_rank_table]},
            outputs={'Out': [x_reordered]})
        return shrink_memory(x_reordered, self.step_idx, self.lod_rank_table)

    @contextlib.contextmanager
    def block(self):
        if self.status != DynamicRNN.BEFORE_RNN:
            raise ValueError("rnn.block() can only be invoke once")
        self.step_idx = fill_constant(
            shape=[1], dtype='int64', value=0, force_cpu=True)
        self.step_idx.stop_gradient = False
        self.status = DynamicRNN.IN_RNN
        with self.while_op.block():
            yield
            increment(x=self.step_idx, value=1.0, in_place=True)

            for new_mem, mem_array in self.mem_link:
                array_write(x=new_mem, i=self.step_idx, array=mem_array)

            less_than(x=self.step_idx, y=self.max_seq_len, cond=self.cond)

        self.status = DynamicRNN.AFTER_RNN
        for each_array in self.output_array:
            self.outputs.append(
                array_to_lod_tensor(
                    x=each_array, table=self.lod_rank_table))

    def __call__(self, *args, **kwargs):
        if self.status != DynamicRNN.AFTER_RNN:
            raise ValueError(("Output of the dynamic RNN can only be visited "
                              "outside the rnn block."))
        if len(self.outputs) == 1:
            return self.outputs[0]
        else:
            return self.outputs

    def memory(self,
               init=None,
               shape=None,
               value=0.0,
               need_reorder=False,
               dtype='float32'):
        self._assert_in_rnn_block_('memory')
        if init is not None:
            if not isinstance(init, Variable):
                raise TypeError(
                    "The input arg `init` of memory() must be a Variable")
            parent_block = self._parent_block_()
            init_tensor = init
            if need_reorder == True:
                if self.lod_rank_table is None:
                    raise ValueError(
                        'If set need_reorder to True, make sure step_input be '
                        'invoked before '
                        'memory(init=init, need_reordered=True, ...).')
                init_reordered = parent_block.create_var(
                    name=unique_name.generate('dynamic_rnn_mem_init_reordered'),
                    type=core.VarDesc.VarType.LOD_TENSOR,
                    dtype=init.dtype)
                parent_block.append_op(
                    type='reorder_lod_tensor_by_rank',
                    inputs={
                        'X': [init_tensor],
                        'RankTable': [self.lod_rank_table]
                    },
                    outputs={'Out': [init_reordered]})
                init_tensor = init_reordered
            mem_array = parent_block.create_var(
                name=unique_name.generate('dynamic_rnn_mem_array'),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=init.dtype)
            parent_block.append_op(
                type='write_to_array',
                inputs={'X': init_tensor,
                        'I': self.zero_idx},
                outputs={'Out': mem_array})
            retv = array_read(array=mem_array, i=self.step_idx)
            retv = shrink_memory(
                x=retv, i=self.step_idx, table=self.lod_rank_table)
            self.mem_dict[retv.name] = mem_array
            return retv
        else:
            if len(self.input_array) == 0:
                raise ValueError(
                    "step_input should be invoked before memory(shape=..., value=...)"
                )
            parent_block = self._parent_block_()
            init = parent_block.create_var(
                name=unique_name.generate('mem_init'), dtype=dtype)
            arr, dtype = self.input_array[0]
            in0 = parent_block.create_var(
                name=unique_name.generate('in0'), dtype=dtype)
            parent_block.append_op(
                type='read_from_array',
                inputs={'X': [arr],
                        'I': [self.zero_idx]},
                outputs={'Out': [in0]})
            parent_block.append_op(
                type='fill_constant_batch_size_like',
                inputs={'Input': [in0]},
                outputs={'Out': [init]},
                attrs={
                    'shape': [-1] + shape,
                    'value': float(value),
                    'dtype': init.dtype
                })
            return self.memory(init=init)

    def update_memory(self, ex_mem, new_mem):
        self._assert_in_rnn_block_('update_memory')
        if not isinstance(ex_mem, Variable):
            raise TypeError("The input arg `ex_mem` of update_memory() must "
                            "be a Variable")
        if not isinstance(new_mem, Variable):
            raise TypeError("The input arg `new_mem` of update_memory() must "
                            "be a Variable")

        mem_array = self.mem_dict.get(ex_mem.name, None)
        if mem_array is None:
            raise ValueError("Please invoke memory before update_memory")
        if self.lod_rank_table is None:
            raise ValueError("Please invoke step_input before update_memory")

        self.mem_link.append((new_mem, mem_array))

    def output(self, *outputs):
        self._assert_in_rnn_block_('output')
        parent_block = self._parent_block_()
        for each in outputs:
            outside_array = parent_block.create_var(
                name=unique_name.generate("_".join(
                    [self.helper.name, "output_array", each.name])),
                type=core.VarDesc.VarType.LOD_TENSOR_ARRAY,
                dtype=each.dtype)
            array_write(x=each, i=self.step_idx, array=outside_array)
            self.output_array.append(outside_array)

    def _parent_block_(self):
        prog = self.helper.main_program
        parent_idx = prog.current_block().parent_idx
        assert parent_idx >= 0
        parent_block = prog.block(parent_idx)

        return parent_block

    def _assert_in_rnn_block_(self, method):
        if self.status != DynamicRNN.IN_RNN:
            raise ValueError("{0} can only be invoked inside rnn block.".format(
                method))


@autodoc()
def reorder_lod_tensor_by_rank(x, rank_table):
    helper = LayerHelper('reorder_lod_tensor_by_rank', **locals())
    helper.is_instance('x', Variable)
    helper.is_instance('rank_table', Variable)

    out = helper.create_tmp_variable(dtype=x.dtype)
    helper.append_op(
        type='reorder_lod_tensor_by_rank',
        inputs={'X': [x],
                'RankTable': [rank_table]},
        outputs={'Out': [out]})
    return out
