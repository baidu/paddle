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

import collections
import contextlib
import re

import numpy as np

import proto.framework_pb2 as framework_pb2
from . import core
import unique_name

__all__ = [
    'Block',
    'Variable',
    'Program',
    'Operator',
    'default_startup_program',
    'default_main_program',
    'program_guard',
    'switch_startup_program',
    'switch_main_program',
    'get_var',
]

EMPTY_VAR_NAME = core.kEmptyVarName()
TEMP_VAR_NAME = core.kTempVarName()
GRAD_VAR_SUFFIX = core.kGradVarSuffix()
ZERO_VAR_SUFFIX = core.kZeroVarSuffix()


def grad_var_name(var_name):
    """
    return gradient name for a certain var name
    """
    return var_name + GRAD_VAR_SUFFIX


def convert_np_dtype_to_dtype_(np_dtype):
    """
    Convert the data type in numpy to the data type in Paddle
    Args:
        np_dtype(np.dtype): the data type in numpy

    Returns(core.VarDesc.VarType): the data type in Paddle

    """
    dtype = np.dtype(np_dtype)
    if dtype == np.float32:
        return core.VarDesc.VarType.FP32
    elif dtype == np.float64:
        return core.VarDesc.VarType.FP64
    elif dtype == np.float16:
        return core.VarDesc.VarType.FP16
    elif dtype == np.int32:
        return core.VarDesc.VarType.INT32
    elif dtype == np.int16:
        return core.VarDesc.VarType.INT16
    elif dtype == np.int64:
        return core.VarDesc.VarType.INT64
    elif dtype == np.bool:
        return core.VarDesc.VarType.BOOL
    else:
        raise ValueError("Not supported numpy dtype " + str(dtype))


def dtype_is_floating(dtype):
    """
    Check the data type is floating or not.
    Args:
        dtype(np.dtype|core.VarDesc.VarType): data type.
            Could be numpy format or Paddle format

    Returns(bool): True if data type is a float value

    """
    if not isinstance(dtype, core.VarDesc.VarType):
        dtype = convert_np_dtype_to_dtype_(dtype)

    return dtype in [
        core.VarDesc.VarType.FP16, core.VarDesc.VarType.FP32,
        core.VarDesc.VarType.FP64
    ]


def _debug_string_(proto, throw_on_error=True):
    """
    Get the debug string of a protobuf message. The message could be not
    initialized.
    Args:
        proto(google.protobuf.message.Message): The protobuf message
        throw_on_error(bool): True if raise an error when the protobuf message
            is not initialized.

    Returns(str): The debug string of the protobuf message

    """
    error_fields = list()
    if not proto.IsInitialized(error_fields) and throw_on_error:
        raise ValueError("{0} are not initialized.\nThe message is {1}:\n".
                         format(error_fields, proto))
    return proto.__str__()


class Variable(object):
    """
    Python variable. Every input and output of an operator is a variable. Every
    variable belongs to a block. The variable has a name and two variables in
    different blocks could have the same name.

    There are many kinds of variables. Please reference the framework.proto for
    details.

    Notes: The constructor of Variable should not be invoked directly. Please
    use `Block.create_var` to create a variable.

    >>> cur_program = Program()
    >>> cur_block = cur_program.current_block()
    >>> new_variable = cur_block.create_var(
    >>>                    name="X", shape=[-1, 23, 48], dtype='float32')

    Args:
        block(Block): The associated block. It will be passed by
            `Block.create_var` automatically.
        type(core.VarDesc.VarType): Variable type. Please reference the
            framework.proto for details.
        shape(tuple|list|None): The shape of variable. -1 means the batch size.
            Some kinds of variable do not contain shape, just set it to None.
        dtype(np.dtype|core.VarDesc.VarType|str): The data type of variable.
        lod_level(int): The level of lod tensor. 0 means it is not a time
            series data.
        capacity(int): The capacity of Channel variable. Ignored
            for other types.
        persistable(bool): True if the variable should be saved as check point.
            Defaults to False.
        stop_gradient(bool): True if the variable will stop to calculate
            gradients when backward. Defaults to False.
    """

    def __init__(self,
                 block,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 name=None,
                 shape=None,
                 dtype=None,
                 lod_level=None,
                 capacity=None,
                 persistable=None,
                 error_clip=None,
                 stop_gradient=False,
                 **kwargs):
        self.block = block
        self.error_clip = error_clip

        if name is None:
            name = unique_name.generate('_generated_var')
        is_new_var = False
        self.desc = self.block.desc.find_var(name)

        if self.desc is None:
            self.desc = self.block.desc.var(name)
            is_new_var = True

        if is_new_var:
            self.desc.set_type(type)
        elif self.desc.type() != type:
            raise ValueError("Variable {0} has been created before. The "
                             "previous type is {1}; the new type is {2}. They"
                             " are not matched".format(self.name,
                                                       self.desc.type(), type))

        if shape is not None:
            if is_new_var:
                self.desc.set_shape(shape)
            else:
                old_shape = self.shape
                shape = tuple(shape)
                if shape != old_shape:
                    raise ValueError(
                        "Variable {0} has been created before. the previous "
                        "shape is {1}; the new shape is {2}. They are not "
                        "matched.".format(self.name, old_shape, shape))
        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)
            if is_new_var:
                self.desc.set_dtype(dtype)
            else:
                old_dtype = self.dtype
                if dtype != old_dtype:
                    raise ValueError("Variable {0} has been created before. "
                                     "The previous data type is {1}; the new "
                                     "data type is {2}. They are not "
                                     "matched.".format(self.name, old_dtype,
                                                       dtype))

        if lod_level is not None:
            if is_new_var:
                self.desc.set_lod_level(lod_level)
            else:
                if lod_level != self.lod_level:
                    raise ValueError("Variable {0} has been created before. "
                                     "The previous lod_level is {1}; the new "
                                     "lod_level is {2}. They are not "
                                     "matched".format(self.name, self.lod_level,
                                                      lod_level))
        if persistable is not None:
            if is_new_var:
                self.desc.set_persistable(persistable)
            else:
                if persistable != self.persistable:
                    raise ValueError(
                        "Variable {0} has been created before."
                        "The previous persistable is {1}; the new "
                        "persistable is {2}. They are not matched".format(
                            self.name, self.persistable, persistable))

        if capacity is not None:
            if is_new_var:
                self.desc.set_capacity(capacity)
            else:
                # TODO(abhinavarora) : Compare with set capacity once,
                # get_capacity is implemented
                pass

        self.block.vars[name] = self
        self.op = None
        self.stop_gradient = stop_gradient

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        Get debug string.

        Args:
            throw_on_error(bool): True if raise an exception when self is not
                intialized.
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.VarDesc.FromString(str(protostr))
        res_str = _debug_string_(proto, throw_on_error)
        if with_details:
            additional_attr = ("error_clip", "stop_gradient")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (attr_name,
                                         str(getattr(self, attr_name)))
        return res_str

    __repr__ = __str__

    def set_desc(self, input):
        self.desc = input

    @property
    def persistable(self):
        return self.desc.persistable()

    @persistable.setter
    def persistable(self, p):
        self.desc.set_persistable(p)

    @property
    def name(self):
        return self.desc.name()

    @name.setter
    def name(self, new_name):
        self.desc.set_name(new_name)

    @property
    def shape(self):
        # convert to tuple, make it as same as numpy API.
        return tuple(self.desc.shape())

    @property
    def dtype(self):
        return self.desc.dtype()

    @property
    def lod_level(self):
        return self.desc.lod_level()

    @property
    def type(self):
        return self.desc.type()

    def set_error_clip(self, error_clip):
        self.error_clip = error_clip


def get_all_op_protos():
    """
    Get all registered op proto from PaddlePaddle C++ end.

    Returns(list): list of OpProto

    """
    protostrs = core.get_all_op_protos()
    ret_values = []
    for pbstr in protostrs:
        op_proto = framework_pb2.OpProto.FromString(str(pbstr))
        ret_values.append(op_proto)
    return ret_values


class OpProtoHolder(object):
    """
    A global variable to hold all OpProtos from C++ as a map
    """

    @classmethod
    def instance(cls):
        if not hasattr(cls, '_instance'):
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        assert not hasattr(
            self.__class__,
            '_instance'), 'Please use `instance()` to get OpProtoHolder object!'
        op_protos = get_all_op_protos()
        self.op_proto_map = {}
        for proto in op_protos:
            self.op_proto_map[proto.type] = proto

    def get_op_proto(self, type):
        """
        Get OpProto by a type string.
        Args:
            type(str): The type that operator registered in C++ side.

        Returns(framework_pb2.OpProto): The OpProto

        """
        if type not in self.op_proto_map:
            raise ValueError("Operator \"%s\" has not been registered." % type)
        return self.op_proto_map[type]


class Operator(object):
    """
    Python Operator class. The operator represents the build in instructions in a
    Block. Users can use the build in instructions to describe their neural
    network.
    """

    def __init__(self,
                 block,
                 desc,
                 type=None,
                 inputs=None,
                 outputs=None,
                 attrs=None):
        """
        Constructor.

        Notes: The constructor of operator should not be invoked directly. Use
        Block.append_op or Block.prepend_op instead.

        >>> cur_program = Program()
        >>> cur_block = cur_program.current_block()
        >>> # var1 += var2 + var3
        >>> cur_block.append_op(type="sum",
        >>>                     inputs={"X": [var1, var2, var3]},
        >>>                     outputs={"Out": [var1]})

        Args:
            block(Block): The block has the current operator.
            desc(core.OpDesc): The protobuf description.
            type(str): The type of operator.
            inputs(dict): The input dictionary. Key is the input parameter name.
                Value is a list of variables.
            outputs(dict): The output dictionary which has the same format with
                           inputs.
            attrs(dict): The attributes dictionary. Key is attribute name. Value
                is the attribute value. The attribute type should be as same as
                the type registered in C++
        """
        self.block = block
        self.desc = desc
        self.attrs = attrs
        if len(self.desc.type()) != 0:
            return
        if type is None:
            raise ValueError(
                "`type` to initilized an Operator can not be None.")
        self.desc.set_type(type)
        proto = OpProtoHolder.instance().get_op_proto(type)

        def find_name(var_list, name):
            for var_name in var_list:
                if var_list[var_name] is not None and var_name == name:
                    return True
            return False

        if inputs is not None:
            for in_proto in proto.inputs:
                found = find_name(inputs, in_proto.name)
                assert found or in_proto.dispensable, "Input {} not found".format(
                    in_proto.name)

                if found:
                    in_args = inputs[in_proto.name]
                    if not isinstance(in_args, list):
                        in_args = [in_args]
                    if not in_proto.duplicable and len(in_args) > 1:
                        raise ValueError(
                            "Input %s expects only one input, but %d are given."
                            % (in_proto.name, len(in_args)))
                    in_arg_names = []
                    for arg in in_args:
                        if isinstance(arg, basestring):
                            in_arg_names.append(arg)
                        else:
                            in_arg_names.append(arg.name)
                    self.desc.set_input(in_proto.name, in_arg_names)
                else:
                    self.desc.set_input(in_proto.name, [])

        if outputs is not None:
            given = set()
            need = set()
            for n in outputs:
                given.add(n)
            for m in proto.outputs:
                need.add(m.name)
            if not given == need:
                raise ValueError(("Incorrect setting for output(s) of "
                                  "operator \"%s\". Need: [%s] Given: [%s]") %
                                 (type, ", ".join(str(e) for e in need),
                                  ", ".join(str(e) for e in given)))

            for out_proto in proto.outputs:
                out_args = outputs[out_proto.name]
                if not isinstance(out_args, list):
                    out_args = [out_args]
                if not out_proto.duplicable and len(out_args) > 1:
                    raise ValueError(
                        "Output %s expects only one output, but %d are given." %
                        (out_proto.name, len(out_args)))
                out_arg_names = []
                for arg in out_args:
                    out_arg_names.append(arg.name)
                    arg.op = self
                self.desc.set_output(out_proto.name, out_arg_names)

        if attrs is not None:
            if not isinstance(attrs, dict):
                raise TypeError("'attrs' should be a dict.")
            for attr in proto.attrs:
                attr_name = attr.name
                if (attr_name not in attrs) or (attrs[attr_name] is None):
                    continue
                if isinstance(attrs[attr_name], Block):
                    self.desc.set_block_attr(attr_name, attrs[attr_name].desc)
                elif isinstance(attrs[attr_name], core.BlockDesc) or \
                   isinstance(attrs[attr_name], core.ProgramDesc):
                    self.desc.set_serialized_attr(
                        attr_name, attrs[attr_name].serialize_to_string())
                else:
                    self.desc.set_attr(attr_name, attrs[attr_name])

        self.desc.check_attrs()
        no_kernel_op_set = {
            'feed', 'fetch', 'save', 'load', 'recurrent', 'go',
            'rnn_memory_helper_grad', 'conditional_block', 'while', 'send',
            'recv', 'listen_and_serv', 'parallel_do', 'save_combine',
            'load_combine', 'ncclInit', 'channel_create', 'channel_close',
            'channel_send', 'channel_recv', 'select'
        }
        if type not in no_kernel_op_set:
            self.desc.infer_var_type(self.block.desc)
            self.desc.infer_shape(self.block.desc)

    def to_string(self, throw_on_error):
        """
        To debug string.
        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True

        Returns(str): The debug string.

        """
        protostr = self.desc.serialize_to_string()
        proto = framework_pb2.OpDesc.FromString(str(protostr))
        return _debug_string_(proto, throw_on_error)

    def __str__(self):
        return self.to_string(True)

    __repr__ = __str__

    @property
    def type(self):
        return self.desc.type()

    def input(self, name):
        """
        Get input arguments by the input parameter name
        Args:
            name(str): The input parameter name

        Returns(list): return the list of argument names associated with the
            specific parameter name.

        """
        return self.desc.input(name)

    def rename_input(self, old_name, new_name):
        self.desc.rename_input(old_name, new_name)

    def rename_output(self, old_name, new_name):
        self.desc.rename_output(old_name, new_name)

    @property
    def input_names(self):
        """
        Get all input parameter names
        Returns(list): return a list of input parameter names

        """
        return self.desc.input_names()

    @property
    def input_arg_names(self):
        return self.desc.input_arg_names()

    @property
    def output_arg_names(self):
        return self.desc.output_arg_names()

    def output(self, name):
        """
        Get output arguments by the output parameter name
        Args:
            name(str): The output parameter name

        Returns(list): return the list of argument names associated with the
            specific parameter name.

        """
        return self.desc.output(name)

    @property
    def output_names(self):
        """
        Get all output parameter names
        Returns(list): return a list of output parameter names

        """
        return self.desc.output_names()

    @property
    def idx(self):
        """
        Return the array index of current operator.
        Returns(int): The array index in block.ops array
        Raises:
            ValueError: when the operator is not found.
        """
        for i, op in enumerate(self.block.ops):
            if op == self:
                return i
        raise ValueError(
            "Can't find op itself in it's block. It could be a bug of Paddle.")

    def has_attr(self, name):
        """
        operator has the attribute with name or not.
        Args:
            name(str): the attribute name

        Returns(bool): True if has this attribute.

        """
        return self.desc.has_attr(name)

    def attr_type(self, name):
        """
        Get the type of attribute by attribute name
        Args:
            name(str): the attribute name

        Returns(core.AttrType): the attribute type

        """
        return self.desc.attr_type(name)

    @property
    def attr_names(self):
        """
        Get all attribute names
        Returns(list): The list of attribute name

        """
        return self.desc.attr_names()

    def attr(self, name):
        """
        Get attribute by name
        Args:
            name(str): the attribute name

        Returns(bool|int|str|float|list): The attribute value. The return value
            can be any valid attribute type.

        """
        return self.desc.attr(name)

    def block_attr(self, name):
        """
        Get the block attribute by name
        Args:
            name(str): the attribute name

        Returns(int): the block index

        """
        return self.desc.block_attr(name)


class Block(object):
    def __init__(self, program, idx):
        self.desc = program.desc.block(idx)
        self.vars = dict()  # var_name --> var
        self.ops = collections.deque()  # operator list
        self.program = program
        self.removed_vars = dict()

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.
        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            re_add_indent = re.compile(r"\n(.)")
            res_str = "blocks {\n  idx: %d\n  parent_idx: %d" % (
                self.idx, self.parent_idx)
            for var in self.vars.itervalues():
                res_str += "\n  vars {\n    %s  }" % re_add_indent.sub(
                    r"\n    \1", var.to_string(throw_on_error, with_details))
            for op in self.ops:
                res_str += "\n  ops {\n    %s  }" % re_add_indent.sub(
                    r"\n    \1", op.to_string(throw_on_error))
            res_str += "\n}"
        else:
            protostr = self.desc.serialize_to_string()
            proto = framework_pb2.BlockDesc.FromString(str(protostr))
            res_str = _debug_string_(proto, throw_on_error)
        return res_str

    __repr__ = __str__

    @property
    def parent_idx(self):
        return self.desc.parent

    @property
    def forward_block_idx(self):
        return self.desc.get_forward_block_idx()

    def set_forward_block_idx(self, idx):
        self.desc.set_forward_block_idx(idx)

    @property
    def idx(self):
        return self.desc.id

    def var(self, name):
        if not isinstance(name, basestring):
            raise TypeError()
        v = self.vars.get(name, None)
        if v is None:
            raise ValueError("var %s not in this block" % name)
        return v

    def var_recursive(self, name):
        frontier = list()
        visited = set()

        frontier.append(self)

        prog = self.program

        while len(frontier) != 0:  # BFS
            cur = frontier[0]
            frontier = frontier[1:]

            if id(cur) in visited:
                continue

            if cur.has_var(name):
                return cur.var(name)

            if cur.parent_idx != -1:
                frontier.append(prog.block(cur.parent_idx))

            if cur.forward_block_idx != -1:
                frontier.append(prog.block(cur.forward_block_idx))

            visited.add(id(cur))

        raise ValueError("Var {0} is not found recursively".format(name))

    def all_parameters(self):
        return list(self.iter_parameters())

    def iter_parameters(self):
        return (item[1] for item in self.vars.iteritems()
                if isinstance(item[1], Parameter))

    def create_var(self, *args, **kwargs):
        var = Variable(block=self, *args, **kwargs)
        if 'initializer' in kwargs:
            kwargs['initializer'](var, self)
        return var

    def has_var(self, name):
        return name in self.vars

    def rename_var(self, name, new_name):
        """
        Rename variable in vars and ops' inputs and outputs
        """
        if not self.has_var(name):
            raise ValueError("var %s is not in current" % name)
        v = self.var(name)
        if type(v) == Parameter:
            var_type = "Parameter"
            stop_gradient = v.stop_gradient
            trainable = v.trainable
            optimize_attr = v.optimize_attr
            regularizer = v.regularizer
            gradient_clip_attr = v.gradient_clip_attr
            error_clip = v.error_clip
        elif type(v) == Variable:
            var_type = "Variable"
            error_clip = v.error_clip
            stop_gradient = v.stop_gradient
        else:
            raise ValueError("unsupported var type: %s", type(v))
        orig_var_type = v.type
        self.desc.rename_var(name, new_name)
        # NOTE: v is destroyed by C++ after calling rename_var.
        d = self.desc.find_var(new_name)
        if var_type == "Parameter":
            var = Parameter(
                self,
                d.shape(),
                d.dtype(),
                type=orig_var_type,
                name=new_name,
                stop_gradient=stop_gradient,
                trainable=trainable,
                optimize_attr=optimize_attr,
                regularizer=regularizer,
                gradient_clip_attr=gradient_clip_attr,
                error_clip=error_clip)
        elif var_type == "Variable":
            var = Variable(
                self,
                type=orig_var_type,
                name=new_name,
                error_clip=error_clip,
                stop_gradient=stop_gradient)

        # rename the python side, sync_with_cpp will only add
        # new vars/ops to python side.
        self.vars[new_name] = var
        del self.vars[name]
        self.sync_with_cpp()

    def create_parameter(self, *args, **kwargs):
        global_block = self.program.global_block()
        param = Parameter(global_block, *args, **kwargs)
        if 'initializer' in kwargs:
            kwargs['initializer'](param, self)
        return param

    def append_op(self, *args, **kwargs):
        op_desc = self.desc.append_op()
        op = Operator(block=self, desc=op_desc, *args, **kwargs)
        self.ops.append(op)
        return op

    def delete_ops(self, ops):
        # remove from cpp
        # FIXME(typhoonzero): remove only the first occurrence.
        try:
            start = list(self.ops).index(ops[0])
            end = list(self.ops).index(ops[-1])
        except Exception, e:
            raise e
        self.desc.remove_op(start, end + 1)

    def slice_ops(self, start, end):
        return list(self.ops)[start:end]

    def prepend_op(self, *args, **kwargs):
        op_desc = self.desc.prepend_op()
        op = Operator(self, op_desc, *args, **kwargs)
        self.ops.appendleft(op)
        return op

    def sync_with_cpp(self):
        """
        Sync with the desc on the c++ end.

        This method is used to synchronize the c++ desc instance generated by backward.
        """
        # sync variables from cpp
        for var in self.desc.all_vars():
            if not self.has_var(var.name()):
                self.create_var(name=var.name(), desc=var, type=var.type())

        # sync operators from cpp
        ops_in_cpp = []
        for op_idx in range(0, self.desc.op_size()):
            ops_in_cpp.append(self.desc.op(op_idx))

        if len(self.ops) != 0:
            first_op_in_python = self.ops[0].desc
            last_op_in_python = self.ops[len(self.ops) - 1].desc
            start_index = None
            end_index = None
            for index in range(len(ops_in_cpp)):
                if first_op_in_python == ops_in_cpp[index]:
                    start_index = index
                if last_op_in_python == ops_in_cpp[index]:
                    end_index = index
            assert start_index is not None
            assert end_index is not None
            assert start_index <= end_index
        else:
            start_index = 0
            end_index = -1

        # sync ops append to the head of cpp_ops
        for index in range((start_index - 1 - 1), -1, -1):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.appendleft(op)

        # sync ops append to the end of cpp_ops
        for index in range((end_index + 1), len(ops_in_cpp)):
            op_desc = ops_in_cpp[index]
            op = Operator(self, op_desc)
            self.ops.append(op)

        assert len(self.ops) == len(ops_in_cpp)
        for index in range(len(self.ops)):
            assert self.ops[index].desc == ops_in_cpp[index]

    def copy_param_info_from(self, other):
        """
        Copy the information of parameters from the other block
        Args:
            other(Block): the other block

        Returns:
            None
        """
        if not isinstance(other, Block):
            raise TypeError("copy_param_info_from should be invoked with Block")
        for p in other.iter_parameters():
            assert isinstance(p, Parameter)
            v = self.vars.get(p.name, None)
            if v is None:
                raise ValueError("copy_param_info_from should be invoked with "
                                 "same topology")
            assert isinstance(v, Variable)
            new_p = Parameter(
                block=self,
                shape=v.shape,
                dtype=v.dtype,
                type=v.type,
                lod_level=v.lod_level,
                stop_gradient=p.stop_gradient,
                trainable=p.trainable,
                optimize_attr=p.optimize_attr,
                regularizer=p.regularizer,
                gradient_clip_attr=p.gradient_clip_attr,
                error_clip=p.error_clip,
                name=v.name)
            self.vars[new_p.name] = new_p


class Program(object):
    def __init__(self):
        self.desc = core.ProgramDesc()
        self.blocks = [Block(self, 0)]
        self.current_block_idx = 0
        self._seed = 0

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.
        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = ""
            for block in self.blocks:
                res_str += block.to_string(throw_on_error, with_details)
        else:
            protostr = self.desc.serialize_to_string()
            proto = framework_pb2.ProgramDesc.FromString(str(protostr))
            res_str = _debug_string_(proto, throw_on_error)
        return res_str

    def get_desc(self):
        return self.desc

    def clone(self, for_test=False):
        """Clone the Program object

        Set for_test to False when we want to clone the program for training.
        Set for_test to True when we want to clone the program for testing.         

        Args:
            for_test(bool): Some operators, such as batch_norm and drop_out ops,
                behave differently in training and testing. If for_test is True,
                the is_test attributes in these operators will be set to True for
                testing purposes, otherwise, they remain unchanged.  
                
        Returns(Program):
            The cloned Program object.
        """
        p = Program()
        if for_test:
            p.desc = core.inference_optimize(self.desc)
        else:
            p.desc = core.ProgramDesc(self.desc)
        p.blocks = [Block(p, i) for i in xrange(self.desc.num_blocks())]
        p.sync_with_cpp()
        p.copy_param_info_from(self)
        return p

    def prune(self, targets):
        if not isinstance(targets, list):
            targets = [targets]
        targets_idx = []
        for t in targets:
            if not isinstance(t, Operator):
                if isinstance(t, Variable):
                    t = t.op
                else:
                    raise ValueError(("All targets of prune() can only be "
                                      "Variable or Operator."))

            targets_idx.append([t.block.idx, t.idx])
        res = Program()
        res.desc = core.prune(self.desc, targets_idx)
        res.blocks = [Block(res, i) for i in xrange(res.desc.num_blocks())]
        res.sync_with_cpp()
        return res

    def inference_optimize(self):
        res = Program()
        res.desc = core.inference_optimize(self.desc)
        res.blocks = [Block(res, i) for i in xrange(res.desc.num_blocks())]
        res.sync_with_cpp()
        return res

    @staticmethod
    def parse_from_string(binary_str):
        p = Program()
        p.desc = core.ProgramDesc(binary_str)
        p.blocks = [Block(p, i) for i in xrange(p.desc.num_blocks())]
        p.sync_with_cpp()
        return p

    @property
    def random_seed(self):
        return self._seed

    @random_seed.setter
    def random_seed(self, seed):
        if not isinstance(seed, int):
            raise ValueError("Seed must be a integer.")
        self._seed = seed

    def __repr__(self):
        return str(self)

    def global_block(self):
        return self.blocks[0]

    def block(self, index):
        return self.blocks[index]

    def current_block(self):
        return self.blocks[self.current_block_idx]

    def append_backward(self, target, no_grad_set=None):
        """
        return map(param_name -> (grad_name, block_index, op_index))
        """
        assert isinstance(target, Variable)
        if no_grad_set is None:
            no_grad_set = set()
        try:
            param_to_grad_info = self.desc.append_backward(target.desc,
                                                           no_grad_set)
        except Exception as e:
            raise core.EnforceNotMet(
                str(e) + "\nCurrent protobuf is\n{0}".format(
                    self.to_string(False)))

        self.sync_with_cpp()
        return param_to_grad_info

    def create_block(self, parent_idx=None):
        new_block_idx = len(self.blocks)
        parent = self.current_block() if parent_idx is None else self.block(
            parent_idx)
        self.desc.append_block(parent.desc)
        self.current_block_idx = new_block_idx
        self.blocks.append(Block(self, self.current_block_idx))
        return self.current_block()

    def rollback(self):
        self.current_block_idx = self.current_block().parent_idx

    def sync_with_cpp(self):
        for block_idx in range(len(self.blocks), self.desc.num_blocks()):
            self.blocks.append(Block(self, block_idx))
        for block in self.blocks:
            block.sync_with_cpp()

    def copy_param_info_from(self, other):
        """
        Copy the information of parameters from other program.
        Args:
            other(Program): Other program

        Returns:
            None
        """
        if not isinstance(other, Program):
            raise TypeError("copy_param_info_from should be invoked with "
                            "Program")

        if len(self.blocks) != len(other.blocks):
            raise ValueError("copy_param_info_from should be invoked with two "
                             "program, with represent the same topology")
        self.global_block().copy_param_info_from(other.global_block())

    def list_vars(self):
        for each_block in self.blocks:
            for each_var in each_block.vars.itervalues():
                yield each_var


class Parameter(Variable):
    def __init__(self, block, shape, dtype, **kwargs):
        if shape is None or dtype is None:
            raise ValueError("Parameter must set shape and dtype")
        if len(shape) == 0:
            raise ValueError("Parameter shape cannot be empty")

        for each in shape:
            if each < 0:
                raise ValueError("Parameter shape should not be related with "
                                 "batch-size")

        Variable.__init__(
            self, block, persistable=True, shape=shape, dtype=dtype, **kwargs)
        self.trainable = kwargs.get('trainable', True)

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})

        self.regularizer = kwargs.get('regularizer', None)

        self.gradient_clip_attr = kwargs.get('gradient_clip_attr', None)

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.
        Args:
            throw_on_error(bool): raise exception when self is not initialized
                when throw_on_error is True
            with_details(bool): more details about variables and parameters
                (e.g. trainable, optimize_attr, ...) will be printed when with_details is True

        Returns(str): The debug string.

        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = Variable.to_string(self, throw_on_error, True)
            additional_attr = ("trainable", "optimize_attr", "regularizer",
                               "gradient_clip_attr")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (attr_name,
                                         str(getattr(self, attr_name)))
        else:
            res_str = Variable.to_string(self, throw_on_error, False)
        return res_str

    __repr__ = __str__


# program is a global instance.
_main_program_ = Program()
_startup_program_ = Program()


def default_startup_program():
    """
    Get default startup program. In startup program, Paddle will initialize
    parameters, initialize nccl handle, etc.

    Returns:
        Program: startup program
    """
    return _startup_program_


def default_main_program():
    """
    Get default main program. The main program is used for training or testing.

    Returns:
        Program: main program
    """
    return _main_program_


def switch_main_program(program):
    """
    Switch the main program to a new program.

    Args:
        program(Program): The new main program

    Returns:
        Program: The previous main program
    """
    global _main_program_
    prev_program = _main_program_
    _main_program_ = program
    return prev_program


def switch_startup_program(program):
    """
    Switch the startup program to a new program
    Args:
        program(Program): The new startup program

    Returns:
        Program: The previous startup program
    """
    global _startup_program_
    prev_program = _startup_program_
    _startup_program_ = program
    return prev_program


@contextlib.contextmanager
def program_guard(main_program, startup_program=None):
    """
    Switch program with `with` statement

    Examples:
        >>> with program_guard(Program()):
        >>>   data = fluid.layers.data(...)
        >>>   hidden = fluid.layers.fc(...)

    Args:
        main_program(Program): New main program inside `with` statement
        startup_program(Program): New startup program inside `with` statement.
            None means do not change startup program.

    Returns:
        None
    """
    if not isinstance(main_program, Program):
        raise TypeError("main_program should be Program")
    main_program = switch_main_program(main_program)
    if startup_program is not None:
        if not isinstance(startup_program, Program):
            raise TypeError("startup_program should be Program")
        startup_program = switch_startup_program(startup_program)
    yield
    switch_main_program(main_program)
    if startup_program is not None:
        switch_startup_program(startup_program)


def get_var(name, program=None):
    """
    Get a variable by name from the global block of a program
    Args:
        name(str): name of the variable
        program(Program|None): program object.
             If None, default_global_program() will be used.

    Returns:
        Variable
    """
    if program is None:
        program = default_main_program()
    assert isinstance(name, str)
    assert isinstance(program, Program)

    return program.global_block().var(name)
