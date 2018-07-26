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

from paddle.fluid import framework as framework
from . import core
import collections
import copy
import six
from . import unique_name

__all__ = ['append_backward']


def _rename_arg_(op_descs, old_name, new_name, begin_idx=None, end_idx=None):
    """
    Traverse all ops in op_descs[begin_idx : end_idx],
    if any op has inputs/outputs named "old_name", rename it as 'new_name'
    """
    if begin_idx is None:
        begin_idx = 0
    if end_idx is None:
        end_idx = len(op_descs)
    for i in range(begin_idx, end_idx):
        op_desc = op_descs[i]
        if isinstance(op_desc, tuple):
            op_desc = op_desc[0]
        op_desc.rename_input(old_name, new_name)
        op_desc.rename_output(old_name, new_name)


def _create_op_desc_(op_type, inputs, outputs, attrs):
    """
    Create a C++ OpDesc object with specified inputs, outputs and attributes.
    """
    op_desc = core.OpDesc()
    op_desc.set_type(op_type)
    for para, args in list(inputs.items()):
        op_desc.set_input(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))
    for para, args in list(outputs.items()):
        op_desc.set_output(
            para,
            list(
                map(lambda arg: arg.decode() if isinstance(arg, six.binary_type) else arg,
                    args)))

    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()

    if op_role_attr_name not in attrs:
        attrs[
            op_role_attr_name] = core.op_proto_and_checker_maker.OpRole.Backward
    for name, val in list(attrs.items()):
        if isinstance(val, framework.Block):
            op_desc.set_block_attr(name, val.desc)
        else:
            op_desc.set_attr(name, val)
    return op_desc


def _infer_var_data_type_(grad_var_name, block):
    """
    Infer the data type of given grad variable
    """
    grad_var = block.desc.find_var(grad_var_name.encode("ascii"))
    fwd_name = _strip_grad_suffix_(grad_var_name.encode("ascii"))
    if block.desc.has_var_recursive(fwd_name):
        fwd_var = block.desc.find_var_recursive(fwd_name.encode("ascii"))
        grad_var.set_dtype(fwd_var.dtype())
    else:
        grad_var.set_dtype(core.VarDesc.VarType.FP32)


def _all_in_set_(cands, s):
    """
    Test if all elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if not c in s:
            return False
    return True


def _some_in_set_(cands, s):
    """
    Test if some elements of 'cands' are in set 's'
    """
    if len(cands) == 0:
        return False
    for c in cands:
        if c in s:
            return True
    return False


def _strip_grad_suffix_(name):
    """
    Strip the grad suffix from the given varibale name
    e.g. x@GRAD ==> x
         y@GRAD@RENAME@1 ==> y
    """
    if isinstance(name, six.text_type):
        name = name.encode()
    pos = name.find(six.b(core.grad_var_suffix()))
    return name[:pos] if pos != -1 else name


def _append_grad_suffix_(name):
    """
    Append grad suffix to the given variable name
    e.g. x ==> x@GRAD
    """
    if isinstance(name, six.text_type):
        name = name.encode()
    return name + six.b(core.grad_var_suffix())


def _addup_repetitive_outputs_(op_descs):
    """
    In backward part, an variable may be the output of more than one ops.
    And one op may yield its multiple outputs to the same variable.
    In these cases, the variable should be the accumulation of all the outputs.
    `sum_op`s are added to implement the accumulate.
    """
    pending_sum_ops = []
    var_rename_count = collections.defaultdict(int)
    renamed_vars = collections.defaultdict(list)
    for idx, op_desc in enumerate(op_descs):
        for var_name in op_desc.input_arg_names():
            if len(renamed_vars[var_name]) > 1:
                pending_sum_ops.append((_create_op_desc_(
                    "sum", {"X": renamed_vars[var_name]}, {"Out": [var_name]},
                    {"use_mkldnn": False}), idx))
                renamed_vars[var_name] = [var_name]
        for param_idx, param_name in enumerate(op_desc.output_names()):
            arg_names = op_desc.output(param_name)
            for arg_idx, var_name in enumerate(arg_names):
                if var_name == core.empty_var_name(
                ) or var_name in op_desc.input_arg_names():
                    # empty variable or inplace op
                    continue
                if len(renamed_vars[var_name]) == 0:
                    # it's the first time we get the variable
                    renamed_vars[var_name] = [var_name]
                else:
                    if len(renamed_vars[var_name]) == 1:
                        new_name = var_name + "@RENAME@" + \
                            str(var_rename_count[var_name])
                        var_rename_count[var_name] += 1
                        # rename original var_name
                        renamed_vars[var_name][0] = new_name
                        _rename_arg_(op_descs, var_name, new_name, 0, idx)
                        _rename_arg_(pending_sum_ops, var_name, new_name)

                        for p in op_desc.output_names()[:param_idx]:
                            p_arg_names = op_desc.output(p)
                            if var_name in p_arg_names:
                                op_desc.set_output(p, [
                                    new_name if x == var_name else x
                                    for x in p_arg_names
                                ])

                        arg_names = [
                            new_name if x == var_name else x
                            for x in arg_names[:arg_idx]
                        ] + arg_names[arg_idx:]

                    new_name = var_name + "@RENAME@" + \
                        str(var_rename_count[var_name])
                    var_rename_count[var_name] += 1
                    arg_names[arg_idx] = new_name
                    op_desc.set_output(param_name, arg_names)
                    renamed_vars[var_name].append(new_name)

    for var_name, inputs in list(renamed_vars.items()):
        if len(inputs) > 1:
            pending_sum_ops.append(
                (_create_op_desc_("sum", {"X": inputs}, {"Out": [var_name]},
                                  {"use_mkldnn": False}), len(op_descs)))
    # sum_op descs are sorted according to their insert position
    for p in reversed(pending_sum_ops):
        op_descs.insert(p[1], p[0])

    return op_descs


def _remove_no_grad_branch_(op_descs, no_grad_set):
    """
    Remove unnecessary grad ops
    A grad op can be removed in two cases:
        1. all outputs of the grad op are in 'no_grad_set'
        2. all grad inputs of the grad op are in 'no_grad_set'
    """

    def _op_can_be_removed_(op_desc, no_grad_set):
        out_arg_names = op_desc.output_arg_names()
        if len(out_arg_names) == 0 or _all_in_set_(out_arg_names, no_grad_set):
            return True
        if _all_in_set_([
                name for name in op_desc.input_arg_names()
                if name.find(core.grad_var_suffix()) != -1
        ], no_grad_set):
            no_grad_set.update(out_arg_names)
            return True
        return False

    # Remove ops whose outputs are all in no_grad_dict
    op_descs = [
        op_desc for op_desc in op_descs
        if not _op_can_be_removed_(op_desc, no_grad_set)
    ]
    # Insert fill_zeros_like_op
    to_insert = []
    for idx, op_desc in enumerate(op_descs):
        for arg in op_desc.input_arg_names():
            if core.grad_var_suffix() in arg and arg in no_grad_set:
                to_insert.append((_create_op_desc_("fill_zeros_like", {
                    "X": [_strip_grad_suffix_(arg)]
                }, {"Out": [arg]}, {}), idx))

    list([op_descs.insert(p[1], p[0]) for p in reversed(to_insert)])

    return op_descs


from .proto import framework_pb2


def serialize_op_decs(op_desc):
    protostr = op_desc.serialize_to_string()
    proto = framework_pb2.OpDesc.FromString(str(protostr))
    return proto.__str__()


def _callback_lookup_(op):
    """
    Only used in _append_backward_ops_
    Build and returns a callback function for certain op. For example

    parallel_do:           AllReduce

    :param op:
    :return: callback function
    """
    if op.type == 'parallel_do' and op.attr('use_nccl'):
        all_vars = op.block.vars
        param_names = set(op.input('parameters'))
        param_names = [
            name for name in param_names
            if all_vars[name].stop_gradient is False
        ]
        param_grad_names = [n + "@GRAD" for n in param_names]

        class ParallelDoCallBack(object):
            def __init__(self, param_grad_names, parallel_scopes_name):
                self.has_inserted_nccl_init = False
                self.param_grad_names = param_grad_names
                self.parallel_scopes_name = parallel_scopes_name

            def __call__(self, block, context):
                if not self.has_inserted_nccl_init:
                    op_desc = _create_op_desc_(
                        "ncclInit",
                        {"parallel_scopes": self.parallel_scopes_name},
                        {"Communicator": ['nccl_com__do_not_change_']}, {})
                    block.program.global_block().desc.append_op().copy_from(
                        op_desc)
                    self.has_inserted_nccl_init = True

                current_op_desc = context["__current_op_desc__"]
                for o_param in current_op_desc.output_names():
                    for o_argu in current_op_desc.output(o_param):
                        if o_argu in self.param_grad_names:
                            allreduce_out_name = o_argu + "__nccl_all_reduce__"
                            op_desc = _create_op_desc_(
                                "ncclReduce",
                                {
                                    "X": [o_argu],
                                    "Communicator":
                                    ['nccl_com__do_not_change_']
                                },
                                {"Out": [allreduce_out_name]},
                                {"reduction": "ncclSum",
                                 "root": 0}, )
                            block.desc.append_op().copy_from(op_desc)

                            op_desc = _create_op_desc_(
                                "assign", {"X": [allreduce_out_name]},
                                {"Out": [o_argu]}, {})
                            block.desc.append_op().copy_from(op_desc)

        return ParallelDoCallBack(param_grad_names,
                                  op.output("parallel_scopes"))
    else:
        return None


def _append_backward_ops_(block,
                          ops,
                          target_block,
                          no_grad_dict,
                          grad_to_var,
                          callbacks=None):
    """
    Create all grad ops, and insert them into given block

    Args:
        block(Block): the block where forward ops are
        ops(Op): the forward operators whose backward ops need to be added
        target_block(Block): the block which is going to hold new generated grad ops
        no_grad_dict(dict):
            key(int)  block index
            val(set) a set of varibale names. These varibales have no gradient
        grad_to_var(dict)(output argument):
            key(str): grad variable name
            val(str): corresponding forward variable name
        callback(callable object): a callable object used to decorate new generated grad ops
    """
    if callbacks is not None:
        assert (isinstance(callbacks, list))
        for cb in callbacks:
            if not hasattr(cb, '__call__'):
                raise ValueError("'callback' must be a callable object.")

    # grad_op_descs holds created grad_op, and will be appended to target_block
    grad_op_descs = []
    program = block.program
    for op in reversed(ops):
        grad_sub_block_list = []
        # If the op has its own sub-block, deal with the sub-block first
        if op.has_attr("sub_block"):
            sub_block = program.block(op.block_attr("sub_block"))
            grad_sub_block = program.create_block()
            grad_sub_block._set_forward_block_idx(sub_block.idx)
            cb = _callback_lookup_(op)
            if cb is not None:
                if callbacks is None:
                    new_callbacks = [cb]
                else:
                    new_callbacks = callbacks + [_callback_lookup_(op)]
                _append_backward_ops_(sub_block, sub_block.ops, grad_sub_block,
                                      no_grad_dict, grad_to_var, new_callbacks)
            else:
                _append_backward_ops_(sub_block, sub_block.ops, grad_sub_block,
                                      no_grad_dict, grad_to_var, callbacks)

            program.rollback()
            grad_sub_block_list.append(grad_sub_block.desc)

        # Getting op's corresponding grad_op
        grad_op_desc, op_grad_to_var = core.get_grad_op_desc(
            op.desc, no_grad_dict[block.idx], grad_sub_block_list)

        grad_op_descs.extend(grad_op_desc)
        grad_to_var.update(op_grad_to_var)

    grad_op_descs = _addup_repetitive_outputs_(grad_op_descs)

    grad_op_descs = _remove_no_grad_branch_(grad_op_descs,
                                            no_grad_dict[block.idx])

    # append op_desc in grad_op_descs to target_block
    op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName()
    backward = core.op_proto_and_checker_maker.OpRole.Backward
    for op_desc in grad_op_descs:
        new_op_desc = target_block.desc.append_op()
        new_op_desc.copy_from(op_desc)
        new_op_desc.set_attr(op_role_attr_name, backward)
        grad_to_var["__current_op_desc__"] = new_op_desc
        if callbacks is not None:
            assert (isinstance(callbacks, list))
            for cb in callbacks:
                cb(block=target_block, context=grad_to_var)


def _append_backward_vars_(block, start_op_idx, grad_to_var, grad_info_map):
    """
    Create new variables required by backward pass.

    Args:
        block(Block): the block where new variables will be created
        start_op_idx(int): Only variables required by ops in block.ops[start_op_idx : ] will be created
        grad_to_var(dict):
            key(str): grad variable name
            val(str): corresponding forward variable name
            In most cases, this dict is generated by _append_backward_ops_()
        grad_info_map(dict)(output argument):
            key(str): forward variable name
            val(tuple): a tuple of (str, Block), str is the corresponding grad name, Block is the block containing grad variable
    """
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        if op_desc.has_attr("sub_block"):
            sub_block = block.program.block(op_desc.block_attr("sub_block"))
            _append_backward_vars_(sub_block, 0, grad_to_var, grad_info_map)
        new_vars = set()
        # create new gradient variables
        for grad_var_name in op_desc.output_arg_names():
            grad_var_name = grad_var_name.encode("ascii")
            if block.desc.has_var_recursive(
                    grad_var_name) or grad_var_name == core.empty_var_name():
                continue
            block.desc.var(grad_var_name)
            new_vars.add(grad_var_name)
            if grad_var_name not in grad_to_var:
                continue
            grad_info_map[grad_to_var[grad_var_name]] = (grad_var_name, block)
        # infer_shape and infer_type
        op_desc.infer_var_type(block.desc)
        op_desc.infer_shape(block.desc)
        # ncclInit dones't need to set data_type
        if op_desc.type() == 'ncclInit':
            continue
        for arg in op_desc.output_arg_names():
            if arg in new_vars:
                _infer_var_data_type_(arg, block)


def _rename_grad_(block, start_op_idx, grad_to_var, target_grad_map):
    var_map = copy.copy(target_grad_map)
    for op_idx in range(start_op_idx, block.desc.op_size()):
        op_desc = block.desc.op(op_idx)
        for name in op_desc.input_arg_names():
            if name in var_map:
                op_desc.rename_input(name, var_map[name])

        for name in op_desc.output_arg_names():
            if block.desc.find_var(name.encode("ascii")):
                new_name = unique_name.generate(name)
                op_desc.rename_output(name, new_name)
                var_map[name] = new_name

    for g, ng in list(var_map.items()):
        if g in grad_to_var:
            grad_to_var[ng] = grad_to_var[g]
            grad_to_var.pop(g)


def _get_stop_gradients_(program):
    no_grad_dict = dict()
    assert isinstance(program, framework.Program)
    for block in program.blocks:
        assert isinstance(block, framework.Block)
        block_no_grad_set = set()
        for var in list(block.vars.values()):
            assert isinstance(var, framework.Variable)
            if var.stop_gradient:
                block_no_grad_set.add(_append_grad_suffix_(var.name))
        no_grad_dict[block.idx] = block_no_grad_set
    return no_grad_dict


def append_backward(loss, parameter_list=None, no_grad_set=None,
                    callbacks=None):
    """
    Append backward part to main_program.

    A complete neural network training is made up of forward and backward
    propagation. However, when we configure a network, we only need to
    specify its forwrd part. The backward part is generated automatically
    according to the forward part by this function.

    In most cases, users do not need to invoke this function manually. It
    will be automatically invoked by the optimizer's `minimize` function.

    Args:
        loss(Variable): The loss variable of the network.
        parameter_list(list[string]|None): Names of parameters that need
                                           to be updated by optimizers.
                                           If it is None, all parameters
                                           will be updated.
                                           Default: None
        no_grad_set(set|None): Variables in the Block 0 whose gradients
                               should be ignored. All variables with
                               `step_gradient=True` from all blocks will
                               be automatically added into this set.
                               Default: None
        callbacks(list[callable object]|None): The callbacks are used for
                                               doing some custom jobs during
                                               backward part building. All
                                               callable objects in it will
                                               be invoked once each time a
                                               new gradient operator is added
                                               into the program. The callable
                                               object must has two input
                                               parameters: 'block' and 'context'.
                                               The 'block' is the block which
                                               the new gradient operator will
                                               be added to. The 'context' is a
                                               map, whose keys are gradient
                                               variable names and values are
                                               corresponding original variables.
                                               In addition to this, the 'context'
                                               has another special key-value pair:
                                               the key is string '__current_op_desc__'
                                               and the value is the op_desc of the
                                               gradient operator who has just
                                               triggered the callable object.

    Returns:
        list[(Variable,Variable)]: Pairs of parameter and its
        corresponding gradients. The key is the parameter and the
        value is gradient variable.

    Raises:
        AssertionError: If `loss` is not an instance of Variable.

    Examples:
        .. code-block:: python

            # network configuration code
            # ...
            avg_loss = fluid.layers.mean(loss)
            param_grad_list = fluid.backward.append_backward(loss=avg_loss)
    """
    assert isinstance(loss, framework.Variable)

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        for op in reversed(loss.block.ops):
            assert isinstance(op, framework.Operator)
            if len(op.output_arg_names) == 1 and op.output_arg_names[
                    0] == loss.name:
                loss.op = op
                break
        if loss.op is None:
            raise ValueError("loss.op is None. Should not happend")

    loss.op.set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
                     int(core.op_proto_and_checker_maker.OpRole.Forward) |
                     int(core.op_proto_and_checker_maker.OpRole.Loss))

    if callbacks is not None:
        isinstance(callbacks, list)

    program = loss.block.program
    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    no_grad_dict = _get_stop_gradients_(program)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    grad_info_map = dict()
    root_block = program.block(0)

    fwd_op_num = root_block.desc.op_size()
    current_block_idx = program.current_block_idx
    grad_to_var = dict()

    op_desc = _create_op_desc_(
        "fill_constant", {}, {"Out": [_append_grad_suffix_(loss.name)]}, {
            "shape": [1],
            "value": 1.0,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName():
            int(core.op_proto_and_checker_maker.OpRole.Backward) |
            int(core.op_proto_and_checker_maker.OpRole.Loss),
        })
    root_block.desc.append_op().copy_from(op_desc)

    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0]))
    op_path = _find_op_path_(root_block, [loss], [], block_no_grad_set)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set)))

    _append_backward_ops_(root_block, op_path, root_block, no_grad_dict,
                          grad_to_var, callbacks)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(root_block, fwd_op_num, grad_to_var, {})

    _append_backward_vars_(root_block, fwd_op_num, grad_to_var, grad_info_map)

    program.current_block_idx = current_block_idx
    program._sync_with_cpp()
    # FIXME(zcd): prevent loss.grad optimized by mem_opt.
    loss.block.var(_append_grad_suffix_(loss.name)).persistable = True

    if parameter_list is not None:
        parameters = parameter_list
    else:
        params = program.global_block().all_parameters()
        parameters = [param.name for param in params]

    params_and_grads = []
    for param in parameters:
        if param not in grad_info_map:
            continue
        grad_info = grad_info_map[param]
        grad_block = grad_info[1]
        if not grad_block.has_var(grad_info[0]):
            raise ValueError("grad block[{0}] did not have grad var {1}".format(
                grad_info[1], grad_info[0]))
        # Get the param var from the global block
        param_var = program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if loss.block.has_var(grad_info[0]):
            params_and_grads.append((param_var, grad_var))
        else:
            params_and_grads.append((param_var, None))

    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
    for p, g in params_and_grads:
        if g is None:
            continue
        for op in reversed(program.global_block().ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op.set_attr(op_role_var_attr_name, attr_val)

    return params_and_grads


def _as_list(x):
    if x is None:
        return []
    return list(x) if isinstance(x, collections.Sequence) else [x]


def _find_op_path_(block, outputs, inputs, no_grad_set):
    """
    no_grad_set will also be changed
    """
    input_names = set([inp.name for inp in inputs])
    output_names = set([out.name for out in outputs])

    relevant_op_flags = [True] * len(block.ops)

    # All the inputs of the block are used if inputs is empty,
    if inputs:
        for i, op in enumerate(block.ops):
            if _some_in_set_(op.desc.input_arg_names(), input_names):
                for name in op.desc.output_arg_names():
                    if name not in no_grad_set:
                        input_names.add(name)
            else:
                relevant_op_flags[i] = False

    for i, op in reversed(list(enumerate(block.ops))):
        if _some_in_set_(op.desc.output_arg_names(), output_names):
            for name in op.desc.input_arg_names():
                if name not in no_grad_set:
                    output_names.add(name)
        else:
            relevant_op_flags[i] = False

    op_path = [
        block.ops[i] for i in range(len(block.ops)) if relevant_op_flags[i]
    ]

    if inputs:
        for op in op_path:
            for name in op.desc.input_arg_names():
                if name not in input_names:
                    no_grad_set.add(name)

    return op_path


def calc_gradient(targets, inputs, target_gradients=None, no_grad_set=None):
    """
    Backpropagate the graidents of targets to inputs.

    Args:
        targets(Variable|list[Variable]): The target variables
        inputs(Variable|list[Variable]): The input variables
        no_grad_set(set[string]): The names of variables that have no gradients
            in Block 0. All variables with `stop_gradient=True` from all blocks
            will be automatically added.

    Return:
        (list[Variable]): list of gradients for inputs
        If an input does not affect targets, the corresponding gradient variable
        will be None
    """
    targets = _as_list(targets)
    inputs = _as_list(inputs)
    target_gradients = _as_list(target_gradients)

    block = targets[0].block
    prog = block.program
    block_idx = block.idx

    if not target_gradients:
        target_gradients = [None] * len(targets)

    if len(targets) != len(target_gradients):
        raise ValueError(
            "Should have the same number of target_gradients as targets")

    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    no_grad_dict = _get_stop_gradients_(prog)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, no_grad_set)))

    fwd_op_num = block.desc.op_size()

    target_grad_map = {}
    for i, grad in enumerate(target_gradients):
        target = targets[i]
        if grad is None:
            grad_name = _append_grad_suffix_(target.name)
            op_desc = _create_op_desc_("fill_constant_batch_size_like",
                                       {"Input": [target.name]},
                                       {"Out": [grad_name]}, {
                                           "shape": target.shape,
                                           "value": 1.0,
                                           "dtype": target.dtype,
                                           'input_dim_idx': 0,
                                           'output_dim_idx': 0
                                       })
            block.desc.append_op().copy_from(op_desc)
        else:
            if target.block.idx != block_idx or target.block.program != prog:
                raise ValueError("all targets must be in the same block")
            if target.shape != grad.shape:
                raise ValueError(
                    "The shapes of target and grad are different: %s %s" % (
                        target.name, grad.name))
            target_grad_map[_append_grad_suffix_(target.name)] = grad.name

    for input in inputs:
        if input.block.program != prog:
            raise "input must be in the same program as targets"

    block_no_grad_set = set(map(_strip_grad_suffix_, no_grad_dict[0]))
    op_path = _find_op_path_(block, targets, inputs, block_no_grad_set)
    no_grad_dict[0].update(list(map(_append_grad_suffix_, block_no_grad_set)))
    grad_to_var = dict()
    grad_info_map = dict()
    _append_backward_ops_(block, op_path, block, no_grad_dict, grad_to_var)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    _rename_grad_(block, fwd_op_num, grad_to_var, target_grad_map)

    _append_backward_vars_(block, fwd_op_num, grad_to_var, grad_info_map)
    prog._sync_with_cpp()

    grad_vars = []
    for input_var in inputs:
        if input_var.name not in grad_info_map:
            grad_vars.append(None)
        else:
            grad_info = grad_info_map[input_var.name]
            grad_block = grad_info[1]
            grad_var = grad_block.var(grad_info[0])
            grad_vars.append(grad_var)

    if len(grad_vars) == 1:
        return grad_vars[0]
    else:
        return grad_vars
