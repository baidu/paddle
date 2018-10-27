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
"""
Steps to transpile trainer:
1. split variable to multiple blocks, aligned by product(dim[1:]) (width).
2. rename splited grad variables to add trainer_id suffix ".trainer_%d".
3. modify trainer program add split_op to each grad variable.
4. append send_op to send splited variables to server and
5. add recv_op to fetch params(splited blocks or origin param) from server.
6. append concat_op to merge splited blocks to update local weights.

Steps to transpile pserver:
1. create new program for parameter server.
2. create params and grad variables that assigned to current server instance.
3. create a sub-block in the server side program
4. append ops that should run on current server instance.
5. add listen_and_serv op
"""

import math
import sys
import numpy as np
import collections
import six

from .ps_dispatcher import RoundRobin, HashName, PSDispatcher
from .. import core, framework
from ..framework import Program, default_main_program, \
    default_startup_program, Block, \
    Parameter, grad_var_name
from .details import *
from functools import reduce

LOOKUP_TABLE_TYPE = "lookup_table"
LOOKUP_TABLE_GRAD_TYPE = "lookup_table_grad"
OP_ROLE_VAR_ATTR_NAME = core.op_proto_and_checker_maker.kOpRoleVarAttrName()
RPC_OP_ROLE_ATTR_NAME = op_role_attr_name = core.op_proto_and_checker_maker.kOpRoleAttrName(
)
OPT_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Optimize
RPC_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.RPC
DIST_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.Dist
LR_SCHED_OP_ROLE_ATTR_VALUE = core.op_proto_and_checker_maker.OpRole.LRSched

PRINT_LOG = False


def log(*args):
    if PRINT_LOG:
        print(args)


class VarBlock:
    def __init__(self, varname, offset, size):
        self.varname = varname
        # NOTE: real offset is offset * size
        self.offset = offset
        self.size = size

    def __str__(self):
        return "%s:%d:%d" % (self.varname, self.offset, self.size)


def same_or_split_var(p_name, var_name):
    return p_name == var_name or p_name.startswith(var_name + ".block")


def slice_variable(var_list, slice_count, min_block_size):
    """
    We may need to split dense tensor to one or more blocks and put
    them equally onto parameter server. One block is a sub-tensor
    aligned by dim[0] of the tensor.

    We need to have a minimal block size so that the calculations in
    the parameter server side can gain better performance. By default
    minimum block size 8K elements (maybe 16bit or 32bit or 64bit).

    Args:
        var_list (list): List of variables.
        slice_count (int): Numel of count that variables will be sliced, which
            could be the pserver services' count.
        min_block_size (int): Minimum splitted block size.
    Returns:
        blocks (list[(varname, block_id, current_block_size)]): A list
            of VarBlocks. Each VarBlock specifies a shard of the var.
    """
    blocks = []
    for var in var_list:
        split_count = slice_count
        var_numel = reduce(lambda x, y: x * y, var.shape)
        max_pserver_count = int(math.floor(var_numel / float(min_block_size)))
        if max_pserver_count == 0:
            max_pserver_count = 1
        if max_pserver_count < slice_count:
            split_count = max_pserver_count
        block_size = int(math.ceil(var_numel / float(split_count)))

        if len(var.shape) >= 2:
            # align by dim1(width)
            dim1 = reduce(lambda x, y: x * y, var.shape[1:])
            remains = block_size % dim1
            if remains != 0:
                block_size += dim1 - remains
        # update split_count after aligning
        split_count = int(math.ceil(var_numel / float(block_size)))
        for block_id in range(split_count):
            curr_block_size = min(block_size, var_numel - (
                (block_id) * block_size))
            block = VarBlock(var.name, block_id, curr_block_size)
            blocks.append(str(block))
    return blocks


class DistributeTranspilerConfig(object):
    """
    slice_var_up (bool): Do Tensor slice for pservers, default is True.
    split_method (PSDispatcher): RoundRobin or HashName can be used
        try to choose the best method to balance loads for pservers.
    min_block_size (int): Minimum splitted element number in block.
        According:https://github.com/PaddlePaddle/Paddle/issues/8638#issuecomment-369912156
        We can use bandwidth effiently when data size is larger than 2MB.If you
        want to change it, please be sure you see the slice_variable function.
    """

    slice_var_up = True
    split_method = None
    min_block_size = 8192
    # supported modes: pserver, nccl2
    mode = "pserver"
    print_log = False


class DistributeTranspiler(object):
    """
    **DistributeTranspiler**

    Convert the fluid program to distributed data-parallelism programs.
    Supports two modes: pserver mode and nccl2 mode.

    In pserver mode, the main_program will be transformed to use a remote
    parameter server to do parameter optimization. And the optimization
    graph will be put into a parameter server program.

    In nccl2 mode, the transpiler will append a NCCL_ID broadcasting
    op in startup_program to share the NCCL_ID across the job nodes.
    After transpile_nccl2 called, you ***must*** pass trainer_id and
    num_trainers argument to ParallelExecutor to enable NCCL2 distributed
    mode.

    Examples:
        .. code-block:: python

           # for pserver mode
           pserver_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
           trainer_endpoints = "192.168.0.1:6174,192.168.0.2:6174"
           current_endpoint = "192.168.0.1:6174"
           trainer_id = 0
           trainers = 4
           role = os.getenv("PADDLE_TRAINING_ROLE")

           t = fluid.DistributeTranspiler()
           t.transpile(
                trainer_id, pservers=pserver_endpoints, trainers=trainers)
           if role == "PSERVER":
                pserver_program = t.get_pserver_program(current_endpoint)
                pserver_startup_program = t.get_startup_program(current_endpoint,
                                                                pserver_program)
           elif role == "TRAINER":
                trainer_program = t.get_trainer_program()

           # for nccl2 mode
           config = fluid.DistributeTranspilerConfig()
           config.mode = "nccl2"
           t = fluid.DistributeTranspiler(config=config)
           t.transpile(trainer_id, workers=workers, current_endpoint=curr_ep)
           exe = fluid.ParallelExecutor(
               use_cuda,
               loss_name=loss_var.name,
               num_trainers=len(trainers.split(",)),
               trainer_id=trainer_id
           )
    """

    def __init__(self, config=None):
        if config is not None:
            self.config = config
        else:
            self.config = DistributeTranspilerConfig()

        if self.config.split_method is None:
            self.config.split_method = RoundRobin

        global PRINT_LOG
        if self.config.print_log:
            PRINT_LOG = True
        assert (self.config.min_block_size >= 8192)
        assert (self.config.split_method.__bases__[0] == PSDispatcher)

    def _transpile_nccl2(self,
                         trainer_id,
                         trainers,
                         current_endpoint,
                         startup_program=None):
        if not startup_program:
            startup_program = default_startup_program()
        if trainer_id >= 0:
            worker_endpoints = trainers.split(",")
            # send NCCL_ID to others or recv from trainer 0
            worker_endpoints.remove(current_endpoint)

            nccl_id_var = startup_program.global_block().create_var(
                name="NCCLID", persistable=True, type=core.VarDesc.VarType.RAW)
            startup_program.global_block().append_op(
                type="gen_nccl_id",
                inputs={},
                outputs={"NCCLID": nccl_id_var},
                attrs={
                    "endpoint": current_endpoint,
                    "endpoint_list": worker_endpoints,
                    "trainer_id": trainer_id
                })
            return nccl_id_var
        else:
            raise ValueError("must set trainer_id > 0")

    def transpile(self,
                  trainer_id,
                  program=None,
                  pservers="127.0.0.1:6174",
                  trainers=1,
                  sync_mode=True,
                  startup_program=None,
                  current_endpoint="127.0.0.1:6174"):
        """
        Run the transpiler.

        Args:
            trainer_id (int): id for current trainer worker, if you have
                n workers, the id may range from 0 ~ n-1
            program (Program|None): program to transpile,
                default is fluid.default_main_program().
            pservers (str): comma separated ip:port string for the pserver
                list.
            trainers (int|str): in pserver mode this is the number of
                trainers, in nccl2 mode this is a string of trainer
                endpoints.
            sync_mode (bool): Do sync training or not, default is True.
            startup_program (Program|None): startup_program to transpile,
                default is fluid.default_main_program().
            current_endpoint (str): need pass current endpoint when
                transpile as nccl2 distributed mode. In pserver mode
                this argument is not used.
        """
        if program is None:
            program = default_main_program()
        if startup_program is None:
            startup_program = default_startup_program()
        self.origin_program = program
        self.startup_program = startup_program
        self.origin_startup_program = self.startup_program.clone()

        if self.config.mode == "nccl2":
            assert (isinstance(trainers, str))
            self._transpile_nccl2(
                trainer_id,
                trainers,
                current_endpoint,
                startup_program=startup_program)
            return

        self.trainer_num = trainers
        self.sync_mode = sync_mode
        self.trainer_id = trainer_id
        pserver_endpoints = pservers.split(",")
        self.pserver_endpoints = pserver_endpoints
        self.optimize_ops, self.params_grads = self._get_optimize_pass()

        ps_dispatcher = self.config.split_method(self.pserver_endpoints)
        self.has_distributed_lookup_table = self._has_distributed_lookup_table()
        self.param_name_to_grad_name = dict()
        self.grad_name_to_param_name = dict()
        for param_var, grad_var in self.params_grads:
            self.param_name_to_grad_name[param_var.name] = grad_var.name
            self.grad_name_to_param_name[grad_var.name] = param_var.name

        # add distributed attrs to program
        self.origin_program._is_distributed = True
        self.origin_program._endpoints = self.pserver_endpoints
        self.origin_program._is_chief = self.trainer_id == 0
        self.origin_program._distributed_lookup_table = self.table_name if self.table_name else None

        # split and create vars, then put splited vars in dicts for later use.
        # step 1: split and create vars, then put splited vars in dicts for later use.
        self._init_splited_vars()

        # step 2: insert send op to send gradient vars to parameter servers
        ps_dispatcher.reset()
        send_vars = []

        # in general cases, the number of pservers is times of 2, and this
        # will lead to uneven distribution among weights and bias:
        #       fc_w@GRAD_trainer_0, fc_w@GRAD_trainer_1 --> pserver1
        #       fc_b@GRAD_trainer_0, fc_b@GRAD_trainer_1 --> pserver2
        # shuffle the map will avoid the uneven distribution above
        grad_var_mapping_items = list(six.iteritems(self.grad_var_mapping))

        if not self.config.slice_var_up:
            np.random.seed(self.origin_program.random_seed)
            np.random.shuffle(grad_var_mapping_items)

        self.grad_name_to_send_dummy_out = dict()
        for grad_varname, splited_vars in grad_var_mapping_items:
            eplist = ps_dispatcher.dispatch(splited_vars)

            if not self.config.slice_var_up:
                assert (len(splited_vars) == 1)

            splited_grad_varname = grad_varname
            if len(splited_vars) == 1:
                splited_grad_varname = splited_vars[0].name
                index = find_op_by_output_arg(
                    program.global_block(), splited_grad_varname, reverse=True)
            elif len(splited_vars) > 1:
                orig_var = program.global_block().vars[splited_grad_varname]
                index = find_op_by_output_arg(
                    program.global_block(), splited_grad_varname, reverse=True)
                self._insert_split_op(program, orig_var, index, splited_vars)
                index += 1
            else:
                AssertionError("Can not insert the send op by original "
                               "variable name :", splited_grad_varname)

            dummy_output = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())
            self.grad_name_to_send_dummy_out[grad_varname] = dummy_output

            # get send op_role_var, if not splited, the grad should have .trainer suffix
            # if splited, grad should be the original grad var name (split_by_ref and send
            # will be on the same place). ParallelExecutor
            # will use op_role_var to get expected device place to run this op.
            program.global_block()._insert_op(
                index=index + 1,
                type="send",
                inputs={"X": splited_vars},
                outputs={"Out": dummy_output},
                attrs={
                    "epmap": eplist,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
                    OP_ROLE_VAR_ATTR_NAME: [
                        self.grad_name_to_param_name[grad_varname],
                        splited_grad_varname
                    ],
                    "sync_mode": not self.sync_mode,
                })
            for _, var in enumerate(splited_vars):
                send_vars.append(var)

        if self.sync_mode:
            send_barrier_out = program.global_block().create_var(
                name=framework.generate_control_dev_var_name())
            if self.has_distributed_lookup_table:
                self.grad_name_to_send_dummy_out[
                    self.table_name] = program.global_block().create_var(
                        name=framework.generate_control_dev_var_name())
            input_deps = list(self.grad_name_to_send_dummy_out.values())

            program.global_block().append_op(
                type="send_barrier",
                inputs={"X": list(input_deps)},
                outputs={"Out": send_barrier_out},
                attrs={
                    "endpoints": pserver_endpoints,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        # step 3: insert recv op to receive parameters from parameter server
        recv_vars = []
        for _, var in enumerate(send_vars):
            recv_vars.append(self.grad_param_mapping[var])
        ps_dispatcher.reset()
        eplist = ps_dispatcher.dispatch(recv_vars)

        for i, ep in enumerate(eplist):
            self.param_grad_ep_mapping[ep]["params"].append(recv_vars[i])
            self.param_grad_ep_mapping[ep]["grads"].append(send_vars[i])

        # step4: Concat the parameters splits together after recv.
        all_recv_outputs = []
        for param_varname, splited_var in six.iteritems(self.param_var_mapping):
            eps = []
            for var in splited_var:
                index = [v.name for v in recv_vars].index(var.name)
                eps.append(eplist[index])
            if self.sync_mode:
                recv_dep_in = send_barrier_out
            else:
                # connect deps to send op in async mode
                recv_dep_in = self.grad_name_to_send_dummy_out[
                    self.param_name_to_grad_name[param_varname]]
            all_recv_outputs.extend(splited_var)
            # get recv op_role_var, if not splited, the grad should have .trainer suffix
            # if splited, grad should be the original grad var name. ParallelExecutor
            # will use op_role_var to get expected device place to run this op.
            orig_grad_name = self.param_name_to_grad_name[param_varname]
            recv_op_role_var_name = orig_grad_name
            splited_trainer_grad = self.grad_var_mapping[orig_grad_name]
            if len(splited_trainer_grad) == 1:
                recv_op_role_var_name = splited_trainer_grad[0].name

            program.global_block().append_op(
                type="recv",
                inputs={"X": [recv_dep_in]},
                outputs={"Out": splited_var},
                attrs={
                    "epmap": eps,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
                    OP_ROLE_VAR_ATTR_NAME:
                    [param_varname, recv_op_role_var_name],
                    "sync_mode": not self.sync_mode
                })

        if self.sync_mode:
            # form a WAW dependency
            program.global_block().append_op(
                type="fetch_barrier",
                inputs={},
                outputs={"Out": all_recv_outputs},
                attrs={
                    "endpoints": pserver_endpoints,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        for param_varname, splited_var in six.iteritems(self.param_var_mapping):
            if len(splited_var) <= 1:
                continue
            orig_param = program.global_block().vars[param_varname]
            program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={
                    "axis": 0,
                    RPC_OP_ROLE_ATTR_NAME: DIST_OP_ROLE_ATTR_VALUE
                })

        self._get_trainer_startup_program(recv_vars=recv_vars, eplist=eplist)

        if self.has_distributed_lookup_table:
            self._replace_lookup_table_op_with_prefetch(program,
                                                        pserver_endpoints)
            self._split_table_grad_and_add_send_vars(program, pserver_endpoints)

    def get_trainer_program(self, wait_port=True):
        """
        Get transpiled trainer side program.

        Returns:
            Program: trainer side program.
        """
        # remove optimize ops and add a send op to main_program
        # FIXME(typhoonzero): Also ops like clip_gradient, lrn_decay?
        lr_ops = self._get_lr_ops()
        delete_ops(self.origin_program.global_block(), self.optimize_ops)
        delete_ops(self.origin_program.global_block(), lr_ops)

        self.origin_program.__str__()

        if wait_port:
            wait_server_ready(self.pserver_endpoints)

        return self.origin_program

    def _get_trainer_startup_program(self, recv_vars, eplist):
        """
        Get transpiled trainer side startup program.

        Args:
            recv_vars (list): Variable list to recv for current trainer_id
            eplist (list): A list of strings indicating

        Returns:
            Program: trainer side startup program.
        """
        startup_program = self.startup_program

        # FIXME(gongwb): delete not need ops.
        # note that: some parameter is not trainable and those ops can't be deleted.

        for varname, splited_var in six.iteritems(self.param_var_mapping):
            # Get the eplist of recv vars
            eps = []
            for var in splited_var:
                index = [v.name for v in recv_vars].index(var.name)
                eps.append(eplist[index])

            for var in splited_var:
                if startup_program.global_block().has_var(var.name):
                    continue

                startup_program.global_block().create_var(
                    name=var.name,
                    persistable=False,
                    type=var.type,
                    dtype=var.dtype,
                    shape=var.shape,
                    lod_level=var.lod_level)

            op = startup_program.global_block().append_op(
                type="recv",
                inputs={"X": []},
                outputs={"Out": splited_var},
                attrs={
                    "epmap": eps,
                    RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
                })

        fetch_barrier_out = startup_program.global_block().create_var(
            name=framework.generate_control_dev_var_name())
        startup_program.global_block().append_op(
            type="fetch_barrier",
            inputs={},
            outputs={"Out": fetch_barrier_out},
            attrs={
                "endpoints": self.pserver_endpoints,
                RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        for varname, splited_var in six.iteritems(self.param_var_mapping):
            # add concat ops to merge splited parameters received from parameter servers.
            if len(splited_var) <= 1:
                continue
            # NOTE: if enable memory optimization, origin vars maybe removed.
            if varname in startup_program.global_block().vars:
                orig_param = startup_program.global_block().vars[varname]
            else:
                origin_param_var = self.origin_program.global_block().vars[
                    varname]
                orig_param = startup_program.global_block().create_var(
                    name=varname,
                    persistable=origin_param_var.persistable,
                    type=origin_param_var.type,
                    dtype=origin_param_var.dtype,
                    shape=origin_param_var.shape)
            startup_program.global_block().append_op(
                type="concat",
                inputs={"X": splited_var},
                outputs={"Out": [orig_param]},
                attrs={"axis": 0})

        return startup_program

    def get_pserver_program(self, endpoint):
        """
        Get parameter server side program.

        Args:
            endpoint (str): current parameter server endpoint.

        Returns:
            Program: the program for current parameter server to run.
        """
        # TODO(panyx0718): Revisit this assumption. what if #blocks > #pservers.
        # NOTE: assume blocks of the same variable is not distributed
        # on the same pserver, only change param/grad varnames for
        # trainers to fetch.
        sys.stderr.write("get_pserver_program() is deprecated, call \
get_pserver_programs() to get pserver main and startup \
in a single call.")
        # step1
        pserver_program = Program()
        pserver_program.random_seed = self.origin_program.random_seed
        # step2: Create vars to receive vars at parameter servers.
        recv_inputs = []
        for v in self.param_grad_ep_mapping[endpoint]["params"]:
            self._clone_var(pserver_program.global_block(), v)
        for v in self.param_grad_ep_mapping[endpoint]["grads"]:
            # create vars for each trainer in global scope, so
            # we don't need to create them when grad arrives.
            # change client side var name to origin name by
            # removing ".trainer_%d" suffix
            suff_idx = v.name.find(".trainer_")
            if suff_idx >= 0:
                orig_var_name = v.name[:suff_idx]
            else:
                orig_var_name = v.name
            # NOTE: single_trainer_var must be created for multi-trainer
            # case to merge grads from multiple trainers
            single_trainer_var = \
                pserver_program.global_block().create_var(
                    name=orig_var_name,
                    persistable=True,
                    type=v.type,
                    dtype=v.dtype,
                    shape=v.shape)
            if self.sync_mode and self.trainer_num > 1:
                for trainer_id in range(self.trainer_num):
                    var = pserver_program.global_block().create_var(
                        name="%s.trainer_%d" % (orig_var_name, trainer_id),
                        persistable=False,
                        type=v.type,
                        dtype=v.dtype,
                        shape=v.shape)
                    recv_inputs.append(var)
            else:
                recv_inputs.append(single_trainer_var)

        # step 3
        # Create a union-find data structure from optimize ops,
        # If two ops are connected, we could add these two ops
        # into one set.
        ufind = self._create_ufind(self.optimize_ops)
        # step 3.2
        # Iterate through the ops and append optimize op which
        # located on current pserver
        opt_op_on_pserver = []
        for _, op in enumerate(self.optimize_ops):
            if self._is_optimizer_op(op) and self._is_opt_op_on_pserver(
                    endpoint, op):
                opt_op_on_pserver.append(op)
        # step 3.3
        # Iterate through the ops, and if an op and the optimize ops
        # which located on current pserver are in one set, then
        # append it into the sub program.

        global_ops = []

        def __append_optimize_op__(op, block, grad_to_block_id, merged_var,
                                   lr_ops):
            if self._is_optimizer_op(op):
                self._append_pserver_ops(block, op, endpoint, grad_to_block_id,
                                         self.origin_program, merged_var)
            elif op not in lr_ops:
                self._append_pserver_non_opt_ops(block, op)

        def __op_have_grad_input__(op):
            for varname in op.input_arg_names:
                if varname.find("@GRAD") >= 0:
                    return varname
            return ""

        def __clone_lr_op_sub_block__(op, program, lr_block):
            if not op.has_attr('sub_block'):
                return

            origin_block_desc = op.attr('sub_block')
            origin_block = self.origin_program.block(origin_block_desc.id)
            assert isinstance(origin_block, Block)
            # we put the new sub block to new block to follow the block
            # hierarchy of the original blocks
            new_sub_block = program._create_block(lr_block.idx)

            # clone vars
            for var in origin_block.vars:
                new_sub_block._clone_variable(var)

            # clone ops
            for origin_op in origin_block.ops:
                cloned_op = self._clone_lr_op(program, new_sub_block, origin_op)
                # clone sub_block of op
                __clone_lr_op_sub_block__(cloned_op, program, new_sub_block)

            # reset the block of op
            op._set_attr('sub_block', new_sub_block)

        # append lr decay ops to the child block if exists
        lr_ops = self._get_lr_ops()
        # record optimize blocks and we can run them on pserver parallel
        optimize_blocks = []
        if len(lr_ops) > 0:
            lr_decay_block = pserver_program._create_block(
                pserver_program.num_blocks - 1)
            optimize_blocks.append(lr_decay_block)
            for _, op in enumerate(lr_ops):
                cloned_op = self._append_pserver_non_opt_ops(lr_decay_block, op)
                # append sub blocks to pserver_program in lr_decay_op
                __clone_lr_op_sub_block__(cloned_op, pserver_program,
                                          lr_decay_block)

        # append op to the current block
        grad_to_block_id = []
        pre_block_idx = pserver_program.num_blocks - 1
        for idx, opt_op in enumerate(opt_op_on_pserver):
            per_opt_block = pserver_program._create_block(pre_block_idx)
            optimize_blocks.append(per_opt_block)
            optimize_target_param_name = opt_op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
            # append grad merging ops before clip and weight decay
            # e.g. merge grad -> L2Decay op -> clip op -> optimize
            merged_var = None
            for _, op in enumerate(self.optimize_ops):
                # find the origin grad var before clipping/L2Decay,
                # merged_var should be the input var name of L2Decaybuil
                grad_varname_for_block = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                if op.attr(OP_ROLE_VAR_ATTR_NAME)[
                        0] == optimize_target_param_name:
                    merged_var = self._append_pserver_grad_merge_ops(
                        per_opt_block, grad_varname_for_block, endpoint,
                        grad_to_block_id, self.origin_program)
                    if merged_var:
                        break  # append optimize op once then append other ops.
            if merged_var:
                for _, op in enumerate(self.optimize_ops):
                    # optimizer is connected to itself
                    if op.attr(OP_ROLE_VAR_ATTR_NAME)[0] == optimize_target_param_name and \
                            op not in global_ops:
                        log("append opt op: ", op.type, op.input_arg_names,
                            merged_var)
                        __append_optimize_op__(op, per_opt_block,
                                               grad_to_block_id, merged_var,
                                               lr_ops)

        # dedup grad to ids list
        grad_to_block_id = list(set(grad_to_block_id))
        # append global ops
        if global_ops:
            opt_state_block = pserver_program._create_block(
                pserver_program.num_blocks - 1)
            optimize_blocks.append(opt_state_block)
            for glb_op in global_ops:
                __append_optimize_op__(glb_op, opt_state_block,
                                       grad_to_block_id, None, lr_ops)

        # process distributed lookup_table
        prefetch_var_name_to_block_id = []
        if self.has_distributed_lookup_table:
            pserver_index = self.pserver_endpoints.index(endpoint)
            table_opt_block = self._create_table_optimize_block(
                pserver_index, pserver_program, pre_block_idx, grad_to_block_id)
            optimize_blocks.append(table_opt_block)
            lookup_table_var_name_to_block_id = self._create_prefetch_block(
                pserver_index, pserver_program, table_opt_block)
            checkpoint_block_id = self._create_checkpoint_save_block(
                pserver_program, table_opt_block.idx)

            pserver_program._distributed_lookup_table = self.table_name
            prefetch_var_name_to_block_id.extend(
                lookup_table_var_name_to_block_id)

        attrs = {
            "optimize_blocks": optimize_blocks,
            "endpoint": endpoint,
            "Fanin": self.trainer_num,
            "sync_mode": self.sync_mode,
            "grad_to_block_id": grad_to_block_id,
        }

        if self.has_distributed_lookup_table:
            attrs['checkpint_block_id'] = checkpoint_block_id

        if len(prefetch_var_name_to_block_id) > 0:
            attrs[
                'prefetch_var_name_to_block_id'] = prefetch_var_name_to_block_id

        # step5 append the listen_and_serv op
        pserver_program.global_block().append_op(
            type="listen_and_serv",
            inputs={'X': recv_inputs},
            outputs={},
            attrs=attrs)

        # add distributed attrs
        pserver_program._slice_vars_and_attrs = self._get_slice_vars_and_attrs(
            endpoint)

        pserver_program._sync_with_cpp()
        # save pserver program to generate pserver side startup relatively.
        self.pserver_program = pserver_program
        return pserver_program

    def get_pserver_programs(self, endpoint):
        """
        Get pserver side main program and startup program for distributed training.

        Args:
            endpoint (str): current pserver endpoint.

        Returns:
            tuple: (main_program, startup_program), of type "Program"
        """
        pserver_prog = self.get_pserver_program(endpoint)
        pserver_startup = self.get_startup_program(
            endpoint, pserver_program=pserver_prog)
        return pserver_prog, pserver_startup

    def get_startup_program(self,
                            endpoint,
                            pserver_program=None,
                            startup_program=None):
        """
        **Deprecated**

        Get startup program for current parameter server.
        Modify operator input variables if there are variables that
        were split to several blocks.

        Args:
            endpoint (str): current pserver endpoint.
            pserver_program (Program): deprecated, call get_pserver_program first.
            startup_program (Program): deprecated, should pass startup_program
                when initalizing

        Returns:
            Program: parameter server side startup program.
        """
        sys.stderr.write("get_startup_program() is deprecated, call \
get_pserver_programs() to get pserver main and startup \
in a single call.")
        if pserver_program != None:
            sys.stderr.write("passing pserver_program to get_startup_program() \
is deprecated, you can use new API get_pserver_programs() to \
get both pserver main program and startup program.")
        if startup_program != None:
            sys.stderr.write("passing startup_program to get_startup_program() \
is deprecated, use fluid.program_guard() or pass this argument \
to transpile() call.")

        s_prog = Program()
        orig_s_prog = self.startup_program
        s_prog.random_seed = orig_s_prog.random_seed
        params = self.param_grad_ep_mapping[endpoint]["params"]

        def _get_splited_name_and_shape(varname):
            for idx, splited_param in enumerate(params):
                pname = splited_param.name
                if same_or_split_var(pname, varname) and varname != pname:
                    return pname, splited_param.shape
            return "", []

        # 1. create vars in pserver program to startup program
        pserver_vars = pserver_program.global_block().vars
        created_var_map = collections.OrderedDict()
        for _, var in six.iteritems(pserver_vars):
            tmpvar = s_prog.global_block()._clone_variable(var)
            created_var_map[var.name] = tmpvar

        # 2. rename op outputs
        for op in orig_s_prog.global_block().ops:
            new_outputs = collections.OrderedDict()
            # do not append startup op if var is not on this pserver
            op_on_pserver = False
            # TODO(gongwb): remove this line.
            if op.type not in ["recv", "fetch_barrier", "concat"]:
                for key in op.output_names:
                    newname, _ = _get_splited_name_and_shape(op.output(key)[0])
                    if newname:
                        op_on_pserver = True
                        new_outputs[key] = created_var_map[newname]
                    elif op.output(key)[0] in pserver_vars:
                        op_on_pserver = True
                        new_outputs[key] = pserver_vars[op.output(key)[0]]

            if op_on_pserver:
                # most startup program ops have no inputs
                new_inputs = self._get_input_map_from_op(pserver_vars, op)

                if op.type in [
                        "gaussian_random", "fill_constant", "uniform_random"
                ]:
                    op._set_attr("shape", list(new_outputs["Out"].shape))
                s_prog.global_block().append_op(
                    type=op.type,
                    inputs=new_inputs,
                    outputs=new_outputs,
                    attrs=op.all_attrs())

        # add slice vars
        s_prog._slice_vars_and_attrs = self._get_slice_vars_and_attrs(endpoint)

        return s_prog

    def _get_slice_vars_and_attrs(self, endpoint):
        slice_vars_and_attrs = []
        block_suffix = "block"
        for param in self.param_grad_ep_mapping[endpoint]["params"]:
            orig_var_name, block_name, _ = self._get_varname_parts(param.name)
            if not block_name:
                continue

            block_idx = int(block_name.split(block_suffix)[1])
            orig_var = self.origin_program.global_block().vars[orig_var_name]

            skip_numel = 0
            slice_vars = self.param_var_mapping[orig_var_name]
            for slice_var in slice_vars[:block_idx]:
                skip_numel += reduce(lambda x, y: x * y, slice_var.shape)
            slice_vars_and_attrs.append([orig_var, skip_numel, param])

        return slice_vars_and_attrs

    # ====================== private transpiler functions =====================

    def _has_distributed_lookup_table(self):
        # process lookup_table_op
        # 1. check all lookup_table_op is distributed
        # 2. check all lookup_table_op share the same table.
        distributed_lookup_table_ops = []
        # support only one distributed_lookup_table now
        self.table_name = None
        for op in self.origin_program.global_block().ops:
            if op.type == LOOKUP_TABLE_TYPE:
                if op.attr('is_distributed') is True:
                    if self.table_name is None:
                        self.table_name = op.input("W")[0]
                    if self.table_name != op.input("W")[0]:
                        raise RuntimeError("all distributed lookup_table_ops"
                                           " should have only one table")
                    distributed_lookup_table_ops.append(op)
                else:
                    if self.table_name is not None:
                        assert op.input("W")[0] != self.table_name

        return len(distributed_lookup_table_ops) > 0

    def _update_dist_lookup_table_vars(self, param_list, grad_list,
                                       params_grads):
        # TODO(wuyi): put find a way to put dist lookup table stuff all together.
        # update self.table_param_grad and self.trainer_side_table_grad_list
        program = self.origin_program
        if self.has_distributed_lookup_table:
            param_list = [
                param for param in param_list if param.name != self.table_name
            ]
            grad_list = [
                grad for grad in grad_list
                if grad.name != grad_var_name(self.table_name)
            ]
            self.table_param_grad = [
                param_grad for param_grad in params_grads
                if param_grad[0].name == self.table_name
            ][0]
            table_grad_var = self.table_param_grad[1]
            if self.sync_mode:
                self.trainer_side_table_grad_list = [
                    program.global_block().create_var(
                        name="%s.trainer_%d.pserver_%d" %
                        (table_grad_var.name, self.trainer_id, index),
                        type=table_grad_var.type,
                        shape=table_grad_var.shape,
                        dtype=table_grad_var.dtype)
                    for index in range(len(self.pserver_endpoints))
                ]
            else:
                self.trainer_side_table_grad_list = [
                    program.global_block().create_var(
                        name="%s.pserver_%d" % (table_grad_var.name, index),
                        type=table_grad_var.type,
                        shape=table_grad_var.shape,
                        dtype=table_grad_var.dtype)
                    for index in range(len(self.pserver_endpoints))
                ]
        return param_list, grad_list

    def _init_splited_vars(self):
        # update these mappings for further transpile:
        # 1. param_var_mapping: param var name -> [splited params vars]
        # 2. grad_var_mapping: grad var name -> [splited grads vars]
        # 3. grad_param_mapping: grad.blockx -> param.blockx
        # 4. param_grad_ep_mapping: ep -> {"params": [], "grads": []}

        param_list = []
        grad_list = []
        param_grad_set = set()
        for p, g in self.params_grads:
            # skip parameter marked not trainable
            if type(p) == Parameter and p.trainable == False:
                continue
            if p.name not in param_grad_set:
                param_list.append(p)
                param_grad_set.add(p.name)
            if g.name not in param_grad_set:
                grad_list.append(g)
                param_grad_set.add(g.name)

        param_list, grad_list = self._update_dist_lookup_table_vars(
            param_list, grad_list, self.params_grads)

        if self.config.slice_var_up:
            # when we slice var up into blocks, we will slice the var according to
            # pserver services' count. A pserver may have two or more listening ports.
            grad_blocks = slice_variable(grad_list,
                                         len(self.pserver_endpoints),
                                         self.config.min_block_size)
            param_blocks = slice_variable(param_list,
                                          len(self.pserver_endpoints),
                                          self.config.min_block_size)
        else:
            # when we do NOT slice var up into blocks, we will always slice params
            # grads into one block.
            grad_blocks = slice_variable(grad_list, 1,
                                         self.config.min_block_size)
            param_blocks = slice_variable(param_list, 1,
                                          self.config.min_block_size)
        assert (len(grad_blocks) == len(param_blocks))

        # origin_param_name -> [splited_param_vars]
        self.param_var_mapping = self._create_vars_from_blocklist(
            self.origin_program, param_blocks)
        # origin_grad_name -> [splited_grad_vars]
        self.grad_var_mapping = self._create_vars_from_blocklist(
            self.origin_program,
            grad_blocks,
            add_trainer_suffix=self.trainer_num > 1)
        # dict(grad_splited_var -> param_splited_var)
        self.grad_param_mapping = collections.OrderedDict()
        for g, p in zip(grad_blocks, param_blocks):
            g_name, g_bid, _ = g.split(":")
            p_name, p_bid, _ = p.split(":")
            self.grad_param_mapping[self.grad_var_mapping[g_name][int(g_bid)]] = \
                self.param_var_mapping[p_name][int(p_bid)]

        # create mapping of endpoint -> split var to create pserver side program
        self.param_grad_ep_mapping = collections.OrderedDict()
        [
            self.param_grad_ep_mapping.update({
                ep: {
                    "params": [],
                    "grads": []
                }
            }) for ep in self.pserver_endpoints
        ]

    # transpiler function for dis lookup_table
    def _replace_lookup_table_op_with_prefetch(self, program,
                                               pserver_endpoints):
        # 1. replace lookup_table_op with split_ids_op -> prefetch_op -> sum_op
        self.all_in_ids_vars = []
        self.all_prefetch_input_vars = []
        self.all_prefetch_output_vars = []
        self.all_out_emb_vars = []
        lookup_table_op_index = -1

        continue_search_lookup_table_op = True
        while continue_search_lookup_table_op:
            continue_search_lookup_table_op = False
            all_ops = program.global_block().ops
            for op in all_ops:
                if op.type == LOOKUP_TABLE_TYPE:
                    continue_search_lookup_table_op = True

                    lookup_table_op_index = lookup_table_op_index if lookup_table_op_index != -1 else list(
                        all_ops).index(op)
                    ids_name = op.input("Ids")
                    out_name = op.output("Out")

                    ids_var = program.global_block().vars[ids_name[0]]
                    self.all_in_ids_vars.append(ids_var)

                    out_var = program.global_block().vars[out_name[0]]
                    self.all_out_emb_vars.append(out_var)

                    # delete lookup_table_op
                    delete_ops(program.global_block(), [op])
                    # break for loop
                    break

        for index in range(len(self.pserver_endpoints)):
            in_var = program.global_block().create_var(
                name=str("prefetch_compress_in_tmp_" + str(index)),
                type=self.all_in_ids_vars[0].type,
                shape=self.all_in_ids_vars[0].shape,
                dtype=self.all_in_ids_vars[0].dtype)
            self.all_prefetch_input_vars.append(in_var)

            out_var = program.global_block().create_var(
                name=str("prefetch_compress_out_tmp_" + str(index)),
                type=self.all_out_emb_vars[0].type,
                shape=self.all_out_emb_vars[0].shape,
                dtype=self.all_out_emb_vars[0].dtype)
            self.all_prefetch_output_vars.append(out_var)

        # insert split_ids_op
        program.global_block()._insert_op(
            index=lookup_table_op_index,
            type="split_ids",
            inputs={'Ids': self.all_in_ids_vars},
            outputs={"Out": self.all_prefetch_input_vars})

        # insert prefetch_op
        program.global_block()._insert_op(
            index=lookup_table_op_index + 1,
            type="prefetch",
            inputs={'X': self.all_prefetch_input_vars},
            outputs={"Out": self.all_prefetch_output_vars},
            attrs={
                "epmap": pserver_endpoints,
                # FIXME(qiao) temporarily disable this config because prefetch
                # is not act as other rpc op, it's more like a forward op
                # RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE
            })

        # insert concat_op
        program.global_block()._insert_op(
            index=lookup_table_op_index + 2,
            type="merge_ids",
            inputs={
                'Ids': self.all_in_ids_vars,
                'Rows': self.all_prefetch_input_vars,
                'X': self.all_prefetch_output_vars
            },
            outputs={"Out": self.all_out_emb_vars})

    def _split_table_grad_and_add_send_vars(self, program, pserver_endpoints):
        # 2. add split_ids_op and send_op to send gradient to pservers

        # there should only be one table_name
        all_ops = program.global_block().ops
        table_grad_name = grad_var_name(self.table_name)
        for op in all_ops:
            if table_grad_name in op.output_arg_names:
                op_index = list(all_ops).index(op)
                # insert split_ids_op
                program.global_block()._insert_op(
                    index=op_index + 1,
                    type="split_ids",
                    inputs={
                        'Ids': [program.global_block().vars[table_grad_name]]
                    },
                    outputs={"Out": self.trainer_side_table_grad_list},
                    attrs={RPC_OP_ROLE_ATTR_NAME: DIST_OP_ROLE_ATTR_VALUE})
                program.global_block()._insert_op(
                    index=op_index + 2,
                    type="send",
                    inputs={'X': self.trainer_side_table_grad_list},
                    outputs={
                        'Out':
                        [self.grad_name_to_send_dummy_out[self.table_name]]
                        if self.sync_mode else []
                    },
                    attrs={
                        "sync_mode": not self.sync_mode,
                        "epmap": pserver_endpoints,
                        RPC_OP_ROLE_ATTR_NAME: RPC_OP_ROLE_ATTR_VALUE,
                        OP_ROLE_VAR_ATTR_NAME: [
                            self.grad_name_to_param_name[table_grad_name],
                            table_grad_name
                        ]
                    })
                break

    def _create_prefetch_block(self, pserver_index, pserver_program,
                               optimize_block):
        # STEP: create prefetch block
        table_var = pserver_program.global_block().vars[self.table_name]
        prefetch_var_name_to_block_id = []
        prefetch_block = pserver_program._create_block(optimize_block.idx)
        trainer_ids = self.all_prefetch_input_vars[pserver_index]
        pserver_ids = pserver_program.global_block().create_var(
            name=trainer_ids.name,
            type=trainer_ids.type,
            shape=trainer_ids.shape,
            dtype=trainer_ids.dtype)
        trainer_out = self.all_prefetch_output_vars[pserver_index]
        pserver_out = pserver_program.global_block().create_var(
            name=trainer_out.name,
            type=trainer_out.type,
            shape=trainer_out.shape,
            dtype=trainer_out.dtype)
        prefetch_block.append_op(
            type="lookup_sparse_table",
            inputs={'Ids': pserver_ids,
                    "W": table_var},
            outputs={"Out": pserver_out},
            attrs={
                "is_sparse": True,  # has no effect on lookup_table op
                "is_distributed": True,
                "padding_idx": -1
            })
        prefetch_var_name_to_block_id.append(trainer_ids.name + ":" + str(
            prefetch_block.idx))
        return prefetch_var_name_to_block_id

    def _create_table_optimize_block(self, pserver_index, pserver_program,
                                     pre_block_idx, grad_to_block_id):
        # STEP: create table optimize block
        table_opt_block = pserver_program._create_block(pre_block_idx)
        # create table param and grad var in pserver program
        # create table optimize block in pserver program
        table_opt_op = [
            op for op in self.optimize_ops
            if 'Param' in op.input_names and op.input("Param")[0] ==
            self.table_name
        ][0]

        origin_param_var = self.origin_program.global_block().vars[
            self.table_name]

        zero_dim = int(math.ceil(origin_param_var.shape[0] * 100 / float(len(self.pserver_endpoints))))
        table_shape = list(origin_param_var.shape)
        table_shape[0] = zero_dim

        param_var = pserver_program.global_block().create_var(
            name=origin_param_var.name,
            shape=table_shape,
            dtype=origin_param_var.dtype,
            type=core.VarDesc.VarType.SELECTED_ROWS,
            persistable=True)

        # parameter must be selected rows
        param_var.desc.set_type(core.VarDesc.VarType.SELECTED_ROWS)
        grad_var = pserver_program.global_block()._clone_variable(
            self.origin_program.global_block().vars[grad_var_name(
                self.table_name)])

        lr_var = pserver_program.global_block()._clone_variable(
            self.origin_program.global_block().vars[table_opt_op.input(
                "LearningRate")[0]])

        if self.sync_mode:
            # create grad vars in pserver program
            table_grad_var = self.table_param_grad[1]
            pserver_side_table_grad_list = [
                pserver_program.global_block().create_var(
                    name="%s.trainer_%d.pserver_%d" %
                    (table_grad_var.name, index, pserver_index),
                    type=table_grad_var.type,
                    shape=table_grad_var.shape,
                    dtype=table_grad_var.dtype)
                for index in range(self.trainer_num)
            ]

            # append sum op for pserver_side_table_grad_list
            table_opt_block.append_op(
                type="sum",
                inputs={"X": pserver_side_table_grad_list},
                outputs={"Out": [grad_var]},
                attrs={"use_mkldnn": False})
        else:
            # in async_mode, for table gradient, it also need to be splited to each parameter server
            origin_grad_name = grad_var.name
            splited_grad_name = self.trainer_side_table_grad_list[
                pserver_index].name
            if not splited_grad_name.startswith(origin_grad_name):
                raise ValueError("origin_grad_var: " + splited_grad_name +
                                 " grad_var:" + grad_var.name)
            grad_var = pserver_program.global_block()._rename_var(
                origin_grad_name, splited_grad_name)

        inputs = {
            "Param": [param_var],
            "Grad": [grad_var],
            "LearningRate": [lr_var]
        }
        outputs = {"ParamOut": [param_var]}
        # only support sgd now
        import logging
        logging.warn(
            "distribute lookup table only support sgd optimizer, change it's optimizer to sgd instead of "
            + table_opt_op.type)
        table_opt_block.append_op(type="sgd", inputs=inputs, outputs=outputs)

        # add table parameter gradient and it's block id to grad_to_block_id
        grad_to_block_id.append(grad_var.name + ":" + str(table_opt_block.idx))

        return table_opt_block

    def _create_checkpoint_save_block(self, pserver_program, pre_block_idx):
        """
        create a new block to handle save checkpoint.
        """
        import os

        pserver_program.global_block().create_var(
            name="kLookupTablePath",
            persistable=True,
            type=core.VarDesc.VarType.RAW)

        checkpoint_save_block = pserver_program._create_block(pre_block_idx)
        # this 'file_path' do not be used in save lookup table variable
        checkpoint_save_block.append_op(
            type='save',
            inputs={'X': [self.table_name]},
            outputs={},
            attrs={'file_path': "none"})

        return checkpoint_save_block.idx

    def _create_vars_from_blocklist(self,
                                    program,
                                    block_list,
                                    add_trainer_suffix=False):
        """
        Create vars for each split.
        NOTE: only grads need to be named for different trainers, use
              add_trainer_suffix to rename the grad vars.
        Args:
            program (ProgramDesc): ProgramDesc which gradients blong.
            block_list (list[(varname, block_id, block_size)]): List of gradient blocks.
            add_trainer_suffix (Bool): Add trainer suffix to new variable's name if set True.
        Returns:
            var_mapping (collections.OrderedDict(varname->[new_varname_variable])):A dict mapping
                from original var name to each var split.
        """

        # varname->[(block_id, current_block_size)]
        block_map = collections.OrderedDict()

        var_mapping = collections.OrderedDict()
        for block_str in block_list:
            varname, offset, size = block_str.split(":")
            if varname not in block_map:
                block_map[varname] = []
            block_map[varname].append((int(offset), int(size)))

        for varname, splited in six.iteritems(block_map):
            orig_var = program.global_block().var(varname)
            if len(splited) == 1:
                if self.sync_mode and add_trainer_suffix:
                    new_var_name = "%s.trainer_%d" % \
                                   (orig_var.name, self.trainer_id)
                    program.global_block()._rename_var(varname, new_var_name)
                    var_mapping[varname] = \
                        [program.global_block().var(new_var_name)]
                else:
                    var_mapping[varname] = \
                        [program.global_block().var(orig_var.name)]
                continue
            var_mapping[varname] = []
            orig_shape = orig_var.shape
            orig_dim1_flatten = 1
            if len(orig_shape) >= 2:
                orig_dim1_flatten = reduce(lambda x, y: x * y, orig_shape[1:])

            for i, block in enumerate(splited):
                size = block[1]
                rows = size // orig_dim1_flatten
                splited_shape = [rows]
                if len(orig_shape) >= 2:
                    splited_shape.extend(orig_shape[1:])
                new_var_name = ""
                if self.sync_mode and add_trainer_suffix:
                    new_var_name = "%s.block%d.trainer_%d" % \
                                   (varname, i, self.trainer_id)
                else:
                    new_var_name = "%s.block%d" % \
                                   (varname, i)
                var = program.global_block().create_var(
                    name=new_var_name,
                    persistable=False,
                    dtype=orig_var.dtype,
                    type=orig_var.type,
                    shape=splited_shape)  # flattend splited var
                var_mapping[varname].append(var)
            program.global_block()._sync_with_cpp()
        return var_mapping

    def _clone_var(self, block, var, persistable=True):
        return block.create_var(
            name=var.name,
            shape=var.shape,
            dtype=var.dtype,
            type=var.type,
            lod_level=var.lod_level,
            persistable=persistable)

    def _insert_split_op(self, program, orig_var, index, splited_vars):
        if orig_var.type == core.VarDesc.VarType.SELECTED_ROWS:
            height_sections = []
            for v in splited_vars:
                height_sections.append(v.shape[0])
            program.global_block()._insert_op(
                index=index + 1,
                type="split_selected_rows",
                inputs={"X": orig_var},
                outputs={"Out": splited_vars},
                attrs={
                    "height_sections": height_sections,
                    RPC_OP_ROLE_ATTR_NAME: DIST_OP_ROLE_ATTR_VALUE
                })
        elif orig_var.type == core.VarDesc.VarType.LOD_TENSOR:
            sections = []
            for v in splited_vars:
                sections.append(v.shape[0])
            program.global_block()._insert_op(
                index=index + 1,
                type="split_byref",
                inputs={"X": orig_var},
                outputs={"Out": splited_vars},
                attrs={
                    "sections": sections,
                    RPC_OP_ROLE_ATTR_NAME: DIST_OP_ROLE_ATTR_VALUE
                })
        else:
            AssertionError("Variable type should be in set "
                           "[LOD_TENSOR, SELECTED_ROWS]")

    def _get_optimizer_input_shape(self, op_type, varkey, orig_shape,
                                   param_shape):
        """
        Returns the shape for optimizer inputs that need to be reshaped when
        Param and Grad is split to multiple servers.
        """
        # HACK(typhoonzero): Should use functions of corresponding optimizer in
        # optimizer.py to get the shape, do not  bind this in the transpiler.
        if op_type == "adam":
            if varkey in ["Moment1", "Moment2"]:
                return param_shape
        elif op_type == "adagrad":
            if varkey == "Moment":
                return param_shape
        elif op_type == "adamax":
            if varkey in ["Moment", "InfNorm"]:
                return param_shape
        elif op_type == "momentum":
            if varkey == "Velocity":
                return param_shape
        elif op_type == "rmsprop":
            if varkey in ["Moment", "MeanSquare"]:
                return param_shape
        elif op_type == "decayed_adagrad":
            if varkey == "Moment":
                return param_shape
        elif op_type == "sgd":
            pass
        return orig_shape

    def _get_varname_parts(self, varname):
        # returns origin, blockid, trainerid
        orig_var_name = ""
        trainer_part = ""
        block_part = ""
        trainer_idx = varname.find(".trainer_")
        if trainer_idx >= 0:
            trainer_part = varname[trainer_idx + 1:]
        else:
            trainer_idx = len(varname)
        block_index = varname.find(".block")
        if block_index >= 0:
            block_part = varname[block_index + 1:trainer_idx]
        else:
            block_index = len(varname)
        orig_var_name = varname[0:min(block_index, trainer_idx)]
        return orig_var_name, block_part, trainer_part

    def _orig_varname(self, varname):
        orig, _, _ = self._get_varname_parts(varname)
        return orig

    def _append_pserver_grad_merge_ops(self, optimize_block,
                                       grad_varname_for_block, endpoint,
                                       grad_to_block_id, origin_program):
        program = optimize_block.program
        pserver_block = program.global_block()
        grad_block = None
        for g in self.param_grad_ep_mapping[endpoint]["grads"]:
            if self._orig_varname(g.name) == \
                    self._orig_varname(grad_varname_for_block):
                grad_block = g
                break
        if not grad_block:
            # do not append this op if current endpoint
            # is not dealing with this grad block
            return None
        orig_varname, block_name, trainer_name = self._get_varname_parts(
            grad_block.name)
        if block_name:
            merged_var_name = '.'.join([orig_varname, block_name])
        else:
            merged_var_name = orig_varname

        merged_var = pserver_block.vars[merged_var_name]
        grad_to_block_id.append(merged_var.name + ":" + str(optimize_block.idx))
        if self.sync_mode and self.trainer_num > 1:
            vars2merge = []
            for i in range(self.trainer_num):
                per_trainer_name = "%s.trainer_%d" % \
                                   (merged_var_name, i)
                vars2merge.append(pserver_block.vars[per_trainer_name])
            optimize_block.append_op(
                type="sum",
                inputs={"X": vars2merge},
                outputs={"Out": merged_var},
                attrs={"use_mkldnn": False})
            optimize_block.append_op(
                type="scale",
                inputs={"X": merged_var},
                outputs={"Out": merged_var},
                attrs={"scale": 1.0 / float(self.trainer_num)})
        return merged_var

    def _append_pserver_ops(self, optimize_block, opt_op, endpoint,
                            grad_to_block_id, origin_program, merged_var):
        program = optimize_block.program
        pserver_block = program.global_block()
        new_inputs = collections.OrderedDict()

        def _get_param_block(opt_op):
            # param is already created on global program
            param_block = None
            for p in self.param_grad_ep_mapping[endpoint]["params"]:
                if same_or_split_var(p.name, opt_op.input("Param")[0]):
                    param_block = p
                    break
            return param_block

        for key in opt_op.input_names:
            if key == "Grad":
                new_inputs[key] = merged_var
            elif key == "Param":
                param_block = _get_param_block(opt_op)
                if not param_block:
                    return
                tmpvar = pserver_block.create_var(
                    name=param_block.name,
                    persistable=True,
                    dtype=param_block.dtype,
                    shape=param_block.shape)
                new_inputs[key] = tmpvar
            elif key == "LearningRate":
                # learning rate variable has already be created by non-optimize op,
                # don't create it once again.
                lr_varname = opt_op.input(key)[0]
                if lr_varname in pserver_block.vars:
                    new_inputs[key] = pserver_block.vars[opt_op.input(key)[0]]
                else:
                    origin_var = origin_program.global_block().vars[lr_varname]
                    tmpvar = pserver_block.create_var(
                        name=origin_var.name,
                        persistable=origin_var.persistable,
                        dtype=origin_var.dtype,
                        shape=origin_var.shape)
                    new_inputs[key] = tmpvar

        for key in opt_op.input_names:
            new_shape = None
            if key in ["Param", "Grad", "LearningRate"]:
                continue
            var = self.origin_program.global_block().vars[opt_op.input(key)[0]]
            # update accumulator variable shape
            param_shape = new_inputs["Param"].shape
            new_shape = self._get_optimizer_input_shape(opt_op.type, key,
                                                        var.shape, param_shape)
            tmpvar = pserver_block.create_var(
                name=var.name,
                persistable=var.persistable,
                dtype=var.dtype,
                shape=new_shape)
            new_inputs[key] = tmpvar

        # change output's ParamOut variable
        outputs = self._get_output_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        outputs["ParamOut"] = new_inputs["Param"]
        optimize_block.append_op(
            type=opt_op.type,
            inputs=new_inputs,
            outputs=outputs,
            attrs=opt_op.all_attrs())

    def _is_splited_grad_var(self, var, var_dict):
        grad_block = None
        for _, g in six.iteritems(var_dict):
            if self._orig_varname(g.name) == self._orig_varname(var.name):
                if g.name.find(".trainer_") == -1:
                    grad_block = g
                    break
        return grad_block

    def _clone_lr_op(self, program, block, op):
        inputs = self._get_input_map_from_op(
            self.origin_program.global_block().vars, op)
        for key, varlist in six.iteritems(inputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                if var not in program.global_block().vars:
                    block._clone_variable(var)

        outputs = self._get_output_map_from_op(
            self.origin_program.global_block().vars, op)
        for key, varlist in six.iteritems(outputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                if var not in program.global_block().vars:
                    block._clone_variable(var)

        return block.append_op(
            type=op.type, inputs=inputs, outputs=outputs, attrs=op.all_attrs())

    def _append_pserver_non_opt_ops(self, optimize_block, opt_op):
        program = optimize_block.program
        # Append the ops for parameters that do not need to be optimized/updated
        inputs = self._get_input_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        for key, varlist in six.iteritems(inputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                # for ops like clipping and weight decay, get the splited var
                # for inputs/outputs
                grad_block = self._is_splited_grad_var(
                    var, program.global_block().vars)
                if grad_block:
                    inputs[key] = grad_block
                elif var.name not in program.global_block().vars:
                    program.global_block().create_var(
                        name=var.name,
                        persistable=var.persistable,
                        dtype=var.dtype,
                        shape=var.shape)

        outputs = self._get_output_map_from_op(
            self.origin_program.global_block().vars, opt_op)
        for key, varlist in six.iteritems(outputs):
            if not isinstance(varlist, list):
                varlist = [varlist]
            for var in varlist:
                grad_block = self._is_splited_grad_var(
                    var, program.global_block().vars)
                if grad_block:
                    outputs[key] = grad_block
                elif var.name not in program.global_block().vars:
                    program.global_block()._clone_variable(var)

        return optimize_block.append_op(
            type=opt_op.type,
            inputs=inputs,
            outputs=outputs,
            attrs=opt_op.all_attrs())

    def _is_op_connected(self, op1, op2):
        # If one op's input is another op's output or
        # one op's output is another op's input, we say
        # the two operator is connected.
        if set(op1.desc.output_arg_names()) & set(op2.desc.input_arg_names()) or \
                set(op1.desc.input_arg_names()) & set(op2.desc.output_arg_names()):
            return True
        return False

    def _create_ufind(self, optimize_ops):
        # Create a unit find data struct by optimize ops
        ufind = UnionFind(optimize_ops)
        for i in range(len(optimize_ops)):
            for j in range(i, len(optimize_ops)):
                op1 = optimize_ops[i]
                op2 = optimize_ops[j]
                if self._is_op_connected(op1, op2):
                    ufind.union(op1, op2)
        return ufind

    def _is_optimizer_op(self, op):
        if "Param" in op.input_names and \
                "LearningRate" in op.input_names:
            return True
        return False

    def _is_opt_op_on_pserver(self, endpoint, op):
        param_names = [
            p.name for p in self.param_grad_ep_mapping[endpoint]["params"]
        ]
        if op.input("Param")[0] in param_names:
            return True
        else:
            for n in param_names:
                param = op.input("Param")[0]
                if same_or_split_var(n, param) and n != param:
                    return True
            return False

    def _get_input_map_from_op(self, varmap, op):
        """Returns a dict from op input name to the vars in varmap."""
        iomap = collections.OrderedDict()
        for key in op.input_names:
            vars = []
            for varname in op.input(key):
                vars.append(varmap[varname])
            if len(vars) == 1:
                iomap[key] = vars[0]
            else:
                iomap[key] = vars
        return iomap

    def _get_output_map_from_op(self, varmap, op):
        """Returns a dict from op output name to the vars in varmap."""
        iomap = collections.OrderedDict()
        for key in op.output_names:
            vars = []
            for varname in op.output(key):
                vars.append(varmap[varname])
            if len(vars) == 1:
                iomap[key] = vars[0]
            else:
                iomap[key] = vars
        return iomap

    def _get_lr_ops(self):
        lr_ops = []
        block = self.origin_program.global_block()
        for op in block.ops:
            role_id = int(op.attr(RPC_OP_ROLE_ATTR_NAME))
            if role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) or \
                role_id == int(LR_SCHED_OP_ROLE_ATTR_VALUE) | \
                    int(OPT_OP_ROLE_ATTR_VALUE):
                lr_ops.append(op)
                log("append lr op: ", op.type)
        return lr_ops

    def _get_lr_ops_deprecated(self):
        lr_ops = []
        # find learning rate variables by optimize op
        lr_vars = set()
        for op in self.optimize_ops:
            if self._is_optimizer_op(op):
                lr_vars.add(op.input("LearningRate")[0])

        find_ops = []
        # find ops which output is lr var
        block = self.origin_program.global_block()
        for op in block.ops:
            if set(op.output_arg_names) & lr_vars:
                find_ops.append(op)
        # make a union find struct by the ops in default_main_program
        ufind = UnionFind(block.ops)

        for op1 in block.ops:
            for op2 in block.ops:
                # NOTE: we need to skip all optimize ops, since it is connected
                # with forward/backward ops and lr ops, we only need the lr ops.
                if op1 != op2 and self._is_op_connected(op1, op2) and \
                        not self._is_optimizer_op(op1) and not self._is_optimizer_op(op2):
                    ufind.union(op1, op2)
        # find all ops which is related with lr var
        for op1 in block.ops:
            for op2 in find_ops:
                if ufind.is_connected(op1, op2):
                    lr_ops.append(op1)
                    # we only need to append op for once
                    break
        return lr_ops

    def _is_opt_role_op(self, op):
        # NOTE: depend on oprole to find out whether this op is for
        # optimize
        op_maker = core.op_proto_and_checker_maker
        optimize_role = core.op_proto_and_checker_maker.OpRole.Optimize
        if op_maker.kOpRoleAttrName() in op.attr_names and \
                int(op.all_attrs()[op_maker.kOpRoleAttrName()]) == int(optimize_role):
            return True
        return False

    def _get_optimize_pass(self):
        """
        Get optimizer operators, parameters and gradients from origin_program
        Returns:
            opt_ops (list): optimize operators.
            params_grads (dict): paramter->gradient.
        """
        block = self.origin_program.global_block()
        opt_ops = []
        params_grads = []
        # tmp set to dedup
        optimize_params = set()
        origin_var_dict = self.origin_program.global_block().vars
        for op in block.ops:
            if self._is_opt_role_op(op):
                opt_ops.append(op)
                if op.attr(OP_ROLE_VAR_ATTR_NAME):
                    param_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[0]
                    grad_name = op.attr(OP_ROLE_VAR_ATTR_NAME)[1]
                    if not param_name in optimize_params:
                        optimize_params.add(param_name)
                        log("adding param_grad pair: ", param_name, grad_name)
                        params_grads.append([
                            origin_var_dict[param_name],
                            origin_var_dict[grad_name]
                        ])
            else:
                pass
        return opt_ops, params_grads
