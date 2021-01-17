# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np
import ascend_parser


class AscendIRParser(object):
    def __init__(self):
        self.graph_idx = 0

    def _construct_input_map(self, input_varlist):
        ret_map = {}
        ge_in_operator = []
        for id, var in enumerate(input_varlist):
            if var.is_data:  # input data
                ge_input = core.GEOperatorFactory.create_operator(
                    var.name, "Data").set_attr_int32("index", id)
                ret_map[var.name] = ge_input
                ge_in_operator.append(ge_input)
            else:  # param, learning ...
                ge_input = core.GEOperatorFactory.create_operator(var.name,
                                                                  "Variable")
                ge_input.update_output_desc("y",
                                            core.GETensorDesc(
                                                core.GEShape(var.shape),
                                                core.GEFormat.FORMAT_ND,
                                                core.GEDataType.DT_FLOAT))
                ret_map[var.name] = ge_input
        return ge_in_operator, ret_map

    def parse_op(self, op):
        if op.type in ascend_parser.registerd_op:
            print("Op[%s] has been registered, begin to parse it" % (op.type))
            op_parser = self.parser_factory.create_parse(
                ascend_parser.registerd_op[op.type])
            op_parser.apply(op)
        else:
            print("Op[%s] has not been registered, so we have to skip it" %
                  (op.type))

    def _parse_program(self,
                       graph_name,
                       program,
                       input_varlist=[],
                       fetch_list=[]):
        begin_graph_idx = self.graph_idx
        ge_in_operator = []
        ge_out_operator = []
        self.var2geop = {}

        block = program.global_block()
        if len(block.ops) == 0:
            print("There is no ops in program %s" % (graph_name))
            return []

        graph = core.GEGraph(graph_name)

        ge_in_operator, self.var2geop = self._construct_input_map(input_varlist)

        self.parser_factory = ascend_parser.AscendParserFactory(graph,
                                                                self.var2geop)
        for i, curop in list(enumerate(block.ops)):
            self.parse_op(curop)

        # Set fetch_var for GE
        for e in fetch_list:
            name = e
            if not isinstance(e, str):
                name = e.name
            ge_out_operator.append(self.var2geop[name])

        # (Debug) If you want to print back prop vars, append/assign the varname in ge_out_operator here, such as: 
        # if graph_name == "main":
        #     ge_out_operator.append(self.var2geop["reduce_sum_0.tmp_0@GRAD"])

        # Add ops that may be input of a graph, such as const.
        for varname, geop in self.var2geop.items():
            if varname.startswith("geinput"):
                ge_in_operator.append(geop)

        graph.set_inputs(ge_in_operator).set_outputs(ge_out_operator)

        # Remove ops of origin program
        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)

        input_varlist = [var for var in input_varlist if var.is_data]

        block.append_op(
            type="ascend_trigger",
            inputs={"FeedList": input_varlist},
            outputs={"FetchList": fetch_list},
            attrs={'graph_idx': self.graph_idx})
        self.graph_idx += 1
        return graph

    def parse_program(self, startup_program, main_program, input_varlist,
                      fetch_list):
        startup_graph = self._parse_program("startup", startup_program)
        main_graph = self._parse_program("main", main_program, input_varlist,
                                         fetch_list)
        return startup_graph, main_graph


# AscendOptimizer is a wrapper for basic optimizer now
# We will make it part of fleet meta_optimizer in the future
class AscendOptimizer(Optimizer):
    def __init__(self, optimizer, fetch_list=[]):
        self.inner_opt = optimizer
        self.fetch_list = fetch_list

    def __del__(self):
        core.ge_finalize()

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False
        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def _get_input_varlist(program):
        ret_list = []
        for var in program.list_vars():
            if var.is_data or var.persistable:
                ret_list.append(var)
        return ret_list

    def minimize(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)

        self.ascend_instance = core.AscendInstance()

        # Config about Graph Engine can be found in https://support.huaweicloud.com/
        config = {
            "ge.exec.deviceId": "0",
            "ge.graphRunMode": "1",
            "ge.exec.precision_mode": "must_keep_origin_dtype"
        }
        core.ge_initialize(config)

        # Init Session
        self.ascend_instance.init_global_resources()

        main_block = loss.block
        self.parser = AscendIRParser()

        input_varlist = _get_input_varlist(main_block.program)
        startup_graph, main_graph = self.parser.parse_program(
            startup_program, main_block.program, input_varlist, self.fetch_list)

        self.ascend_instance.add_ascend_subgraph(0, startup_graph)
        self.ascend_instance.add_ascend_subgraph(1, main_graph)

        return minimized
