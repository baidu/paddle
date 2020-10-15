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

from __future__ import print_function

import gast
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code


class LogicalTransformer(gast.NodeTransformer):
    """
    Transform python boolean op into Paddle logical op
    """

    def __init__(self, wrapper_root):
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node

    def transform(self):
        return self.visit(self.root)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.Not):
            arg = ast_to_source_code(node.operand)
            new_node_str = "paddle.jit.dy2static.convert_logical_not({})".format(
                arg)
            # NOTE: gast.parse returns Module(body=[expr(value=...)])
            new_node = gast.parse(new_node_str).body[0].value
            return new_node
        return node

    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, gast.And):
            new_node = self._create_bool_op_node(node.values, 'and')
        elif isinstance(node.op, gast.Or):
            new_node = self._create_bool_op_node(node.values, 'or')
        else:
            raise TypeError(
                "Only supports and/or syntax in control flow if statement.")
        return new_node

    def _create_bool_op_node(self, nodes, api_type):
        assert len(
            nodes
        ) > 1, "The length of BoolOp should be at least 2, but received {}.".format(
            len(nodes))
        if len(nodes) > 2:
            # Creates logic_and/logic_or node recursively.
            pre_logic_node = self._create_bool_op_node(nodes[:2], api_type)
            if len(nodes[2:]) == 1:
                post_logic_node = nodes[2]
            else:
                post_logic_node = self._create_bool_op_node(nodes[2:], api_type)
            nodes = [pre_logic_node] + [post_logic_node]

        args = [ast_to_source_code(child) for child in nodes]
        new_node_str = "paddle.jit.dy2static.convert_logical_{}(x={}, y={})".format(
            api_type, args[0], args[1])
        # NOTE: gast.parse return Module(body=[expr(...)])
        new_node = gast.parse(new_node_str).body[0].value
        return new_node
