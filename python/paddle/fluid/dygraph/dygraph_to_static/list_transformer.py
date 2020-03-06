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

import copy
import gast
import astor

from collections import defaultdict
from paddle.fluid import unique_name
from paddle.fluid.dygraph.dygraph_to_static.ast_utils import create_funcDef_node
from paddle.fluid.dygraph.dygraph_to_static.ast_utils import generate_name_node
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import create_static_variable_gast_node
from paddle.fluid.dygraph.dygraph_to_static.variable_trans_func import to_static_variable_gast_node
from .static_analysis import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor


class ListTransformer(gast.NodeTransformer):
    """
    This class transforms python list used in control flow into Static Graph Ast
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of ListTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_of_list_set = set()

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        # TODO: Consider that Tensor.shape is used in sub function and sub_scopes is empty
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        self.visit(self.root)

    def visit_Assign(self, node):
        if self._update_name_of_list_set(node):
            node = self._create_tensor_array(node)
            return node
        return node

    def visit_Expr(self, node):
        value_node = node.value
        if isinstance(value_node, gast.Call):
            if self._is_list_append_tensor(value_node):
                node.value = self._to_array_write_node(value_node)
            return node
        return node

    # def visit_If(self, node):
    #     self.generic_visit(node)
    #     cond = node.test
    #     for child_node in gast.walk(cond):
    #         if isinstance(child_node, gast.Attribute):
    #             if self.is_tensor_shape(child_node):
    #                 wrapper_node = self.node_to_wrapper_map.get(child_node)
    #                 parent_node = wrapper_node.parent.node
    #                 for field, value in gast.iter_fields(parent_node):
    #                     if value is child_node:
    #                         setattr(parent_node, field,
    #                                 create_api_shape_node(child_node))
    #                         break

    def _is_list_append_tensor(self, node):
        """
        a.append(b): a is list, b is Tensor
        """
        assert isinstance(node, gast.Call)
        # 1. is .append
        if not isinstance(node.func, gast.Attribute):
            return False
        if node.func.attr != 'append':
            return False

        # 2. is list
        value_name = astor.to_source(gast.gast_to_ast(node.func.value))

        # if not isinstance(node.func.value, gast.Name):
        #     return False
        for ele in self.name_of_list_set:
            print(ele, len(ele))

        if value_name.strip() not in self.name_of_list_set:
            return False

        # 3. append Variable
        # Only one argument is supported in Python list.append()
        if len(node.args) != 1:
            # print("len(node.args) != 1")
            return False

        arg = node.args[0]
        if isinstance(arg, gast.Name):
            # TODO: `arg.id` may be not in scope_var_type_dict if `arg.id` is the arg of decorated function
            # Need a better way to confirm whether `arg.id` is a Tensor.
            try:
                var_type_set = self.scope_var_type_dict[arg.id]
            except KeyError:
                return False

            if NodeVarType.NUMPY_NDARRAY in var_type_set:
                return False
            if NodeVarType.TENSOR not in var_type_set and NodeVarType.PADDLE_RETURN_TYPES not in var_type_set:
                return False
        else:
            # Todo: Consider that `arg` may be a gast.Call about Paddle Api.
            # eg: list_a.append(fluid.layers.reshape(x))
            return True
        return True

    def _create_tensor_array(self, node):
        assert isinstance(node, gast.Assign)
        target = node.targets[0].id
        x = "fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)"
        i = "fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)"
        func_code = "{} = fluid.layers.array_write(x={}, i={}, array=None)".format(
            target, x, i)
        return gast.parse(func_code)

    def _to_array_write_node(self, node):
        assert isinstance(node, gast.Call)
        array = astor.to_source(gast.gast_to_ast(node.func.value))
        x = astor.to_source(gast.gast_to_ast(node.args[0]))
        i = "fluid.layers.array_length({})".format(array)
        new_code = "fluid.layers.array_write(x={}, i={}, array={})".format(
            x, i, array)
        return gast.parse(new_code)

    def _update_name_of_list_set(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        # TODO: Consider node has more than one target. eg: x, y = a, []
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value
        if isinstance(value_node, gast.List):
            self.name_of_list_set.add(target_id)
            return True
        elif target_id in self.name_of_list_set:
            self.name_of_list_set.remove(target_id)
        return False
