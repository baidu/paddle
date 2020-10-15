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

from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_api
from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.static_analysis import StaticAnalysisVisitor


def create_convert_shape_node(var_shape_node):
    assert isinstance(var_shape_node, (gast.Attribute, gast.Subscript))

    convert_var_shape_func = "paddle.jit.dy2static.convert_var_shape"

    if isinstance(var_shape_node, gast.Attribute):
        api_shape_node = gast.Call(
            func=gast.parse(convert_var_shape_func).body[0].value,
            args=[var_shape_node.value],
            keywords=[])
        return api_shape_node

    if isinstance(var_shape_node, gast.Subscript):
        result_node = copy.deepcopy(var_shape_node)
        result_node.value = create_convert_shape_node(result_node.value)
        return result_node


class TensorShapeTransformer(gast.NodeTransformer):
    """
    This class transforms variable.shape used in Paddle Apis or control flow conditions into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of TensorShapeTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.name_to_var_shape = {}

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)

    def visit_Assign(self, node):
        if self._update_name_to_var_shape(node):
            return node
        self.generic_visit(node)
        return node

    def visit_Attribute(self, node):
        if self._used_by_paddle_api(node):
            if self.is_var_shape(node):
                return create_convert_shape_node(node)
        return node

    def visit_Name(self, node):
        if node.id in self.name_to_var_shape:
            if self._used_by_paddle_api(node):
                var_shape_node = self.name_to_var_shape[node.id]
                return create_convert_shape_node(var_shape_node)
        return node

    def visit_Call(self, node):
        assert isinstance(node, gast.Call)
        if is_paddle_api(node):
            # Visit gast.Attribute and gast.Name to replace var.shape if necessary.
            self.generic_visit(node)

        return node

    def visit_If(self, node):
        # Call generic_visit first to transform var.shape that is used in Paddle Api.
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)

        return node

    def visit_While(self, node):
        self.generic_visit(node)
        cond = node.test
        self._transform_var_shape_if_necessary(cond)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        iter = node.iter
        self._transform_var_shape_if_necessary(iter)

        # If var.shape is a gast.Name and it is used in range function, transform it
        self._transform_var_shape_in_range(node)
        return node

    def _transform_var_shape_in_range(self, node):
        assert isinstance(node, gast.For)
        if not isinstance(node.iter, gast.Call):
            return False
        if not isinstance(node.iter.func, gast.Name):
            return False
        if node.iter.func.id != "range":
            return False
        args = node.iter.args
        for idx, arg in enumerate(args):
            if isinstance(arg, gast.Name) and arg.id in self.name_to_var_shape:
                args[idx] = create_convert_shape_node(self.name_to_var_shape[
                    arg.id])

        return True

    def _transform_var_shape_if_necessary(self, cond):
        need_transformed = False
        for child_node in gast.walk(cond):
            var_shape_node = None
            if isinstance(child_node, (gast.Attribute)):
                if self.is_var_shape(child_node):
                    var_shape_node = child_node
            elif isinstance(child_node, (gast.Name)):
                if child_node.id in self.name_to_var_shape:
                    var_shape_node = self.name_to_var_shape[child_node.id]

            if var_shape_node:
                need_transformed = True
                wrapper_node = self.node_to_wrapper_map.get(child_node)
                parent_node = wrapper_node.parent.node
                for field, value in gast.iter_fields(parent_node):
                    if child_node is value:
                        setattr(parent_node, field,
                                create_convert_shape_node(var_shape_node))
                        break
        return need_transformed

    def _used_by_paddle_api(self, node):
        assert isinstance(node, (gast.Attribute, gast.Name))
        wrapper_node = self.node_to_wrapper_map.get(node)
        if not wrapper_node:
            # Transformed node is not in node_to_wrapper_map
            return False
        while wrapper_node.parent:
            parent_node = wrapper_node.parent.node
            if isinstance(parent_node, gast.Call):
                if is_paddle_api(parent_node):
                    return True
                else:
                    return False
            wrapper_node = wrapper_node.parent

        return False

    def is_var_shape(self, node):
        """
        Return True if node is like `x.shape`, return False otherwise.
        """
        assert isinstance(node, gast.Attribute)

        if node.attr != 'shape':
            return False

        try:
            value_id = node.value.id
        except AttributeError:
            return False

        if value_id in self.name_to_var_shape:
            return True

        return True

    def _update_name_to_var_shape(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value

        if isinstance(value_node, gast.Name):
            if value_node.id in self.name_to_var_shape:
                self.name_to_var_shape[target_id] = self.name_to_var_shape[
                    value_node.id]
                return True
        if isinstance(value_node, gast.Attribute):
            if self.is_var_shape(value_node):  # eg: x.shape
                self.name_to_var_shape[target_id] = value_node
                return True
        if isinstance(value_node, gast.Subscript):
            if isinstance(value_node.value, gast.Attribute):
                if self.is_var_shape(value_node.value):  # eg: x.shape[0]
                    self.name_to_var_shape[target_id] = value_node
                    return True
        return False
