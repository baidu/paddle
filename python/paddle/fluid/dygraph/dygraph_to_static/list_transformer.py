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

import astor
import gast

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper, NodeVarType, StaticAnalysisVisitor
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code, is_control_flow_to_transform
from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer
from paddle.fluid.framework import core, Variable
from paddle.fluid.layers import array_length, array_read, array_write, create_array
from paddle.fluid.layers import assign, fill_constant, slice
from paddle.fluid.layers.control_flow import cond, while_loop, less_than, increment


# TODO(liym27): A better way to slice tensor array.
#  Maybe support start == end for slice op.
def slice_tensor_array(array, start, end):
    def true_fn():
        null_array = create_array("float32")
        return null_array

    def false_fn(array, start, end):
        new_array = slice(array, starts=[start], ends=[end], axes=[0])
        return new_array

    new_array = cond(start == end, true_fn, lambda: false_fn(array, start, end))
    return new_array


def tensor_array_pop(array, idx):
    assert isinstance(idx, int)

    def cond(i, new_array):
        return less_than(i, arr_len)

    def body(i, new_array):
        item = array_read(array=array, i=i)
        array_write(item, array_length(new_array), new_array)
        i = increment(i)
        return i, new_array

    arr_len = array_length(array)
    if idx < 0:
        idx = idx + arr_len
    else:
        idx = fill_constant(shape=[1], dtype="int64", value=idx)

    pop_item = array_read(array, idx)

    new_array = slice_tensor_array(array, 0, idx)
    i = idx + 1
    _, new_array = while_loop(cond, body, [i, new_array])
    assign(input=new_array, output=array)

    return pop_item


def convert_list_pop(target, idx=None):
    """
    Convert list pop.
    """

    if idx is None:
        idx = -1

    is_variable = isinstance(target, Variable)
    if is_variable:
        is_tensor_array = target.type == core.VarDesc.VarType.LOD_TENSOR_ARRAY
    if is_variable and is_tensor_array:
        result = tensor_array_pop(target, idx)
    else:
        result = target.pop(idx)
    return result


class ListTransformer(gast.NodeTransformer):
    """
    This class transforms python list used in control flow into Static Graph Ast.
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of ListTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.list_name_to_updated = dict()
        self.list_nodes = set()

        self.static_analysis_visitor = StaticAnalysisVisitor(self.root)
        self.node_to_wrapper_map = self.static_analysis_visitor.get_node_to_wrapper_map(
        )
        var_env = self.static_analysis_visitor.get_var_env()
        var_env.cur_scope = var_env.cur_scope.sub_scopes[0]
        self.scope_var_type_dict = var_env.get_scope_var_type()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)
        self.replace_list_with_tensor_array(self.root)

    def visit_Call(self, node):
        if isinstance(node.func, gast.Attribute):
            func_name = node.func.attr
            if func_name == "pop":
                node = self._replace_list_pop(node)
        return node

    def visit_Assign(self, node):
        if self._update_list_name_to_updated(node):
            return node

        if self._need_to_array_write_node(node):
            return self._transform_slice_to_tensor_write(node)

        self.generic_visit(node)
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        if is_control_flow_to_transform(node, self.static_analysis_visitor,
                                        self.scope_var_type_dict):
            self._transform_list_append_in_control_flow(node)
        return node

    def replace_list_with_tensor_array(self, node):
        for child_node in gast.walk(node):
            if isinstance(child_node, gast.Assign):
                if self._need_to_create_tensor_array(child_node):
                    child_node.value = self._create_tensor_array()

    def _transform_list_append_in_control_flow(self, node):
        for child_node in gast.walk(node):
            if self._need_to_array_write_node(child_node):
                child_node.value = \
                    self._to_array_write_node(child_node.value)

    def _need_to_array_write_node(self, node):
        if isinstance(node, gast.Expr):
            if isinstance(node.value, gast.Call):
                if self._is_list_append_tensor(node.value):
                    return True

        if isinstance(node, gast.Assign):
            target_node = node.targets[0]
            if isinstance(target_node, gast.Subscript):
                list_name = ast_to_source_code(target_node.value).strip()
                if list_name in self.list_name_to_updated:
                    if self.list_name_to_updated[list_name] == True:
                        return True
        return False

    def _transform_slice_to_tensor_write(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        target_name = target_node.value.id
        slice_node = target_node.slice

        if isinstance(slice_node, gast.Slice):
            pass
        elif isinstance(slice_node, gast.Index):
            value_code = ast_to_source_code(node.value)
            i = "paddle.cast(" \
                "x=paddle.jit.dy2static.to_static_variable({})," \
                "dtype='int64')".format(ast_to_source_code(slice_node))
            assign_code = "{} = fluid.layers.array_write(x={}, i={}, array={})" \
                .format(target_name, value_code, i, target_name)
            assign_node = gast.parse(assign_code).body[0]
        return assign_node

    def _is_list_append_tensor(self, node):
        """
        a.append(b): a is list, b is Tensor
        self.x.append(b): self.x is list, b is Tensor
        """
        assert isinstance(node, gast.Call)
        # 1. The func is `append`.
        if not isinstance(node.func, gast.Attribute):
            return False
        if node.func.attr != 'append':
            return False

        # 2. It's a `python list` to call append().
        value_name = astor.to_source(gast.gast_to_ast(node.func.value)).strip()
        if value_name not in self.list_name_to_updated:
            return False

        # 3. The number of arg of append() is one
        # Only one argument is supported in Python list.append()
        if len(node.args) != 1:
            return False

        # TODO(liym27): The arg of append() should be Tensor. But because the type of arg is often wrong with static analysis,
        #   the arg is not required to be Tensor here.
        # 4. The arg of append() is Tensor
        # arg = node.args[0]
        # if isinstance(arg, gast.Name):
        #     # TODO: `arg.id` may be not in scope_var_type_dict if `arg.id` is the arg of decorated function
        #     # Need a better way to confirm whether `arg.id` is a Tensor.
        #     try:
        #         var_type_set = self.scope_var_type_dict[arg.id]
        #     except KeyError:
        #         return False
        #     if NodeVarType.NUMPY_NDARRAY in var_type_set:
        #         return False
        #     if NodeVarType.TENSOR not in var_type_set and NodeVarType.PADDLE_RETURN_TYPES not in var_type_set:
        #         return False
        # # TODO: Consider that `arg` may be a gast.Call about Paddle Api. eg: list_a.append(fluid.layers.reshape(x))
        # # else:
        # # return True
        self.list_name_to_updated[value_name.strip()] = True
        return True

    def _need_to_create_tensor_array(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        if self.list_name_to_updated.get(target_id) and node in self.list_nodes:
            return True
        return False

    def _create_tensor_array(self):
        # Although `dtype='float32'`, other types such as `int32` can also be supported
        func_code = "fluid.layers.create_array(dtype='float32')"
        func_node = gast.parse(func_code).body[0].value
        return func_node

    def _to_array_write_node(self, node):
        assert isinstance(node, gast.Call)
        array = astor.to_source(gast.gast_to_ast(node.func.value))
        x = astor.to_source(gast.gast_to_ast(node.args[0]))
        i = "fluid.layers.array_length({})".format(array)
        func_code = "fluid.layers.array_write(x={}, i={}, array={})".format(
            x, i, array)
        return gast.parse(func_code).body[0].value

    def _update_list_name_to_updated(self, node):
        assert isinstance(node, gast.Assign)
        target_node = node.targets[0]
        # NOTE: Code like `x, y = a, []` has been transformed to `x=a; y=[]`
        try:
            target_id = target_node.id
        except AttributeError:
            return False
        value_node = node.value
        if isinstance(value_node, gast.List):
            self.list_name_to_updated[target_id] = False
            self.list_nodes.add(node)
            return True
        elif target_id in self.list_name_to_updated and \
                self.list_name_to_updated[target_id] == False:
            del self.list_name_to_updated[target_id]
        return False

    def _replace_list_pop(self, node):
        assert isinstance(node, gast.Call)
        assert isinstance(node.func, gast.Attribute)

        target_node = node.func.value
        target_str = ast_to_source_code(target_node).strip()

        if node.args:
            idx_node = node.args[0]
            idx_str = ast_to_source_code(idx_node).strip()
        else:
            idx_str = "None"

        new_call_str = "fluid.dygraph.dygraph_to_static.list_transformer.convert_list_pop({}, {})".format(
            target_str, idx_str)
        new_call_node = gast.parse(new_call_str).body[0].value
        return new_call_node
