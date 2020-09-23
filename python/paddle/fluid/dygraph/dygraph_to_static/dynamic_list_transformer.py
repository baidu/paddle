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

from paddle.fluid.dygraph.dygraph_to_static.static_analysis import AstNodeWrapper
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_source_code
from paddle.fluid.dygraph.dygraph_to_static.utils import RenameTransformer
from paddle.fluid.dygraph.dygraph_to_static.utils import SplitAssignTransformer


class DynamicListTransformer(gast.NodeTransformer):
    """
    A class transforms returning DynamicList to LoDTensorArray to match user call
    """

    def __init__(self, wrapper_root):
        assert isinstance(
            wrapper_root, AstNodeWrapper
        ), "Input non-AstNodeWrapper node for the initialization of DynamicListTransformer."
        self.wrapper_root = wrapper_root
        self.root = wrapper_root.node
        self.dynamic_list_name = set()

    def transform(self):
        SplitAssignTransformer(self.root).transform()
        self.visit(self.root)

    def visit_Assign(self, node):
        self.generic_visit(node)
        value_str = ast_to_source_code(node.value).strip()
        target_name = ast_to_source_code(node.targets[0]).strip()
        if value_str == "fluid.dygraph.dygraph_to_static.variable_trans_func.DynamicList()" or value_str in self.dynamic_list_name:
            self.dynamic_list_name.add(target_name)
        else:
            self.dynamic_list_name.discard(target_name)
        return node

    def visit_Return(self, node):
        self.generic_visit(node)
        rename_transformer = RenameTransformer(node)
        for name in self.dynamic_list_name:
            rename_transformer.rename(
                name,
                "fluid.dygraph.dygraph_to_static.variable_trans_func.dynamic_list_as_tensor_array({})".
                format(name))
        return node
