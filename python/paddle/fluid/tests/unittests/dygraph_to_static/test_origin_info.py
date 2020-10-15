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

import sys
import unittest

from paddle.fluid.dygraph.dygraph_to_static.ast_transformer import DygraphToStaticAst
from paddle.fluid.dygraph.dygraph_to_static.origin_info import *
from paddle.fluid.dygraph.dygraph_to_static.utils import ast_to_func
from paddle.fluid.dygraph.jit import declarative


def simple_func(x):
    y = x + 1
    return y


def nested_func(x):
    def f1(a):
        return a

    result = f1(x)
    return result


@declarative
def decorated_func(x):
    return x


@declarative
@declarative
def decorated_func2(x):
    return x


class TestOriginInfo(unittest.TestCase):
    def setUp(self):
        self.set_test_func()
        self.dygraph_func = unwrap(self.func)
        self.dygraph_filepath = inspect.getfile(self.dygraph_func)
        self.source_code = inspect.getsource(self.dygraph_func)
        lines, self.start_lineno = inspect.getsourcelines(self.dygraph_func)
        lines = [line.strip("\n") for line in lines]
        self.lines = [line for line in lines
                      if line != ""]  # Delete empty lines

        self.set_static_lineno()
        self.set_dygraph_info()

    def set_test_func(self):
        self.func = simple_func

    def set_static_lineno(self):
        self.static_abs_lineno_list = [3, 4, 5]

    def set_dygraph_info(self):
        self.line_num = 3
        self.line_index_list = [0, 1, 2]
        self.dy_rel_lineno_list = [0, 1, 2]
        self.dy_abs_col_offset = [0, 4, 4]
        self.dy_func_name = [self.dygraph_func.__name__] * 3

    def set_origin_info_list(self, dygraph_ast):
        assert isinstance(dygraph_ast, gast.Module)
        self.transformed_node_list = [
            dygraph_ast.body[0], dygraph_ast.body[0].body[0],
            dygraph_ast.body[0].body[1]
        ]

    def _get_OriginInfo_map(self):
        # step1
        dygraph_ast = gast.parse(self.source_code)
        dygraph_ast = attach_origin_info(dygraph_ast, self.dygraph_func)

        # step2
        transformed_ast = DygraphToStaticAst().get_static_ast(dygraph_ast).node

        # step3
        self.static_func, _ = ast_to_func(transformed_ast, self.dygraph_func)
        info_map = create_and_update_origin_info_map(dygraph_ast,
                                                     self.static_func)

        return info_map

    def test_origin_info_map(self):
        self.set_static_lineno()
        origin_info_map = self._get_OriginInfo_map()
        static_filepath = inspect.getfile(self.static_func)
        start_lineno = self.start_lineno
        dygraph_abs_lineno_list = [
            start_lineno + lineno for lineno in self.dy_rel_lineno_list
        ]

        for i in range(self.line_num):
            static_lineno = self.static_abs_lineno_list[i]
            staic_loc = Location(static_filepath, static_lineno)
            self.assertIn(staic_loc.line_location, origin_info_map)

            dy_lineno = dygraph_abs_lineno_list[i]
            dy_col_offset = self.dy_abs_col_offset[i]
            line_idx = self.line_index_list[i]
            code = self.lines[line_idx]
            origin_info = OriginInfo(
                Location(self.dygraph_filepath, dy_lineno, dy_col_offset),
                self.dy_func_name[i], code)
            self.assertEqual(
                str(origin_info_map[staic_loc.line_location]), str(origin_info))

    def test_attach_origin_info(self):
        dygraph_ast = gast.parse(self.source_code)
        dygraph_ast = attach_origin_info(dygraph_ast, self.dygraph_func)
        self.set_origin_info_list(dygraph_ast)
        start_lineno = self.start_lineno

        filepath = inspect.getfile(self.dygraph_func)

        for i in range(self.line_num):
            node = self.transformed_node_list[i]
            origin_info = getattr(node, ORIGI_INFO)
            dy_rel_lineno = self.dy_rel_lineno_list[i]
            dy_abs_lineno = start_lineno + dy_rel_lineno
            dy_col_offset = self.dy_abs_col_offset[i]
            func_name = self.dy_func_name[i]
            line_idx = self.line_index_list[i]
            code = self.lines[line_idx]
            self.assertEqual(origin_info.location.filepath, filepath)
            self.assertEqual(origin_info.location.lineno, dy_abs_lineno)
            self.assertEqual(origin_info.location.col_offset, dy_col_offset)
            self.assertEqual(origin_info.function_name, func_name)
            self.assertEqual(origin_info.source_code, code)


class TestOriginInfoWithNestedFunc(TestOriginInfo):
    def set_test_func(self):
        self.func = nested_func

    def set_static_lineno(self):
        self.static_abs_lineno_list = [3, 5, 6, 7, 8]

    def set_dygraph_info(self):
        self.line_num = 5
        self.line_index_list = [0, 1, 2, 3, 4]
        self.dy_rel_lineno_list = [0, 1, 2, 4, 5]
        self.dy_abs_col_offset = [0, 4, 8, 4, 4]
        self.dy_func_name = [self.dygraph_func.__name__] + \
                            ["f1"] * 2 + \
                            [self.dygraph_func.__name__] * 2

    def set_origin_info_list(self, dygraph_ast):
        assert isinstance(dygraph_ast, gast.Module)
        self.transformed_node_list = [
            dygraph_ast.body[0], dygraph_ast.body[0].body[0],
            dygraph_ast.body[0].body[0].body[0], dygraph_ast.body[0].body[1],
            dygraph_ast.body[0].body[2]
        ]


class TestOriginInfoWithDecoratedFunc(TestOriginInfo):
    def set_test_func(self):
        self.func = decorated_func

    def set_static_lineno(self):
        self.static_abs_lineno_list = [3, 4]

    def set_dygraph_info(self):
        self.line_num = 2

        # NOTE(liym27):
        #   There are differences in ast_node.lineno between PY3.8+ and PY3.8-.
        #   If the first gast.FunctionDef has decorator, the lineno of gast.FunctionDef is differs.
        #       1. < PY3.8
        #           its lineno equals to the lineno of the first decorator node, which is not right.
        #       2. >= PY3.8
        #           its lineno is the actual lineno, which is right.
        if sys.version_info >= (3, 8):
            self.line_index_list = [1, 2]
            self.dy_rel_lineno_list = [1, 2]
        else:
            self.line_index_list = [0, 2]
            self.dy_rel_lineno_list = [0, 2]
        self.dy_abs_col_offset = [0, 4]
        self.dy_func_name = [self.dygraph_func.__name__] * self.line_num

    def set_origin_info_list(self, dygraph_ast):
        assert isinstance(dygraph_ast, gast.Module)
        self.transformed_node_list = [
            dygraph_ast.body[0],
            dygraph_ast.body[0].body[0],
        ]


class TestOriginInfoWithDecoratedFunc2(TestOriginInfo):
    def set_test_func(self):
        self.func = decorated_func2

    def set_static_lineno(self):
        self.static_abs_lineno_list = [3, 4]

    def set_dygraph_info(self):
        self.line_num = 2

        if sys.version_info >= (3, 8):
            self.line_index_list = [2, 3]
            self.dy_rel_lineno_list = [2, 3]
        else:
            self.line_index_list = [0, 3]
            self.dy_rel_lineno_list = [0, 3]
        self.dy_abs_col_offset = [0, 4]
        self.dy_func_name = [self.dygraph_func.__name__] * self.line_num

    def set_origin_info_list(self, dygraph_ast):
        assert isinstance(dygraph_ast, gast.Module)
        self.transformed_node_list = [
            dygraph_ast.body[0],
            dygraph_ast.body[0].body[0],
        ]


if __name__ == '__main__':
    unittest.main()
