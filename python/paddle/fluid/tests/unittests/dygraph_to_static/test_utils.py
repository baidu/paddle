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

import types
import unittest

from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.dygraph_to_static.utils import index_in_list
from paddle.fluid.dygraph.dygraph_to_static.utils import is_paddle_func

from test_program_translator import get_source_code


class TestIndexInList(unittest.TestCase):
    def test_index_in_list(self):
        list_to_test = [1, 2, 3, 4, 5]
        self.assertEqual(index_in_list(list_to_test, 4), 3)
        self.assertEqual(index_in_list(list_to_test, 1), 0)
        self.assertEqual(index_in_list(list_to_test, 5), 4)
        self.assertEqual(index_in_list(list_to_test, 0), -1)
        self.assertEqual(index_in_list(list_to_test, 6), -1)


def dyfunc_assign(input):
    a = b = 1
    c, d = e, f = a, b
    z = [3, 4]
    [x, y] = m, n = z


class StaticCode():
    def dyfunc_assign(self, input):
        b = 1
        a = b
        e = a
        f = b
        c = e
        d = f
        z = [3, 4]
        m, n = z
        x = m
        y = n


class TestSplitAssignTransformer(unittest.TestCase):
    def test_code(self):
        answer = get_source_code(StaticCode.dyfunc_assign)
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_assign)
        self.assertEqual(answer, code)


class TestIsPaddle(unittest.TestCase):
    def fake_module(self):
        return types.ModuleType('paddlenlp')

    def test_func(self):
        m = self.fake_module()
        self.assertFalse(is_paddle_func(m))


if __name__ == '__main__':
    unittest.main()
