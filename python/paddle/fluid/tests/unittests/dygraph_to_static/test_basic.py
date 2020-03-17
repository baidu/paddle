#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle.fluid as fluid
import unittest

from paddle.fluid.dygraph.jit import dygraph_to_static_graph

from ifelse_simple_func import *

np.random.seed(1)

if fluid.is_compiled_with_cuda():
    place = fluid.CUDAPlace(0)
else:
    place = fluid.CPUPlace()


class TestDygraphIfElse(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x_v = fluid.layers.assign(self.x)
            # Transform into static graph
            out = dygraph_to_static_graph(self.dyfunc)(x_v)
            exe = fluid.Executor(place)
            ret = exe.run(main_program, fetch_list=out)
            return ret

    def _run_dygraph(self):
        with fluid.dygraph.guard(place):
            x_v = fluid.dygraph.to_variable(self.x)
            ret = self.dyfunc(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


class TestDygraphIfElse2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = dyfunc_with_if_else2


class TestDygraphIfElse3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = nested_if_else


class TestDygraphIfElseWithAndOr(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or


class TestDygraphIfElseWithAndOr1(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_1


class TestDygraphIfElseWithAndOr2(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_2


class TestDygraphIfElseWithAndOr3(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_3


class TestDygraphIfElseWithAndOr4(TestDygraphIfElse):
    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.dyfunc = if_with_and_or_4


class TestDygraphIfElseNet(unittest.TestCase):
    """
    TestCase for the transformation from control flow `if/else`
    dependent on tensor in Dygraph into Static `fluid.layers.cond`.
    """

    def setUp(self):
        self.x = np.random.random([10, 16]).astype('float32')
        self.Net = NetWithControlFlowIf

    def _run_static(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            net = self.Net()
            x_v = fluid.layers.assign(self.x)
            # Transform into static graph
            out = net(x_v)
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            ret = exe.run(main_program, fetch_list=out)
            return ret[0]

    def _run_dygraph(self):
        with fluid.dygraph.guard(place):
            net = self.Net()
            x_v = fluid.dygraph.to_variable(self.x)
            ret = net(x_v)
            return ret.numpy()

    def test_ast_to_func(self):
        self.assertTrue((self._run_dygraph() == self._run_static()).all())


if __name__ == '__main__':
    unittest.main()
