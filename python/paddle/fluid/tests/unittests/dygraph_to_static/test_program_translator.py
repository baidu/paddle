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
import inspect
import numpy as np
import textwrap
import unittest

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.dygraph_to_static import ProgramTranslator
from paddle.fluid.dygraph.jit import declarative
from paddle.fluid.dygraph.nn import Linear

from ifelse_simple_func import dyfunc_with_if_else

np.random.seed(0)


# TODO(Aurelius): Currently, `declarative` don't support decorate the function
# that contains layers with initialized operation, like `fc = linear(10, 3)`.
# Because initialized ops will be added into program and be executed many times.
# The parameters are assumed to initialized outside of the function.
def simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = fluid.layers.mean(y)
    return z


@declarative
def decorated_simple_func(x, weight_numpy):
    x = fluid.dygraph.to_variable(x)
    w = fluid.dygraph.to_variable(weight_numpy)
    y = fluid.layers.matmul(x, w)
    z = fluid.layers.mean(y)
    return z


def get_source_code(func):
    raw_code = inspect.getsource(func)
    code = textwrap.dedent(raw_code)
    root = gast.parse(code)
    source_code = astor.to_source(gast.gast_to_ast(root))
    return source_code


class StaticCode1():
    def dyfunc_with_if_else(self, x_v, label=None):
        __return_value_init_0 = paddle.fluid.layers.fill_constant(
            shape=[1], dtype='float64', value=0.0)
        __return_value_0 = __return_value_init_0

        def true_fn_0(x_v):
            x_v = x_v - 1
            return x_v

        def false_fn_0(x_v):
            x_v = x_v + 1
            return x_v

        x_v = paddle.jit.dy2static.convert_ifelse(
            fluid.layers.mean(x_v)[0] > 5, true_fn_0, false_fn_0, (x_v, ),
            (x_v, ), (x_v, ))
        __return_0 = paddle.jit.dy2static.create_bool_as_type(label is not None,
                                                              False)

        def true_fn_1(__return_0, __return_value_0, label, x_v):
            loss = fluid.layers.cross_entropy(x_v, label)
            __return_0 = paddle.jit.dy2static.create_bool_as_type(
                label is not None, True)
            __return_value_0 = loss
            return __return_0, __return_value_0

        def false_fn_1(__return_0, __return_value_0):
            return __return_0, __return_value_0

        __return_0, __return_value_0 = (paddle.jit.dy2static.convert_ifelse(
            label is not None, true_fn_1, false_fn_1,
            (__return_0, __return_value_0, label, x_v),
            (__return_0, __return_value_0), (__return_0, __return_value_0)))

        def true_fn_2(__return_0, __return_value_0, x_v):
            __return_1 = paddle.jit.dy2static.create_bool_as_type(
                paddle.jit.dy2static.convert_logical_not(__return_0), True)
            __return_value_0 = x_v
            return __return_value_0

        def false_fn_2(__return_value_0):
            return __return_value_0

        __return_value_0 = paddle.jit.dy2static.convert_ifelse(
            paddle.jit.dy2static.convert_logical_not(__return_0), true_fn_2,
            false_fn_2, (__return_0, __return_value_0,
                         x_v), (__return_value_0, ), (__return_value_0, ))
        return __return_value_0


class StaticCode2():
    # TODO: Transform return statement
    def dyfunc_with_if_else(self, x_v, label=None):
        __return_value_init_1 = paddle.fluid.layers.fill_constant(
            shape=[1], dtype='float64', value=0.0)
        __return_value_1 = __return_value_init_1

        def true_fn_3(x_v):
            x_v = x_v - 1
            return x_v

        def false_fn_3(x_v):
            x_v = x_v + 1
            return x_v

        x_v = paddle.jit.dy2static.convert_ifelse(
            fluid.layers.mean(x_v)[0] > 5, true_fn_3, false_fn_3, (x_v, ),
            (x_v, ), (x_v, ))
        __return_2 = paddle.jit.dy2static.create_bool_as_type(label is not None,
                                                              False)

        def true_fn_4(__return_2, __return_value_1, label, x_v):
            loss = fluid.layers.cross_entropy(x_v, label)
            __return_2 = paddle.jit.dy2static.create_bool_as_type(
                label is not None, True)
            __return_value_1 = loss
            return __return_2, __return_value_1

        def false_fn_4(__return_2, __return_value_1):
            return __return_2, __return_value_1

        __return_2, __return_value_1 = paddle.jit.dy2static.convert_ifelse(
            label is not None, true_fn_4, false_fn_4, (
                __return_2, __return_value_1, label, x_v),
            (__return_2, __return_value_1), (__return_2, __return_value_1))

        def true_fn_5(__return_2, __return_value_1, x_v):
            __return_3 = paddle.jit.dy2static.create_bool_as_type(
                paddle.jit.dy2static.convert_logical_not(__return_2), True)
            __return_value_1 = x_v
            return __return_value_1

        def false_fn_5(__return_value_1):
            return __return_value_1

        __return_value_1 = paddle.jit.dy2static.convert_ifelse(
            paddle.jit.dy2static.convert_logical_not(__return_2), true_fn_5,
            false_fn_5, (__return_2, __return_value_1,
                         x_v), (__return_value_1, ), (__return_value_1, ))
        return __return_value_1


class NetWithError(fluid.dygraph.layers.Layer):
    @declarative
    def forward(self, x):
        linear = fluid.dygraph.Linear(32, 64)
        y = linear(x)
        return y


class TestDygraphToStaticCode(unittest.TestCase):
    def setUp(self):
        # set to print all string diff when assertEqual fails
        self.maxDiff = None

    def test_decorator(self):
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        answer = get_source_code(StaticCode1.dyfunc_with_if_else)
        self.assertEqual(answer, code)

    def test_program_translator(self):
        answer = get_source_code(StaticCode2.dyfunc_with_if_else)
        program_translator = ProgramTranslator()
        code = program_translator.get_code(dyfunc_with_if_else)
        self.assertEqual(answer, code)


class TestEnableDeclarative(unittest.TestCase):
    def setUp(self):
        self.x = np.random.randn(30, 10, 32).astype('float32')
        self.weight = np.random.randn(32, 64).astype('float32')
        self.program_translator = ProgramTranslator()

    def test_raise_error(self):
        with fluid.dygraph.guard():
            self.program_translator.enable(True)
            net = NetWithError()
            with self.assertRaises(ValueError):
                net(fluid.dygraph.to_variable(self.x))

    def test_enable_disable_get_output(self):
        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_output(
                simple_func, self.x, self.weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))

    def test_enable_disable_get_func(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(static_func))
            static_output = static_func(self.x, self.weight)
            self.assertTrue(isinstance(static_output, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_func = self.program_translator.get_func(simple_func)
            self.assertTrue(callable(dygraph_func))
            dygraph_output = dygraph_func(self.x, self.weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_get_program(self):

        self.program_translator.enable(True)
        static_output = self.program_translator.get_program(simple_func, self.x,
                                                            self.weight)
        self.assertTrue(isinstance(static_output, tuple))
        self.assertEqual(len(static_output), 4)
        self.assertTrue(isinstance(static_output[0], fluid.Program))
        self.assertTrue(isinstance(static_output[1], fluid.Program))
        # Check all inputs and outputs are Variable
        for var in static_output[2]:
            self.assertTrue(isinstance(var, fluid.Variable))

        for var in static_output[3]:
            self.assertTrue(isinstance(var, fluid.Variable))

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = self.program_translator.get_program(
                simple_func, self.x, self.weight)
            self.assertTrue(isinstance(dygraph_output, fluid.core.VarBase))

    def test_enable_disable_declarative(self):

        self.program_translator.enable(True)
        with fluid.dygraph.guard():
            static_output = decorated_simple_func(self.x, self.weight)

        self.program_translator.enable(False)
        with fluid.dygraph.guard():
            dygraph_output = decorated_simple_func(self.x, self.weight)
            self.assertTrue(
                np.allclose(
                    static_output.numpy(), dygraph_output.numpy(), atol=1e-4))


class Net(fluid.dygraph.layers.Layer):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        return x + 1


class TestErrorWithInitFromStaticMode(unittest.TestCase):
    def setUp(self):
        self.program_translator = ProgramTranslator()
        self.x = np.random.randn(10, 32).astype('float32')

    def test_raise_error(self):
        # disable imperative
        paddle.enable_static()
        net = Net()

        self.program_translator.enable(True)
        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            self.program_translator.get_output(net.forward, self.x)

        with self.assertRaisesRegexp(RuntimeError,
                                     "only available in dynamic mode"):
            self.program_translator.get_program(net.forward, self.x)


if __name__ == '__main__':
    unittest.main()
