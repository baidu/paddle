# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.static import InputSpec
from paddle.fluid.dygraph.dygraph_to_static.function_spec import FunctionSpec

from test_declarative import foo_func

import unittest

paddle.enable_static()


class TestFunctionSpec(unittest.TestCase):
    def test_constructor(self):
        foo_spec = FunctionSpec(foo_func)
        args_name = foo_spec.args_name
        self.assertListEqual(args_name, ['a', 'b', 'c', 'd'])
        self.assertTrue(foo_spec.dygraph_function == foo_func)
        self.assertTrue(foo_spec.input_spec is None)

    def test_verify_input_spec(self):
        a_spec = InputSpec([None, 10], name='a')
        b_spec = InputSpec([10], name='b')

        # type(input_spec) should be list or tuple
        with self.assertRaises(TypeError):
            foo_spec = FunctionSpec(foo_func, input_spec=a_spec)

        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        self.assertTrue(len(foo_spec.flat_input_spec) == 2)

    def test_unified_args_and_kwargs(self):
        foo_spec = FunctionSpec(foo_func)
        # case 1: foo(10, 20, c=4)
        args, kwargs = foo_spec.unified_args_and_kwargs([10, 20], {'c': 4})
        self.assertTupleEqual(args, (10, 20, 4, 2))
        self.assertTrue(len(kwargs) == 0)

        # case 2: foo(a=10, b=20, d=4)
        args, kwargs = foo_spec.unified_args_and_kwargs(
            [], {'a': 10,
                 'b': 20,
                 'd': 4})
        self.assertTupleEqual(args, (10, 20, 1, 4))
        self.assertTrue(len(kwargs) == 0)

        # case 3: foo(10, b=20)
        args, kwargs = foo_spec.unified_args_and_kwargs([10], {'b': 20})
        self.assertTupleEqual(args, (10, 20, 1, 2))
        self.assertTrue(len(kwargs) == 0)

        # assert len(self._arg_names) >= len(args)
        with self.assertRaises(ValueError):
            foo_spec.unified_args_and_kwargs([10, 20, 30, 40, 50], {'c': 4})

        # assert arg_name should be in kwargs
        with self.assertRaises(ValueError):
            foo_spec.unified_args_and_kwargs([10], {'c': 4})

    def test_args_to_input_spec(self):
        a_spec = InputSpec([None, 10], name='a')
        b_spec = InputSpec([10], name='b')

        a_tensor = paddle.static.data(name='a_var', shape=[4, 10])
        b_tensor = paddle.static.data(name='b_var', shape=[4, 10])
        kwargs = {'c': 1, 'd': 2}

        # case 1
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        input_with_spec, _ = foo_spec.args_to_input_spec(
            (a_tensor, b_tensor, 1, 2), {})

        self.assertTrue(len(input_with_spec) == 4)
        self.assertTrue(input_with_spec[0] == a_spec)  # a
        self.assertTrue(input_with_spec[1] == b_spec)  # b
        self.assertTrue(input_with_spec[2] == 1)  # c
        self.assertTrue(input_with_spec[3] == 2)  # d

        # case 2
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec])
        input_with_spec, _ = foo_spec.args_to_input_spec((a_tensor, b_tensor),
                                                         {})
        self.assertTrue(len(input_with_spec) == 2)
        self.assertTrue(input_with_spec[0] == a_spec)  # a
        self.assertTupleEqual(input_with_spec[1].shape, (4, 10))  # b.shape
        self.assertEqual(input_with_spec[1].name, 'b_var')  # b.name

        # case 3
        # assert kwargs is None if set `input_spec`
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec])
        with self.assertRaises(ValueError):
            input_with_spec = foo_spec.args_to_input_spec((a_tensor, b_tensor),
                                                          {'c': 4})

        # case 4
        # assert len(args) >= len(self._input_spec)
        foo_spec = FunctionSpec(foo_func, input_spec=[a_spec, b_spec])
        with self.assertRaises(ValueError):
            input_with_spec = foo_spec.args_to_input_spec((a_tensor, ), {})


if __name__ == '__main__':
    unittest.main()
