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

import numpy

import unittest
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative


def dyfunc_tensor_shape_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=x.shape)
    return res


def dyfunc_tensor_shape_2(x):
    x = paddle.to_tensor(x)
    shape = x.shape
    shape2 = shape
    res = paddle.reshape(x, shape2)
    return res


def dyfunc_tensor_shape_3(x):
    # Transform y.shape but run y.shape actually because y is not Tensor
    x = fluid.dygraph.to_variable(x)
    y = numpy.ones(5)
    res = fluid.layers.reshape(x, shape=y.shape)
    return res


def dyfunc_tensor_shape_4(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=(-1, x.shape[0], len(x.shape)))
    return res


def dyfunc_tensor_shape_5(x):
    # `res = fluid.layers.reshape(x, shape=(-1, s))` to
    # `res = fluid.layers.reshape(x, shape=(-1,
    #           paddle.jit.dy2static.convert_var_shape(x)[0]))`
    x = fluid.dygraph.to_variable(x)
    s = x.shape[0]
    res = fluid.layers.reshape(x, shape=(-1, s))
    return res


def dyfunc_tensor_shape_6(x):
    # `res = fluid.layers.reshape(x, shape=(-1, s))` to
    # `res = fluid.layers.reshape(x, shape=(-1,
    #           paddle.jit.dy2static.convert_var_shape(x)[0:]))`
    x = fluid.dygraph.to_variable(x)
    s = x.shape[0:]
    res = fluid.layers.reshape(x, shape=s)
    return res


def dyfunc_tuple_shape_1(x):
    x = paddle.to_tensor(x)
    a, b = x.shape
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_tuple_shape_2(x):
    x = paddle.to_tensor(x)
    shape = x.shape
    a, b = shape
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_tuple_shape_3(x):
    x = paddle.to_tensor(x)
    a, b = paddle.shape(x)
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_paddle_shape_api(x):
    x = paddle.to_tensor(x)
    # paddle.shape will not be converted.
    a = paddle.shape(x)[0]
    # alias api will also not be converted.
    alias_old_api = paddle.fluid.layers
    b = alias_old_api.shape(x)[1]
    res = paddle.reshape(x, shape=(b, a))
    return res


def dyfunc_with_if_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, [-1, 1])
    x_shape_0 = x.shape[0]
    if x_shape_0 < 1:
        # `res.shape[0]` is transformed into
        #   `paddle.jit.dy2static.convert_var_shape(res)[0]`
        if res.shape[0] > 1:
            res = fluid.layers.fill_constant(
                value=2, shape=x.shape, dtype="int32")
        else:
            res = fluid.layers.fill_constant(
                value=3, shape=x.shape, dtype="int32")
    return res


def dyfunc_with_if_2(x):
    x = fluid.dygraph.to_variable(x)
    # `len(x.shape)` will not be transformed because x.shape is not used by Paddle api.
    if len(x.shape) < 1:
        res = x
    else:
        res = fluid.layers.fill_constant(value=8, shape=x.shape, dtype="int32")

    return res


def dyfunc_with_for_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")
    # `x.shape[0]` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    for i in range(x.shape[0]):
        res += 1
    return res


def dyfunc_with_for_2(x):
    x = fluid.dygraph.to_variable(x)
    x_shape_0 = x.shape[0]
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")

    # `x_shape_0` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    for i in range(x_shape_0):
        res += 1
    return res


def dyfunc_with_for_3(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")
    # `len(x.shape)` is not transformed.
    for i in range(len(x.shape)):
        res += 1

    return res


def dyfunc_with_while_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")
    # `x.shape[0]` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    i = 1
    while i < x.shape[0]:
        res += 1
        i = i + 2
    return res


def dyfunc_with_while_2(x):
    x = fluid.dygraph.to_variable(x)
    x_shape_0 = x.shape[0]
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")
    i = 1
    # `x_shape_0` is transformed into `paddle.jit.dy2static.convert_var_shape(x)[0]`
    while i < x_shape_0:
        res += 1
        i = i + 2
    return res


def dyfunc_with_while_3(x):
    x = fluid.dygraph.to_variable(x)
    x_shape = x.shape
    res = fluid.layers.fill_constant(value=0, shape=[1], dtype="int32")
    i = 1

    # `len(x.shape)` is not transformed.
    while len(x_shape) > i:
        res += 1
        i += 1
    return res


def dyfunc_with_while_4(x):
    x = paddle.to_tensor(x)
    y = numpy.ones(5)
    y_shape_0 = y.shape[0]
    i = 1

    # Transform y_shape_0 but run y.shape[0] actually because y is not Tensor
    while y_shape_0 > i:
        x += 1
        i += 1
    return x


def dyfunc_change_shape_after_assign(x):
    x = paddle.to_tensor(x)
    a, b = x.shape
    x = paddle.reshape(x, shape=(-1, 1))
    res = paddle.reshape(x, shape=(b, a))
    return res


# 1. Basic tests without control flow
class TestTensorShapeBasic(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self._set_input_spec()
        self._set_expected_op_num()
        self.init_test_func()

    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_1

    def _set_input_spec(self):
        self.input_spec = [paddle.static.InputSpec(shape=[5], dtype="int32")]

    def _run(self, to_static):
        with fluid.dygraph.guard():
            if to_static:
                res = declarative(self.dygraph_func)(self.input).numpy()
            else:
                res = self.dygraph_func(self.input).numpy()
            return res

    def get_dygraph_output(self):
        return self._run(to_static=False)

    def get_static_output(self):
        return self._run(to_static=True)

    def test_transformed_static_result(self):
        static_res = self.get_static_output()
        dygraph_res = self.get_dygraph_output()
        self.assertTrue(
            numpy.allclose(dygraph_res, static_res),
            msg='dygraph res is {}\nstatic_res is {}'.format(dygraph_res,
                                                             static_res))

    def _set_expected_op_num(self):
        self.expected_op_num = 2
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0

    def _compute_op_num(self, program):
        self.op_num = sum([len(block.ops) for block in program.blocks])
        self.shape_op_num = 0
        self.slice_op_num = 0

        for block in program.blocks:
            self.shape_op_num += len(
                [op for op in block.ops if op.type == "shape"])
            self.slice_op_num += len(
                [op for op in block.ops if op.type == "slice"])

    def test_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program
        self._compute_op_num(program)
        self.assertEqual(self.op_num, self.expected_op_num)
        self.assertEqual(self.shape_op_num, self.expected_shape_op_num)
        self.assertEqual(self.slice_op_num, self.expected_slice_op_num)


class TestTensorShapeBasic2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_2

    def _set_expected_op_num(self):
        self.expected_op_num = 3
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 0


class TestTensorShapeBasic3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_3


class TestTensorShapeBasic4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_4


class TestTensorShapeBasic5(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_5

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestTensorShapeBasic6(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_6

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestTupleShape1(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = numpy.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_tuple_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2


class TestTupleShape2(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = numpy.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_tuple_shape_2

    def _set_expected_op_num(self):
        self.expected_op_num = 5
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 2


class TestTupleShape3(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = numpy.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_tuple_shape_3

    def _set_expected_op_num(self):
        self.expected_op_num = 5
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 2


class TestPaddleShapeApi(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = numpy.ones((5, 7)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[5, 7], dtype="int32")]
        self.dygraph_func = dyfunc_paddle_shape_api

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2


# 2. Tests with control flow if
class TestTensorShapeInIf1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_1

    def _set_expected_op_num(self):
        self.expected_op_num = 4
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestTensorShapeInIf2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_2

    def _set_expected_op_num(self):
        self.expected_op_num = 14
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 1


# 3. Tests with control flow for loop
class TestTensorShapeInFor1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_1

    def _set_expected_op_num(self):
        self.expected_op_num = 22
        self.expected_shape_op_num = 3
        self.expected_slice_op_num = 3


class TestTensorShapeInFor2(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_2

    def _set_expected_op_num(self):
        self.expected_op_num = 9
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestTensorShapeInFor3(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_3

    def _set_expected_op_num(self):
        self.expected_op_num = 25
        self.expected_shape_op_num = 6
        self.expected_slice_op_num = 3


# 4. Tests with control flow while loop
class TestTensorShapeInWhile1(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_1


class TestTensorShapeInWhile2(TestTensorShapeInFor1):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_2

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestTensorShapeInWhile3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_3

    def _set_expected_op_num(self):
        self.expected_op_num = 3
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 0


class TestTensorShapeInWhile4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_4

    def _set_expected_op_num(self):
        self.expected_op_num = 5
        self.expected_shape_op_num = 0
        self.expected_slice_op_num = 0


# 5. Test op num for negetive dim
class TestOpNumBasicWithTensorShape(unittest.TestCase):
    def setUp(self):
        self._set_input_spec()
        self._set_test_func()
        self._set_expected_op_num()

    def _set_input_spec(self):
        self.input_spec = [
            paddle.static.InputSpec(
                shape=[-1, 5], dtype="int32")
        ]

    def _set_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 3
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 0

    def _compute_op_num(self, program):
        self.op_num = sum([len(block.ops) for block in program.blocks])
        self.shape_op_num = 0
        self.slice_op_num = 0

        for block in program.blocks:
            self.shape_op_num += len(
                [op for op in block.ops if op.type == "shape"])
            self.slice_op_num += len(
                [op for op in block.ops if op.type == "slice"])

    def test_op_num(self):
        static_layer = paddle.jit.to_static(self.dygraph_func, self.input_spec)
        program = static_layer.main_program

        self._compute_op_num(program)
        self.assertEqual(self.op_num, self.expected_op_num)
        self.assertEqual(self.shape_op_num, self.expected_shape_op_num)
        self.assertEqual(self.slice_op_num, self.expected_slice_op_num)


class TestOpNumBasicWithTensorShape4(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_4

    def _set_expected_op_num(self):
        self.expected_op_num = 6
        self.expected_shape_op_num = 1
        self.expected_slice_op_num = 1


class TestOpNumWithTensorShapeTuple1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_tuple_shape_1

    def _set_expected_op_num(self):
        self.expected_op_num = 7
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2


class TestOpNumWithTensorShapeInIf1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_if_1

    def _set_expected_op_num(self):
        self.expected_op_num = 28
        self.expected_shape_op_num = 4
        self.expected_slice_op_num = 2


class TestOpNumWithTensorShapeInFor1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_for_1

    def _set_expected_op_num(self):
        self.expected_op_num = 22
        self.expected_shape_op_num = 3
        self.expected_slice_op_num = 3


class TestOpNumWithTensorShapeInWhile1(TestOpNumBasicWithTensorShape):
    def _set_test_func(self):
        self.dygraph_func = dyfunc_with_while_1

    def _set_expected_op_num(self):
        self.expected_op_num = 22
        self.expected_shape_op_num = 3
        self.expected_slice_op_num = 3


class TestChangeShapeAfterAssign(TestTensorShapeBasic):
    def init_test_func(self):
        self.input = numpy.ones((2, 3)).astype("int32")
        self.input_spec = [paddle.static.InputSpec(shape=[2, 3], dtype="int32")]
        self.dygraph_func = dyfunc_change_shape_after_assign

    def _set_expected_op_num(self):
        self.expected_op_num = 7
        self.expected_shape_op_num = 2
        self.expected_slice_op_num = 2


def dyfunc_with_static_convert_var_shape(x):
    # Note: this will create `batch_size__static_convert_var_shape_suffix_0` firstly.
    batch_size = x.shape[0]
    if len(x.shape) < 1:
        res = x
    else:
        # Test for correctly to find `batch_size__static_convert_var_shape_suffix_0` in
        # deeply nested scope.
        res = fluid.layers.fill_constant(
            value=8, shape=[batch_size], dtype="int32")

    return res


class TestFindStatiConvertVarShapeSuffixVar(unittest.TestCase):
    def test(self):
        x_spec = paddle.static.InputSpec(shape=[None, 10])
        func = paddle.jit.to_static(dyfunc_with_if_2, input_spec=[x_spec])
        # Call this function to trigger program translation.
        func.concrete_program


if __name__ == '__main__':
    unittest.main()
