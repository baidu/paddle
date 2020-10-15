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
import paddle.fluid as fluid
from paddle.fluid.dygraph.jit import declarative


def dyfunc_tensor_shape_1(x):
    x = fluid.dygraph.to_variable(x)
    res = fluid.layers.reshape(x, shape=x.shape)
    return res


def dyfunc_tensor_shape_2(x):
    x = fluid.dygraph.to_variable(x)
    shape = x.shape
    shape2 = shape
    res = fluid.layers.reshape(x, shape2)
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
    # TODO(liym27):
    #  It will fail to run because `for i in range(len(x.shape))` will be transformed into Paddle while_loop.
    #  Here the python list x.shape will be added to loop_vars. However, loop_vars doesn't support python list.
    #  And the condition of `for i in range(len(x.shape))` only uses the length of x.shape, so it doesn't have to be transformed into Paddle while_loop.
    #  After the AST tranformation of for loop is improved, add TestTensorShapeInFor3.
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
    x = fluid.dygraph.to_variable(x)
    y = numpy.ones(5)
    y_shape_0 = y.shape[0]
    i = 1

    # Transform y_shape_0 but run y.shape[0] actually because y is not Tensor
    while y_shape_0 > i:
        x += 1
        i += 1
    return x


# 1. Basic tests without control flow
class TestTensorShapeBasic(unittest.TestCase):
    def setUp(self):
        self.input = numpy.ones(5).astype("int32")
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        self.init_test_func()

    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_1

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


class TestTensorShapeBasic2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_2


class TestTensorShapeBasic3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_3


class TestTensorShapeBasic4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_4


class TestTensorShapeBasic5(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_tensor_shape_5


# 2. Tests with control flow if
class TestTensorShapeInIf1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_1


class TestTensorShapeInIf2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_if_2


# 3. Tests with control flow for loop
class TestTensorShapeInFor1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_1


class TestTensorShapeInFor2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_for_2


# 4. Tests with control flow while loop
class TestTensorShapeInWhile1(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_1


class TestTensorShapeInWhile2(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_2


class TestTensorShapeInWhile3(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_3


class TestTensorShapeInWhile4(TestTensorShapeBasic):
    def init_test_func(self):
        self.dygraph_func = dyfunc_with_while_4


if __name__ == '__main__':
    unittest.main()
