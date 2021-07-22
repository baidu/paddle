# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import numpy as np

import paddle
from paddle.autograd import PyLayer
import warnings, time, sys


def add_warn(msg, line):
    time.sleep(0.005)
    warnings.warn(msg + str(line))
    time.sleep(0.005)


class FakeTensor(paddle.fluid.core.VarBase):
    def __init__(self):
        pass


class TestPyLayer(unittest.TestCase):
    def test_simple_pylayer_multiple_output(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return y1, 1, y2, None

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, y2 = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return re1, re2

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input1, paddle.tanh, paddle.square)
        z = z[0] + z[2]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input2)
        z2.mean().backward()

        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_simple_pylayer_return_none_with_no_grad(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                ctx.save_for_backward(y1, y2)
                return 1, None, y1, y2, ''

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, y2 = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return re1, None

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input3 = input1.detach().clone()
        input4 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        input3.stop_gradient = True
        input4.stop_gradient = True
        z = tanh.apply(input1, input3, paddle.tanh, paddle.square)
        z = z[2] + z[3]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input4)
        z2.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)
        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)

    def test_simple_pylayer_single_output(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                ctx.save_for_backward(y1)
                return y1

            @staticmethod
            def backward(ctx, dy1):
                y1, = ctx.saved_tensor()
                re1 = dy1 * (1 - ctx.func(y1))
                return re1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(x1=input1, func1=paddle.tanh)
        z.mean().backward()
        z2 = paddle.tanh(input2)
        z2.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)

        self.assertTrue(
            np.max(np.abs((input1.grad.numpy() - input2.grad.numpy()))) < 1e-10)

    def test_pylayer_num_output_match(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(
                    ctx,
                    x1,
                    x2, ):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                return dy1 + 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input2)

        with self.assertRaises(ValueError):
            z.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_dtype(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x, dtype):
                y = paddle.cast(x, dtype)
                return y

            @staticmethod
            def backward(ctx, dy1):
                return dy1

        dtypes = [
            'bool', 'float16', 'float32', 'float64', 'uint8', 'int32', 'int64'
        ]
        for dtype in dtypes:
            input1 = (paddle.randn([2, 3]))
            input1.stop_gradient = False
            self.assertTrue(input1.grad is None)

            z = tanh.apply(input1, dtype)
            z = paddle.cast(z, "float32")
            z.sum().backward()
            self.assertTrue(input1.grad is not None)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_Exception_forward(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Layer_None1(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return None

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(ValueError):
            z = Layer_None1.apply(input1)

        class Layer_None2(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return [None, args[0]]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # return None
        z = Layer_None2.apply(input1)

        class Layer_one1(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return 1

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # At least one output of `PyLayer.backward` is a `Tensor`
        with self.assertRaises(ValueError):
            z = Layer_one1.apply(input1)

        class Layer_one2(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return [1, 2, args[0]]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        # return int 
        z = Layer_one2.apply(input1)

        class Layer_no_fw(PyLayer):
            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_no_fw.apply(input1)

        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_nograd(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square, xx=None):
                ctx.func = func2
                y1 = func1(x1)
                return y1

            @staticmethod
            def backward(ctx, x1, y1, dy1):
                re1 = dy1 * (1 - ctx.func(y1))
                return re1

        input1 = paddle.randn([2, 3]).astype("float64")
        z = tanh.apply(input1, paddle.tanh, paddle.square)
        z.mean().backward()
        self.assertTrue(z.grad is None)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_Exception_bk(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Layer_bk_none1(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, dy1):
                return None

        input2 = paddle.randn([2, 3]).astype("float64")
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input2)

        with self.assertRaises(ValueError):
            z.sum().backward()

        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Layer_bk_none2(PyLayer):
            add_warn("pylayer: ", sys._getframe().f_lineno)

            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy1):
                return None, dy1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input1)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_one1(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x + x

            @staticmethod
            def backward(ctx, dy):
                return 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_one1.apply(input1)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_one2(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2):
                return x1 * 2, x2 * 5

            @staticmethod
            def backward(ctx, *args):
                return 1, 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False

        y = Layer_bk_one2.apply(input1, input1)
        z = y[0] + y[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_no_bk(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_no_bk.apply(input1)

        with self.assertRaises(OSError):
            z = z[0] + z[1]
            z.mean().backward()

        class Layer_bk_match(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy2 * 2, dy1 * 2

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_match.apply(input1)
        with self.assertRaises(ValueError):
            z = z[0] + z[1]
            z.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_bk_return_none(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Layer_bk_none1(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + x2

            @staticmethod
            def backward(ctx, dy):
                return 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input1, input2)

        with self.assertRaises(ValueError):
            z.mean().backward()

        class Layer_bk_none2(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2):
                return x1 * 2, x2 * 5

            @staticmethod
            def backward(ctx, *args):
                return 1, 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = True
        input2.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input2)
        z = z[0] + z[1]
        with self.assertRaises(ValueError):
            z.mean().backward()

        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_pylayer_inplace(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class cus_tanh(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x

            @staticmethod
            def backward(ctx, dy):
                return dy

        class Layer(paddle.nn.Layer):
            def __init__(self):
                super(Layer, self).__init__()

            def forward(self, data):
                data = paddle.nn.functional.relu(data)
                z = paddle.tanh(data)
                z = cus_tanh.apply(data)
                return z.mean()

        for i in range(2):
            data = paddle.ones([2, 3], dtype="float64") / (i + 1)
            data.stop_gradient = False
            layer = Layer()
            z = layer(data)
            z.backward()
            self.assertTrue(data.grad is not None)

        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_backward_in_backward(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class cus_tanh(PyLayer):
            @staticmethod
            def forward(ctx, x):
                temp = x.detach()
                ctx.inputs = temp
                return x.mean()

            @staticmethod
            def backward(ctx, dy):
                with paddle.set_grad_enabled(True):
                    temp = ctx.inputs
                    temp.stop_gradient = False
                    z = paddle.tanh(temp)
                    z.backward()
                    self.assertTrue(temp.grad is not None)
                    return paddle.to_tensor(temp.grad)

        for i in range(2):
            data = paddle.ones([2, 3], dtype="float32") / (i + 1)
            data.stop_gradient = False
            data = paddle.nn.functional.relu(data)
            z = paddle.tanh(data)
            z = cus_tanh.apply(data)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_return_to_tensor(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):
                y1 = paddle.tanh(x1)
                ctx.save_for_backward(y1)
                tensor_1 = paddle.to_tensor([1, 2], dtype='float32')
                return y1, 5, None, "helloworld", tensor_1

            @staticmethod
            def backward(ctx, dy1, dy2):
                y1, = ctx.saved_tensor()
                re1 = dy1 * (1 - paddle.square(y1))
                return dy1

        input1 = paddle.randn([2, 3]).astype("float32")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z, number, none_item, string_item, tensor1 = Tanh.apply(x1=input1)
        z.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)


class TestPyLayerReturnType(unittest.TestCase):
    def test_forward_args_fake_tensor(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):
                y1 = FakeTensor()
                return y1, x1

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = FakeTensor()

        with self.assertRaises(ValueError):
            y1, y2 = Tanh.apply(input1)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_forward_kwargs_fake_tensor(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):

                return x1

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = FakeTensor()

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_forward_return_fake_tensor(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):

                return FakeTensor()

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = paddle.randn([3, 2])

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_forward_return_fake_tensor_tuple(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):

                return FakeTensor(), FakeTensor()

            @staticmethod
            def backward(ctx, dy1, dy2):
                return dy1

        input1 = paddle.randn([3, 2])

        with self.assertRaises(ValueError):
            y = Tanh.apply(x1=input1)
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_backward_return_fake_tensor_tuple(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + 1, x1 + 2

            @staticmethod
            def backward(ctx, dy1, dy2):

                return FakeTensor(), 2

        input1 = paddle.randn([3, 2])
        input1.stop_gradient = False
        y, _ = Tanh.apply(input1, 1 + input1)

        with self.assertRaises(ValueError):
            y.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)

    def test_backward_return_fake_tensor(self):
        add_warn("pylayer: ", sys._getframe().f_lineno)

        class Tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1):
                return x1 + 1, x1 + 2

            @staticmethod
            def backward(ctx, dy1, dy2):
                return FakeTensor()

        input1 = paddle.randn([3, 2])
        input1.stop_gradient = False
        y, _ = Tanh.apply(input1)

        with self.assertRaises(ValueError):
            y.mean().backward()
        add_warn("pylayer: ", sys._getframe().f_lineno)


if __name__ == '__main__':
    unittest.main()
