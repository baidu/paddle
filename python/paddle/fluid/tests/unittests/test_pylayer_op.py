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
from paddle.nn.layer import PyLayer


class TestPyLayer(unittest.TestCase):
    def test_simple_pylayer_multiple_output(self):
        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                y2 = func1(x2)
                return y1, y2

            @staticmethod
            def backward(ctx, x1, x2, y1, y2, dy1, dy2):

                re1 = dy1 * (1 - ctx.func(y1))
                re2 = dy2 * (1 - paddle.square(y2))
                return re1, re2

        input1 = paddle.randn([2, 3]).astype("float64")
        input2 = input1.detach().clone()
        input1.stop_gradient = False
        input2.stop_gradient = False
        z = tanh.apply(input1, input1, paddle.tanh, paddle.square)
        z = z[0] + z[1]
        z.mean().backward()

        z2 = paddle.tanh(input2) + paddle.tanh(input2)
        z2.mean().backward()

        self.assertTrue(np.max(np.abs((input1.grad - input2.grad))) < 1e-10)

    def test_simple_pylayer_single_output(self):
        class tanh(PyLayer):
            @staticmethod
            def forward(ctx, x1, func1, func2=paddle.square):
                ctx.func = func2
                y1 = func1(x1)
                return y1

            @staticmethod
            def backward(ctx, x1, y1, dy1):
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

        self.assertTrue(np.max(np.abs((input1.grad - input2.grad))) < 1e-10)

    def test_pylayer_Exception_forward(self):
        class Layer_None1(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return None

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_None1.apply(input1)

        class Layer_None2(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return [None, None]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_None2.apply(input1)

        class Layer_one1(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return 1

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_one1.apply(input1)

        class Layer_one2(PyLayer):
            @staticmethod
            def forward(ctx, *args):
                return [1, 2]

            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_one2.apply(input1)

        class Layer_no_fw(PyLayer):
            @staticmethod
            def backward(ctx, *args):
                return args

        input1 = paddle.randn([2, 3]).astype("float64")
        with self.assertRaises(NotImplementedError):
            z = Layer_no_fw.apply(input1)

    def test_pylayer_nograd(self):
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

    def test_pylayer_Exception_bk(self):
        class Layer_bk_none1(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2

            @staticmethod
            def backward(ctx, x1, y1, dy1):
                return None

        input2 = paddle.randn([2, 3]).astype("float64")
        input2.stop_gradient = False
        z = Layer_bk_none1.apply(input2)

        with self.assertRaises(NotImplementedError):
            with paddle.fluid.dygraph.guard():
                z.sum().backward()

        class Layer_bk_none2(PyLayer):
            @staticmethod
            def forward(ctx, x1, x2):
                return x1 + x2

            @staticmethod
            def backward(ctx, x1, x2, y1, dy1):
                return None, x1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_none2.apply(input1, input1)
        with self.assertRaises(NotImplementedError):
            with paddle.fluid.dygraph.guard():
                z.mean().backward()

        class Layer_bk_one1(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x + x

            @staticmethod
            def backward(ctx, x, y, dy):
                return 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_one1.apply(input1)
        with self.assertRaises(NotImplementedError):
            with paddle.fluid.dygraph.guard():
                z.mean().backward()

        class Layer_bk_one2(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

            @staticmethod
            def backward(ctx, *args):
                return 1, 1

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_one1.apply(input1)
        with self.assertRaises(NotImplementedError):
            with paddle.fluid.dygraph.guard():
                z.mean().backward()

        class Layer_no_bk(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_no_bk.apply(input1)

        with self.assertRaises(NotImplementedError):
            with paddle.fluid.dygraph.guard():
                z = z[0] + z[1]
                z.mean().backward()

        class Layer_bk_match(PyLayer):
            @staticmethod
            def forward(ctx, x):
                return x * 2, x * 5

            @staticmethod
            def backward(ctx, x, y1, y2, dy1, dy2):
                return x * 2, y1 * 2

        input1 = paddle.randn([2, 3]).astype("float64")
        input1.stop_gradient = False
        z = Layer_bk_match.apply(input1)
        with self.assertRaises(ValueError):
            with paddle.fluid.dygraph.guard():
                z = z[0] + z[1]
                z.mean().backward()


if __name__ == '__main__':
    unittest.main()
