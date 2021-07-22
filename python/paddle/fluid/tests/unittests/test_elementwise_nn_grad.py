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

import unittest
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
import gradient_checker

from decorator_helper import prog_scope


class TestElementwiseMulDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 7, 3, 3]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseMulBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 7, 3, 3]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_mul(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseAddBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_add(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseSubBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.005
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_sub(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[:-1]).astype(dtype)

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseDivDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape, False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=0)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestElementwiseDivBroadcastDoubleGradCheck(unittest.TestCase):
    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        shape = [2, 3, 4, 5]
        eps = 0.0001
        dtype = np.float64

        x = layers.data('x', shape, False, dtype)
        y = layers.data('y', shape[1:-1], False, dtype)
        x.persistable = True
        y.persistable = True
        out = layers.elementwise_div(x, y, axis=1)
        x_arr = np.random.uniform(-1, 1, shape).astype(dtype)
        y_arr = np.random.uniform(-1, 1, shape[1:-1]).astype(dtype)
        y_arr[np.abs(y_arr) < 0.005] = 0.02

        gradient_checker.double_grad_check(
            [x, y], out, x_init=[x_arr, y_arr], place=place, eps=eps, atol=1e-3)

    def test_grad(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == "__main__":
    unittest.main()
