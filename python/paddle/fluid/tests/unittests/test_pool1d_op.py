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

import numpy as np
import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool1D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=False,
                             adaptive=False,
                             data_type=np.float64):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     L - ksize[0] + 2 * paddings[0]) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        out[:, :, i] = np.max(x_masked, axis=(2))
    return out


def avg_pool1D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=False,
                             adaptive=False,
                             data_type=np.float64):
    N, C, L = x.shape
    if global_pool == 1:
        ksize = [L]
    if adaptive:
        L_out = ksize[0]
    else:
        L_out = (L - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     L - ksize[0] + 2 * paddings[0]) // strides[0] + 1

    out = np.zeros((N, C, L_out))
    for i in range(L_out):
        if adaptive:
            r_start = adaptive_start_index(i, L, ksize[0])
            r_end = adaptive_end_index(i, L, ksize[0])
        else:
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], L))
        x_masked = x[:, :, r_start:r_end]

        field_size = (r_end - r_start) \
            if (exclusive or adaptive) else (ksize[0])
        if data_type == np.int8 or data_type == np.uint8:
            out[:, :, i] = (np.rint(
                np.sum(x_masked, axis=(2, 3)) / field_size)).astype(data_type)
        else:
            out[:, :, i] = (np.sum(x_masked, axis=(2)) /
                            field_size).astype(data_type)
    return out


class TestPool1d_API(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_avg_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 32], dtype="float32")
            result = F.avg_pool1d(input, kernel_size=2, stride=2, padding=0)

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0], ceil_mode=False)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_avg_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.avg_pool1d(input, kernel_size=2, stride=2, padding=[0])

            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0])

            self.assertTrue(np.allclose(result.numpy(), result_np))

            avg_pool1d_dg = paddle.nn.layer.AvgPool1d(
                kernel_size=2, stride=None, padding=0)
            result = avg_pool1d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_max_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 32], dtype="float32")
            result = F.max_pool1d(input, kernel_size=2, stride=2, padding=[0])

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0])

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_max_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.max_pool1d(input, kernel_size=2, stride=2, padding=0)

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0])

            self.assertTrue(np.allclose(result.numpy(), result_np))

            max_pool1d_dg = paddle.nn.layer.MaxPool1d(
                kernel_size=2, stride=None, padding=0)
            result = max_pool1d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_adaptive_max_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.adaptive_max_pool1d(input, output_size=16)

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[16], strides=[0], paddings=[0], adaptive=True)
            self.assertTrue(np.allclose(result.numpy(), result_np))

            ada_max_pool1d_dg = paddle.nn.layer.AdaptiveMaxPool1d(
                output_size=16)
            result = ada_max_pool1d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_adaptive_avg_dygraph_results(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.adaptive_avg_pool1d(input, output_size=16)
            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[16], strides=[0], paddings=[0], adaptive=True)

            self.assertTrue(np.allclose(result.numpy(), result_np))

            ada_max_pool1d_dg = paddle.nn.layer.AdaptiveAvgPool1d(
                output_size=16)
            result = ada_max_pool1d_dg(input)
            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_adaptive_max_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 32], dtype="float32")
            result = F.adaptive_max_pool1d(input, output_size=16)

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = max_pool1D_forward_naive(
                input_np, ksize=[16], strides=[2], paddings=[0], adaptive=True)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_adaptive_avg_static_results(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[2, 3, 32], dtype="float32")
            result = F.adaptive_avg_pool1d(input, output_size=16)

            input_np = np.random.random([2, 3, 32]).astype("float32")
            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[16], strides=[2], paddings=[0], adaptive=True)

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input": input_np},
                              fetch_list=[result])
            self.assertTrue(np.allclose(fetches[0], result_np))

    def check_max_dygraph_padding_same(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.max_pool1d(
                input, kernel_size=2, stride=2, padding="SAME")

            result_np = max_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0])

            self.assertTrue(np.allclose(result.numpy(), result_np))

    def check_avg_dygraph_padding_same(self, place):
        with fluid.dygraph.guard(place):
            input_np = np.random.random([2, 3, 32]).astype("float32")
            input = fluid.dygraph.to_variable(input_np)
            result = F.avg_pool1d(
                input, kernel_size=2, stride=2, padding="SAME")

            result_np = avg_pool1D_forward_naive(
                input_np, ksize=[2], strides=[2], paddings=[0])

            self.assertTrue(np.allclose(result.numpy(), result_np))

    def test_pool1d(self):
        for place in self.places:

            self.check_max_dygraph_results(place)
            self.check_avg_dygraph_results(place)
            self.check_max_static_results(place)
            self.check_avg_static_results(place)
            self.check_adaptive_max_dygraph_results(place)
            self.check_adaptive_avg_dygraph_results(place)
            self.check_adaptive_max_static_results(place)
            self.check_adaptive_avg_static_results(place)
            self.check_max_dygraph_padding_same(place)
            self.check_avg_dygraph_padding_same(place)


class TestPool2dError_API(unittest.TestCase):
    def test_error_api(self):
        def run1():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = [[2]]
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding)

        self.assertRaises(ValueError, run1)

        def run2():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = [[2]]
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding)

        self.assertRaises(ValueError, run2)

        def run3():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = "padding"
                res_pd = F.max_pool1d(
                    input_pd, kernel_size=2, stride=2, padding=padding)

        self.assertRaises(ValueError, run3)

        def run4():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.max_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True)

        self.assertRaises(ValueError, run4)

        def run5():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.max_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True)

        self.assertRaises(ValueError, run5)

        def run6():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = "VALID"
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True)

        self.assertRaises(ValueError, run6)

        def run7():
            with fluid.dygraph.guard():
                input_np = np.random.uniform(-1, 1,
                                             [2, 3, 32]).astype(np.float32)
                input_pd = fluid.dygraph.to_variable(input_np)
                padding = "paddle"
                res_pd = F.avg_pool1d(
                    input_pd,
                    kernel_size=2,
                    stride=2,
                    padding=padding,
                    ceil_mode=True)

        self.assertRaises(ValueError, run7)


if __name__ == '__main__':
    unittest.main()
