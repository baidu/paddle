#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from op_test import OpTest
from test_collective_base import TestDistBase

import random
random.seed(2021)

paddle.enable_static()


def find_output_shape(input_list):
    """Infer output tensor shape according to bcast semantics"""
    output_rank = 0
    for x in input_list:
        rank = len(x.shape)
        output_rank = max(output_rank, rank)

    output_shape = [0 for i in range(output_rank)]
    for i in range(output_rank):
        for x in input_list:
            shape = list(reversed(x.shape))
            size = 1
            if i < len(shape):
                size = shape[i]
            output_shape[i] = max(output_shape[i], size)

    return list(reversed(output_shape))


def make_inputs_outputs(input_shapes, dtype):
    """Automatically generate formatted inputs and outputs from input_shapes"""
    input_list = [
        np.random.random(shape).astype(dtype) for shape in input_shapes
    ]
    output_shape = find_output_shape(input_list)
    output_list = [
        x + np.zeros(output_shape).astype(x.dtype) for x in input_list
    ]

    output_formatted = {
        "Out": [(f"out{i}", output_list[i]) for i in range(len(output_list))]
    }
    input_formatted = {
        "X": [(f"x{i}", input_list[i]) for i in range(len(input_list))]
    }

    return input_formatted, output_formatted


def gen_rank_diff_test(dtype):
    input_shapes = [(2, 60, 1), (6, 2, 1, 10)]
    return make_inputs_outputs(input_shapes, dtype)


def gen_no_broadcast_test(dtype):
    input_shapes = [(12, 1, 10, 1), (12, 1, 10, 1)]
    return make_inputs_outputs(input_shapes, dtype)


def gen_mixed_tensors_test(dtype):
    input_shapes = [(2, 60, 1), (2, 2, 1, 30), (1, 2, 60, 1)]
    return make_inputs_outputs(input_shapes, dtype)


class TestCPUBroadcastTensorsOp(OpTest):
    def set_place(self):
        self.place = core.CPUPlace()

    def set_dtypes(self):
        self.dtypes = ['float64']

    def setUp(self):
        self.op_type = "broadcast_tensors"
        self.use_mkldnn = False
        self.attrs = {'use_mkldnn': self.use_mkldnn}
        self.test_gen_func_list = [
            gen_rank_diff_test, gen_no_broadcast_test, gen_mixed_tensors_test
        ]
        self.set_place()
        self.set_dtypes()

    def run_test(self, test_func, args):
        for dtype in self.dtypes:
            for gen_func in self.test_gen_func_list:
                self.inputs, self.outputs = gen_func(dtype)
                test_func(**args)

    def test_check_output(self):
        self.run_test(self.check_output_with_place,
                      {"place": self.place,
                       "atol": 1e-1})

    def test_check_grad_normal(self):
        self.run_test(self.check_grad_with_place, {
            "place": self.place,
            "inputs_to_check": ['x0', 'x1'],
            "output_names": ['out0', 'out1'],
            "max_relative_error": 0.05,
        })


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestCUDABroadcastTensorsOp(TestCPUBroadcastTensorsOp):
    def set_place(self):
        self.place = core.CUDAPlace(0)

    def set_dtypes(self):
        self.dtypes = ['float64']
        if core.is_float16_supported(self.place):
            self.dtypes.append('float16')


class TestBroadcastTensorsAPI(unittest.TestCase):
    def test_api(self):
        def test_static():
            inputs = [
                paddle.fluid.layers.data(
                    shape=[4, 1, 4, 1], dtype='float32', name="x0"),
                paddle.fluid.layers.data(
                    shape=[1, 4, 1, 4], dtype='float32', name="x1")
            ]
            paddle.broadcast_tensors(inputs)

        def test_dynamic():
            paddle.disable_static()
            try:
                inputs = [
                    paddle.to_tensor(
                        np.random.random([4, 1, 4, 1]).astype("float32")),
                    paddle.to_tensor(
                        np.random.random([1, 4, 1, 4]).astype("float32"))
                ]
                paddle.broadcast_tensors(inputs)
            finally:
                paddle.enable_static()

        test_static()
        test_dynamic()


class TestRaiseBroadcastTensorsError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            inputs = [
                paddle.fluid.layers.data(
                    shape=[1, 1, 1, 1], dtype='float32', name="x4"),
                paddle.fluid.layers.data(
                    shape=[1, 4, 1, 1], dtype='float64', name="x5")
            ]
            paddle.broadcast_tensors(inputs)

        def test_dtype():
            inputs = [
                paddle.fluid.layers.data(
                    shape=[1, 1, 1, 1], dtype='int8', name="x6"),
                paddle.fluid.layers.data(
                    shape=[1, 4, 1, 1], dtype='int8', name="x7")
            ]
            paddle.broadcast_tensors(inputs)

        def test_bcast_semantics():
            inputs = [
                paddle.fluid.layers.data(
                    shape=[1, 3, 1, 1], dtype='float32', name="x9"),
                paddle.fluid.layers.data(
                    shape=[1, 8, 1, 1], dtype='float32', name="x10")
            ]
            paddle.broadcast_tensors(inputs)

        self.assertRaises(TypeError, test_type)
        self.assertRaises(TypeError, test_dtype)
        self.assertRaises(TypeError, test_bcast_semantics)


if __name__ == '__main__':
    unittest.main()
