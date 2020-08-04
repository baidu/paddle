#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import random
import paddle.fluid.core as core
from paddle.fluid.op import Operator
from op_test import OpTest


def update_step(param, grad, rows, sq_accum, lin_accum, lr, l1, l2, lr_power):
    smooth_factor = 1e-6

    param_hit = param[rows]
    sq_accum_hit = sq_accum[rows]
    lin_accum_hit = lin_accum[rows]

    new_accum = sq_accum_hit + grad * grad
    if lr_power == -0.5:
        lin_accum_updated = lin_accum_hit + grad - (
            (np.sqrt(new_accum) - np.sqrt(sq_accum_hit)) / lr) * param_hit
    else:
        lin_accum_updated = lin_accum_hit + grad - (
            (np.power(new_accum, -lr_power) - np.power(sq_accum_hit, -lr_power)
             ) / lr) * param_hit

    param_updated = np.zeros(lin_accum_updated.shape).astype("float32")
    for i in range(len(rows)):
        if (np.abs(lin_accum_updated[i]) > l1).sum() == 0:
            continue

        x = l1 * np.sign(lin_accum_updated[i]) - lin_accum_updated[i]
        x *= 1 - l1 / (np.linalg.norm(x) + smooth_factor)

        if lr_power == -0.5:
            y = (np.sqrt(new_accum[i]) / lr) + l2
        else:
            y = (np.power(new_accum[i], -lr_power) / lr) + l2

        param_updated[i] = x / (y + smooth_factor)

    sq_accum_updated = sq_accum_hit + grad * grad

    param_out = param.copy()
    sq_accum_out = sq_accum.copy()
    lin_accum_out = lin_accum.copy()

    for i in range(len(rows)):
        param_out[rows[i]] = param_updated[i]
        sq_accum_out[rows[i]] = sq_accum_updated[i]
        lin_accum_out[rows[i]] = lin_accum_updated[i]

    return param_out, sq_accum_out, lin_accum_out


class TestGFTRLOp(OpTest):
    def setUp(self):
        self.op_type = "gftrl"
        self.conf()
        row_height = 100
        row_numel = 16
        w = np.random.random((row_height, row_numel)).astype("float32")
        g = np.random.random((row_height, row_numel)).astype("float32")
        sq_accum = np.full((row_height, row_numel), 0.1).astype("float32")
        linear_accum = np.full((row_height, row_numel), 0.1).astype("float32")
        lr = np.array([0.01]).astype("float32")
        l1 = self.l1
        l2 = 0.2
        lr_power = self.lr_power

        self.inputs = {
            'Param': w,
            'SquaredAccumulator': sq_accum,
            'LinearAccumulator': linear_accum,
            'Grad': g,
            'LearningRate': lr
        }
        self.attrs = {
            'l1': l1,
            'l2': l2,
            'lr_power': lr_power,
            'learning_rate': lr
        }

        param_out, sq_accum_out, lin_accum_out = update_step(
            w, g,
            range(row_height), sq_accum, linear_accum, lr, l1, l2, lr_power)

        self.outputs = {
            'ParamOut': param_out,
            'SquaredAccumOut': sq_accum_out,
            'LinearAccumOut': lin_accum_out
        }

    def conf(self):
        self.l1 = 0.1
        self.lr_power = -0.5

    def test_check_output(self):
        self.check_output()


class TestGFTRLOp2(TestGFTRLOp):
    def conf(self):
        self.l1 = 0.1
        self.lr_power = -0.6


class TestGFTRLOp3(TestGFTRLOp):
    def conf(self):
        self.l1 = 100000000.0
        self.lr_power = -0.5


class TestSparseGFTRLOp(unittest.TestCase):
    def setUp(self):
        self.l1 = 0.1
        self.lr_power = -0.5

    def check_with_place(self, place):
        self.init_kernel()
        scope = core.Scope()

        height = 100
        rows = random.sample(range(height), 30)
        row_numel = 16
        l1 = self.l1
        l2 = 0.2
        lr_power = self.lr_power

        # create and initialize Param Variable
        param = scope.var('Param').get_tensor()
        param_array = np.random.random((height, row_numel)).astype("float32")
        param.set(param_array, place)

        # create and initialize Grad Variable
        grad = scope.var('Grad').get_selected_rows()
        grad.set_height(height)
        grad.set_rows(rows)
        grad_array = np.random.random((len(rows), row_numel)).astype("float32")

        grad_tensor = grad.get_tensor()
        grad_tensor.set(grad_array, place)

        # create and initialize SquaredAccumulator Variable
        sq_accum = scope.var('SquaredAccumulator').get_tensor()
        sq_accum_array = np.full((height, row_numel), 0.1).astype("float32")
        sq_accum.set(sq_accum_array, place)

        # create and initialize LinearAccumulator Variable
        lin_accum = scope.var('LinearAccumulator').get_tensor()
        lin_accum_array = np.full((height, row_numel), 0.1).astype("float32")
        lin_accum.set(lin_accum_array, place)

        # create and initialize LeraningRate Variable
        lr = scope.var('LearningRate').get_tensor()
        lr_array = np.array([0.01]).astype("float32")
        lr.set(lr_array, place)

        # calculate ground-truth answer
        param_out, sq_accum_out, lin_accum_out = update_step(
            param_array, grad_array, rows, sq_accum_array, lin_accum_array, lr,
            l1, l2, lr_power)

        # create and run operator
        op = Operator(
            "gftrl",
            Param='Param',
            Grad='Grad',
            ParamOut='Param',
            SquaredAccumulator='SquaredAccumulator',
            SquaredAccumOut='SquaredAccumulator',
            LinearAccumulator='LinearAccumulator',
            LinearAccumOut='LinearAccumulator',
            LearningRate='LearningRate',
            l1=l1,
            l2=l2,
            lr_power=lr_power)

        op.run(scope, place)

        # get and compare param result
        param_array = np.array(param)
        sq_accum_array = np.array(sq_accum)
        lin_accum_array = np.array(lin_accum)

        for i in range(height):
            for j in range(row_numel):
                self.assertAlmostEqual(
                    param_out[i][j], param_array[i][j], places=4)
                self.assertAlmostEqual(
                    sq_accum_out[i][j], sq_accum_array[i][j], places=4)
                self.assertAlmostEqual(
                    lin_accum_out[i][j], lin_accum_array[i][j], places=4)

    def init_kernel(self):
        pass

    def test_sparse_gftrl(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place)


class TestSparseGFTRLOp2(TestSparseGFTRLOp):
    def init_kernel(self):
        self.l1 = 0.1
        self.lr_power = -0.6


class TestSparseGFTRLOp3(TestSparseGFTRLOp):
    def init_kernel(self):
        self.l1 = 100000000.0
        self.lr_power = -0.5


if __name__ == "__main__":
    unittest.main()
