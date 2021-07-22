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
from op_test import OpTest
import math
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
from paddle.fluid import Program, program_guard


def stable_softmax_comm(x):
    shiftx = (x - np.max(x))
    deno = np.log(np.sum(np.exp(shiftx)))
    comm = shiftx - deno
    return comm


def margin_softmax_with_cross_entropy(logits,
                                      label,
                                      axis,
                                      margin1,
                                      margin2,
                                      margin3,
                                      scale,
                                      reduction=None):
    one_hot_label = np.zeros_like(logits, dtype=logits.dtype)
    for i, lb in enumerate(label):
        one_hot_label[i, lb] = 1.0

    # add arcface margin to logit
    theta = np.arccos(logits)
    if margin1 != 1.0:
        theta = margin1 * theta
    if margin2 != 0.0:
        theta = theta + margin2
    margin_cos = np.cos(theta)
    if margin3 != 0.0:
        margin_cos = margin_cos - margin3
    diff = one_hot_label * (margin_cos - logits)
    arc_logits = (logits + diff) * scale

    comm = np.apply_along_axis(stable_softmax_comm, axis, arc_logits)
    loss = (-one_hot_label * comm).sum(axis=axis, keepdims=True)
    softmax = np.exp(comm)
    if reduction == 'mean':
        loss = np.mean(loss)
    elif reduction == 'sum':
        loss = np.sum(loss)
    return loss, softmax


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOp(OpTest):
    def initParams(self):
        self.op_type = "margin_softmax_with_cross_entropy"
        self.axis = -1
        self.batch_dim = 20
        self.feat_dim = 41
        self.num_class = 37

    def init_loss_params(self):
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        self.dtype = np.float64

    def setUp(self):
        self.initParams()
        self.init_loss_params()
        self.init_dtype()

        datas = np.random.uniform(
            -0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
        datas = datas / np.sqrt(np.sum(np.square(datas), axis=1, keepdims=True))
        weights = np.random.uniform(
            -0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
        weights = weights / np.sqrt(
            np.sum(np.square(weights), axis=0, keepdims=True))
        logits = np.matmul(datas, weights)

        labels = np.random.randint(
            0, self.num_class, (self.batch_dim, ), dtype="int64")

        loss, softmax = margin_softmax_with_cross_entropy(
            logits, labels, self.axis, self.margin1, self.margin2, self.margin3,
            self.scale)

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            'margin1': self.margin1,
            'margin2': self.margin2,
            'margin3': self.margin3,
            'scale': self.scale,
        }

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), atol=1e-5)

    def test_check_grad(self):
        self.check_grad_with_place(core.CUDAPlace(0), ["Logits"], "Loss")


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpFP32(
        TestMarginSoftmaxWithCrossEntropyOp):
    def init_dtype(self):
        self.dtype = np.float32

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CUDAPlace(0), ["Logits"],
            "Loss",
            numeric_grad_delta=5e-2,
            max_relative_error=5e-2)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpFP16(
        TestMarginSoftmaxWithCrossEntropyOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        self.check_output_with_place(core.CUDAPlace(0), atol=5e-2)

    def test_check_grad(self):
        self.check_grad_with_place(
            core.CUDAPlace(0), ["Logits"],
            "Loss",
            numeric_grad_delta=6e-1,
            max_relative_error=6e-1)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpCosFace(
        TestMarginSoftmaxWithCrossEntropyOp):
    def init_loss_params(self):
        self.margin1 = 1.0
        self.margin2 = 0.0
        self.margin3 = 0.35
        self.scale = 2.0


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpSphereFace(
        TestMarginSoftmaxWithCrossEntropyOp):
    def init_loss_params(self):
        self.margin1 = 1.35
        self.margin2 = 0.0
        self.margin3 = 0.0
        self.scale = 2.0


class TestMarginSoftmaxWithCrossEntropyOpCPU(
        TestMarginSoftmaxWithCrossEntropyOp):
    def test_check_output(self):
        try:
            self.check_output_with_place(core.CPUPlace(), atol=1e-5)
        except RuntimeError:
            pass

    def test_check_grad(self):
        try:
            self.check_grad_with_place(core.CPUPlace(), ["Logits"], "Loss")
        except RuntimeError:
            pass


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpV2(unittest.TestCase):
    def setUp(self):
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(self.seed)
        self.places = []
        if core.is_compiled_with_cuda():
            self.places.append(paddle.fluid.CUDAPlace(0))

    def initParams(self):
        self.seed = 2021
        self.axis = -1
        self.batch_dim = 20
        self.feat_dim = 41
        self.num_class = 37
        self.init_loss_params()
        self.init_dtype()
        self.init_reduction()

    def init_loss_params(self):
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        self.dtype = np.float64

    def init_reduction(self):
        self.reduction = None

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def check_static_result(self, place):
        with program_guard(Program(), Program()):
            datas = np.random.uniform(
                -0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
            datas = datas / np.sqrt(
                np.sum(np.square(datas), axis=1, keepdims=True))
            weights = np.random.uniform(
                -0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
            weights = weights / np.sqrt(
                np.sum(np.square(weights), axis=0, keepdims=True))

            logits_np = np.matmul(datas, weights)
            labels_np = np.random.randint(
                0, self.num_class, (self.batch_dim, ), dtype="int64")

            loss_np, softmax_np = margin_softmax_with_cross_entropy(
                logits_np, labels_np, self.axis, self.margin1, self.margin2,
                self.margin3, self.scale, self.reduction)

            logits = paddle.static.data(
                name='logits',
                shape=[self.batch_dim, self.num_class],
                dtype=self.dtype)
            label = paddle.static.data(
                name='label', shape=[self.batch_dim], dtype="int64")
            loss, softmax = paddle.nn.functional.margin_softmax_with_cross_entropy(
                logits,
                label,
                margin1=self.margin1,
                margin2=self.margin2,
                margin3=self.margin3,
                scale=self.scale,
                return_softmax=True,
                reduction=self.reduction)

            exe = paddle.fluid.Executor(place)
            [loss_res, softmax_res] = exe.run(
                paddle.fluid.default_main_program(),
                feed={'logits': logits_np,
                      'label': labels_np},
                fetch_list=[loss, softmax])
            np.testing.assert_allclose(loss_res, loss_np)
            np.testing.assert_allclose(softmax_res, softmax_np)

    def test_dynamic(self):
        for place in self.places:
            self.check_dynamic_result(place=place)

    def check_dynamic_result(self, place):
        with paddle.fluid.dygraph.guard(place):
            datas = np.random.uniform(
                -0.99, 0.99, [self.batch_dim, self.feat_dim]).astype(self.dtype)
            datas = datas / np.sqrt(
                np.sum(np.square(datas), axis=1, keepdims=True))
            weights = np.random.uniform(
                -0.99, 0.99, [self.feat_dim, self.num_class]).astype(self.dtype)
            weights = weights / np.sqrt(
                np.sum(np.square(weights), axis=0, keepdims=True))

            logits_np = np.matmul(datas, weights)
            labels_np = np.random.randint(
                0, self.num_class, (self.batch_dim, ), dtype="int64")

            loss_np, softmax_np = margin_softmax_with_cross_entropy(
                logits_np, labels_np, self.axis, self.margin1, self.margin2,
                self.margin3, self.scale, self.reduction)

            logits = paddle.to_tensor(logits_np, dtype=self.dtype)
            labels = paddle.to_tensor(labels_np, dtype="int64")

            loss, softmax = paddle.nn.functional.margin_softmax_with_cross_entropy(
                logits,
                labels,
                margin1=self.margin1,
                margin2=self.margin2,
                margin3=self.margin3,
                scale=self.scale,
                return_softmax=True,
                reduction=self.reduction)

            loss_res = loss.numpy()
            softmax_res = softmax.numpy()
            np.testing.assert_allclose(loss_res, loss_np)
            np.testing.assert_allclose(softmax_res, softmax_np)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpV3(
        TestMarginSoftmaxWithCrossEntropyOpV2):
    def init_reduction(self):
        self.reduction = 'mean'


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpV4(
        TestMarginSoftmaxWithCrossEntropyOpV2):
    def init_reduction(self):
        self.reduction = 'sum'


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestMarginSoftmaxWithCrossEntropyOpAPIError(unittest.TestCase):
    def setUp(self):
        self.initParams()
        np.random.seed(self.seed)
        paddle.framework.random._manual_program_seed(self.seed)
        self.places = []
        if core.is_compiled_with_cuda():
            self.places.append(paddle.fluid.CUDAPlace(0))

    def initParams(self):
        self.seed = 2021
        self.axis = -1
        self.batch_dim = 20
        self.feat_dim = 41
        self.num_class = 37
        self.init_loss_params()
        self.init_dtype()

    def init_loss_params(self):
        self.margin1 = 1.0
        self.margin2 = 0.5
        self.margin3 = 0.0
        self.scale = 2.0

    def init_dtype(self):
        self.dtype = np.float64

    def test_dynamic_errors(self):
        def test_dim():
            for place in self.places:
                with paddle.fluid.dygraph.guard(place):
                    labels_np = np.random.randint(
                        0, self.num_class, (self.batch_dim, 2), dtype="int64")
                    logits_np = np.random.uniform(
                        -0.99, 0.99,
                        [self.batch_dim, self.num_class]).astype(self.dtype)
                    labels = paddle.to_tensor(labels_np)
                    logits = paddle.to_tensor(logits_np)

                    loss, softmax = paddle.nn.functional.margin_softmax_with_cross_entropy(
                        logits,
                        labels,
                        margin1=self.margin1,
                        margin2=self.margin2,
                        margin3=self.margin3,
                        scale=self.scale,
                        return_softmax=True,
                        reduction=None)

        self.assertRaises(ValueError, test_dim)


if __name__ == '__main__':
    unittest.main()
