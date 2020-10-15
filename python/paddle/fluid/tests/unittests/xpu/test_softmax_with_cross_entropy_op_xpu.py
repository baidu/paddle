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

import unittest
import numpy as np
import sys
sys.path.append("..")
import paddle
import paddle.fluid.core as core

from op_test import OpTest
from test_softmax_op import stable_softmax


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)


class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.soft_label = False
        self.dtype = np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]

    def setUp(self):
        self.initParams()

        logits = getattr(
            self, "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype))
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype(self.dtype)
            labels /= np.sum(labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                             self.ignore_index)

        self.inputs = {"Logits": logits, "Label": labels}
        self.outputs = {
            "Softmax": softmax.astype(self.dtype),
            "Loss": loss.astype(self.dtype)
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
        }
        if self.ignore_index >= 0:
            self.attrs['ignore_index'] = self.ignore_index
        if self.axis != -1:
            self.attrs['axis'] = self.axis

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=1e-2)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ["Logits"], "Loss", max_relative_error=0.1)


class TestXPUSoftmaxWithCrossEntropyOp(TestSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=1e-2)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ["Logits"], "Loss", max_relative_error=0.1)


class TestXPUSoftmaxWithCrossEntropyOp2(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with soft labels.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.dtype = np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [41, 37]

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=1e-2)

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ["Logits"], "Loss", max_relative_error=0.1)


class TestXPUSoftmaxWithCrossEntropyOp3(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with ignore_index.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [41, 37]
        self.ignore_index = 5
        self.axis = -1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpAxis1(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 0
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestXPUSoftmaxWithCrossEntropyOpAxis2(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestXPUSoftmaxWithCrossEntropyOpAxis3(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 2
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestXPUSoftmaxWithCrossEntropyOpAxis4(TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = 3
        self.ignore_index = -1
        self.shape = [3, 5, 7, 11]


class TestXPUSoftmaxWithCrossEntropyOpAxisDimEqualOne(
        TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    Given axis != -1
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.dtype = np.float64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [3, 5, 7, 1]


class TestXPUSoftmaxWithCrossEntropyOpSoftLabelAxis1(
        TestXPUSoftmaxWithCrossEntropyOp):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 0
        self.ignore_index = -1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpSoftLabelAxis2(
        TestXPUSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 1
        self.ignore_index = -1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpSoftLabelAxis3(
        TestXPUSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 2
        self.ignore_index = -1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpSoftLabelAxis4(
        TestXPUSoftmaxWithCrossEntropyOp2):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = True
        self.shape = [3, 5, 7, 11]
        self.axis = 3
        self.ignore_index = -1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis1(
        TestXPUSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 1
        self.axis = 0
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis2(
        TestXPUSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 0
        self.axis = 1
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis3(
        TestXPUSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 3
        self.axis = 2
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpIgnoreIndexNoCudnnAxis4(
        TestXPUSoftmaxWithCrossEntropyOp3):
    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.ignore_index = 3
        self.axis = 3
        self.dtype = np.float64


class TestXPUSoftmaxWithCrossEntropyOpBoundary0(
        TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test stable softmax with cross entropy operator will not product INF
    with small logits value.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float64
        self.logits = np.full(self.shape, -500.0).astype(self.dtype)


class TestXPUSoftmaxWithCrossEntropyOpBoundary1(
        TestXPUSoftmaxWithCrossEntropyOp):
    """
    Test stable softmax with cross entropy operator will not product INF
    with small logits value.
    """

    def initParams(self):
        self.op_type = "softmax_with_cross_entropy"
        self.numeric_stable_mode = True
        self.soft_label = False
        self.shape = [3, 5, 7, 11]
        self.axis = -1
        self.ignore_index = -1
        self.dtype = np.float64
        self.logits = np.full(self.shape, 1000.0).astype(self.dtype)
        self.logits[:, :, 0, :] = -1000.0


if __name__ == "__main__":
    unittest.main()
