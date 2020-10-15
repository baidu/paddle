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

import paddle
import numpy as np
import sys
import unittest
from functools import reduce
sys.path.append("..")
from op_test import OpTest
from operator import mul

paddle.enable_static()


def ref_layer_norm(x, scale, bias, epsilon, begin_norm_axis=1):
    x_shape = x.shape
    left = reduce(mul, x_shape[0:begin_norm_axis], 1)
    right = reduce(mul, x_shape[begin_norm_axis:len(x_shape)], 1)
    x.shape = [left, right]
    mean = np.mean(x, axis=1)
    variance = np.var(x, axis=1) + epsilon
    y = np.divide((x - mean.reshape([left, 1])),
                  (np.sqrt(variance)).reshape([left, 1]))
    if scale is not None:
        y = scale.reshape([1, right]) * y
    if bias is not None:
        y = y + bias.reshape([1, right])
    x.shape, y.shape = x_shape, x_shape
    return y, mean, variance


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULayerNormOp(OpTest):
    def setUp(self):
        self.op_type = "layer_norm"
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.epsilon = 1e-05
        self.begin_norm_axis = 1
        self.set_attrs()

        right = reduce(mul, self.shape[self.begin_norm_axis:len(self.shape)], 1)
        np.random.seed(10)
        x_np = np.random.uniform(0.1, 1, self.shape).astype(self.dtype)
        scale_np = np.random.uniform(0.1, 1, [right]).astype(self.dtype)
        bias_np = np.random.uniform(0.1, 1, [right]).astype(self.dtype)
        ref_y_np, ref_mean_np, ref_variance_np = ref_layer_norm(
            x_np, scale_np, bias_np, self.epsilon, self.begin_norm_axis)

        self.inputs = {'X': x_np, 'Scale': scale_np, 'Bias': bias_np}
        self.outputs = {
            'Y': ref_y_np,
            'Mean': ref_mean_np,
            'Variance': ref_variance_np
        }
        self.attrs = {'begin_norm_axis': self.begin_norm_axis, 'use_xpu': True}

    def set_attrs(self):
        pass

    def test_check_output(self):
        self.check_output_with_place(paddle.XPUPlace(0), atol=1e-4)

    def test_check_grad(self):
        self.check_grad_with_place(
            paddle.XPUPlace(0), ['X'], 'Y', max_relative_error=0.02)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULayerNormOpAxis2(TestXPULayerNormOp):
    def set_attrs(self):
        self.begin_norm_axis = 2


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULayerNormOpAxis3(TestXPULayerNormOp):
    def set_attrs(self):
        self.begin_norm_axis = 3


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULayerNormOp2D(TestXPULayerNormOp):
    def set_attrs(self):
        self.shape = [10, 12]


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULayerNormOp3D(TestXPULayerNormOp):
    def set_attrs(self):
        self.shape = [4, 5, 6]


if __name__ == "__main__":
    unittest.main()
