# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest
from paddle import enable_static
from paddle.fluid.tests.unittests.test_conv2d_transpose_op import conv2dtranspose_forward_naive, TestConv2DTransposeOp


def conv2d_bias_naive(out, bias):
    _, out_c, _, _ = out.shape

    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out


class TestConv2DTransposeMKLDNNOp(TestConv2DTransposeOp):
    def test_check_grad(self):
        return

    def test_check_grad_no_input(self):
        return

    def test_check_grad_no_filter(self):
        return

    def init_op_type(self):
        self.data_format = "NCHW"
        self.op_type = "conv2d_transpose"
        self._cpu_only = True

    def init_test_case(self):
        self.use_mkldnn = True
        self.is_test = True
        self.pad = [0, 0]
        self.fuse_bias = False
        self.bias_size = None
        self.fuse_activation = ""
        self.fuse_alpha = 0.0
        self.fuse_beta = 0.0
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 3, 3]
        self.groups = 1
        self.dtype = np.float32

    def setUp(self):

        TestConv2DTransposeOp.setUp(self)

        output = self.outputs['Output']

        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(self.dtype)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(self.dtype)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)

        if self.fuse_activation == "relu":
            output = np.maximum(output, 0).astype(self.dtype)
        output = output.astype(self.dtype)

        self.attrs['fuse_activation'] = self.fuse_activation
        self.attrs['fuse_alpha'] = self.fuse_alpha
        self.attrs['fuse_beta'] = self.fuse_beta
        self.attrs['mkldnn_data_type'] = 'float32'
        self.attrs['force_fp32_output'] = False

        self.outputs['Output'] = output


class TestMKLDNNFuseBias(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]


class TestMKLDNNWithPad(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.input_size = [2, 3, 10, 10]


class TestMKLDNNWithStride(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW


class TestMKLDNNWithAsymPad(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestMKLDNNWithSamePad(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestMKLDNNWithValidPad(TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.padding_algorithm = "VALID"


class TestMKLDNNWithValidPad_NHWC(TestMKLDNNWithValidPad):
    def init_test_case(self):
        super(TestMKLDNNWithValidPad_NHWC, self).init_test_case()
        self.data_format = "NHWC"
        N, C, H, W = self.input_size
        self.input_size = [N, H, W, C]


class TestConv2DTransposeMKLDNNWithDilationsExplicitPad(
        TestConv2DTransposeMKLDNNOp):
    def init_test_case(self):
        TestConv2DTransposeMKLDNNOp.init_test_case(self)
        self.stride = [2, 1]
        self.dilations = [1, 2]
        self.groups = 1
        self.input_size = [4, 3, 8, 7]  # NCHW
        f_c = self.input_size[1]
        self.filter_size = [f_c, 6, 4, 3]
        self.pad = [1, 3, 2, 1]
        self.padding_algorithm = "EXPLICIT"


if __name__ == '__main__':
    enable_static()
    unittest.main()
