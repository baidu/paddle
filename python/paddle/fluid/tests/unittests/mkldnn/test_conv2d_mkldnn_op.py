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
from paddle.fluid.tests.unittests.test_conv2d_op import TestConv2dOp, TestConv2dOp_v2


def conv2d_bias_naive(out, bias):
    _, out_c, _, _ = out.shape

    for l in range(out_c):
        out[:, l, :, :] = out[:, l, :, :] + bias[l]
    return out


def conv2d_residual_naive(out, residual):
    assert out.shape == residual.shape
    out = np.add(out, residual)
    return out


class TestConv2dMKLDNNOp(TestConv2dOp):
    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        self.data_format = "NCHW"
        self.use_mkldnn = True
        self._cpu_only = True

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def setUp(self):
        self.fuse_bias = False
        self.bias_size = None
        self.fuse_activation = ""
        self.fuse_alpha = 0
        self.fuse_beta = 0
        self.fuse_brelu_threshold = 6.0
        self.fuse_residual_connection = False
        self.input_residual_size = None
        self.dtype = np.float32
        TestConv2dOp.setUp(self)

        output = self.outputs['Output']

        #mkldnn only support either conv-sum-relu, or conv-relu.
        if self.fuse_bias and self.bias_size is not None:
            bias = np.random.random(self.bias_size).astype(self.dtype)
            output = conv2d_bias_naive(output, bias)
            output = output.astype(self.dtype)
            self.attrs['fuse_bias'] = self.fuse_bias
            self.inputs['Bias'] = OpTest.np_dtype_to_fluid_dtype(bias)

        if self.fuse_residual_connection and self.input_residual_size is not None:
            input_residual = np.random.random(self.input_residual_size).astype(
                self.dtype)
            output = conv2d_residual_naive(output, input_residual)

            self.attrs[
                'fuse_residual_connection'] = self.fuse_residual_connection
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                input_residual)

        if self.fuse_activation == "relu":
            output = np.maximum(output, 0).astype(self.dsttype)

        if self.fuse_activation == "relu6":
            output = np.minimum(np.maximum(output, 0),
                                self.fuse_alpha).astype(self.dsttype)
        output = output.astype(self.dtype)

        self.attrs['fuse_bias'] = self.fuse_bias
        self.attrs['fuse_activation'] = self.fuse_activation
        self.attrs['fuse_alpha'] = self.fuse_alpha
        self.attrs['fuse_beta'] = self.fuse_beta
        self.attrs['fuse_brelu_threshold'] = self.fuse_brelu_threshold
        self.attrs['fuse_residual_connection'] = self.fuse_residual_connection

        self.outputs['Output'] = output


class TestWithbreluFusion(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.fuse_activation = "relu6"
        self.fuse_alpha = 6.0
        self.dsttype = np.float32

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass


class TestWithFuse(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.fuse_bias = True
        self.bias_size = [6]
        self.fuse_residual_connection = True
        self.input_residual_size = [2, 6, 5, 5]

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass


class TestWithPadWithBias(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.input_size = [2, 3, 6, 6]


class TestWithStride(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]


class TestWithGroup(TestConv2dMKLDNNOp):
    def init_group(self):
        self.groups = 3


class TestWith1x1(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.filter_size = [6, 3, 1, 1]


class TestWithInput1x1Filter1x1(TestConv2dMKLDNNOp):
    def init_test_case(self):
        TestConv2dMKLDNNOp.init_test_case(self)
        self.input_size = [2, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestConv2dOp_AsyPadding_MKLDNN(TestConv2dOp_v2):
    def init_kernel_type(self):
        self.use_mkldnn = True

    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestConv2dOp_Same_MKLDNN(TestConv2dOp_AsyPadding_MKLDNN):
    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestConv2dOp_Valid_MKLDNN(TestConv2dOp_AsyPadding_MKLDNN):
    def init_paddings(self):
        self.pad = [1, 1]
        self.padding_algorithm = "VALID"


class TestConv2dOp_Valid_NHWC_MKLDNN(TestConv2dOp_Valid_MKLDNN):
    def init_data_format(self):
        self.data_format = "NHWC"

    def init_test_case_2(self):
        N, C, H, W = self.input_size
        self.input_size = [N, H, W, C]

    #TODO(jczaja): Enable once GRAD op is adjusted
    def test_check_grad(self):
        pass

    #TODO(jczaja): Enable once GRAD op is adjusted
    def test_check_grad_no_filter(self):
        pass

    #TODO(jczaja): Enable once GRAD op is adjusted
    def test_check_grad_no_input(self):
        pass


class TestConv2dOp_Same_NHWC_MKLDNN(TestConv2dOp_Valid_NHWC_MKLDNN):
    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "SAME"


class TestConv2dOp_AsyPadding_NHWC_MKLDNN(TestConv2dOp_Valid_NHWC_MKLDNN):
    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


if __name__ == '__main__':
    unittest.main()
