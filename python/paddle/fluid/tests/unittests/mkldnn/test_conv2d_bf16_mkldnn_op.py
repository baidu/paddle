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
import struct

import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16
from paddle.fluid.tests.unittests.test_conv2d_op import conv2d_forward_naive, TestConv2dOp


def conv2d_forward_refer(input, filter, group, conv_param):
    out, in_n, out_h, out_w, out_c = conv2d_forward_naive(input, filter, group,
                                                          conv_param)
    return out


def conv2d_residual_naive(out, residual):
    assert out.shape == residual.shape
    out = np.add(out, residual)
    return out


@unittest.skipIf(not core.supports_bfloat16(),
                 "place does not support BF16 evaluation")
class TestConv2dBf16Op(TestConv2dOp):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = True
        self._cpu_only = True
        self.weight_type = np.float32
        self.input_type = np.float32
        self.mkldnn_data_type = "bfloat16"
        self.force_fp32_output = False
        self.init_group()
        self.init_dilation()
        self.init_test_case()
        self.init_fuse_relu()
        self.init_fuse_residual()
        self.init_data_type()
        self.init_force_fp32_output()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }
        self.input = np.random.random(self.input_size).astype(np.float32)
        self.filter = np.random.random(self.filter_size).astype(np.float32)
        conv_out, _, _, _, _ = conv2d_forward_naive(self.input, self.filter,
                                                    self.groups, conv2d_param)
        self.conv_output_float = conv_out

        if self.fuse_residual:
            self.input_residual = np.random.random(
                self.input_residual_size).astype(np.float32)
            self.conv_output_float = conv2d_residual_naive(
                self.conv_output_float, self.input_residual)
            self.conv_output = convert_float_to_uint16(self.conv_output_float)
            self.outputs = {'Output': self.conv_output}
        elif self.force_fp32_output:
            self.outputs = {'Output': self.conv_output_float.astype(np.float32)}

        if self.input_type is not np.float32:
            self.input = convert_float_to_uint16(self.input)

        self.inputs = {
            'Input': self.input.view(self.input_type),
            'Filter': OpTest.np_dtype_to_fluid_dtype(
                self.filter.astype(self.weight_type))
        }

        if self.fuse_residual:
            self.inputs['ResidualData'] = OpTest.np_dtype_to_fluid_dtype(
                convert_float_to_uint16(self.input_residual))

        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'mkldnn_data_type': self.mkldnn_data_type,
            'force_fp32_output': self.force_fp32_output,
            'fuse_residual_connection': self.fuse_residual
        }

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())

    def test_check_grad(self):
        pass

    def test_check_grad_no_filter(self):
        pass

    def test_check_grad_no_input(self):
        pass

    def init_test_case(self):
        TestConv2dOp.init_test_case(self)
        self.input_size = [1, 1, 5, 5]  # NCHW
        f_c = self.input_size[1] // self.groups
        self.input_residual_size = [1, 2, 3, 3]
        self.filter_size = [2, f_c, 3, 3]

    def init_data_type(self):
        self.weight_type = np.float32
        self.input_type = np.float32

    def init_force_fp32_output(self):
        self.force_fp32_output = False

    def init_fuse_relu(self):
        self.fuse_activation = "relu"

    def init_fuse_residual(self):
        self.fuse_residual = True


class TestConv2d(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_data_type(self):
        self.input_type = np.uint16


class TestWithPad(TestConv2d):
    def init_test_case(self):
        TestConv2d.init_test_case(self)
        self.pad = [1, 1]
        self.input_residual_size = [2, 6, 5, 5]


class TestWithGroup(TestConv2d):
    def init_group(self):
        self.groups = 3


class TestWithStride(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]
        self.input_residual_size = [2, 6, 3, 3]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_data_type(self):
        self.input_type = np.uint16


class TestWith1x1ForceFP32Output(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [1, 3, 5, 5]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_force_fp32_output(self):
        self.force_fp32_output = True

    def init_fuse_residual(self):
        self.fuse_residual = False


class TestWithInput1x1Filter1x1(TestConv2dBf16Op):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 1, 1]
        self.input_residual_size = [2, 6, 1, 1]
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


if __name__ == '__main__':
    unittest.main()
