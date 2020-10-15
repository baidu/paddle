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

import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest


def dmc_bilinear(data_im, height, width, h, w):
    h_low = int(np.floor(h))
    w_low = int(np.floor(w))
    h_high = h_low + 1
    w_high = w_low + 1

    lh = h - h_low
    lw = w - w_low
    hh = 1 - lh
    hw = 1 - lw

    v1 = 0
    if h_low >= 0 and w_low >= 0:
        v1 = data_im[h_low, w_low]
    v2 = 0
    if h_low >= 0 and w_high <= width - 1:
        v2 = data_im[h_low, w_high]
    v3 = 0
    if h_high <= height - 1 and w_low >= 0:
        v3 = data_im[h_high, w_low]
    v4 = 0
    if h_high <= height - 1 and w_high <= width - 1:
        v4 = data_im[h_high, w_high]

    w1, w2, w3, w4 = hh * hw, hh * lw, lh * hw, lh * lw
    val = w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4

    return val


def dconv_im2col_gemm(input, offset, mask, filter, group, conv_param):
    in_n, in_c, in_h, in_w = input.shape
    out_c, f_c, f_h, f_w = filter.shape

    assert offset.shape == (in_n, 2 * f_h * f_w, in_h, in_w)
    assert mask.shape == (in_n, f_h * f_w, in_h, in_w)
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0

    stride, pad, dilation = conv_param['stride'], conv_param['pad'],\
        conv_param['dilation']
    out_h = 1 + (in_h + 2 * pad[0] - (dilation[0] * (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + 2 * pad[1] - (dilation[1] * (f_w - 1) + 1)) // stride[1]
    assert out_h == in_h
    assert out_w == in_w

    col_buffer = np.zeros((in_n, in_c * f_h * f_w, in_h * in_w))
    for n in range(in_n):
        for c in range(in_c):
            for h in range(out_h):
                for w in range(out_w):
                    for kh in range(f_h):
                        for kw in range(f_w):
                            offset_h_table = \
                                    offset[n, ::2, h, w].reshape(f_h, f_w)
                            offset_w_table = \
                                    offset[n, 1::2, h, w].reshape(f_h, f_w)
                            mask_table = \
                                mask[n, :, h, w].reshape(f_h, f_w)
                            offset_h = offset_h_table[kh, kw]
                            offset_w = offset_w_table[kh, kw]
                            val = 0
                            im_h = h * stride[0] + kh * dilation[0] \
                                + offset_h - pad[0]
                            im_w = w * stride[0] + kw * dilation[0] \
                                + offset_w - pad[1]
                            if im_h > -1 and im_w > -1 and \
                                im_h < in_h and im_w < in_h:
                                val = dmc_bilinear(input[n, c], in_h, in_w,
                                                   im_h, im_w)
                            val_out = val * mask_table[kh, kw]
                            col_buffer[n, c * f_h * f_w + kh * f_w + kw, h *
                                       in_w + w] = val_out

    out = np.zeros((in_n, group, int(out_c // group), out_h * out_w))
    weight = filter.reshape(group, int(out_c // group), f_c * f_h * f_w)
    col_buffer = col_buffer.reshape(
        (in_n, group, int(in_c // group * f_h * f_w), in_h * in_w))
    for n in range(in_n):
        for g in range(group):
            out[n, g] = np.matmul(weight[g], col_buffer[n, g])
    out = out.reshape(in_n, out_c, out_h, out_w)
    return out


class TestModulatedDeformableConvOp(OpTest):
    def setUp(self):
        self.op_type = "deformable_conv"
        self.dtype = np.float32
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        offset = 10 * np.random.random(self.offset_size).astype(self.dtype)
        mask = 10 * np.random.random(self.mask_size).astype(self.dtype)
        filter = np.random.random(self.filter_size).astype(self.dtype)

        output = dconv_im2col_gemm(input, offset, mask, filter, self.groups,
                                   conv_param)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Offset': OpTest.np_dtype_to_fluid_dtype(offset),
            'Mask': OpTest.np_dtype_to_fluid_dtype(mask),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'deformable_groups': self.deformable_groups,
            'im2col_step': self.im2col_step,
            'dilations': self.dilations,
        }
        self.outputs = {'Output': output}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(
            {'Input', 'Offset', 'Mask', 'Filter'},
            'Output',
            max_relative_error=0.05)

    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.dilations = [1, 1]
        self.input_size = [2, 8, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [4, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1


class TestWithStride(TestModulatedDeformableConvOp):
    def init_test_case(self):
        self.pad = [3, 3]
        self.stride = [2, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]


class TestWithDilation(TestModulatedDeformableConvOp):
    def init_test_case(self):
        self.pad = [2, 2]
        self.stride = [1, 1]
        self.input_size = [4, 3, 4, 4]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]

    def init_dilation(self):
        self.dilations = [2, 2]


class TestWith3x3(TestModulatedDeformableConvOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]
        self.im2col_step = 1
        self.deformable_groups = 1
        offset_c = 2 * self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        mask_c = self.deformable_groups * self.filter_size[
            2] * self.filter_size[3]
        self.offset_size = [
            self.input_size[0], offset_c, self.input_size[2], self.input_size[3]
        ]
        self.mask_size = [
            self.input_size[0], mask_c, self.input_size[2], self.input_size[3]
        ]


class TestWithGroup(TestModulatedDeformableConvOp):
    def init_group(self):
        self.groups = 2


class TestModulatedDeformableConvInvalidInput(unittest.TestCase):
    def test_error(self):
        def test_invalid_input():
            paddle.enable_static()
            input = [1, 3, 32, 32]
            offset = fluid.data(
                name='offset', shape=[None, 3, 32, 32], dtype='float32')
            mask = fluid.data(
                name='mask', shape=[None, 3, 32, 32], dtype='float32')
            loss = fluid.layers.deformable_conv(
                input, offset, mask, num_filters=4, filter_size=1)

        self.assertRaises(TypeError, test_invalid_input)

        def test_invalid_offset():
            paddle.enable_static()
            input = fluid.data(
                name='input', shape=[None, 3, 32, 32], dtype='int32')
            offset = fluid.data(
                name='offset', shape=[None, 3, 32, 32], dtype='float32')
            mask = fluid.data(
                name='mask', shape=[None, 3, 32, 32], dtype='float32')
            loss = fluid.layers.deformable_conv(
                input, offset, mask, num_filters=4, filter_size=1)

        self.assertRaises(TypeError, test_invalid_offset)


class TestDeformConv2dAPI(unittest.TestCase):
    def test_api(self):
        def test_deform_conv2d_v1():
            paddle.enable_static()
            input = paddle.static.data(
                name='input_v1', shape=[None, 3, 32, 32], dtype='float32')
            offset = paddle.static.data(
                name='offset_v1', shape=[None, 4, 32, 32], dtype='float32')
            out = paddle.static.nn.deform_conv2d(
                input, offset, None, num_filters=4, filter_size=1)

            assert (out.shape == (-1, 4, 32, 32))

        test_deform_conv2d_v1()

        def test_deform_conv2d_v2():
            paddle.enable_static()
            input = paddle.static.data(
                name='input_v2', shape=[None, 3, 32, 32], dtype='float32')
            offset = paddle.static.data(
                name='offset_v2', shape=[None, 4, 32, 32], dtype='float32')
            mask = paddle.static.data(
                name='mask_v2', shape=[None, 2, 32, 32], dtype='float32')
            out = paddle.static.nn.deform_conv2d(
                input, offset, mask, num_filters=4, filter_size=1)

            assert (out.shape == (-1, 4, 32, 32))

        test_deform_conv2d_v2()


if __name__ == '__main__':
    unittest.main()
