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
import sys
sys.path.append("..")
import unittest
import numpy as np

import paddle.fluid.core as core
import paddle.fluid as fluid
from op_test import OpTest
import paddle
from paddle.fluid import Program, program_guard


def conv2d_forward_naive(input,
                         filter,
                         group,
                         conv_param,
                         padding_algorithm='EXPLICIT',
                         data_format='NCHW'):
    if padding_algorithm not in ["SAME", "VALID", "EXPLICIT"]:
        raise ValueError("Unknown Attr(padding_algorithm): '%s'. "
                         "It can only be 'SAME' or 'VALID'." %
                         str(padding_algorithm))

    if data_format not in ["NCHW", "NHWC"]:
        raise ValueError("Unknown Attr(data_format): '%s' ."
                         "It can only be 'NCHW' or 'NHWC'." % str(data_format))

    channel_last = (data_format == "NHWC")
    if channel_last:
        input = np.transpose(input, [0, 3, 1, 2])

    in_n, in_c, in_h, in_w = input.shape
    f_n, f_c, f_h, f_w = filter.shape
    out_n = in_n
    out_c = f_n
    assert f_c * group == in_c
    assert np.mod(out_c, group) == 0
    sub_out_c = out_c // group
    sub_f_n = f_n // group

    stride, pad, dilation = conv_param['stride'], conv_param['pad'], conv_param[
        'dilation']

    # update pad and dilation
    def _get_padding_with_SAME(input_shape, pool_size, pool_stride):
        padding = []
        for input_size, filter_size, stride_size in zip(input_shape, pool_size,
                                                        pool_stride):
            out_size = int((input_size + stride_size - 1) / stride_size)
            pad_sum = np.max((
                (out_size - 1) * stride_size + filter_size - input_size, 0))
            pad_0 = int(pad_sum / 2)
            pad_1 = int(pad_sum - pad_0)
            padding.append(pad_0)
            padding.append(pad_1)
        return padding

    ksize = filter.shape[2:4]
    if padding_algorithm == "VALID":
        pad = [0, 0, 0, 0]
    elif padding_algorithm == "SAME":
        dilation = [1, 1]
        input_data_shape = input.shape[2:4]
        pad = _get_padding_with_SAME(input_data_shape, ksize, stride)

    pad_h_0, pad_h_1 = pad[0], pad[0]
    pad_w_0, pad_w_1 = pad[1], pad[1]
    if len(pad) == 4:
        pad_h_0, pad_h_1 = pad[0], pad[1]
        pad_w_0, pad_w_1 = pad[2], pad[3]
    out_h = 1 + (in_h + pad_h_0 + pad_h_1 - (dilation[0] *
                                             (f_h - 1) + 1)) // stride[0]
    out_w = 1 + (in_w + pad_w_0 + pad_w_1 - (dilation[1] *
                                             (f_w - 1) + 1)) // stride[1]
    out = np.zeros((out_n, out_c, out_h, out_w))

    d_bolck_h = (dilation[0] * (f_h - 1) + 1)
    d_bolck_w = (dilation[1] * (f_w - 1) + 1)

    input_pad = np.pad(input, ((0, 0), (0, 0), (pad_h_0, pad_h_1),
                               (pad_w_0, pad_w_1)),
                       mode='constant',
                       constant_values=0)

    filter_dilation = np.zeros((f_n, f_c, d_bolck_h, d_bolck_w))
    filter_dilation[:, :, 0:d_bolck_h:dilation[0], 0:d_bolck_w:dilation[
        1]] = filter

    for i in range(out_h):
        for j in range(out_w):
            for g in range(group):
                input_pad_masked = \
                    input_pad[:, g * f_c:(g + 1) * f_c,
                    i * stride[0]:i * stride[0] + d_bolck_h,
                    j * stride[1]:j * stride[1] + d_bolck_w]

                f_sub = filter_dilation[g * sub_f_n:(g + 1) * sub_f_n, :, :, :]
                # sub_f_n == sub_out_c
                for k in range(sub_out_c):
                    # Multiplication of Corresponding Elements, then sum all
                    out[:, g * sub_out_c + k, i, j] = \
                        np.sum(input_pad_masked * f_sub[k, :, :, :],
                               axis=(1, 2, 3))

    if channel_last:
        out = np.transpose(out, [0, 2, 3, 1])

    return out, in_n, out_h, out_w, out_c


def create_test_channel_last_class(parent):
    class TestChannelLastCase(parent):
        def init_data_format(self):
            self.data_format = "NHWC"

        def init_test_case_2(self):
            N, C, H, W = self.input_size
            self.input_size = [N, H, W, C]

    cls_name = "{0}_{1}".format(parent.__name__, "ChannelLast")
    TestChannelLastCase.__name__ = cls_name
    globals()[cls_name] = TestChannelLastCase


def create_test_padding_SAME_class(parent):
    class TestPaddingSMAECase(parent):
        def init_paddings(self):
            self.pad = [0, 0]
            self.padding_algorithm = "SAME"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingSAMEOp")
    TestPaddingSMAECase.__name__ = cls_name
    globals()[cls_name] = TestPaddingSMAECase


def create_test_padding_VALID_class(parent):
    class TestPaddingVALIDCase(parent):
        def init_paddings(self):
            self.pad = [1, 1]
            self.padding_algorithm = "VALID"

    cls_name = "{0}_{1}".format(parent.__name__, "PaddingVALIDOp")
    TestPaddingVALIDCase.__name__ = cls_name
    globals()[cls_name] = TestPaddingVALIDCase


class TestConv2dOp(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.data_format = "AnyLayout"
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_test_case()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)

        output, _, _, _, _ = conv2d_forward_naive(input2, filter, self.groups,
                                                  conv2d_param)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'fuse_relu_before_depthwise_conv':
            self.fuse_relu_before_depthwise_conv,
            'exhaustive_search': self.exhaustive_search
        }
        self.outputs = {'Output': output}

    def has_cuda(self):
        return core.is_compiled_with_cuda() and (self.use_cudnn or
                                                 self.use_cuda)

    def test_check_output(self):
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output')

    def test_check_grad_no_filter(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ['Input'], 'Output', no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        if self.dtype == np.float16 or (hasattr(self, "no_need_check_grad") and
                                        self.no_need_check_grad == True):
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ['Filter'], 'Output', no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_test_case_2(self):
        pass

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass


class TestWithPad(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithStride(TestConv2dOp):
    def init_test_case(self):
        self.pad = [1, 1]
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]


class TestWithGroup(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [18, f_c, 3, 3]


class TestWith1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


class TestWithDilation(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [12, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3


class TestWithInput1x1Filter1x1(TestConv2dOp):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 1]
        self.input_size = [100, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3


# Please Don't remove the following code.
# Currently, CI use cudnn V5.0 which not support dilation conv.
# class TestCUDNNWithDilation(TestWithDilation):
#     def init_op_type(self):
#         self.op_type = "conv_cudnn"

# ---- test asymmetric padding ----


class TestConv2dOp_v2(OpTest):
    def setUp(self):
        self.op_type = "conv2d"
        self.use_cudnn = False
        self.exhaustive_search = False
        self.use_cuda = False
        self.use_mkldnn = False
        self.fuse_relu_before_depthwise_conv = False
        self.dtype = np.float64
        self.init_kernel_type()
        self.init_group()
        self.init_dilation()
        self.init_data_format()
        self.init_test_case()
        self.init_paddings()
        self.init_test_case_2()

        conv2d_param = {
            'stride': self.stride,
            'pad': self.pad,
            'dilation': self.dilations
        }

        input = np.random.random(self.input_size).astype(self.dtype)
        if not self.has_cuda():
            self.fuse_relu_before_depthwise_conv = False
        if self.fuse_relu_before_depthwise_conv:
            input = input - 0.5
            input -= (input < 0) * 0.1
            input += (input >= 0) * 0.1
            input2 = np.maximum(input, 0.0)
        else:
            input2 = input
        filter = np.random.uniform(-1, 1, self.filter_size).astype(self.dtype)
        output, _, _, _, _ = conv2d_forward_naive(
            input2, filter, self.groups, conv2d_param, self.padding_algorithm,
            self.data_format)
        output = output.astype(self.dtype)

        self.inputs = {
            'Input': OpTest.np_dtype_to_fluid_dtype(input),
            'Filter': OpTest.np_dtype_to_fluid_dtype(filter)
        }
        self.attrs = {
            'strides': self.stride,
            'paddings': self.pad,
            'padding_algorithm': self.padding_algorithm,
            'groups': self.groups,
            'dilations': self.dilations,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'data_format': self.data_format,
            'fuse_relu_before_depthwise_conv':
            self.fuse_relu_before_depthwise_conv,
            'exhaustive_search': self.exhaustive_search
        }
        self.outputs = {'Output': output}

    def has_cuda(self):
        return core.is_compiled_with_cuda() and (self.use_cudnn or
                                                 self.use_cuda)

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_check_grad(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, {'Input', 'Filter'}, 'Output')

    def test_check_grad_no_filter(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ['Input'], 'Output', no_grad_set=set(['Filter']))

    def test_check_grad_no_input(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        if self.dtype == np.float16:
            return
        if core.is_compiled_with_xpu():
            paddle.enable_static()
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(
                place, ['Filter'], 'Output', no_grad_set=set(['Input']))

    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 4, 3]

    def init_dilation(self):
        self.dilations = [1, 1]

    def init_group(self):
        self.groups = 1

    def init_kernel_type(self):
        pass

    def init_paddings(self):
        self.pad = [0, 0]
        self.padding_algorithm = "EXPLICIT"

    def init_data_format(self):
        self.data_format = "NCHW"

    def init_test_case_2(self):
        pass


class TestConv2dOp_AsyPadding(TestConv2dOp_v2):
    def init_paddings(self):
        self.pad = [0, 0, 1, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithPad_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithStride_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.stride = [2, 2]
        self.input_size = [2, 3, 6, 6]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [6, f_c, 3, 3]

    def init_paddings(self):
        self.pad = [2, 1, 3, 2]
        self.padding_algorithm = "EXPLICIT"


class TestWithGroup_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.pad = [0, 0]
        self.stride = [1, 2]
        self.input_size = [2, 3, 5, 5]  # NCHW
        self.group = 3
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 4, 3]


class TestWith1x1_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 5, 5]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [2, 2, 4, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithDilation_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [2, 3, 10, 10]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [24, f_c, 3, 3]

    def init_dilation(self):
        self.dilations = [2, 2]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 1, 3, 0]
        self.padding_algorithm = "EXPLICIT"


class TestWithInput1x1Filter1x1_AsyPadding(TestConv2dOp_v2):
    def init_test_case(self):
        self.stride = [1, 1]
        self.input_size = [40, 3, 1, 1]  # NCHW
        assert np.mod(self.input_size[1], self.groups) == 0
        f_c = self.input_size[1] // self.groups
        self.filter_size = [120, f_c, 1, 1]

    def init_group(self):
        self.groups = 3

    def init_paddings(self):
        self.pad = [0, 3, 4, 0]
        self.padding_algorithm = "EXPLICIT"


#---------- test SAME VALID -----------
create_test_padding_SAME_class(TestConv2dOp_AsyPadding)
create_test_padding_SAME_class(TestWithPad_AsyPadding)
create_test_padding_SAME_class(TestWithStride_AsyPadding)
create_test_padding_SAME_class(TestWithGroup_AsyPadding)
create_test_padding_SAME_class(TestWithInput1x1Filter1x1_AsyPadding)

create_test_padding_VALID_class(TestConv2dOp_AsyPadding)
create_test_padding_VALID_class(TestWithPad_AsyPadding)
create_test_padding_VALID_class(TestWithStride_AsyPadding)
create_test_padding_VALID_class(TestWithGroup_AsyPadding)
create_test_padding_VALID_class(TestWithInput1x1Filter1x1_AsyPadding)

# ------------ test channel last ---------
create_test_channel_last_class(TestConv2dOp_AsyPadding)
create_test_channel_last_class(TestWithPad_AsyPadding)
create_test_channel_last_class(TestWithGroup_AsyPadding)
create_test_channel_last_class(TestWith1x1_AsyPadding)
create_test_channel_last_class(TestWithInput1x1Filter1x1_AsyPadding)

if __name__ == '__main__':
    unittest.main()
