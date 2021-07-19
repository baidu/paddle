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

import math
import numpy as np
import unittest
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core


class TestUnfoldOp(OpTest):
    """
    This is for test on unfold Op
    """

    def init_data(self):
        self.batch_size = 3
        self.input_channels = 3
        self.input_height = 20
        self.input_width = 20
        self.kernel_size = [3, 3]
        self.stride = [1, 1]
        self.padding = [1, 1, 1, 1]
        self.dilation = [1, 1]
        input_shape = [
            self.batch_size, self.input_channels, self.input_height,
            self.input_width
        ]
        self.x = np.random.rand(*input_shape).astype(np.float64)

    def calc_unfold(self):
        output_shape = [0] * 3
        output_shape[0] = self.batch_size
        output_shape[1] = self.input_channels * self.kernel_size[
            0] * self.kernel_size[1]
        dkernel_h = self.dilation[0] * (self.kernel_size[0] - 1) + 1
        dkernel_w = self.dilation[1] * (self.kernel_size[1] - 1) + 1
        out_height = int((self.input_height + self.padding[0] +
                          self.padding[2] - dkernel_h) / self.stride[0]) + 1
        out_width = int((self.input_width + self.padding[1] + self.padding[3]
                         - dkernel_w) / self.stride[1]) + 1
        output_shape[2] = out_height * out_width
        output = np.zeros(output_shape).astype(np.float64)
        ############ calculate output ##############
        for i in range(output_shape[0]):
            for j in range(output_shape[1]):
                for k in range(output_shape[2]):
                    h_out = int(k / out_width)
                    w_out = k % out_width
                    w_offset = j % self.kernel_size[1]
                    h_offset = int(j /
                                   self.kernel_size[1]) % self.kernel_size[0]
                    c_in = int(j /
                               (self.kernel_size[0] * self.kernel_size[1]))
                    h_in = h_offset * self.dilation[0] + h_out * self.stride[
                        0] - self.padding[0]
                    w_in = w_offset * self.dilation[1] + w_out * self.stride[
                        1] - self.padding[1]
                    if (h_in>=0 and h_in<self.input_height) and \
                         (w_in>=0 and w_in<self.input_width):
                        output[i, j, k] = self.x[i, c_in, h_in, w_in]

        self.outputs = output

    def set_data(self):
        self.init_data()
        self.calc_unfold()

        self.inputs = {'X': self.x}
        self.attrs = {
            'kernel_size': self.kernel_size,
            'padding': self.padding,
            'dilation': self.dilation,
            'stride': self.stride
        }
        self.outputs = {'Y': self.outputs}

    def setUp(self):
        self.op_type = 'unfold'
        self.set_data()

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Y')


class TestUnfoldAPI(TestUnfoldOp):
    """
    This is for test on paddle.nn.Unfold
    """

    def setUp(self):
        self.op_type = 'unfold'
        self.set_data()
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                input = fluid.dygraph.to_variable(self.inputs['X'])
                m = paddle.nn.Unfold(**self.attrs)
                m.eval()
                result = m(input)
                self.assertTrue(np.allclose(result.numpy(), self.outputs['Y']))

    def test_info(self):
        str(paddle.nn.Unfold(**self.attrs))


if __name__ == '__main__':
    unittest.main()
