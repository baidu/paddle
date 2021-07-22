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
import paddle.fluid.core as core
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle


@OpTestTool.skip_if(core.is_compiled_with_cuda())
class TestSliceOneDNNOp(OpTest):
    def setUp(self):
        self.op_type = "slice"
        self.config()
        self.set_inputs()
        self.outputs = {'Out': self.out}
        self.attrs = {
            'axes': self.axes,
            'starts': self.starts,
            'ends': self.ends,
            'infer_flags': self.infer_flags,
            'use_mkldnn': True
        }
        self.set_attrs()

    def set_inputs(self):
        self.inputs = {'Input': self.input}

    def set_attrs(self):
        pass

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [3, 3, 4]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1:3, 0:3, 2:4, :]

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['Input'], 'Out')


class TestSliceOneDNNOp1(TestSliceOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, 2:-1, :]


class TestSliceOneDNNOp2(TestSliceOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-3, 0, 2]
        self.ends = [3, 100, -1]
        self.axes = [0, 1, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-3:3, 0:100, :, 2:-1]


class TestSliceDecrease1AxisOneDNNOp(TestSliceOneDNNOp):
    def set_attrs(self):
        self.attrs['decrease_axis'] = self.decrease_axis

    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 3, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0:3, 2:4, :]


class TestSliceDecrease2AxesOneDNNOp(TestSliceDecrease1AxisOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [1, 0, 2]
        self.ends = [2, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[1, 0, 2:4, :]


class TestSliceDecrease3AxesOneDNNOp(TestSliceDecrease1AxisOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6]).astype("float32")
        self.starts = [-1, 0, 2]
        self.ends = [1000000, 1, 4]
        self.axes = [0, 1, 2]
        self.decrease_axis = [0, 1]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[-1, 0, 2:4, :]


class TestSliceDecrease4AxesOneDNNOp(TestSliceDecrease1AxisOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 7]).astype("float32")
        self.starts = [0, 1, 2, 3]
        self.ends = [1, 2, 3, 4]
        self.axes = [0, 1, 2, 3]
        self.decrease_axis = [0, 1, 2, 3]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[0, 1, 2, 3:4]


class TestSlice5DOneDNNOp(TestSliceDecrease1AxisOneDNNOp):
    def config(self):
        self.input = np.random.random([3, 4, 5, 6, 7]).astype("float32")
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [4]
        self.decrease_axis = [4]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, :, :, -1]


class TestSlice3DOneDNNOp(TestSliceDecrease1AxisOneDNNOp):
    def config(self):
        self.input = np.random.random([5, 4, 5]).astype("float32")
        self.starts = [-1]
        self.ends = [1000000]
        self.axes = [2]
        self.decrease_axis = [2]
        self.infer_flags = [1, 1, 1]
        self.out = self.input[:, :, -1]


#   BF16 TESTS
def create_bf16_test_class(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestSliceBF16OneDNNOp(parent):
        def set_inputs(self):
            self.dtype = np.uint16
            self.inputs = {'Input': convert_float_to_uint16(self.input)}

        def calculate_grads(self):
            self.dout = self.out
            self.dx = np.zeros(shape=self.input.shape)

            begin = [None] * self.input.ndim
            end = [None] * self.input.ndim

            for i in range(len(self.axes)):
                begin[self.axes[i]] = self.starts[i]
                end[self.axes[i]] = self.ends[i]
            print(self.dout.shape)
            self.dx[begin[0]:end[0], begin[1]:end[1], begin[2]:end[2], begin[3]:
                    end[3]] = self.dout

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["Input"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[convert_float_to_uint16(self.dout)])

    cls_name = "{0}_{1}".format(parent.__name__, "BF16")
    TestSliceBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestSliceBF16OneDNNOp


create_bf16_test_class(TestSliceOneDNNOp)
create_bf16_test_class(TestSliceOneDNNOp1)
create_bf16_test_class(TestSliceDecrease1AxisOneDNNOp)
create_bf16_test_class(TestSliceDecrease2AxesOneDNNOp)
create_bf16_test_class(TestSliceDecrease3AxesOneDNNOp)
create_bf16_test_class(TestSliceDecrease4AxesOneDNNOp)

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
