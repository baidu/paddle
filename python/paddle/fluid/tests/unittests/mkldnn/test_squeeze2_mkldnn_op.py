# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from paddle.fluid.tests.unittests.op_test import OpTest, OpTestTool, convert_float_to_uint16


@OpTestTool.skip_if(core.is_compiled_with_cuda(),
                    "CUDA has to be skipped because it forces dygraph")
class TestSqueeze2OneDNNOp(OpTest):
    def set_op_type(self):
        self.op_type = "squeeze2"

    def init_test_case(self):
        self.ori_shape = (1, 3, 1, 40)
        self.axes = (0, 2)
        self.new_shape = (3, 40)

    def set_inputs(self):
        self.inputs = {"X": self.x}

    def init_attrs(self):
        self.attrs = {"axes": self.axes, 'use_mkldnn': True}

    def set_outputs(self):
        self.outputs = {
            "Out": self.x.reshape(self.new_shape),
            "XShape": np.random.random(self.ori_shape).astype("float32")
        }

    def setUp(self):
        self.set_op_type()
        self.init_test_case()
        self.x = np.random.random(self.ori_shape).astype("float32")
        self.set_inputs()
        self.init_attrs()
        self.set_outputs()

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad_with_place(core.CPUPlace(), ["X"], "Out")


class TestSqueezeOneDNNOp(TestSqueeze2OneDNNOp):
    def set_op_type(self):
        self.op_type = "squeeze"

    def set_outputs(self):
        self.outputs = {"Out": self.x.reshape(self.new_shape)}

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace())


class TestSqueeze2OneDNNOp1(TestSqueeze2OneDNNOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


class TestSqueezeOneDNNOp1(TestSqueezeOneDNNOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = (0, -2)
        self.new_shape = (20, 5)


class TestSqueeze2OneDNNOp2(TestSqueeze2OneDNNOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


class TestSqueezeOneDNNOp2(TestSqueezeOneDNNOp):
    def init_test_case(self):
        self.ori_shape = (1, 20, 1, 5)
        self.axes = ()
        self.new_shape = (20, 5)


class TestSqueeze2OneDNNOp3(TestSqueeze2OneDNNOp):
    def init_test_case(self):
        self.ori_shape = (25, 1, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (25, 1, 4)


class TestSqueezeOneDNNOp3(TestSqueezeOneDNNOp):
    def init_test_case(self):
        self.ori_shape = (25, 1, 1, 4, 1)
        self.axes = (1, -1)
        self.new_shape = (25, 1, 4)


#   BF16 TESTS
def create_squeeze_bf16_test_classes(parent):
    @OpTestTool.skip_if_not_cpu_bf16()
    class TestSqueeze2BF16OneDNNOp(parent):
        def set_inputs(self):
            self.dtype = np.uint16
            self.inputs = {"X": convert_float_to_uint16(self.x)}

        def calculate_grads(self):
            self.dout = self.outputs['Out']
            self.dx = np.reshape(self.dout, self.ori_shape)

        def test_check_grad(self):
            self.calculate_grads()
            self.check_grad_with_place(
                core.CPUPlace(), ["X"],
                "Out",
                user_defined_grads=[self.dx],
                user_defined_grad_outputs=[self.dout])

    cls_name = "{0}_{1}".format(parent.__name__, "Squeeze2_BF16")
    TestSqueeze2BF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestSqueeze2BF16OneDNNOp

    class TestSqueezeBF16OneDNNOp(TestSqueeze2BF16OneDNNOp):
        def set_op_type(self):
            self.dtype = np.uint16
            self.op_type = "squeeze"

        def set_outputs(self):
            self.outputs = {"Out": self.x.reshape(self.new_shape)}

        def test_check_output(self):
            self.check_output_with_place(core.CPUPlace())

    cls_name = "{0}_{1}".format(parent.__name__, "Squeeze_BF16")
    TestSqueezeBF16OneDNNOp.__name__ = cls_name
    globals()[cls_name] = TestSqueezeBF16OneDNNOp


create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp1)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp2)
create_squeeze_bf16_test_classes(TestSqueeze2OneDNNOp3)

if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
