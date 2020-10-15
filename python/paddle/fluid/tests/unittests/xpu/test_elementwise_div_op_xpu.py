#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
sys.path.append("..")
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from op_test import OpTest, skip_check_grad_ci
from elementwise import TestXPUElementwiseOpBase
paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseDivOp(OpTest, TestXPUElementwiseOpBase):
    def setUp(self):
        TestXPUElementwiseOpBase.setUp(self, "elementwise_div")
        self.make_input()
        self.make_output()

    def make_output(self, x_shape=None, y_shape=None):
        x, y = self.reshape_input(x_shape, y_shape)
        self.outputs = {'Out': np.divide(x, y)}


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_scalar(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_scalar, self).setUp()
        self.grad_implemented = False
        self.make_input([20, 3, 4], [1])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_Vector(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_Vector, self).setUp()
        self.make_input([100, ], [100, ])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_0(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_0, self).setUp()
        self.attrs['axis'] = 0
        self.make_input([100, 3, 4], [100, ])
        self.make_output(y_shape=[100, 1, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_1(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_1, self).setUp()
        self.attrs['axis'] = 1
        self.make_input([2, 100, 4], [100, ])
        self.make_output(y_shape=[1, 100, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_2(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_2, self).setUp()
        self.make_input([2, 3, 100], [100, ])
        self.make_output(y_shape=[1, 1, 100])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_3(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_3, self).setUp()
        self.attrs['axis'] = 1
        self.make_input([2, 10, 12, 5], [10, 12])
        self.make_output(y_shape=[1, 10, 12, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_4(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_4, self).setUp()
        self.is_common_broadcast = True
        self.make_input([2, 3, 50], [2, 1, 50])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_broadcast_5(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_broadcast_5, self).setUp()
        self.is_common_broadcast = True
        self.make_input([2, 3, 4, 20], [2, 3, 1, 20])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_commonuse_1(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_commonuse_1, self).setUp()
        self.is_common_broadcast = True
        self.make_input([2, 3, 100], [1, 1, 100])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseDivOp_xsize_lessthan_ysize(TestXPUElementwiseDivOp):
    def setUp(self):
        super(TestElementwiseDivOp_xsize_lessthan_ysize, self).setUp()
        self.is_x_size_less_than_y = True
        self.attrs['axis'] = 2
        self.make_input([10, 12], [2, 3, 10, 12])
        self.make_output(x_shape=[1, 1, 10, 12])


if __name__ == '__main__':
    unittest.main()
