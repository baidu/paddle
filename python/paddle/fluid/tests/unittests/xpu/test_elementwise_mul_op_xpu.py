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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle
from elementwise import TestXPUElementwiseOpBase
paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp(OpTest, TestXPUElementwiseOpBase):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        TestXPUElementwiseOpBase.setUp(self, "elementwise_mul")
        self.init_kernel_type()
        self.init_axis()
        self.attrs['axis'] = self.axis
        self.attrs['use_mkldnn'] = self.use_mkldnn
        self.grad_implemented = True
        self.make_input()
        self.make_output()

    def make_output(self, x_shape=None, y_shape=None):
        x, y = self.reshape_input(x_shape, y_shape)
        self.outputs = {'Out': np.multiply(x, y)}


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp_scalar(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestXPUElementwiseMulOp_scalar, self).setUp()
        self.make_input((10, 3, 4), (1, ))
        self.make_output()
        self.grad_implemented = False


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp_Vector(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestXPUElementwiseMulOp_Vector, self).setUp()
        self.make_input((100, ), (100, ))
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp_broadcast_0(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestXPUElementwiseMulOp_broadcast_0, self).setUp()
        self.make_input((100, 2, 3), (100, ))
        self.make_output(y_shape=(100, 1, 1))
        self.y_grad_implemented = False

    def init_axis(self):
        self.axis = 0


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMulOp_broadcast_1(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestElementwiseMulOp_broadcast_1, self).setUp()
        self.attrs['axis'] = 1
        self.y_grad_implemented = False
        self.make_input((2, 100, 3), (100, ))
        self.make_output(y_shape=(1, 100, 1))


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMulOp_broadcast_2(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestElementwiseMulOp_broadcast_2, self).setUp()
        self.y_grad_implemented = False
        self.make_input((2, 3, 100), (100, ))
        self.make_output(y_shape=(1, 1, 100))


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMulOp_broadcast_3(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestElementwiseMulOp_broadcast_3, self).setUp()
        self.attrs['axis'] = 1
        self.y_grad_implemented = False
        self.make_input((2, 10, 12, 3), (10, 12))
        self.make_output(y_shape=(1, 10, 12, 1))


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMulOp_broadcast_4(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestElementwiseMulOp_broadcast_4, self).setUp()
        self.is_common_broadcast = True
        self.make_input((10, 2, 11), (10, 1, 11))
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMulOp_broadcast_5(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestElementwiseMulOp_broadcast_5, self).setUp()
        self.is_common_broadcast = True
        self.make_input((10, 4, 2, 3), (10, 4, 1, 3))
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp_commonuse_1(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestXPUElementwiseMulOp_commonuse_1, self).setUp()
        self.is_common_broadcast = True
        self.make_input((2, 3, 100), (1, 1, 100))
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseMulOp_xsize_lessthan_ysize(TestXPUElementwiseMulOp):
    def setUp(self):
        super(TestXPUElementwiseMulOp_xsize_lessthan_ysize, self).setUp()
        self.attrs['axis'] = 2
        self.is_x_size_less_than_y = True
        self.make_input((10, 10), (2, 2, 10, 10))
        self.make_output(x_shape=(1, 1, 10, 10))


if __name__ == '__main__':
    unittest.main()
