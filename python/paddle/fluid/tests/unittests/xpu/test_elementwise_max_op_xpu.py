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
import sys
sys.path.append("..")
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
from elementwise import TestXPUElementwiseOpBase
paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUElementwiseOp(OpTest, TestXPUElementwiseOpBase):
    def setUp(self):
        TestXPUElementwiseOpBase.setUp(self, "elementwise_max")
        self.make_input()
        self.make_output()

    def make_input(self, x_shape=[13, 17], y_shape=[13, 17], idx_list=None):
        x = np.random.random(x_shape).astype(self.dtype)
        sgn = np.random.choice([-1, 1], y_shape).astype(self.dtype)
        if idx_list is None:
            y = x + sgn * np.random.uniform(0.1, 1, y_shape).astype(self.dtype)
        else:
            x_temp = x
            for idx in idx_list:
                x_temp = np.take(x_temp, [0], axis=idx)
            sgn = sgn.reshape(x_temp.shape)
            y = x_temp + sgn * np.random.uniform(0.1, 1, x_temp.shape)
            y = y.reshape(y_shape).astype(self.dtype)

        self.inputs = {'X': x, 'Y': y}

    def make_output(self, x_shape=None, y_shape=None):
        x, y = self.reshape_input(x_shape, y_shape)
        self.outputs = {'Out': np.maximum(x, y)}


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_scalar(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_scalar, self).setUp()
        self.make_input([2, 3, 20], [1])
        self.make_output()
        self.grad_implemented = False


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_Vector(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_Vector, self).setUp()
        self.make_input([100, ], [100, ])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_0(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_0, self).setUp()
        self.attrs['axis'] = 0
        self.make_input([100, 5, 2], [100, ], [1, 2])
        self.make_output(y_shape=[100, 1, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_1(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_1, self).setUp()
        self.attrs['axis'] = 1
        self.make_input([2, 100, 3], [100, ], [0, 2])
        self.make_output(y_shape=[1, 100, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_2(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_2, self).setUp()
        self.make_input([1, 3, 100], [100, ], [0, 1])
        self.make_output(y_shape=[1, 1, 100])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_3(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_3, self).setUp()
        self.attrs['axis'] = 1
        self.make_input([2, 50, 2, 1], [50, 2], [0, 3])
        self.make_output(y_shape=[1, 50, 2, 1])


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_4(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_4, self).setUp()
        self.make_input([2, 3, 4, 5], [2, 3, 1, 5])
        self.make_output()


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestElementwiseMaxOp_broadcast_5(TestXPUElementwiseOp):
    def setUp(self):
        super(TestElementwiseMaxOp_broadcast_5, self).setUp()
        self.make_input([2, 3, 100], [1, 1, 100])
        self.make_output()


if __name__ == '__main__':
    unittest.main()
