#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
import paddle


class CwiseMulOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "cwise_mul"
        self.dtype = np.float64
        self.axis = -1
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}

    def test_check_output(self):
        # TODO(wangzhongpu): support mkldnn op in dygraph mode
        self.check_output(check_dygraph=(self.use_mkldnn == False))

    def test_dygraph(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = fluid.dygraph.to_variable(self.x)
            y = fluid.dygraph.to_variable(self.y)
            out = core.ops.cwise_mul(x, y)
            self.assertTrue(np.array_equal(out.numpy(), self.out))

    def test_api_dygraph(self):
        with fluid.dygraph.guard(fluid.CPUPlace()):
            x = fluid.dygraph.to_variable(self.x)
            y = fluid.dygraph.to_variable(self.y)
            out = paddle.cwise_mul(x, y, axis=-1)
            self.assertTrue(np.array_equal(out.numpy(), self.out))

    def test_api_static(self):
        x = fluid.data("x", shape=[13, 17], dtype='float64')
        y = fluid.data("y", shape=[13, 17], dtype='float64')
        out = paddle.cwise_mul(x, y, axis=-1)
        exe = fluid.Executor(fluid.CPUPlace())
        res = exe.run(feed={"x": self.x, "y": self.y}, fetch_list=[out])
        self.assertTrue(np.allclose(res, self.out))  # why not equal ?

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.multiply(self.x, self.y)

    def init_dtype(self):
        pass

    def init_axis(self):
        pass


if __name__ == '__main__':
    unittest.main()
