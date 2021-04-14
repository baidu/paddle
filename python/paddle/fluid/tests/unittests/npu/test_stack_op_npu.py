#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestStack1(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (5, 6, 7)
        self.axis = 0
        self.dtype = 'float32'

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append('x{}'.format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.set_npu()
        self.op_type = "stack"
        self.place = paddle.NPUPlace(0)

        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestStack2(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (2, 3, 4)
        self.axis = -1
        self.dtype = 'float32'

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append('x{}'.format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.set_npu()
        self.op_type = "stack"
        self.place = paddle.NPUPlace(0)

        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestStack3(OpTest):
    def initDefaultParameters(self):
        self.num_inputs = 4
        self.input_dim = (2, 3, 4)
        self.axis = 1
        self.dtype = 'float32'

    def get_x_names(self):
        x_names = []
        for i in range(self.num_inputs):
            x_names.append('x{}'.format(i))
        return x_names

    def setUp(self):
        self.initDefaultParameters()
        self.set_npu()
        self.op_type = "stack"
        self.place = paddle.NPUPlace(0)

        self.x = []
        for i in range(self.num_inputs):
            self.x.append(
                np.random.random(size=self.input_dim).astype(self.dtype))

        tmp = []
        x_names = self.get_x_names()
        for i in range(self.num_inputs):
            tmp.append((x_names[i], self.x[i]))

        self.inputs = {'X': tmp}
        self.outputs = {'Y': np.stack(self.x, axis=self.axis)}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
