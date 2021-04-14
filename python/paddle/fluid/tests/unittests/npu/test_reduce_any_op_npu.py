#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.framework import convert_np_dtype_to_dtype_

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAny8DOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reduce_any"
        self.place = paddle.NPUPlace(0)
        self.inputs = {
            'X': np.random.randint(0, 2,
                                   (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {'dim': (3, 5, 4)}
        self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAnyOpWithDim(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reduce_any"
        self.place = paddle.NPUPlace(0)
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': [1]}
        self.outputs = {'Out': self.inputs['X'].any(axis=1)}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAny8DOpWithDim(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reduce_any"
        self.place = paddle.NPUPlace(0)
        self.inputs = {
            'X': np.random.randint(0, 2,
                                   (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {'dim': (3, 6)}
        self.outputs = {'Out': self.inputs['X'].any(axis=self.attrs['dim'])}

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAnyOpWithKeepDim(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reduce_any"
        self.place = paddle.NPUPlace(0)
        self.inputs = {'X': np.random.randint(0, 2, (5, 6, 10)).astype("bool")}
        self.attrs = {'dim': (1, ), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].any(axis=self.attrs['dim']), axis=1)
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


class TestAny8DOpWithKeepDim(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "reduce_any"
        self.place = paddle.NPUPlace(0)
        self.inputs = {
            'X': np.random.randint(0, 2,
                                   (2, 5, 3, 2, 2, 3, 4, 2)).astype("bool")
        }
        self.attrs = {'dim': (1, ), 'keep_dim': True}
        self.outputs = {
            'Out': np.expand_dims(
                self.inputs['X'].any(axis=self.attrs['dim']), axis=1)
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
