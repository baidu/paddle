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
from op_test import OpTest, _set_use_system_allocator
import paddle
import paddle.fluid as fluid

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestTransposeOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "transpose2"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.attrs = {'axis': [0, 2, 1, 3], 'data_format': 'AnyLayout'}
        self.outputs = {'Out': self.out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [8, 512, 12, 64]).astype(self.dtype)
        self.out = np.transpose(self.x, [0, 2, 1, 3])

    def init_dtype(self):
        self.dtype = np.float32

    def init_axis(self):
        self.axis = -1

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestTransposeOpFP16(TestTransposeOp):
    no_need_check_grad = True

    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
