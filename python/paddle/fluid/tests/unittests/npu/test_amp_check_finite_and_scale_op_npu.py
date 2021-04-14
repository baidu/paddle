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

import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
import paddle
import paddle.fluid as fluid

paddle.enable_static()


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCheckFiniteAndUnscaleOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "check_finite_and_unscale"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x / scale)],
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCheckFiniteAndUnscaleOpWithNan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "check_finite_and_unscale"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.nan
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains nan, do not check the output, 
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(
            self.place, check_dygraph=False, no_check_set=['Out'])


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestCheckFiniteAndUnscaleOpWithInf(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "check_finite_and_unscale"
        self.place = paddle.NPUPlace(0)
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        x[128][128] = np.inf
        scale = np.random.random((1)).astype(self.dtype)

        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([1]),
            'Out': [('out0', x)],
        }

    def set_npu(self):
        self.__class__.use_npu = True

    def init_kernel_type(self):
        self.use_mkldnn = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        # When input contains inf, do not check the output, 
        # since the output may be nondeterministic and will be discarded.
        self.check_output_with_place(
            self.place, check_dygraph=False, no_check_set=['Out'])


if __name__ == '__main__':
    unittest.main()
