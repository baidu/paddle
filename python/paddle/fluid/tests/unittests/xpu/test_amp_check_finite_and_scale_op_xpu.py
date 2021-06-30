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
'''
Author: your name
Date: 2021-06-21 11:36:48
LastEditTime: 2021-06-21 16:33:57
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /xpu/paddle2.0/Paddle/python/paddle/fluid/tests/unittests/xpu/test_amp_check_finite_and_scale_op_xpu.py
'''
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
import paddle
import unittest
import numpy as np
from op_test_xpu import XPUOpTest
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
paddle.enable_static()


class TestCheckFiniteAndUnscaleOp(XPUOpTest):
    def setUp(self):
        self.op_type = "check_finite_and_unscale"
        self.init_dtype()
        x = np.random.random((1024, 1024)).astype(self.dtype)
        scale = np.random.random((1)).astype(self.dtype)
        # self.attrs = {'stop_gradient': True}
        self.inputs = {'X': [('x0', x)], 'Scale': scale}
        self.outputs = {
            'FoundInfinite': np.array([0]),
            'Out': [('out0', x / scale)],
        }

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)


# class TestCheckFiniteAndUnscaleOpWithNan(XPUOpTest):
#     def setUp(self):
#         self.op_type = "check_finite_and_unscale"
#         self.init_dtype()
#         x = np.random.random((1024, 1024)).astype(self.dtype)
#         x[128][128] = np.nan
#         print("x shape = ", x.shape)
#         print(x)
#         scale = np.random.random((1)).astype(self.dtype)

#         self.inputs = {'X': [('x0', x)], 'Scale': scale}
#         self.outputs = {
#             'FoundInfinite': np.array([1]),
#             'Out': [('out0', x)],
#         }

#     def init_dtype(self):
#         self.dtype = np.float32

#     def test_check_output(self):
#         # When input contains nan, do not check the output, 
#         # since the output may be nondeterministic and will be discarded.
#         if paddle.is_compiled_with_xpu():
#             place = paddle.XPUPlace(0)
#             self.check_output_with_place(place, no_check_set=['Out'])

# class TestCheckFiniteAndUnscaleOpWithInf(XPUOpTest):
#     def setUp(self):
#         self.op_type = "check_finite_and_unscale"
#         self.init_dtype()
#         x = np.random.random((1024, 1024)).astype(self.dtype)
#         x[128][128] = np.inf
#         scale = np.random.random((1)).astype(self.dtype)

#         self.inputs = {'X': [('x0', x)], 'Scale': scale}
#         self.outputs = {
#             'FoundInfinite': np.array([1]),
#             'Out': [('out0', x)],
#         }

#     def init_dtype(self):
#         self.dtype = np.float32

#     def test_check_output(self):
#         # When input contains inf, do not check the output, 
#         # since the output may be nondeterministic and will be discarded.
#         if paddle.is_compiled_with_xpu():
#             place = paddle.XPUPlace(0)
#             self.check_output_with_place(place, no_check_set=['Out'])

if __name__ == '__main__':
    unittest.main()
