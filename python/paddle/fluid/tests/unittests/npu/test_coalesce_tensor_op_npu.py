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
from paddle.fluid import core

paddle.enable_static()
SEED = 2021
alignment = 512


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAllocContinuousSpace(OpTest):
    def setUp(self):
        self.__class__.use_npu = True
        self.op_type = "coalesce_tensor"
        self.dtype, self.fluid_dtype = self.init_dtype()
        attrs = self.init_attr()
        self.copy_data = attrs["copy_data"]
        self.constant = attrs["constant"]
        self.set_constant = attrs["set_constant"]
        self.Inputs = self.init_input()
        self.Outputs, self.FusedOutput = self.init_output(
            self.Inputs, self.set_constant, self.constant)
        self.inputs = {'Input': self.Inputs}
        self.attrs = attrs
        self.outputs = {'Output': self.Outputs, 'FusedOutput': self.FusedOutput}

    def init_dtype(self):
        return np.float32, int(core.VarDesc.VarType.FP32)

    def init_input(self):
        inputs = []
        inputs.append(("x1", np.zeros([20, 3]).astype(self.dtype)))
        inputs.append(("x2", np.zeros([20, 3]).astype(self.dtype)))
        return inputs

    def init_attr(self):
        return {
            "copy_data": False,
            "set_constant": False,
            "constant": 0.0,
            "use_align": True,
            "dtype": self.fluid_dtype
        }

    def init_output(self, input_list, set_constant, constant):
        inputs = []
        outputs = input_list

        for input in input_list:
            length = len(input[1].flatten())
            aligned_len = (length + alignment) / alignment * alignment
            out = np.zeros(int(aligned_len), dtype=self.dtype)
            out[0:length] = input[1].flatten()
            inputs.append(out)

        coalesce_tensor_var = np.concatenate([input for input in inputs])
        return outputs, coalesce_tensor_var

    def test_check_output(self):
        self.check_output_with_place(
            place=paddle.NPUPlace(0),
            no_check_set=["FusedOutput"],
            atol=1e-5,
            check_dygraph=False)


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestAllocContinuousSpace2(TestAllocContinuousSpace):
    def init_attr(self):
        return {
            "copy_data": True,
            "set_constant": False,
            "constant": 0.5,
            "use_align": True,
            "dtype": self.fluid_dtype
        }

    def test_check_output(self):
        self.check_output_with_place(
            place=paddle.NPUPlace(0),
            no_check_set=["FusedOutput"],
            atol=1e-5,
            check_dygraph=False)


if __name__ == '__main__':
    unittest.main()
