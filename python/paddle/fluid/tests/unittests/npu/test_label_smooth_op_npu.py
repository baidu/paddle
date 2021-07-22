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

paddle.enable_static()
SEED = 2021


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
class TestLabelSmoothOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "label_smooth"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)

        self.set_inputs()
        self.set_attrs()
        self.set_outputs()

    def calc_out(self, label, epsilon, dist=None):
        label_dim = label.shape[-1]
        y = (1 - epsilon) * label
        if dist is not None:
            y += epsilon * dist
        else:
            y += epsilon / label_dim
        return y.astype(self.dtype)

    def set_inputs(self):
        batch_size, label_dim = 10, 12
        x = np.zeros((batch_size, label_dim)).astype(self.dtype)
        nonzero_index = np.random.randint(label_dim, size=(batch_size))
        x[np.arange(batch_size), nonzero_index] = 1
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}

    def set_attrs(self):
        epsilon = 0.1
        self.attrs = {"epsilon": epsilon}

    def set_outputs(self):
        dist = None if 'PriorDist' not in self.inputs else self.inputs[
            'PriorDist']
        out = self.calc_out(self.inputs['X'], self.attrs['epsilon'], dist)
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place, check_dygraph=False)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad_with_place(self.place, ['X'], 'Out')


class TestLabelSmoothOpWithPriorDist(TestLabelSmoothOp):
    def set_inputs(self):
        super(TestLabelSmoothOpWithPriorDist, self).set_inputs()
        label_dim = self.inputs['X'].shape[-1]
        dist = np.random.random((1, label_dim)).astype(self.dtype)
        self.inputs['PriorDist'] = dist


class TestLabelSmoothOp3D(TestLabelSmoothOp):
    def set_inputs(self):
        super(TestLabelSmoothOp3D, self).set_inputs()
        self.inputs['X'].reshape([2, -1, self.inputs['X'].shape[-1]])


class TestLabelSmoothOpWithPriorDist3D(TestLabelSmoothOpWithPriorDist):
    def set_inputs(self):
        super(TestLabelSmoothOpWithPriorDist3D, self).set_inputs()
        self.inputs['X'].reshape([2, -1, self.inputs['X'].shape[-1]])


class TestLabelSmoothOpFP16(TestLabelSmoothOp):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDistFP16(TestLabelSmoothOpWithPriorDist):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOp3DFP16(TestLabelSmoothOp3D):
    def init_dtype(self):
        self.dtype = np.float16


class TestLabelSmoothOpWithPriorDist3DFP16(TestLabelSmoothOpWithPriorDist3D):
    def init_dtype(self):
        self.dtype = np.float16


if __name__ == '__main__':
    unittest.main()
