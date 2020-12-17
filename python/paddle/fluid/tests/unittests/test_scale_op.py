#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.op import Operator


class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.float64
        self.init_dtype_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestScaleOpComplex(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.complex64
        self.init_dtype_type()
        self.inputs = {'X': np.random.random((10, 10)).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.inputs['X'] * self.dtype(self.attrs['scale'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestScaleOpScaleVariable(OpTest):
    def setUp(self):
        self.op_type = "scale"
        self.dtype = np.float64
        self.init_dtype_type()
        self.scale = -2.3
        self.inputs = {
            'X': np.random.random((10, 10)).astype(self.dtype),
            'ScaleTensor': np.array([self.scale]).astype('float64')
        }
        self.attrs = {}
        self.outputs = {'Out': self.inputs['X'] * self.dtype(self.scale)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestScaleOpSelectedRows(unittest.TestCase):
    def init_dtype_type(self):
        pass

    def check_with_place(self, place, in_name, out_name):
        scope = core.Scope()

        self.dtype = np.float64
        self.init_dtype_type()

        # create and initialize Grad Variable
        in_height = 10
        in_rows = [0, 4, 7]
        in_row_numel = 12
        scale = 2.0

        in_selected_rows = scope.var(in_name).get_selected_rows()
        in_selected_rows.set_height(in_height)
        in_selected_rows.set_rows(in_rows)
        in_array = np.random.random(
            (len(in_rows), in_row_numel)).astype(self.dtype)

        in_tensor = in_selected_rows.get_tensor()
        in_tensor.set(in_array, place)

        # create and initialize Param Variable
        out_selected_rows = scope.var(out_name).get_selected_rows()
        out_tensor = out_selected_rows.get_tensor()
        out_tensor._set_dims(in_tensor._get_dims())

        # create and run sgd operator
        scale_op = Operator("scale", X=in_name, Out=out_name, scale=scale)
        scale_op.run(scope, place)

        # get and compare result
        out_height = out_selected_rows.height()
        out_rows = out_selected_rows.rows()
        result_array = np.array(out_tensor)

        assert (in_array * scale == result_array).all()
        assert in_height == out_height
        assert in_rows == out_rows

    def test_scale_selected_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, 'in', 'out')

    def test_scale_selected_rows_inplace(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            self.check_with_place(place, 'in', 'in')


class TestScaleRaiseError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.scale([10])

        self.assertRaises(TypeError, test_type)


# Add FP16 test
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestScaleFp16Op(TestScaleOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=0.002)

    def test_check_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad_with_place(
                place, ["X"], "Out", max_relative_error=0.05)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestScaleFp16OpSelectedRows(TestScaleOpSelectedRows):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_scale_selected_rows(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_with_place(place, 'in', 'out')

    def test_scale_selected_rows_inplace(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_with_place(place, 'in', 'in')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
