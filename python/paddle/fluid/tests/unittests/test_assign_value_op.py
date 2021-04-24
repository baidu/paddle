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
import numpy

import op_test
import paddle.fluid as fluid
import paddle.fluid.framework as framework
import paddle.fluid.layers as layers


class TestAssignValueOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign_value"
        self.inputs = {}
        self.attrs = {}
        self.init_data()
        self.attrs["shape"] = self.value.shape
        self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(
            self.value.dtype)
        self.outputs = {"Out": self.value}

    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.float32)
        self.attrs["fp32_values"] = [float(v) for v in self.value.flat]

    def test_forward(self):
        self.check_output()


class TestAssignValueOp2(TestAssignValueOp):
    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.int32)
        self.attrs["int32_values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp3(TestAssignValueOp):
    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.int64)
        self.attrs["int64_values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp4(TestAssignValueOp):
    def init_data(self):
        self.value = numpy.random.choice(
            a=[False, True], size=(2, 5)).astype(numpy.bool)
        self.attrs["bool_values"] = [bool(v) for v in self.value.flat]


class TestAssignValueOp5(TestAssignValueOp):
    def init_data(self):
        self.value = numpy.random.random(size=(2, 5)).astype(numpy.float64)
        self.attrs["fp64_values"] = [float(v) for v in self.value.flat]


class TestAssignApi(unittest.TestCase):
    def setUp(self):
        self.init_dtype()
        self.value = (
            -100 + 200 * numpy.random.random(size=(2, 5))).astype(self.dtype)
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        main_program = fluid.Program()
        with fluid.program_guard(main_program):
            x = layers.create_tensor(dtype=self.dtype)
            layers.assign(input=self.value, output=x)

        exe = fluid.Executor(self.place)
        [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
        if self.dtype == 'float64':
            numpy.testing.assert_allclose(fetched_x, self.value)
        else:
            self.assertTrue(
                numpy.array_equal(fetched_x, self.value),
                "fetch_x=%s val=%s" % (fetched_x, self.value))
        self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):
    def setUp(self):
        self.init_dtype()
        self.value = numpy.random.choice(
            a=[False, True], size=(2, 5)).astype(numpy.bool)
        self.place = fluid.CUDAPlace(0) if fluid.is_compiled_with_cuda(
        ) else fluid.CPUPlace()

    def init_dtype(self):
        self.dtype = "bool"


class TestAssignApi5(TestAssignApi):
    def init_dtype(self):
        self.dtype = "float64"


if __name__ == '__main__':
    unittest.main()
