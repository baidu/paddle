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


class TestSumOp(OpTest):
    def setUp(self):
        self.op_type = "sum"
        self.init_kernel_type()
        self.use_mkldnn = False
        self.init_kernel_type()
        x0 = np.random.random((3, 40)).astype(self.dtype)
        x1 = np.random.random((3, 40)).astype(self.dtype)
        x2 = np.random.random((3, 40)).astype(self.dtype)
        self.inputs = {"X": [("x0", x0), ("x1", x1), ("x2", x2)]}
        y = x0 + x1 + x2
        self.outputs = {'Out': y}
        self.attrs = {'use_mkldnn': self.use_mkldnn}

    def init_kernel_type(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')

    def init_kernel_type(self):
        pass


class TestSelectedRowsSumOp(unittest.TestCase):
    def setUp(self):
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 3, 4, 5, 6]
        self.dtype = np.float64
        self.init_kernel_type()

    def check_with_place(self, place, inplace):
        self.check_input_and_optput(core.Scope(), place, inplace, True, True,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, True,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, False,
                                    True)
        self.check_input_and_optput(core.Scope(), place, inplace, False, False,
                                    False)

    def init_kernel_type(self):
        pass

    def _get_array(self, rows, row_numel):
        array = np.ones((len(rows), row_numel)).astype(self.dtype)
        for i in range(len(rows)):
            array[i] *= rows[i]
        return array

    def check_input_and_optput(self,
                               scope,
                               place,
                               inplace,
                               w1_has_data=False,
                               w2_has_data=False,
                               w3_has_data=False):

        self.create_selected_rows(scope, place, "W1", w1_has_data)
        self.create_selected_rows(scope, place, "W2", w2_has_data)
        self.create_selected_rows(scope, place, "W3", w3_has_data)

        # create Out Variable
        if inplace:
            out_var_name = "W1"
        else:
            out_var_name = "Out"
        out = scope.var(out_var_name).get_selected_rows()

        # create and run sum operator
        sum_op = Operator("sum", X=["W1", "W2", "W3"], Out=out_var_name)
        sum_op.run(scope, place)

        has_data_w_num = 0
        for has_data in [w1_has_data, w2_has_data, w3_has_data]:
            if has_data:
                has_data_w_num += 1

        if has_data_w_num > 0:
            self.assertEqual(len(out.rows()), 7)
            self.assertTrue(
                np.array_equal(
                    np.array(out.get_tensor()),
                    self._get_array(self.rows, self.row_numel) *
                    has_data_w_num))
        else:
            self.assertEqual(len(out.rows()), 0)

    def create_selected_rows(self, scope, place, var_name, has_data):
        # create and initialize W Variable
        if has_data:
            rows = self.rows
        else:
            rows = []

        var = scope.var(var_name)
        w_selected_rows = var.get_selected_rows()
        w_selected_rows.set_height(self.height)
        w_selected_rows.set_rows(rows)
        w_array = self._get_array(self.rows, self.row_numel)
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

        return var

    def test_w_is_selected_rows(self):
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            for inplace in [True, False]:
                self.check_with_place(place, inplace)


class TestLoDTensorAndSelectedRowsOp(TestSelectedRowsSumOp):
    def setUp(self):
        self.height = 10
        self.row_numel = 12
        self.rows = [0, 1, 2, 2, 4, 5, 6]
        self.dtype = np.float64

    def check_with_place(self, place, inplace):
        scope = core.Scope()
        if inplace:
            self.create_lod_tensor(scope, place, "x1")
            self.create_selected_rows(scope, place, "x2", True)
            out = scope.var("x1").get_tensor()
            out_name = "x1"
        else:
            self.create_selected_rows(scope, place, "x1", True)
            self.create_lod_tensor(scope, place, "x2")
            out = scope.var("out").get_tensor()
            out_name = "out"

        # create and run sum operator
        sum_op = Operator("sum", X=["x1", "x2"], Out=out_name)
        sum_op.run(scope, place)

        result = np.ones((1, self.height)).astype(np.int32).tolist()[0]
        for ele in self.rows:
            result[ele] += 1

        out_t = np.array(out)
        self.assertEqual(out_t.shape[0], self.height)
        self.assertTrue(
            np.array_equal(out_t,
                           self._get_array([i for i in range(
                               self.height)], self.row_numel) * np.tile(
                                   np.array(result).reshape(self.height, 1),
                                   self.row_numel)))

    def create_lod_tensor(self, scope, place, var_name):
        var = scope.var(var_name)
        w_tensor = var.get_tensor()
        w_array = self._get_array([i for i in range(self.height)],
                                  self.row_numel)
        w_tensor.set(w_array, place)
        return var


#----------- test fp16 -----------
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16SumOp(TestSumOp):
    def init_kernel_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=2e-2)

    # FIXME: Because of the precision fp16, max_relative_error
    # should be 0.15 here.
    def test_check_grad(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_grad(['x0'], 'Out', max_relative_error=0.15)


def create_test_sum_fp16_class(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestSumFp16Case(parent):
        def init_kernel_type(self):
            self.dtype = np.float16

        def test_w_is_selected_rows(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                for inplace in [True, False]:
                    self.check_with_place(place, inplace)

    cls_name = "{0}_{1}".format(parent.__name__, "SumFp16Test")
    TestSumFp16Case.__name__ = cls_name
    globals()[cls_name] = TestSumFp16Case


class API_Test_Add_n(unittest.TestCase):
    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input0 = fluid.layers.fill_constant(
                shape=[2, 3], dtype='int64', value=5)
            input1 = fluid.layers.fill_constant(
                shape=[2, 3], dtype='int64', value=3)
            expected_result = np.empty((2, 3))
            expected_result.fill(8)
            sum_value = paddle.add_n([input0, input1])
            exe = fluid.Executor(fluid.CPUPlace())
            result = exe.run(fetch_list=[sum_value])

            self.assertEqual((result == expected_result).all(), True)

        with fluid.dygraph.guard():
            input0 = paddle.ones(shape=[2, 3], dtype='float32')
            expected_result = np.empty((2, 3))
            expected_result.fill(2)
            sum_value = paddle.add_n([input0, input0])

            self.assertEqual((sum_value.numpy() == expected_result).all(), True)


class TestRaiseSumError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.sum([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sum([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sum(data1)

        self.assertRaises(TypeError, test_dtype1)


class TestRaiseSumsError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.sums([11, 22])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            data2 = fluid.data(name="input2", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2])

        self.assertRaises(TypeError, test_dtype)

        def test_dtype1():
            data1 = fluid.data(name="input1", shape=[10], dtype="int8")
            fluid.layers.sums(data1)

        self.assertRaises(TypeError, test_dtype1)

        def test_out_type():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            fluid.layers.sums([data1, data2], out=[10])

        self.assertRaises(TypeError, test_out_type)

        def test_out_dtype():
            data1 = fluid.data(name="input1", shape=[10], dtype="flaot32")
            data2 = fluid.data(name="input2", shape=[10], dtype="float32")
            out = fluid.data(name="out", shape=[10], dtype="int8")
            fluid.layers.sums([data1, data2], out=out)

        self.assertRaises(TypeError, test_out_dtype)


class TestSumOpError(unittest.TestCase):
    def test_errors(self):
        def test_empty_list_input():
            with fluid.dygraph.guard():
                fluid.core.ops.sum([])

        def test_list_of_none_input():
            with fluid.dygraph.guard():
                fluid.core.ops.sum([None])

        self.assertRaises(Exception, test_empty_list_input)
        self.assertRaises(Exception, test_list_of_none_input)


create_test_sum_fp16_class(TestSelectedRowsSumOp)
create_test_sum_fp16_class(TestLoDTensorAndSelectedRowsOp)

if __name__ == "__main__":
    unittest.main()
