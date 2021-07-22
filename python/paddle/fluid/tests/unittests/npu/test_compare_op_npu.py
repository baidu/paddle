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

callback_dict = {
    'less_than': lambda _a, _b: _a < _b,
    'less_equal': lambda _a, _b: _a <= _b,
    'greater_than': lambda _a, _b: _a > _b,
    'greater_equal': lambda _a, _b: _a >= _b,
    'equal': lambda _a, _b: _a == _b,
    'not_equal': lambda _a, _b: _a != _b
}


class TestEqual(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x == y  # all elements are not equal

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestLessthan(OpTest):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = x < y

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}

    def set_npu(self):
        self.__class__.use_npu = True

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)


class TestEqual2(TestEqual):
    def setUp(self):
        self.set_npu()
        self.op_type = "equal"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x == y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestLessthan2(TestLessthan):
    def setUp(self):
        self.set_npu()
        self.op_type = "less_than"
        self.place = paddle.NPUPlace(0)

        self.init_dtype()
        np.random.seed(SEED)
        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        y = x.copy()
        y[0][1] = 1
        out = x < y  # all elements are equal, except position [0][1]

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'Y': OpTest.np_dtype_to_fluid_dtype(y)
        }
        self.outputs = {'Out': out}


class TestEqual2FP16(TestEqual2):
    def init_dtype(self):
        self.dtype = np.float16


class TestEqual2Int(TestEqual2):
    def init_dtype(self):
        self.dtype = np.int32


class TestLessthan2FP16(TestLessthan2):
    def init_dtype(self):
        self.dtype = np.float16


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
def create_paddle_op_case(op_type, dtype):
    callback = callback_dict[op_type]

    class PaddleClsOp(OpTest):
        def setUp(self):
            self.__class__.use_npu = True
            self.op_type = op_type
            self.place = paddle.NPUPlace(0)
            self.dtype = dtype
            np.random.seed(SEED)
            self.get_inputs([11, 17], [11, 17])
            self.get_outputs()
            self.inputs = {
                'X': OpTest.np_dtype_to_fluid_dtype(self.input_x),
                'Y': OpTest.np_dtype_to_fluid_dtype(self.input_y),
            }
            self.outputs = {'Out': self.real_result}

        def get_inputs(self, shape_x, shape_y):
            self.input_x = np.random.uniform(0, 5, shape_x).astype(self.dtype)
            self.input_y = np.random.uniform(0, 5, shape_y).astype(self.dtype)

        def get_outputs(self):
            self.real_result = callback(self.input_x, self.input_y)

        def test_check_output(self):
            self.check_output_with_place(self.place)

    cls_op_name = 'TestCaseOp_%s_%s' % (op_type, dtype)
    PaddleClsOp.__name__ = cls_op_name
    globals()[cls_op_name] = PaddleClsOp


@unittest.skipIf(not paddle.is_compiled_with_npu(),
                 "core is not compiled with NPU")
def create_paddle_api_case(op_type, dtype):
    callback = callback_dict[op_type]

    class PaddleClsAPI(unittest.TestCase):
        def setUp(self):
            self.__class__.use_npu = True
            self.op_type = op_type
            self.place = paddle.NPUPlace(0)
            self.dtype = dtype
            np.random.seed(SEED)

        def get_inputs(self, shape_x, shape_y):
            self.input_x = np.random.uniform(0, 5, shape_x).astype(self.dtype)
            self.input_y = np.random.uniform(0, 5, shape_y).astype(self.dtype)

        def get_outputs(self):
            self.real_result = callback(self.input_x, self.input_y)

        def _test_static(self, shape_x, shape_y):
            paddle.enable_static()
            self.get_inputs(shape_x, shape_y)
            self.get_outputs()
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = fluid.data(name='x', shape=shape_x, dtype=self.dtype)
                y = fluid.data(name='y', shape=shape_y, dtype=self.dtype)
                op = eval('paddle.%s' % (self.op_type))
                out = op(x, y)
                exe = fluid.Executor(self.place)
                res, = exe.run(feed={'x': self.input_x,
                                     'y': self.input_y},
                               fetch_list=[out])
            self.assertEqual((res == self.real_result).all(), True)

        def _test_dynamic(self, shape_x, shape_y):
            paddle.disable_static()
            self.get_inputs(shape_x, shape_y)
            self.get_outputs()
            x = paddle.to_tensor(self.input_x)
            y = paddle.to_tensor(self.input_y)
            op = eval("paddle.%s" % (self.op_type))
            out = op(x, y)
            self.assertEqual((out.numpy() == self.real_result).all(), True)
            paddle.enable_static()

        def test_api(self):
            self._test_static([4], [4])

        def test_dynamic_api(self):
            self._test_dynamic([4], [4])

        def test_broadcast_api_1(self):
            self._test_static([1, 2, 1, 3], [1, 2, 3])

        def test_broadcast_api_2(self):
            self._test_static([1, 2, 3], [1, 2, 1, 3])

        def test_broadcast_api_3(self):
            self._test_static([5], [3, 1])

        def test_attr_name(self):
            paddle.enable_static()
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                x = fluid.layers.data(name='x', shape=[4], dtype=self.dtype)
                y = fluid.layers.data(name='y', shape=[4], dtype=self.dtype)
                op = eval("paddle.%s" % (self.op_type))
                out = op(x=x, y=y, name="name_%s" % (self.op_type))
            self.assertEqual("name_%s" % (self.op_type) in out.name, True)

    cls_api_name = 'TestCaseAPI_%s_%s' % (op_type, dtype)
    PaddleClsAPI.__name__ = cls_api_name
    globals()[cls_api_name] = PaddleClsAPI


create_paddle_op_case('greater_than', 'float32')
create_paddle_op_case('greater_than', 'float16')
create_paddle_op_case('greater_than', 'int32')
create_paddle_op_case('not_equal', 'float32')
create_paddle_op_case('not_equal', 'float16')
create_paddle_op_case('not_equal', 'int32')
create_paddle_op_case('less_equal', 'float32')
create_paddle_op_case('less_equal', 'float16')
create_paddle_op_case('less_equal', 'int32')

create_paddle_api_case('greater_than', 'float32')
create_paddle_api_case('greater_than', 'int32')
create_paddle_api_case('not_equal', 'float32')
create_paddle_api_case('not_equal', 'int32')
create_paddle_api_case('less_equal', 'float32')
create_paddle_api_case('less_equal', 'int32')

if __name__ == '__main__':
    unittest.main()
