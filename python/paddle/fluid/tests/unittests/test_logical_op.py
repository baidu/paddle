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

import op_test
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.static import Program, program_guard

SUPPORTED_DTYPES = [
    bool, np.int8, np.int16, np.int32, np.int64, np.float32, np.float64
]

TEST_META_OP_DATA = [{
    'op_str': 'logical_and',
    'binary_op': True
}, {
    'op_str': 'logical_or',
    'binary_op': True
}, {
    'op_str': 'logical_xor',
    'binary_op': True
}, {
    'op_str': 'logical_not',
    'binary_op': False
}]

TEST_META_SHAPE_DATA = {
    'XDimLargerThanYDim1': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 5]
    },
    'XDimLargerThanYDim2': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 1]
    },
    'XDimLargerThanYDim3': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [1, 4, 1]
    },
    'XDimLargerThanYDim4': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [3, 4, 1]
    },
    'XDimLargerThanYDim5': {
        'x_shape': [2, 3, 1, 5],
        'y_shape': [3, 1, 1]
    },
    'XDimLessThanYDim1': {
        'x_shape': [4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim2': {
        'x_shape': [1, 4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim3': {
        'x_shape': [3, 4, 1],
        'y_shape': [2, 3, 4, 5]
    },
    'XDimLessThanYDim4': {
        'x_shape': [3, 1, 1],
        'y_shape': [2, 3, 1, 5]
    },
    'XDimLessThanYDim5': {
        'x_shape': [4, 5],
        'y_shape': [2, 3, 4, 5]
    },
    'Axis1InLargerDim': {
        'x_shape': [1, 4, 5],
        'y_shape': [2, 3, 1, 5]
    },
    'EqualDim1': {
        'x_shape': [10, 7],
        'y_shape': [10, 7]
    },
    'EqualDim2': {
        'x_shape': [1, 1, 4, 5],
        'y_shape': [2, 3, 1, 5]
    }
}

TEST_META_WRONG_SHAPE_DATA = {
    'ErrorDim1': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [3, 4]
    },
    'ErrorDim2': {
        'x_shape': [2, 3, 4, 5],
        'y_shape': [4, 3]
    }
}


def run_static(x_np, y_np, op_str, use_gpu=False, binary_op=True):
    paddle.enable_static()
    startup_program = fluid.Program()
    main_program = fluid.Program()
    place = paddle.CPUPlace()
    if use_gpu and fluid.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = fluid.Executor(place)
    with fluid.program_guard(main_program, startup_program):
        x = paddle.static.data(name='x', shape=x_np.shape, dtype=x_np.dtype)
        op = getattr(paddle, op_str)
        feed_list = {'x': x_np}
        if not binary_op:
            res = op(x)
        else:
            y = paddle.static.data(name='y', shape=y_np.shape, dtype=y_np.dtype)
            feed_list['y'] = y_np
            res = op(x, y)
        exe.run(startup_program)
        static_result = exe.run(main_program, feed=feed_list, fetch_list=[res])
    return static_result


def run_dygraph(x_np, y_np, op_str, use_gpu=False, binary_op=True):
    place = paddle.CPUPlace()
    if use_gpu and fluid.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    op = getattr(paddle, op_str)
    x = paddle.to_tensor(x_np, dtype=x_np.dtype)
    if not binary_op:
        dygraph_result = op(x)
    else:
        y = paddle.to_tensor(y_np, dtype=y_np.dtype)
        dygraph_result = op(x, y)
    return dygraph_result


def np_data_generator(np_shape, dtype, *args, **kwargs):
    if dtype == bool:
        return np.random.choice(a=[True, False], size=np_shape).astype(bool)
    else:
        return np.random.randn(*np_shape).astype(dtype)


def test(unit_test, use_gpu=False, test_error=False):
    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        meta_data['use_gpu'] = use_gpu
        np_op = getattr(np, meta_data['op_str'])
        META_DATA = dict(TEST_META_SHAPE_DATA)
        if test_error:
            META_DATA = dict(TEST_META_WRONG_SHAPE_DATA)
        for shape_data in META_DATA.values():
            for data_type in SUPPORTED_DTYPES:
                meta_data['x_np'] = np_data_generator(
                    shape_data['x_shape'], dtype=data_type)
                meta_data['y_np'] = np_data_generator(
                    shape_data['y_shape'], dtype=data_type)
                if meta_data['binary_op'] and test_error:
                    # catch C++ Exception
                    unit_test.assertRaises(BaseException, run_static,
                                           **meta_data)
                    unit_test.assertRaises(BaseException, run_dygraph,
                                           **meta_data)
                    continue
                static_result = run_static(**meta_data)
                dygraph_result = run_dygraph(**meta_data)
                if meta_data['binary_op']:
                    np_result = np_op(meta_data['x_np'], meta_data['y_np'])
                else:
                    np_result = np_op(meta_data['x_np'])
                unit_test.assertTrue((static_result == np_result).all())
                unit_test.assertTrue((dygraph_result.numpy() == np_result).all(
                ))


def test_type_error(unit_test, use_gpu, type_str_map):
    def check_type(op_str, x, y, binary_op):
        op = getattr(paddle, op_str)
        error_type = ValueError
        if isinstance(x, np.ndarray):
            x = paddle.to_tensor(x)
            y = paddle.to_tensor(y)
            error_type = BaseException
        if binary_op:
            if type_str_map['x'] != type_str_map['y']:
                unit_test.assertRaises(error_type, op, x=x, y=y)
            if not fluid.in_dygraph_mode():
                error_type = TypeError
                unit_test.assertRaises(error_type, op, x=x, y=y, out=1)
        else:
            if not fluid.in_dygraph_mode():
                error_type = TypeError
                unit_test.assertRaises(error_type, op, x=x, out=1)

    place = paddle.CPUPlace()
    if use_gpu and fluid.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    for op_data in TEST_META_OP_DATA:
        meta_data = dict(op_data)
        binary_op = meta_data['binary_op']

        paddle.disable_static(place)
        x = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map['x'])
        y = np.random.choice(a=[0, 1], size=[10]).astype(type_str_map['y'])
        check_type(meta_data['op_str'], x, y, binary_op)

        paddle.enable_static()
        startup_program = paddle.static.Program()
        main_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            x = paddle.static.data(
                name='x', shape=[10], dtype=type_str_map['x'])
            y = paddle.static.data(
                name='y', shape=[10], dtype=type_str_map['y'])
            check_type(meta_data['op_str'], x, y, binary_op)


def type_map_factory():
    return [{
        'x': x_type,
        'y': y_type
    } for x_type in SUPPORTED_DTYPES for y_type in SUPPORTED_DTYPES]


class TestCPU(unittest.TestCase):
    def test(self):
        test(self)

    def test_error(self):
        test(self, False, True)

    def test_type_error(self):
        type_map_list = type_map_factory()
        for type_map in type_map_list:
            test_type_error(self, False, type_map)


class TestCUDA(unittest.TestCase):
    def test(self):
        test(self, True)

    def test_error(self):
        test(self, True, True)

    def test_type_error(self):
        type_map_list = type_map_factory()
        for type_map in type_map_list:
            test_type_error(self, True, type_map)


if __name__ == '__main__':
    unittest.main()
