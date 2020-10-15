# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import unittest
import numpy as np


def run_static(x_np, dtype, op_str, use_gpu=False):
    paddle.enable_static()
    startup_program = fluid.Program()
    main_program = fluid.Program()
    place = paddle.CPUPlace()
    if use_gpu and fluid.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    exe = fluid.Executor(place)
    with fluid.program_guard(main_program, startup_program):
        x = paddle.fluid.data(name='x', shape=x_np.shape, dtype=dtype)
        res = getattr(paddle.tensor, op_str)(x)
        exe.run(startup_program)
        static_result = exe.run(main_program,
                                feed={'x': x_np},
                                fetch_list=[res])
    return static_result


def run_dygraph(x_np, op_str, use_gpu=True):
    place = paddle.CPUPlace()
    if use_gpu and fluid.core.is_compiled_with_cuda():
        place = paddle.CUDAPlace(0)
    paddle.disable_static(place)
    x = paddle.to_tensor(x_np)
    dygraph_result = getattr(paddle.tensor, op_str)(x)
    return dygraph_result


def np_data_generator(low, high, np_shape, type, sv_list, op_str, *args,
                      **kwargs):
    x_np = np.random.uniform(low, high, np_shape).astype(getattr(np, type))
    # x_np.shape[0] >= len(sv_list)
    if type in ['float16', 'float32', 'float64']:
        for i, v in enumerate(sv_list):
            x_np[i] = v
    ori_shape = x_np.shape
    x_np = x_np.reshape((np.product(ori_shape), ))
    np.random.shuffle(x_np)
    x_np = x_np.reshape(ori_shape)
    result_np = getattr(np, op_str)(x_np)
    return x_np, result_np


TEST_META_DATA = [
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [8, 17, 5, 6, 7],
        'type': 'float16',
        'sv_list': [np.inf, np.nan]
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [11, 17],
        'type': 'float32',
        'sv_list': [np.inf, np.nan]
    },
    {
        'low': 0.1,
        'high': 1,
        'np_shape': [2, 3, 4, 5],
        'type': 'float64',
        'sv_list': [np.inf, np.nan]
    },
    {
        'low': 0,
        'high': 100,
        'np_shape': [11, 17, 10],
        'type': 'int32',
        'sv_list': [np.inf, np.nan]
    },
    {
        'low': 0,
        'high': 999,
        'np_shape': [132],
        'type': 'int64',
        'sv_list': [np.inf, np.nan]
    },
]


def test(test_case, op_str, use_gpu=False):
    for meta_data in TEST_META_DATA:
        meta_data = dict(meta_data)
        meta_data['op_str'] = op_str
        x_np, result_np = np_data_generator(**meta_data)
        static_result = run_static(x_np, meta_data['type'], op_str, use_gpu)
        dygraph_result = run_dygraph(x_np, op_str, use_gpu)
        test_case.assertTrue((static_result == result_np).all())
        test_case.assertTrue((dygraph_result.numpy() == result_np).all())


class TestCPUNormal(unittest.TestCase):
    def test_inf(self):
        test(self, 'isinf')

    def test_nan(self):
        test(self, 'isnan')

    def test_finite(self):
        test(self, 'isfinite')


class TestCUDANormal(unittest.TestCase):
    def test_inf(self):
        test(self, 'isinf', True)

    def test_nan(self):
        test(self, 'isnan', True)

    def test_finite(self):
        test(self, 'isfinite', True)


class TestError(unittest.TestCase):
    def test_bad_input(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):

            def test_isinf_bad_x():
                x = [1, 2, 3]
                result = paddle.tensor.isinf(x)

            self.assertRaises(TypeError, test_isinf_bad_x)

            def test_isnan_bad_x():
                x = [1, 2, 3]
                result = paddle.tensor.isnan(x)

            self.assertRaises(TypeError, test_isnan_bad_x)

            def test_isfinite_bad_x():
                x = [1, 2, 3]
                result = paddle.tensor.isfinite(x)

            self.assertRaises(TypeError, test_isfinite_bad_x)


if __name__ == '__main__':
    unittest.main()
