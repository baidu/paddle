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
import paddle
import copy

np.random.seed(10)
paddle.manual_seed(10)


class TestNormalAPI(unittest.TestCase):
    def setUp(self):
        self.mean = 1.0
        self.std = 0.0
        self.shape = None
        self.repeat_num = 2000
        self.set_attrs()
        self.dtype = self.get_dtype()
        self.place=paddle.CUDAPlace(0) \
            if paddle.fluid.core.is_compiled_with_cuda() \
            else paddle.CPUPlace()

    def set_attrs(self):
        self.shape = [8, 12]

    def get_shape(self):
        if isinstance(self.mean, np.ndarray):
            shape = self.mean.shape
        elif isinstance(self.std, np.ndarray):
            shape = self.std.shape
        else:
            shape = self.shape
        return list(shape)

    def get_dtype(self):
        if isinstance(self.mean, np.ndarray):
            return self.mean.dtype
        elif isinstance(self.std, np.ndarray):
            return self.std.dtype
        else:
            return 'float32'

    def static_api(self):
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)
        if isinstance(self.mean, np.ndarray) \
            and isinstance(self.std, np.ndarray):
            with paddle.static.program_guard(paddle.static.Program()):
                mean = paddle.fluid.data('Mean', self.mean.shape, self.mean.dtype)
                std = paddle.fluid.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={
                        'Mean': self.mean,
                        'Std': self.std.reshape(shape)
                    },
                                  fetch_list=[out])
                    ret_all[i] = ret[0]
            return ret_all
        elif isinstance(self.mean, np.ndarray):
            with paddle.static.program_guard(paddle.static.Program()):
                mean = paddle.fluid.data('Mean', self.mean.shape, self.mean.dtype)
                out = paddle.normal(mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={'Mean': self.mean}, fetch_list=[out])
                    ret_all[i] = ret[0]
            return ret_all
        elif isinstance(self.std, np.ndarray):
            with paddle.static.program_guard(paddle.static.Program()):
                std = paddle.fluid.data('Std', self.std.shape, self.std.dtype)
                out = paddle.normal(self.mean, std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(feed={'Std': self.std}, fetch_list=[out])
                    ret_all[i] = ret[0]
            return ret_all
        else:
            with paddle.static.program_guard(paddle.static.Program()):
                out = paddle.normal(self.mean, self.std, self.shape)

                exe = paddle.static.Executor(self.place)
                for i in range(self.repeat_num):
                    ret = exe.run(fetch_list=[out])
                    ret_all[i] = ret[0]
            return ret_all

    def dygraph_api(self):
        paddle.disable_static(self.place)
        shape = self.get_shape()
        ret_all_shape = copy.deepcopy(shape)
        ret_all_shape.insert(0, self.repeat_num)
        ret_all = np.zeros(ret_all_shape, self.dtype)

        mean = paddle.to_tensor(self.mean) \
            if isinstance(self.mean, np.ndarray) else self.mean
        std = paddle.to_tensor(self.std) \
            if isinstance(self.std, np.ndarray) else self.std
        for i in range(self.repeat_num):
            out = paddle.normal(mean, std, self.shape)
            ret_all[i] = out.numpy()
        paddle.enable_static()
        return ret_all

    def test_api(self):
        ret_static = self.static_api()
        ret_dygraph = self.dygraph_api()
        for ret in [ret_static, ret_dygraph]:
            shape_ref = self.get_shape()
            self.assertEqual(shape_ref, list(ret[0].shape))

            ret = ret.flatten().reshape([self.repeat_num, -1])
            mean = np.mean(ret, axis=0)
            std = np.std(ret, axis=0)
            mean_ref=self.mean.reshape([1, -1]) \
                if isinstance(self.mean, np.ndarray) else self.mean
            std_ref=self.std.reshape([1, -1]) \
                if isinstance(self.std, np.ndarray) else self.std
            self.assertTrue(np.allclose(mean_ref, mean, 0.2, 0.2))
            self.assertTrue(np.allclose(std_ref, std, 0.2, 0.2))


class TestNormalAPI_mean_is_tensor(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(-2, -1, [2, 3, 4, 5]).astype('float64')


class TestNormalAPI_std_is_tensor(TestNormalAPI):
    def set_attrs(self):
        self.std = np.random.uniform(0.7, 1, [2, 3, 17]).astype('float64')


class TestNormalAPI_mean_std_are_tensor(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(1, 2, [1, 100]).astype('float64')
        self.std = np.random.uniform(0.5, 1, [1, 100]).astype('float64')


class TestNormalAPI_mean_std_are_tensor_with_different_dtype(TestNormalAPI):
    def set_attrs(self):
        self.mean = np.random.uniform(1, 2, [100]).astype('float64')
        self.std = np.random.uniform(1, 2, [100]).astype('float32')


class TestNormalAlias(unittest.TestCase):
    def test_alias(self):
        paddle.disable_static()
        shape = [1, 2, 3]
        out1 = paddle.normal(shape=shape)
        out2 = paddle.tensor.normal(shape=shape)
        out3 = paddle.tensor.random.normal(shape=shape)
        paddle.enable_static()


class TestNormalErrors(unittest.TestCase):
    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            mean = [1, 2, 3]
            self.assertRaises(TypeError, paddle.normal, mean)

            std = [1, 2, 3]
            self.assertRaises(TypeError, paddle.normal, std=std)

            mean = paddle.fluid.data('Mean', [100], 'int32')
            self.assertRaises(TypeError, paddle.normal, mean)

            std = paddle.fluid.data('Std', [100], 'int32')
            self.assertRaises(TypeError, paddle.normal, mean=1.0, std=std)

            self.assertRaises(TypeError, paddle.normal, shape=1)

            self.assertRaises(TypeError, paddle.normal, shape=[1.0])

            shape = paddle.fluid.data('Shape', [100], 'float32')
            self.assertRaises(TypeError, paddle.normal, shape=shape)


if __name__ == "__main__":
    unittest.main()
