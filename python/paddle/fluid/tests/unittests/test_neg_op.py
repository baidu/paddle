#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class TestNegOp(unittest.TestCase):
    def setUp(self):
        self.init_dtype_type()
        self.input = (np.random.random((32, 8)) * 100).astype(self.dtype)

    def init_dtype_type(self):
        self.dtype = np.float64

    def run_imperative(self):
        input = paddle.to_tensor(self.input)
        dy_result = paddle.neg(input)
        expected_result = np.negative(self.input)
        self.assertTrue(np.allclose(dy_result.numpy(), expected_result))

    def run_static(self, use_gpu=False):
        input = paddle.fluid.data(name='input', shape=[32, 8], dtype=self.dtype)
        result = paddle.neg(input)

        place = paddle.CUDAPlace(0) if use_gpu else paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        exe.run(paddle.static.default_startup_program())
        st_result = exe.run(feed={"input": self.input}, fetch_list=[result])
        expected_result = np.negative(self.input)
        self.assertTrue(np.allclose(st_result[0], expected_result))

    def test_cpu(self):
        paddle.disable_static(place=paddle.CPUPlace())
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static()

    def test_gpu(self):
        if not paddle.fluid.core.is_compiled_with_cuda():
            return

        paddle.disable_static(place=paddle.CUDAPlace(0))
        self.run_imperative()
        paddle.enable_static()

        with paddle.static.program_guard(paddle.static.Program()):
            self.run_static(use_gpu=True)


class TestNegOpFp32(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.float32


class TestNegOpInt64(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int64


class TestNegOpInt32(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int32


class TestNegOpInt16(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int16


class TestNegOpInt8(TestNegOp):
    def init_dtype_type(self):
        self.dtype = np.int8


if __name__ == "__main__":
    unittest.main()
