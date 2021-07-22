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
import six
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from tf32_precision import tf32_on_and_off


class TestTF32Switch(unittest.TestCase):
    def test_on_off(self):
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            self.assertTrue(core.get_cublas_switch())  # default
            core.set_cublas_switch(False)
            self.assertFalse(core.get_cublas_switch())  # turn off
            core.set_cublas_switch(True)
            self.assertTrue(core.get_cublas_switch())  # turn on

            core.set_cublas_switch(True)  # restore the switch
        else:
            pass


class TestTF32OnMatmul(unittest.TestCase):
    def test_dygraph_without_out(self):
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            core.set_cublas_switch(False)  # turn off
            with fluid.dygraph.guard(place):
                input_array1 = np.random.rand(4, 12, 64, 88).astype("float32")
                input_array2 = np.random.rand(4, 12, 88, 512).astype("float32")
                data1 = paddle.to_tensor(input_array1)
                data2 = paddle.to_tensor(input_array2)
                out = paddle.matmul(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
            self.assertTrue(np.allclose(expected_result, out.numpy(), 1e-03))
            core.set_cublas_switch(True)  # restore the switch
        else:
            pass


class TestTF32Precision(unittest.TestCase):
    def setUp(self):
        self.precision = 1e-6

    @tf32_on_and_off(0.001)
    def test_dygraph_without_out(self):
        if core.is_compiled_with_cuda():
            place = fluid.CUDAPlace(0)
            with fluid.dygraph.guard(place):
                input_array1 = np.random.rand(8, 8).astype("float32")
                input_array2 = np.random.rand(8, 8).astype("float32")
                data1 = fluid.dygraph.to_variable(input_array1)
                data2 = fluid.dygraph.to_variable(input_array2)
                out = paddle.matmul(data1, data2)
                expected_result = np.matmul(input_array1, input_array2)
            self.assertTrue(
                np.allclose(expected_result, out.numpy(), self.precision))
        else:
            pass


if __name__ == '__main__':
    unittest.main()
