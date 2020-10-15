# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard


class TestInf(OpTest):
    def setUp(self):
        self.op_type = "isinf"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.inf
        x[-1] = np.inf

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(True).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


class TestRaiseError(unittest.TestCase):
    def test_errors(self):
        def test_type():
            fluid.layers.isfinite([10])

        self.assertRaises(TypeError, test_type)

        def test_dtype():
            data = fluid.data(shape=[10], dtype="float16", name="input")
            fluid.layers.isfinite(data)

        self.assertRaises(TypeError, test_dtype)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16Inf(TestInf):
    def init_dtype(self):
        self.dtype = np.float16


class TestNAN(OpTest):
    def setUp(self):
        self.op_type = "isnan"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.nan
        x[-1] = np.nan

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(True).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16NAN(TestNAN):
    def init_dtype(self):
        self.dtype = np.float16


class TestIsfinite(OpTest):
    def setUp(self):
        self.op_type = "isfinite"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        x[0] = np.inf
        x[-1] = np.nan
        out = np.isinf(x) | np.isnan(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': np.array(False).astype(self.dtype)}

    def init_dtype(self):
        pass

    def test_output(self):
        self.check_output()


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16Isfinite(TestIsfinite):
    def init_dtype(self):
        self.dtype = np.float16


class BadInputTest(unittest.TestCase):
    def test_error(self):
        with fluid.program_guard(fluid.Program()):

            def test_has_inf_bad_x():
                data = [1, 2, 3]
                result = fluid.layers.has_inf(data)

            self.assertRaises(TypeError, test_has_inf_bad_x)

            def test_has_nan_bad_x():
                data = [1, 2, 3]
                result = fluid.layers.has_nan(data)

            self.assertRaises(TypeError, test_has_nan_bad_x)

        with fluid.dygraph.guard():
            data = paddle.zeros([2, 3])
            result = paddle.fluid.layers.has_inf(data)
            expect_value = np.array([False])
            self.assertEqual((result.numpy() == expect_value).all(), True)
            result = paddle.fluid.layers.has_nan(data)
            self.assertEqual((result.numpy() == expect_value).all(), True)


if __name__ == '__main__':
    unittest.main()
