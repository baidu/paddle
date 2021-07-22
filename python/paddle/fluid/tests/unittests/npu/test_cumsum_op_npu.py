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

from __future__ import print_function

import unittest
import numpy as np
from paddle.fluid.tests.unittests.op_test import OpTest
import paddle
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()


class TestCumsumOp(unittest.TestCase):
    def run_cases(self):
        data_np = np.arange(12).reshape(3, 4)
        data = paddle.to_tensor(data_np)

        y = paddle.cumsum(data)
        z = np.cumsum(data_np)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, axis=0)
        z = np.cumsum(data_np, axis=0)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, axis=-1)
        z = np.cumsum(data_np, axis=-1)
        self.assertTrue(np.array_equal(z, y.numpy()))

        y = paddle.cumsum(data, dtype='float32')
        self.assertTrue(y.dtype == core.VarDesc.VarType.FP32)

        y = paddle.cumsum(data, dtype=np.int32)
        self.assertTrue(y.dtype == core.VarDesc.VarType.INT32)

        y = paddle.cumsum(data, axis=-2)
        z = np.cumsum(data_np, axis=-2)
        self.assertTrue(np.array_equal(z, y.numpy()))

    def run_static(self, use_npu=False):
        with fluid.program_guard(fluid.Program()):
            data_np = np.random.random((100, 100)).astype(np.float32)
            x = paddle.static.data('X', [100, 100])
            y = paddle.cumsum(x)
            y2 = paddle.cumsum(x, axis=0)
            y3 = paddle.cumsum(x, axis=-1)
            y4 = paddle.cumsum(x, dtype='float32')
            y5 = paddle.cumsum(x, dtype=np.int32)
            y6 = paddle.cumsum(x, axis=-2)

            place = fluid.NPUPlace(0) if use_npu else fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            out = exe.run(feed={'X': data_np},
                          fetch_list=[
                              y.name, y2.name, y3.name, y4.name, y5.name,
                              y6.name
                          ])

            z = np.cumsum(data_np)
            self.assertTrue(np.allclose(z, out[0]))
            z = np.cumsum(data_np, axis=0)
            self.assertTrue(np.allclose(z, out[1]))
            z = np.cumsum(data_np, axis=-1)
            self.assertTrue(np.allclose(z, out[2]))
            self.assertTrue(out[3].dtype == np.float32)
            self.assertTrue(out[4].dtype == np.int32)
            z = np.cumsum(data_np, axis=-2)
            self.assertTrue(np.allclose(z, out[5]))

    def test_npu(self):
        # Now, npu tests need setting paddle.enable_static()

        self.run_static(use_npu=True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = paddle.static.data('x', [3, 4])
            y = paddle.cumsum(x, name='out')
            self.assertTrue('out' in y.name)


class TestNPUCumSumOp1(OpTest):
    def setUp(self):
        self.op_type = "cumsum"
        self.set_npu()
        self.init_dtype()
        self.init_testcase()

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_npu(self):
        self.__class__.use_npu = True
        self.place = paddle.NPUPlace(0)

    def init_dtype(self):
        self.dtype = np.float32

    def init_testcase(self):
        self.attrs = {'axis': 2}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=2)}


class TestNPUCumSumOp2(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': -1, 'reverse': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {
            'Out': np.flip(
                np.flip(
                    self.inputs['X'], axis=2).cumsum(axis=2), axis=2)
        }


class TestNPUCumSumOp3(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 1}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


class TestNPUCumSumOp4(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 0}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


class TestNPUCumSumOp5(TestNPUCumSumOp1):
    def init_testcase(self):
        self.inputs = {'X': np.random.random((5, 20)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=1)}


class TestNPUCumSumOp7(TestNPUCumSumOp1):
    def init_testcase(self):
        self.inputs = {'X': np.random.random((100)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum(axis=0)}


class TestNPUCumSumExclusive1(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 65)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumExclusive2(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumExclusive3(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 888)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumExclusive4(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((1, 1, 3049)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (1, 1, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumExclusive5(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, "exclusive": True}
        a = np.random.random((4, 5, 3096)).astype(self.dtype)
        self.inputs = {'X': a}
        self.outputs = {
            'Out': np.concatenate(
                (np.zeros(
                    (4, 5, 1), dtype=self.dtype), a[:, :, :-1].cumsum(axis=2)),
                axis=2)
        }


class TestNPUCumSumReverseExclusive(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'axis': 2, 'reverse': True, "exclusive": True}
        a = np.random.random((4, 5, 6)).astype(self.dtype)
        self.inputs = {'X': a}
        a = np.flip(a, axis=2)
        self.outputs = {
            'Out': np.concatenate(
                (np.flip(
                    a[:, :, :-1].cumsum(axis=2), axis=2), np.zeros(
                        (4, 5, 1), dtype=self.dtype)),
                axis=2)
        }


class TestNPUCumSumWithFlatten1(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'flatten': True}
        self.inputs = {'X': np.random.random((5, 6)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum()}


class TestNPUCumSumWithFlatten2(TestNPUCumSumOp1):
    def init_testcase(self):
        self.attrs = {'flatten': True}
        self.inputs = {'X': np.random.random((5, 6, 10)).astype(self.dtype)}
        self.outputs = {'Out': self.inputs['X'].cumsum()}


if __name__ == '__main__':
    unittest.main()
