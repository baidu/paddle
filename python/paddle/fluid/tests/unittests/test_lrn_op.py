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

import paddle
import unittest
import numpy as np
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard


class TestLRNOp(OpTest):
    def get_input(self):
        ''' TODO(gongweibao): why it's grad diff is so large?
        x = np.ndarray(
            shape=(self.N, self.C, self.H, self.W), dtype=float, order='C')
        for m in range(0, self.N):
            for i in range(0, self.C):
                for h in range(0, self.H):
                    for w in range(0, self.W):
                        x[m][i][h][w] = m * self.C * self.H * self.W +  \
                                        i * self.H * self.W +  \
                                        h * self.W + w + 1
        '''
        x = np.random.rand(self.N, self.C, self.H, self.W).astype("float32")
        return x + 1

    def get_out(self):
        start = -(self.n - 1) // 2
        end = start + self.n

        mid = np.empty((self.N, self.C, self.H, self.W)).astype("float32")
        mid.fill(self.k)
        for m in range(0, self.N):
            for i in range(0, self.C):
                for c in range(start, end):
                    ch = i + c
                    if ch < 0 or ch >= self.C:
                        continue

                    s = mid[m][i][:][:]
                    r = self.x[m][ch][:][:]
                    s += np.square(r) * self.alpha

        mid2 = np.power(mid, -self.beta)
        return np.multiply(self.x, mid2), mid

    def get_attrs(self):
        attrs = {
            'n': self.n,
            'k': self.k,
            'alpha': self.alpha,
            'beta': self.beta,
            'data_format': self.data_format
        }
        return attrs

    def setUp(self):
        self.op_type = "lrn"
        self.init_test_case()

        self.N = 2
        self.C = 3
        self.H = 5
        self.W = 5

        self.n = 5
        self.k = 2.0
        self.alpha = 0.0001
        self.beta = 0.75
        self.x = self.get_input()
        self.out, self.mid_out = self.get_out()
        if self.data_format == 'NHWC':
            self.x = np.transpose(self.x, [0, 2, 3, 1])
            self.out = np.transpose(self.out, [0, 2, 3, 1])
            self.mid_out = np.transpose(self.mid_out, [0, 2, 3, 1])

        self.inputs = {'X': self.x}
        self.outputs = {'Out': self.out, 'MidOut': self.mid_out}
        self.attrs = self.get_attrs()

    def init_test_case(self):
        self.data_format = 'NCHW'

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out')


class TestLRNOpAttrDataFormat(TestLRNOp):
    def init_test_case(self):
        self.data_format = 'NHWC'


class TestLRNAPI(unittest.TestCase):
    def test_case(self):
        data1 = fluid.data(name='data1', shape=[2, 4, 5, 5], dtype='float32')
        data2 = fluid.data(name='data2', shape=[2, 5, 5, 4], dtype='float32')
        out1 = fluid.layers.lrn(data1, data_format='NCHW')
        out2 = fluid.layers.lrn(data2, data_format='NHWC')
        data1_np = np.random.random((2, 4, 5, 5)).astype("float32")
        data2_np = np.transpose(data1_np, [0, 2, 3, 1])

        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
        else:
            place = core.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        results = exe.run(fluid.default_main_program(),
                          feed={"data1": data1_np,
                                "data2": data2_np},
                          fetch_list=[out1, out2],
                          return_numpy=True)

        self.assertTrue(
            np.allclose(results[0], np.transpose(results[1], (0, 3, 1, 2))))

    def test_exception(self):
        input1 = fluid.data(name="input1", shape=[2, 4, 5, 5], dtype="float32")
        input2 = fluid.data(
            name="input2", shape=[2, 4, 5, 5, 5], dtype="float32")

        def _attr_data_fromat():
            out = fluid.layers.lrn(input1, data_format='NDHW')

        def _input_dim_size():
            out = fluid.layers.lrn(input2)

        self.assertRaises(ValueError, _attr_data_fromat)
        self.assertRaises(ValueError, _input_dim_size)


class TestLRNOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # the input must be float32
            in_w = fluid.data(name="in_w", shape=[None, 3, 3, 3], dtype="int64")
            self.assertRaises(TypeError, fluid.layers.lrn, in_w)


class TestLocalResponseNormFAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_3d_input(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            in_np1 = np.random.random([3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 1))

            input1 = fluid.data(
                name="input1", shape=[3, 40, 40], dtype="float32")
            input2 = fluid.data(
                name="input2", shape=[3, 40, 40], dtype="float32")
            res1 = paddle.nn.functional.local_response_norm(
                x=input1, size=5, data_format='NCL')
            res2 = paddle.nn.functional.local_response_norm(
                x=input2, size=5, data_format='NLC')
            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input1": in_np1,
                                    "input2": in_np2},
                              fetch_list=[res1, res2])

            fetches1_tran = np.transpose(fetches[1], (0, 2, 1))
            self.assertTrue(np.allclose(fetches[0], fetches1_tran))

    def check_static_4d_input(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input1 = fluid.data(
                name="input1", shape=[3, 3, 40, 40], dtype="float32")
            input2 = fluid.data(
                name="input2", shape=[3, 40, 40, 3], dtype="float32")

            res1 = paddle.nn.functional.local_response_norm(
                x=input1, size=5, data_format='NCHW')
            res2 = paddle.nn.functional.local_response_norm(
                x=input2, size=5, data_format='NHWC')

            in_np1 = np.random.random([3, 3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 3, 1))

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input1": in_np1,
                                    "input2": in_np2},
                              fetch_list=[res1, res2])

            fetches1_tran = np.transpose(fetches[1], (0, 3, 1, 2))
            self.assertTrue(np.allclose(fetches[0], fetches1_tran))

    def check_static_5d_input(self, place):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input1 = fluid.data(
                name="input1", shape=[3, 3, 3, 40, 40], dtype="float32")
            input2 = fluid.data(
                name="input2", shape=[3, 3, 40, 40, 3], dtype="float32")
            res1 = paddle.nn.functional.local_response_norm(
                x=input1, size=5, data_format='NCDHW')
            res2 = paddle.nn.functional.local_response_norm(
                x=input2, size=5, data_format='NDHWC')

            in_np1 = np.random.random([3, 3, 3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 3, 4, 1))

            exe = fluid.Executor(place)
            fetches = exe.run(fluid.default_main_program(),
                              feed={"input1": in_np1,
                                    "input2": in_np2},
                              fetch_list=[res1, res2])

            fetches1_tran = np.transpose(fetches[1], (0, 4, 1, 2, 3))
            self.assertTrue(np.allclose(fetches[0], fetches1_tran))

    def test_static(self):
        for place in self.places:
            self.check_static_3d_input(place=place)
            self.check_static_4d_input(place=place)
            self.check_static_5d_input(place=place)

    def check_dygraph_3d_input(self, place):
        with fluid.dygraph.guard(place):
            in_np1 = np.random.random([3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 1))

            in1 = paddle.to_tensor(in_np1)
            in2 = paddle.to_tensor(in_np2)

            res1 = paddle.nn.functional.local_response_norm(
                x=in1, size=5, data_format='NCL')
            res2 = paddle.nn.functional.local_response_norm(
                x=in2, size=5, data_format='NLC')

            res2_tran = np.transpose(res2.numpy(), (0, 2, 1))
            self.assertTrue(np.allclose(res1.numpy(), res2_tran))

    def check_dygraph_4d_input(self, place):
        with fluid.dygraph.guard(place):
            in_np1 = np.random.random([3, 3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 3, 1))

            in1 = paddle.to_tensor(in_np1)
            in2 = paddle.to_tensor(in_np2)

            res1 = paddle.nn.functional.local_response_norm(
                x=in1, size=5, data_format='NCHW')
            res2 = paddle.nn.functional.local_response_norm(
                x=in2, size=5, data_format='NHWC')

            res2_tran = np.transpose(res2.numpy(), (0, 3, 1, 2))
            self.assertTrue(np.allclose(res1.numpy(), res2_tran))

    def check_dygraph_5d_input(self, place):
        with fluid.dygraph.guard(place):
            in_np1 = np.random.random([3, 3, 3, 40, 40]).astype("float32")
            in_np2 = np.transpose(in_np1, (0, 2, 3, 4, 1))

            in1 = paddle.to_tensor(in_np1)
            in2 = paddle.to_tensor(in_np2)

            res1 = paddle.nn.functional.local_response_norm(
                x=in1, size=5, data_format='NCDHW')
            res2 = paddle.nn.functional.local_response_norm(
                x=in2, size=5, data_format='NDHWC')

            res2_tran = np.transpose(res2.numpy(), (0, 4, 1, 2, 3))
            self.assertTrue(np.allclose(res1.numpy(), res2_tran))

    def test_dygraph(self):
        for place in self.places:
            self.check_dygraph_3d_input(place)
            self.check_dygraph_4d_input(place)
            self.check_dygraph_5d_input(place)


class TestLocalResponseNormFAPIError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_Variable():
                # the input of lrn must be Variable.
                x1 = fluid.create_lod_tensor(
                    np.array([-1, 3, 5, 5]), [[1, 1, 1, 1]], fluid.CPUPlace())
                paddle.nn.functional.local_response_norm(x1, size=5)

            self.assertRaises(TypeError, test_Variable)

            def test_datatype():
                x = fluid.data(name='x', shape=[3, 4, 5, 6], dtype="int32")
                paddle.nn.functional.local_response_norm(x, size=5)

            self.assertRaises(TypeError, test_datatype)

            def test_dataformat():
                x = fluid.data(name='x', shape=[3, 4, 5, 6], dtype="float32")
                paddle.nn.functional.local_response_norm(
                    x, size=5, data_format="NCTHW")

            self.assertRaises(ValueError, test_dataformat)

            def test_dim():
                x = fluid.data(name='x', shape=[3, 4], dtype="float32")
                paddle.nn.functional.local_response_norm(x, size=5)

            self.assertRaises(ValueError, test_dim)


class TestLocalResponseNormCAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(123)
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                in1 = paddle.rand(shape=(3, 3, 40, 40), dtype="float32")
                in2 = paddle.transpose(in1, [0, 2, 3, 1])

                m1 = paddle.nn.LocalResponseNorm(size=5, data_format='NCHW')
                m2 = paddle.nn.LocalResponseNorm(size=5, data_format='NHWC')

                res1 = m1(in1)
                res2 = m2(in2)

                res2_tran = np.transpose(res2.numpy(), (0, 3, 1, 2))
                self.assertTrue(np.allclose(res1.numpy(), res2_tran))


if __name__ == "__main__":
    unittest.main()
