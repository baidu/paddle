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

from __future__ import print_function, division

import unittest
from decorator_helper import prog_scope
import paddle
import paddle.fluid as fluid
import numpy
import numpy as np


class TestMathOpPatches(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    @prog_scope()
    def test_add_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a + 10
        ab = fluid.layers.concat(input=[a, b], axis=1)
        c = ab + 10
        d = ab + a
        # e = a + ab
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np, c_np, d_np = exe.run(fluid.default_main_program(),
                                   feed={"a": a_np},
                                   fetch_list=[b, c, d])
        self.assertTrue(numpy.allclose(a_np + 10, b_np))
        ab_np = numpy.concatenate([a_np, b_np], axis=1)
        self.assertTrue(numpy.allclose(ab_np + 10, c_np))
        d_expected = ab_np + numpy.concatenate([a_np, a_np], axis=1)
        self.assertTrue(numpy.allclose(d_expected, d_np))

    @prog_scope()
    def test_radd_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 + a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np + 10, b_np))

    @prog_scope()
    def test_sub_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a - 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np - 10, b_np))

    @prog_scope()
    def test_radd_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 - a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 - a_np, b_np))

    @prog_scope()
    def test_mul_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a * 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np * 10, b_np))

    @prog_scope()
    def test_rmul_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 * a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 * a_np, b_np))

    @prog_scope()
    def test_div_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = a / 10
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np / 10, b_np))

    @prog_scope()
    def test_rdiv_scalar(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = 10 / a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32') + 1e-2

        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(10 / a_np, b_np))

    @prog_scope()
    def test_div_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a / b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32') + 1e-2
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np / b_np, c_np))

    @prog_scope()
    def test_mul_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a * b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np * b_np, c_np))

    @prog_scope()
    def test_add_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a + b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np + b_np, c_np))

    @prog_scope()
    def test_sub_two_tensor(self):
        a = fluid.layers.data(name="a", shape=[1])
        b = fluid.layers.data(name="b", shape=[1])
        c = a - b
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.random(size=[10, 1]).astype('float32')
        b_np = numpy.random.random(size=[10, 1]).astype('float32')
        c_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np,
                             'b': b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np - b_np, c_np))

    @prog_scope()
    def test_integer_div(self):
        a = fluid.layers.data(name="a", shape=[1], dtype='int64')
        b = a / 7
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.array([3, 4, 10, 14, 9, 18]).astype('int64')
        b_np, = exe.run(fluid.default_main_program(),
                        feed={"a": a_np},
                        fetch_list=[b])

        b_np_actual = (a_np / 7).astype('float32')
        self.assertTrue(numpy.allclose(b_np, b_np_actual))

    @prog_scope()
    def test_equal(self):
        a = fluid.layers.data(name="a", shape=[1], dtype='float32')
        b = fluid.layers.data(name="b", shape=[1], dtype='float32')
        c = (a == b)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.array([3, 4, 10, 14, 9, 18]).astype('float32')
        b_np = numpy.array([3, 4, 11, 15, 8, 18]).astype('float32')

        c_np, = exe.run(fluid.default_main_program(),
                        feed={"a": a_np,
                              "b": b_np},
                        fetch_list=[c])

        self.assertTrue(numpy.array_equal(c_np, a_np == b_np))
        self.assertEqual(c.dtype, fluid.core.VarDesc.VarType.BOOL)

    @prog_scope()
    def test_equal_and_cond(self):
        a = fluid.layers.data(name="a", shape=[1], dtype='float32')
        b = fluid.layers.data(name="b", shape=[1], dtype='float32')

        one = fluid.layers.ones(shape=[1], dtype='int32')
        zero = fluid.layers.zeros(shape=[1], dtype='int32')
        cond = (one == zero)
        c = fluid.layers.cond(cond, lambda: a + b, lambda: a - b)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.array([3, 4, 10, 14, 9, 18]).astype('float')
        b_np = numpy.array([3, 4, 11, 15, 8, 18]).astype('float')
        c_np, = exe.run(fluid.default_main_program(),
                        feed={"a": a_np,
                              "b": b_np},
                        fetch_list=[c])

        self.assertTrue(numpy.array_equal(c_np, a_np - b_np))

    @prog_scope()
    def test_neg(self):
        a = fluid.layers.data(name="a", shape=[10, 1])
        b = -a
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.uniform(-1, 1, size=[10, 1]).astype('float32')

        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(-a_np, b_np))

    @prog_scope()
    def test_astype(self):
        a = fluid.layers.data(name="a", shape=[10, 1])
        b = a.astype('float32')
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        a_np = numpy.random.uniform(-1, 1, size=[10, 1]).astype('float64')

        b_np = exe.run(fluid.default_main_program(),
                       feed={"a": a_np},
                       fetch_list=[b])
        self.assertTrue(numpy.allclose(a_np.astype('float32'), b_np))

    def test_bitwise_and(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np & y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x & y

        exe = fluid.Executor()
        out = exe.run(fluid.default_main_program(),
                      feed={"x": x_np,
                            "y": y_np},
                      fetch_list=[z])
        self.assertTrue(np.array_equal(out[0], out_np))

    @prog_scope()
    def test_bitwise_or(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np | y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x | y

        exe = fluid.Executor()
        out = exe.run(fluid.default_main_program(),
                      feed={"x": x_np,
                            "y": y_np},
                      fetch_list=[z])
        self.assertTrue(np.array_equal(out[0], out_np))

    @prog_scope()
    def test_bitwise_xor(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        y_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = x_np ^ y_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        y = paddle.static.data(name="y", shape=[2, 3, 5], dtype="int32")
        z = x ^ y

        exe = fluid.Executor()
        out = exe.run(fluid.default_main_program(),
                      feed={"x": x_np,
                            "y": y_np},
                      fetch_list=[z])
        self.assertTrue(np.array_equal(out[0], out_np))

    @prog_scope()
    def test_bitwise_not(self):
        x_np = np.random.randint(-100, 100, [2, 3, 5]).astype("int32")
        out_np = ~x_np

        x = paddle.static.data(name="x", shape=[2, 3, 5], dtype="int32")
        z = ~x

        exe = fluid.Executor()
        out = exe.run(fluid.default_main_program(),
                      feed={"x": x_np},
                      fetch_list=[z])
        self.assertTrue(np.array_equal(out[0], out_np))

    @prog_scope()
    def test_ndim(self):
        a = paddle.static.data(name="a", shape=[10, 1])
        self.assertEqual(a.dim(), 2)
        self.assertEqual(a.ndimension(), 2)
        self.assertEqual(a.ndim, 2)

    @prog_scope()
    def test_matmul(self):
        a = paddle.static.data(name='a', shape=[2, 3], dtype='float32')
        b = paddle.static.data(name='b', shape=[3, 5], dtype='float32')
        c = a @b  # __matmul__
        a_np = numpy.random.uniform(-1, 1, size=[2, 3]).astype('float32')
        b_np = numpy.random.uniform(-1, 1, size=[3, 5]).astype('float32')
        place = paddle.CPUPlace()
        exe = paddle.static.Executor(place)
        c_np = exe.run(paddle.static.default_main_program(),
                       feed={"a": a_np,
                             "b": b_np},
                       fetch_list=[c])
        self.assertTrue(numpy.allclose(a_np @b_np, c_np))


if __name__ == '__main__':
    unittest.main()
