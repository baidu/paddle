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

from __future__ import print_function

import unittest
import numpy as np

import paddle
import paddle.fluid.core as core


class TestInplace(unittest.TestCase):
    def test_forward_version(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(np.ones((4, 2, 3)).astype(np.float32))
            self.assertEqual(var.inplace_version, 0)

            var[0] = 1.1
            self.assertEqual(var.inplace_version, 1)

            paddle.assign(paddle.ones(shape=[3]), var)

            # NOTE(liym27): assign(input, output) is an inplace operation for output.
            # There is inplace-related processing for api assign, var.inplace_version should be 2 not 1.
            self.assertEqual(var.inplace_version, 2)

            var[2] = 3
            self.assertEqual(var.inplace_version, 3)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            var_b[1:2] = 3.3  # var_b is modified inplace after using it

            var_d = var_b**2

            loss = paddle.nn.functional.relu(var_c + var_d)
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()

    def test_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2
            var_b[1:2] = 3  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            loss = var_c.sum()
            loss.backward()

    def test_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.ones(shape=[4, 2, 3], dtype="float32")
            var_a.stop_gradient = False

            var_b = var_a**2

            var_b[1:2] = 3  # var_b is modified inplace before using it

            var_c = var_b + var_b  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_c.sum()

            var_b[1:2] = 3  # var_b is modified inplace after using it

            loss.backward()


class TestDygraphInplace(unittest.TestCase):
    def setUp(self):
        self.init_data()
        self.set_np_compare_func()

    def init_data(self):
        self.input_var_numpy = np.random.uniform(-5, 5, [10, 20, 1])
        self.dtype = "float32"

    def set_np_compare_func(self):
        self.np_compare = np.array_equal

    def non_inplace_api_processing(self, var):
        return paddle.squeeze(var)

    def inplace_api_processing(self, var):
        return paddle.squeeze_(var)

    def test_inplace_api(self):
        var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
        inplace_var = self.inplace_api_processing(var)
        self.assertTrue(id(var) == id(inplace_var))

        inplace_var[0] = 2.
        self.assertTrue(np.array_equal(var.numpy(), inplace_var.numpy()))

    def test_forward_version(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            self.assertEqual(var.inplace_version, 0)

            inplace_var = self.inplace_api_processing(var)
            self.assertEqual(var.inplace_version, 1)

            inplace_var[0] = 2.
            self.assertEqual(var.inplace_version, 2)

            inplace_var = self.inplace_api_processing(inplace_var)
            self.assertEqual(var.inplace_version, 3)

    def test_leaf_inplace_var_error(self):
        with paddle.fluid.dygraph.guard():
            var = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var.stop_gradient = False

            def leaf_inplace_error():
                self.inplace_api_processing(var)

            self.assertRaises(ValueError, leaf_inplace_error)

    def test_backward_error(self):
        # It raises an error because the inplace operator will result
        # in incorrect gradient computation.
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            # Here, the gradient computation will use the value of var_b
            var_c = var_b**2
            self.inplace_api_processing(var_b)

            loss = paddle.nn.functional.relu(var_c)
            with self.assertRaisesRegexp(
                    RuntimeError,
                    "received tensor_version:{} != wrapper_version_snapshot:{}".
                    format(1, 0)):
                loss.backward()

    def test_backward_success_1(self):
        # var_b is modified inplace before using it, the inplace operator doesn't result
        # in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.inplace_api_processing(
                var_b)  # var_b is modified inplace before using it

            # Here, the gradient computation will use the value of var_b
            var_d = var_c**2
            loss = var_d.sum()
            loss.backward()
            grad_var_a_inplace = var_a.grad

        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2
            var_c = self.non_inplace_api_processing(var_b)
            var_d = var_c**2
            loss = var_d.sum()
            loss.backward()
            grad_var_a = var_a.grad

        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))

    def test_backward_success_2(self):
        # Although var_b is modified inplace after using it, it does not used in gradient computation.
        # The inplace operator doesn't result in incorrect gradient computation.
        grad_var_a, grad_var_a_inplace = 0, 1
        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.inplace_api_processing(
                var_b)  # var_b is modified inplace before using it

            var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_d.sum()

            loss.backward()
            grad_var_a_inplace = var_a.grad

        with paddle.fluid.dygraph.guard():
            var_a = paddle.to_tensor(self.input_var_numpy).astype(self.dtype)
            var_a.stop_gradient = False

            var_b = var_a**2

            var_c = self.non_inplace_api_processing(
                var_b)  # var_b is modified inplace before using it

            var_d = var_c + var_c  # Here, the grad op of sum doesn't use the value of var_b
            loss = var_d.sum()

            loss.backward()
            grad_var_a = var_a.grad
        self.assertTrue(self.np_compare(grad_var_a_inplace, grad_var_a))


class TestDygraphInplaceUnsqueeze(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.unsqueeze(var, -1)

    def inplace_api_processing(self, var):
        return paddle.unsqueeze_(var, -1)


class TestDygraphInplaceReshape(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.reshape(var, [-1])

    def inplace_api_processing(self, var):
        return paddle.reshape_(var, [-1])


class TestDygraphInplaceFlatten(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.flatten(var)

    def inplace_api_processing(self, var):
        return paddle.flatten_(var)


class TestDygraphInplaceScatter(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.array([[1, 1], [2, 2], [3, 3]])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

        return paddle.scatter(var, index, updates, overwrite=False)

    def inplace_api_processing(self, var):
        index = paddle.to_tensor([2, 1, 0, 1], dtype='int64')
        updates = paddle.to_tensor(
            [[1, 1], [2, 2], [3, 3], [4, 4]], dtype='float32')

        return paddle.scatter_(var, index, updates, overwrite=False)


class TestDygraphInplaceElu(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.elu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.elu_(var)


class TestDygraphInplaceRelu(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.relu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.relu_(var)


class TestDygraphInplaceSoftmax(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.softmax(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_(var)


class TestDygraphInplaceRelu6(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.relu6(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.relu6_(var)


class TestDygraphInplaceHardsigmoid(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.hardsigmoid(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.hardsigmoid_(var)


class TestDygraphInplaceLeakyRelu(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.leaky_relu(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.leaky_relu_(var)


class TestDygraphInplaceSoftmaxWithCrossEntropyForward(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.rand(64, 100)
        label_numpy = np.random.rand(64, 1).astype("int64")
        self.label_var = paddle.to_tensor(label_numpy)
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_with_cross_entropy(
            var, self.label_var, return_softmax=True)[1]

    def inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_with_cross_entropy_(
            var, self.label_var, return_softmax=True)[1]

    # grad_var of `Softmax` in non-inplace API is None.
    # grad_var of `Softmax` in inplace API is grad_var of `Logits`.
    def test_backward_success_1(self):
        pass

    def test_backward_success_2(self):
        pass


class TestDygraphInplaceSoftmaxWithCrossEntropyBackward(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.rand(64, 100)
        label_numpy = np.random.rand(64, 1).astype("int64")
        self.label_var = paddle.to_tensor(label_numpy)
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_with_cross_entropy(
            var, self.label_var, return_softmax=False)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.softmax_with_cross_entropy_(
            var, self.label_var, return_softmax=False)

    def test_inplace_api(self):
        pass

    def test_forward_version(self):
        pass


class TestDygraphInplaceTanh(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.tanh(var)

    def inplace_api_processing(self, var):
        return paddle.tanh_(var)


class TestDygraphInplaceCeil(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.ceil(var)

    def inplace_api_processing(self, var):
        return paddle.ceil_(var)


class TestDygraphInplaceFloor(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.floor(var)

    def inplace_api_processing(self, var):
        return paddle.floor_(var)


class TestDygraphInplaceExp(TestDygraphInplace):
    def set_np_compare_func(self):
        self.np_compare = np.allclose

    def non_inplace_api_processing(self, var):
        return paddle.exp(var)

    def inplace_api_processing(self, var):
        return paddle.exp_(var)


class TestDygraphInplaceReciprocal(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.reciprocal(var)

    def inplace_api_processing(self, var):
        return paddle.reciprocal_(var)


class TestDygraphInplaceRound(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.round(var)

    def inplace_api_processing(self, var):
        return paddle.round_(var)


class TestDygraphInplaceSqrt(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.uniform(0, 5, [10, 20, 1])
        self.dtype = "float32"

    def non_inplace_api_processing(self, var):
        return paddle.sqrt(var)

    def inplace_api_processing(self, var):
        return paddle.sqrt_(var)


class TestDygraphInplaceRsqrt(TestDygraphInplaceSqrt):
    def non_inplace_api_processing(self, var):
        return paddle.rsqrt(var)

    def inplace_api_processing(self, var):
        return paddle.rsqrt_(var)


class TestDygraphInplaceSigmoid(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.nn.functional.sigmoid(var)

    def inplace_api_processing(self, var):
        return paddle.nn.functional.sigmoid_(var)


class TestDygraphInplaceClip(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.clip(var, 0.6, 1.5)

    def inplace_api_processing(self, var):
        return paddle.clip_(var, 0.6, 1.5)


class TestDygraphInplaceScale(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.scale(var, scale=2.0, bias=3.0)

    def inplace_api_processing(self, var):
        return paddle.scale_(var, scale=2.0, bias=3.0)


class TestDygraphInplaceAdd(TestDygraphInplace):
    def init_data(self):
        self.input_var_numpy = np.random.rand(2, 3, 4)
        self.dtype = "float32"
        input_var_numpy_2 = np.random.rand(2, 3, 4).astype(self.dtype)
        self.input_var_2 = paddle.to_tensor(input_var_numpy_2)

    def non_inplace_api_processing(self, var):
        return paddle.add(var, self.input_var_2)

    def inplace_api_processing(self, var):
        return paddle.add_(var, self.input_var_2)


class TestDygraphInplaceSubtract(TestDygraphInplaceAdd):
    def non_inplace_api_processing(self, var):
        return paddle.subtract(var, self.input_var_2)

    def inplace_api_processing(self, var):
        return paddle.subtract_(var, self.input_var_2)


class TestDygraphInplaceAddN(TestDygraphInplace):
    def non_inplace_api_processing(self, var):
        return paddle.add_n(var)

    def inplace_api_processing(self, var):
        return paddle.add_n_(var)


class TestDygraphInplaceAddNMultiInput(TestDygraphInplaceAdd):
    def non_inplace_api_processing(self, var):
        return paddle.add_n([var, self.input_var_2])

    def inplace_api_processing(self, var):
        return paddle.add_n_([var, self.input_var_2])


if __name__ == '__main__':
    unittest.main()
