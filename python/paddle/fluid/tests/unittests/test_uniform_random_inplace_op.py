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
import paddle
import paddle.fluid as fluid
import numpy as np


class TestUniformRandomInplaceOpDtype(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_dtype(self):
        def test_fp32():
            tensor_fp32 = paddle.ones(self.shape, dtype=paddle.float32)
            tensor_fp32.uniform_()
            self.assertEqual(tensor_fp32.dtype, paddle.float32)

        def test_fp64():
            tensor_fp64 = paddle.ones(self.shape, paddle.float64)
            tensor_fp64.uniform_()
            self.assertEqual(tensor_fp64.dtype, paddle.float64)

        places = ['cpu']
        if fluid.core.is_compiled_with_cuda():
            places.append('gpu')
        for place in places:
            paddle.set_device(place)
            test_fp32()
            test_fp64()


class TestUniformRandomInplaceOpIsInplace(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_is_inplace(self):
        tensor_a = paddle.ones(self.shape)
        tensor_b = tensor_a.uniform_()
        self.assertTrue(tensor_a is tensor_b)


class TestUniformRandomInplaceOpSeedIsZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.seed = 0

    def test_uniform_random_inplace_op_seed_is_zero(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(seed=self.seed)
        tensor_data_first = tensor.numpy()
        tensor.uniform_(seed=self.seed)
        tensor_data_second = tensor.numpy()
        self.assertFalse((tensor_data_first == tensor_data_second).all())


class TestUniformRandomInplaceOpSeedIsNotZero(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.seed = 10

    def test_uniform_random_inplace_op_seed_is_not_zero(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(seed=self.seed)
        tensor_data_first = tensor.numpy()
        tensor.uniform_(seed=self.seed)
        tensor_data_second = tensor.numpy()
        self.assertTrue((tensor_data_first == tensor_data_second).all())


class TestUniformRandomInplaceOpWithinRange(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.min = -2
        self.max = 1
        self.seed = 10

    def test_uniform_random_inplace_op_within_range(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(min=self.min, max=self.max, seed=self.seed)
        tensor_data = tensor.numpy()
        self.assertTrue((tensor_data > self.min).all() and
                        (tensor_data < self.max).all())


class TestUniformRandomInplaceOpShape(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_shape(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_()
        tensor_shape_np = np.array(tensor.shape)
        origin_shape = np.array(self.shape)
        self.assertTrue((tensor_shape_np == origin_shape).all())


class TestUniformRandomInplaceOpDistribution(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)
        self.min = -3
        self.max = 5
        self.seed = 10
        self.bins = 100

    def test_uniform_random_inplace_op_distribution(self):
        tensor = paddle.ones(self.shape)
        tensor.uniform_(self.min, self.max, self.seed)

        hist, _ = np.histogram(tensor.numpy()[0], bins=self.bins)
        prob = hist / float(self.shape[0])
        prob_expect = np.ones((self.bins, )) / float(self.bins)
        self.assertTrue(np.allclose(prob, prob_expect, rtol=0, atol=1e-2))


class TestUniformRandomInplaceOpError(unittest.TestCase):
    def setUp(self):
        self.shape = (1000, 784)

    def test_uniform_random_inplace_op_error(self):
        def test_attr_error():
            tensor = paddle.ones(self.shape)
            tensor.uniform_(shape=self.shape, min=-2, max=2)

        self.assertRaises(TypeError, test_attr_error)


if __name__ == '__main__':
    unittest.main()
