# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2021 NVIDIA Corporation.  All rights reserved.
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
import threading, time
import paddle
from paddle.fluid.contrib import sparsity
import numpy as np


class TestASPUtils(unittest.TestCase):
    def test_get_check_method(self):
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(sparsity.MaskAlgo.MASK_1D),
            sparsity.CheckMethod.CHECK_1D)
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(
                sparsity.MaskAlgo.MASK_2D_GREEDY),
            sparsity.CheckMethod.CHECK_2D)
        self.assertEqual(
            sparsity.CheckMethod.get_checking_method(
                sparsity.MaskAlgo.MASK_2D_BEST), sparsity.CheckMethod.CHECK_2D)

    def test_density(self):
        x = np.array([[1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertEqual(sparsity.calculate_density(x), 0.56)
        x[:, 0] = 0.0
        self.assertEqual(sparsity.calculate_density(x), 0.4)

    def test_check_mask_1d(self):
        x = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [1.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertTrue(sparsity.check_mask_1d(x, 2, 4))
        self.assertFalse(sparsity.check_mask_1d(x, 3, 4))
        self.assertTrue(sparsity.check_mask_1d(x, 2, 5))
        self.assertFalse(sparsity.check_mask_1d(x, 3, 5))
        self.assertTrue(sparsity.check_mask_1d(x, 3, 6))
        self.assertFalse(sparsity.check_mask_1d(x, 4, 6))

    def test_get_mask_1d(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_1d(x, 2, 4)
            self.assertTrue(sparsity.check_mask_1d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_1d(x, 2, 4)
            self.assertTrue(sparsity.check_mask_1d(x, 2, 4))

    def test_check_mask_2d(self):
        x = np.array([[1.0, 0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0, 1.0]])
        self.assertTrue(sparsity.check_mask_2d(x, 2, 4))
        self.assertFalse(sparsity.check_mask_2d(x, 3, 4))
        self.assertTrue(sparsity.check_mask_2d(x, 2, 5))
        self.assertFalse(sparsity.check_mask_2d(x, 3, 5))
        self.assertTrue(sparsity.check_mask_2d(x, 3, 6))
        self.assertFalse(sparsity.check_mask_2d(x, 4, 6))

    def test_get_mask_2d_greedy(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_2d_greedy(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_2d_greedy(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

    def test_get_mask_2d_best(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5, 5))
            x = sparsity.get_mask_2d_best(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

            x = np.random.randn(5, 4)
            x = sparsity.get_mask_2d_best(x, 2, 4)
            self.assertTrue(sparsity.check_mask_2d(x, 2, 4))

    def test_threadsafe_valid_2d_patterns(self):
        def get_reference(m=4, n=2):
            from itertools import permutations

            patterns = np.zeros(m)
            patterns[:n] = 1
            patterns = list(set(permutations(patterns.tolist())))
            patterns = patterns + patterns
            patterns = np.asarray(list(set(permutations(patterns, m))))

            valid = ((patterns.sum(axis=1) <= n).sum(axis=1) == m
                     ).nonzero()[0].reshape(-1)
            valid_patterns = np.empty((valid.shape[0], m, m))
            valid_patterns[:] = patterns[valid[:]]
            return valid_patterns

        for _ in range(4):
            computing_thread = threading.Thread(
                target=paddle.fluid.contrib.sparsity.utils.
                _compute_valid_2d_patterns,
                args=(2, 4))
            computing_thread.start()
        time.sleep(3)
        patterns_map = paddle.fluid.contrib.sparsity.utils._valid_2d_patterns
        reference_patterns = get_reference()
        reference_key = '4_2'

        self.assertTrue(reference_key in patterns_map)
        self.assertTrue(len(patterns_map) == 1)
        self.assertTrue((reference_patterns == patterns_map[reference_key]).all(
        ))

    def test_check_sparsity(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5))
            x_2d = x.reshape(1, x.shape[0])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5))
            x_2d = x
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5, 5))
            x_2d = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

            x = np.random.randint(10, size=(5, 5, 5, 5))
            x_2d = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            self.__test_1D_2D_sparsity_checking_methods(x_2d)

    def test_create_mask(self):
        for _ in range(10):
            x = np.random.randint(10, size=(5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

            x = np.random.randint(10, size=(5, 5, 5, 5))
            self.__test_1D_2D_sparse_mask_generation_methods(x)

    def __test_1D_2D_sparsity_checking_methods(self, x_2d):
        mask = sparsity.get_mask_1d(x_2d, 2, 4)
        self.assertEqual(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_1D, n=2, m=4),
            sparsity.check_mask_1d(mask, 2, 4))
        mask = sparsity.get_mask_2d_best(x_2d, 2, 4)
        self.assertEqual(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4),
            sparsity.check_mask_2d(mask, 2, 4))

    def __test_1D_2D_sparse_mask_generation_methods(self, x):
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_1D, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_1D, n=2, m=4))
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_2D_GREEDY, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4))
        mask = sparsity.create_mask(
            x, func_name=sparsity.MaskAlgo.MASK_2D_BEST, n=2, m=4)
        self.assertTrue(
            sparsity.check_sparsity(
                mask, func_name=sparsity.CheckMethod.CHECK_2D, n=2, m=4))
