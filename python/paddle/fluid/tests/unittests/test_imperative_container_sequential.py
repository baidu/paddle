# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import numpy as np


class TestImperativeContainerSequential(unittest.TestCase):
    def test_sequential(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with fluid.dygraph.guard():
            data = fluid.dygraph.to_variable(data)
            model1 = fluid.dygraph.Sequential(
                fluid.Linear(10, 1), fluid.Linear(1, 2))
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 2])
            model1[1] = fluid.Linear(1, 3)
            res1 = model1(data)
            self.assertListEqual(res1.shape, [5, 3])
            loss1 = fluid.layers.reduce_mean(res1)
            loss1.backward()

            l1 = fluid.Linear(10, 1)
            l2 = fluid.Linear(1, 3)
            model2 = fluid.dygraph.Sequential(('l1', l1), ('l2', l2))
            self.assertEqual(len(model2), 2)
            res2 = model2(data)
            self.assertTrue(l1 is model2.l1)
            self.assertListEqual(res2.shape, res1.shape)
            self.assertEqual(len(model1.parameters()), len(model2.parameters()))
            del model2['l2']
            self.assertEqual(len(model2), 1)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 1])
            model2.add_sublayer('l3', fluid.Linear(1, 3))
            model2.add_sublayer('l4', fluid.Linear(3, 4))
            self.assertEqual(len(model2), 3)
            res2 = model2(data)
            self.assertListEqual(res2.shape, [5, 4])

            loss2 = fluid.layers.reduce_mean(res2)
            loss2.backward()


class TestImperativeContainerSequentialCase1(unittest.TestCase):
    def test_sequential(self):
        data = np.random.uniform(-1, 1, [5, 10]).astype('float32')
        with fluid.dygraph.guard():
            data = fluid.dygraph.to_variable(data)
            model1 = fluid.dygraph.Sequential(
                fluid.Linear(10, 1),
                fluid.Linear(1, 2), fluid.Linear(2, 2), fluid.Linear(2, 2))
            actual_model = model1[:3]
            res1 = actual_model(data)
            res1 = model1[-1](res1)
            self.assertListEqual(res1.shape, [5, 2])
            loss1 = fluid.layers.reduce_mean(res1)
            loss1.backward()


class TestImperativeContainerSequentialCase2(unittest.TestCase):
    def test_sequential(self):
        with fluid.dygraph.guard():
            model1 = fluid.dygraph.Sequential(
                fluid.Linear(10, 1), fluid.Linear(1, 2))
            with self.assertRaises(IndexError):
                layer = model1[2]


if __name__ == '__main__':
    unittest.main()
