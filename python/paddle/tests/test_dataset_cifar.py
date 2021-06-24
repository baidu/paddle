# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.vision.datasets import Cifar10, Cifar100


class TestCifar10Train(unittest.TestCase):
    def test_main(self):
        cifar = Cifar10(mode='train')
        self.assertTrue(len(cifar) == 50000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 50000)
        data, label = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)


class TestCifar10Test(unittest.TestCase):
    def test_main(self):
        cifar = Cifar10(mode='test')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 9)

        # test cv2 backend
        cifar = Cifar10(mode='test', backend='cv2')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 99)

        with self.assertRaises(ValueError):
            cifar = Cifar10(mode='test', backend=1)


class TestCifar100Train(unittest.TestCase):
    def test_main(self):
        cifar = Cifar100(mode='train')
        self.assertTrue(len(cifar) == 50000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 50000)
        data, label = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 99)


class TestCifar100Test(unittest.TestCase):
    def test_main(self):
        cifar = Cifar100(mode='test')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        data = np.array(data)
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 99)

        # test cv2 backend
        cifar = Cifar100(mode='test', backend='cv2')
        self.assertTrue(len(cifar) == 10000)

        # traversal whole dataset may cost a
        # long time, randomly check 1 sample
        idx = np.random.randint(0, 10000)
        data, label = cifar[idx]
        self.assertTrue(len(data.shape) == 3)
        self.assertTrue(data.shape[2] == 3)
        self.assertTrue(data.shape[1] == 32)
        self.assertTrue(data.shape[0] == 32)
        self.assertTrue(len(label.shape) == 1)
        self.assertTrue(label.shape[0] == 1)
        self.assertTrue(0 <= int(label) <= 99)

        with self.assertRaises(ValueError):
            cifar = Cifar100(mode='test', backend=1)


if __name__ == '__main__':
    unittest.main()
