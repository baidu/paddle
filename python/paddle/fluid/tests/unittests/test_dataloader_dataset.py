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

from __future__ import division

import sys
import unittest
import numpy as np

import paddle
import paddle.vision.transforms as transforms
import paddle.fluid as fluid
from paddle.io import *


class TestDatasetAbstract(unittest.TestCase):
    def test_main(self):
        dataset = Dataset()
        try:
            d = dataset[0]
            self.assertTrue(False)
        except NotImplementedError:
            pass

        try:
            l = len(dataset)
            self.assertTrue(False)
        except NotImplementedError:
            pass


class TestDatasetWithDiffOutputPlace(unittest.TestCase):
    def get_dataloader(self, num_workers):
        dataset = paddle.vision.datasets.MNIST(
            mode='test', transform=transforms.ToTensor())
        loader = paddle.io.DataLoader(
            dataset, batch_size=32, num_workers=num_workers, shuffle=True)
        return loader

    def run_check_on_cpu(self):
        paddle.set_device('cpu')
        loader = self.get_dataloader(0)
        for image, label in loader:
            self.assertTrue(image.place.is_cpu_place())
            self.assertTrue(label.place.is_cpu_place())
            break

    def test_single_process(self):
        self.run_check_on_cpu()
        if paddle.is_compiled_with_cuda():
            # Get (image, label) tuple from MNIST dataset
            # - the image is on CUDAPlace, label is on CPUPlace
            paddle.set_device('gpu')
            loader = self.get_dataloader(0)
            for image, label in loader:
                self.assertTrue(image.place.is_gpu_place())
                self.assertTrue(label.place.is_cuda_pinned_place())
                break

    def test_multi_process(self):
        # DataLoader with multi-process mode is not supported on MacOs and Windows currently
        if sys.platform != 'darwin' and sys.platform != 'win32':
            self.run_check_on_cpu()
            if paddle.is_compiled_with_cuda():
                # Get (image, label) tuple from MNIST dataset
                # - the image and label are on CPUPlace
                paddle.set_device('gpu')
                loader = self.get_dataloader(1)
                for image, label in loader:
                    self.assertTrue(image.place.is_cuda_pinned_place())
                    self.assertTrue(label.place.is_cuda_pinned_place())
                    break


if __name__ == '__main__':
    unittest.main()
